# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
from enum import auto, Enum
from typing import List, Optional, Type

import float8_experimental.config as fp8_config

import torch
import torch.distributed as dist
import torch.nn as nn
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear import Float8Linear

from float8_experimental.float8_utils import amax_history_to_scale_stack
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.distributed_c10d import _get_default_group

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class LinearType(Enum):
    DELAYED = auto()
    DYNAMIC = auto()


REQUIRES_SYNC = {LinearType.DELAYED}


def get_float8_linear(
    linear_type: LinearType,
    linear_ref: torch.nn.Linear,
    emulate: bool = False,
    use_activation_hooks: bool = False,
):
    """Returns a Float8Linear module of the given type, initialized from linear_ref.
    Args:
        linear_type: The type of Float8Linear to return.
        linear_ref: The linear module to initialize from.
        emulate: Whether to emulate the fp8 matmul logic in float32.
        use_activation_hooks: Whether to use activation hooks for dynamic linear.
    """
    LINEAR_TYPE_MAP = {
        LinearType.DELAYED: Float8Linear,
        LinearType.DYNAMIC: Float8DynamicLinear,
    }
    if linear_type not in LINEAR_TYPE_MAP:
        raise ValueError(f"linear_type must be one of {LINEAR_TYPE_MAP.keys()}")
    if use_activation_hooks and linear_type != LinearType.DYNAMIC:
        raise ValueError("use_activation_hooks is only supported for dynamic linear")
    return LINEAR_TYPE_MAP[linear_type].from_float(
        copy.deepcopy(linear_ref),
        emulate=emulate,
        use_activation_hooks=use_activation_hooks,
    )


def linear_requires_sync(linear_type: LinearType):
    """Returns whether the given linear_type requires sync before forward."""
    return linear_type in REQUIRES_SYNC


def _update_history_with_new_amax(new_amax, amax_history):
    """
    Updates `amax_history` (the last N cur_amax values) inplace with the value
    of `new_amax`.
    """
    new_amax_history = torch.roll(amax_history, 1)
    new_amax_history[0] = new_amax
    amax_history.copy_(new_amax_history)


def _update_history_stack(
    new_amax: torch.Tensor, amax_history_stack: torch.Tensor
) -> torch.Tensor:
    """
    Updates `amax_history` (the last N cur_amax values) inplace with the value
    of `new_amax`.

    Args:
        new_amax (torch.Tensor): The new amax value to add to the history. (n_amaxes, 1)
        amax_history_stack (torch.Tensor): The history of amax values. (n_amaxes, history_length)
    """
    assert amax_history_stack.dim() == 2, "amax_history_stack must be 2D"
    assert new_amax.size(0) == amax_history_stack.size(
        0
    ), "new_amax must have the same size as the second dimension of amax_history_stack"
    new_amax_history_stack = torch.roll(amax_history_stack, 1, dims=1)
    new_amax_history_stack[:, 0] = new_amax.squeeze(-1)
    amax_history_stack.copy_(new_amax_history_stack)


def swap_linear_with_float8_linear(
    module: nn.Module,
    module_cls: Type[nn.Module],
    emulate: bool = False,
    skip_fqn_list: Optional[List[str]] = None,
) -> nn.Module:
    """
    Replaces all instances of ``torch.nn.Linear`` in ``module`` with instances
    of ``module_cls`` (either ``Float8Linear`` or ``Float8DynamicLinear``).

    Args:
        module (torch.nn.Module): Module to modify.
        module_cls (Union[Type[Float8Linear], Type[Float8DynamicLinear]]): Float8 linear class for the swap.
        emulate (bool, optional): Whether to emulate the fp8 matmul logic in fp32.
        skip_fqn_list (List[str], optional): If specified, a list of module FQNs to skip.
            Linear submodules of these skipped modules will also be skipped.
    """
    module_names_to_skip = set(skip_fqn_list or [])
    if isinstance(module, nn.Linear):
        if len(list(module.children())) > 0:
            raise AssertionError(
                f"Does not support a root nn.Linear with children: {module}"
            )
        return module_cls.from_float(module, emulate)

    # Mark all modules to skip as visited
    root_module = module
    visited_modules = {root_module}
    for module_name, module in root_module.named_modules():
        if module_name in module_names_to_skip:
            visited_modules.add(module)

    # Run a post-order traversal to swap linears
    def post_order_traversal(
        module: nn.Module, module_name: str, parent_module: Optional[nn.Module]
    ):
        nonlocal visited_modules
        for child_module_name, child_module in module.named_children():
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                post_order_traversal(child_module, child_module_name, module)
        if isinstance(module, nn.Linear):
            assert (
                parent_module is not None
            ), f"Linear root module should return early: {module}"
            setattr(parent_module, module_name, module_cls.from_float(module, emulate))

    post_order_traversal(root_module, "", None)
    # Without this explicit `del`, this set only gets deleted upon an explicit
    # garbage collection (not from when its refcount hits zero)
    del visited_modules
    return root_module


def get_float8_layers(model: torch.nn.Module):
    """Iterates through the model and returns all the Float8Linear layers.
    Args:
        model (torch.nn.Module): The model to look for Float8Linear layers in.
    """

    # Get all fp8 layers and tensors
    fp8_layers = [
        child for _, child in model.named_modules() if isinstance(child, Float8Linear)
    ]

    return fp8_layers


def sync_float8_amax_and_scale_history(model: torch.nn.Module, fp8_layers=None) -> None:
    """
    Manages the float8 amax and scale bookkeeping. In detail, it does the
    following:
    1. in distributed contexts, syncs amax values across workers
    2. adds the `amax` values to history
    3. calculates the scales to be used for next iteration
    4. sets the `amax_and_scale_synced` flag on the Float8Linear modules
       to signal that they have been synced

    TODO(future): design the UX for this (context manager, etc)

    PERFORMANCE NOTE:
        When you can it is much more efficient to call te get_float8_layers once a
        the beginning of the training loop and pass the result to this function.
        Because of how this interacts with torch.compile

    Args:
        model (torch.nn.Module): The model to track amaxes for
        fp8_layers (optional): If fp8_layers are provided, fp8_classes are ignored,
            and we loop over all fp8_layers to sync and update amax scale histories.
            Users can use get_float8_layers to get all fp8 layers.
    """
    if fp8_layers is None:
        fp8_layers = get_float8_layers(model)

    if len(fp8_layers) == 0:
        log.warn(
            "Calling sync_float8_amax_and_scale_history on a module with no Float8Linear layers"
        )
        return

    # Loop over all fp8 layers and grab the needed tensors
    fp8_amax_x_tensor_list = [None] * len(fp8_layers)
    fp8_amax_w_tensor_list = [None] * len(fp8_layers)
    fp8_amax_dL_dY_tensor_list = [None] * len(fp8_layers)

    fp8_x_amax_history_stack = [None] * len(fp8_layers)
    fp8_w_amax_history_stack = [None] * len(fp8_layers)
    fp8_dL_dY_amax_history_stack = [None] * len(fp8_layers)

    x_dtypes = set()
    scale_fn_recipes = set()

    for idx, child in enumerate(fp8_layers):
        fp8_amax_x_tensor_list[idx] = child.fp8_amax_x
        fp8_amax_w_tensor_list[idx] = child.fp8_amax_w
        fp8_amax_dL_dY_tensor_list[idx] = child.fp8_amax_dL_dY

        fp8_x_amax_history_stack[idx] = child.fp8_amax_history_x
        fp8_w_amax_history_stack[idx] = child.fp8_amax_history_w
        fp8_dL_dY_amax_history_stack[idx] = child.fp8_amax_history_dL_dY

        x_dtypes.add(child.last_seen_input_dtype)
        scale_fn_recipes.add(child.recipe.scale_fn_name)

    # TODO This way to get the activation dtype is not ideal
    assert len(x_dtypes) == 1, "All layers must have the same last seen input_dtype"
    x_dtype = next(iter(x_dtypes))

    assert len(scale_fn_recipes) == 1, "All layers must have the same scale_fn recipe"
    scale_fn_recipe = next(iter(scale_fn_recipes))

    assert (
        len(fp8_amax_x_tensor_list)
        == len(fp8_amax_w_tensor_list)
        == len(fp8_amax_dL_dY_tensor_list)
    ), "Mismatched lengths of amax tensors."

    if dist.is_initialized():
        # Combine all the amax tensors into one tensor and reduce it
        all_amax_tensors = torch.cat(
            fp8_amax_x_tensor_list + fp8_amax_w_tensor_list + fp8_amax_dL_dY_tensor_list
        )
        all_reduced_amax_tensor = all_reduce(
            all_amax_tensors, "MAX", _get_default_group()
        )
        (
            reduced_fp8_amax_tensor,
            reduced_fp8_amax_w_tensor,
            reduced_fp8_amax_dL_dY_tensor,
        ) = torch.split(all_reduced_amax_tensor, len(fp8_amax_x_tensor_list))

        # TODO foreach is not supported with AsyncCollectiveTensor
        for idx, child in enumerate(fp8_layers):
            child.fp8_amax_x.copy_(reduced_fp8_amax_tensor[idx])
            child.fp8_amax_w.copy_(reduced_fp8_amax_w_tensor[idx])
            child.fp8_amax_dL_dY.copy_(reduced_fp8_amax_dL_dY_tensor[idx])

    # We create two stacked tensors, one for the amax history and one for the current scales
    fp8_amax_x_tensors = torch.vstack(fp8_amax_x_tensor_list)
    fp8_amax_w_tensors = torch.vstack(fp8_amax_w_tensor_list)
    fp8_amax_dL_dY_tensors = torch.vstack(fp8_amax_dL_dY_tensor_list)

    fp8_x_amax_history_stack = torch.vstack(fp8_x_amax_history_stack)
    fp8_w_amax_history_stack = torch.vstack(fp8_w_amax_history_stack)
    fp8_dL_dY_amax_history_stack = torch.vstack(fp8_dL_dY_amax_history_stack)

    _update_history_stack(fp8_amax_x_tensors, fp8_x_amax_history_stack)
    _update_history_stack(fp8_amax_w_tensors, fp8_w_amax_history_stack)
    _update_history_stack(fp8_amax_dL_dY_tensors, fp8_dL_dY_amax_history_stack)

    # We are not reading the
    new_x_scales = amax_history_to_scale_stack(
        fp8_x_amax_history_stack, torch.float8_e4m3fn, x_dtype, scale_fn_recipe
    )
    new_w_scales = amax_history_to_scale_stack(
        fp8_w_amax_history_stack, torch.float8_e4m3fn, x_dtype, scale_fn_recipe
    )
    new_dL_dY_scales = amax_history_to_scale_stack(
        fp8_dL_dY_amax_history_stack, torch.float8_e5m2, x_dtype, scale_fn_recipe
    )

    # Iterate through the layers and update the scales, and set the flag to signal that the amaxes/scales are ready
    for idx, child in enumerate(fp8_layers):
        child.fp8_scale_x.copy_(new_x_scales[idx])
        child.fp8_scale_w.copy_(new_w_scales[idx])
        child.fp8_scale_dL_dY.copy_(new_dL_dY_scales[idx])

        # 4. set a flag to signal amaxes/scales are ready
        # We only update the flag if we know it will be checked by the modules
        if fp8_config.enable_amax_init:
            child.amax_and_scale_synced = True
