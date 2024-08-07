# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from float8_experimental.float8_dynamic_utils import cast_to_float8_e4m3_dynamic

from float8_experimental.float8_tensor import (
    Float8Tensor,
    merge_mm_configs,
    ScaledMMConfig,
)

from float8_experimental.float8_utils import e4m3_dtype, EPS
from torch._prims_common import suggest_memory_format


@torch.no_grad()
def precompute_float8_dynamic_scale_for_fsdp(module: nn.Module) -> None:
    """
    Calculate scale dynamically for all float8 parameters.
    This should be run after the optimizer step. It performs a single all-reduce to compute the
    scales for all float8 weights.
    Example usage:
        model(input).sum().backward()
        optim.step()
        precompute_float8_dynamic_scale_for_fsdp(model)
    """
    from float8_experimental.float8_linear import Float8Linear, TensorScalingType
    from torch.distributed._tensor import DTensor

    if any(
        isinstance(m, Float8Linear) and m.scaling_type_w is TensorScalingType.DELAYED
        for m in module.modules()
    ):
        raise NotImplementedError("Only supports delayed scaling")
    float8_linears: List[Float8Linear] = [
        m
        for m in module.modules()
        if isinstance(m, Float8Linear)
        and isinstance(m.weight, DTensor)
        and isinstance(m.weight._local_tensor, WeightWithDynamicFloat8CastTensor)
    ]
    weights: List[DTensor] = [float8_linear.weight for float8_linear in float8_linears]

    if not weights:
        return

    # inf-norm is equivalent to max(abs(w))
    max_weights = torch._foreach_norm(weights, ord=math.inf)  # Partial
    amax_tensor = torch.vstack(max_weights)  # Partial
    # clamp is dispatched through DTensor
    # it will issue a single all-reduce
    amax_tensor = torch.clamp(amax_tensor, EPS)  # Replicate
    scale_tensor = torch.finfo(torch.float8_e4m3fn).max / amax_tensor  # Replicate
    if amax_tensor.dtype is torch.float16:
        scale_tensor = torch.clamp(scale_tensor, max=torch.finfo(torch.float16).max)
    scales = torch.split(scale_tensor, 1)  # Replicate
    for scale, float8_linear in zip(scales, float8_linears):
        float8_linear.weight._local_tensor._precomputed_scale = (
            scale._local_tensor.squeeze()
        )


# FSDP pads its local tensor on dim-0. The subclass should be preserved such
# that the padded local tensor (and any transformations like copying to GPU)
# is of the subclass as well.
_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
}


class WeightWithDynamicFloat8CastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        mm_config: ScaledMMConfig,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        mm_config: ScaledMMConfig,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        self._tensor = tensor
        self._mm_config = mm_config
        # for dynamic scaling
        # `precompute_float8_dynamic_scale_for_fsdp` calculates scales
        # for all float8 parameters after optimizer step
        self._precomputed_scale = precomputed_scale

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDynamicFloat8CastTensor(
                args[0]._tensor, args[0]._mm_config
            )
        mm_config: Optional[ScaledMMConfig] = None

        def unwrap(t):
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._mm_config
            else:
                mm_config = merge_mm_configs(mm_config, t._mm_config)
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithDynamicFloat8CastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor, lambda x: WeightWithDynamicFloat8CastTensor(x, mm_config), out
        )

    def __tensor_flatten__(self):
        if self._precomputed_scale:
            return ["_tensor", "_precomputed_scale"], self._mm_config
        else:
            return ["_tensor"], self._mm_config

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        mm_config = flatten_spec
        return WeightWithDynamicFloat8CastTensor(
            inner_tensors["_tensor"],
            mm_config,
            getattr(inner_tensors, "_precomputed_scale", None),
        )

    def __repr__(self):
        return f"WeightWithDynamicFloat8CastTensor(tensor={self._tensor}, mm_config={self._mm_config})"

    def fsdp_pre_all_gather(self, mesh):
        if self._precomputed_scale is not None:
            float8_tensor = Float8Tensor.to_float8(
                self._tensor,
                self._precomputed_scale,
                torch.float8_e4m3fn,
                mm_config=self._mm_config,
            )
        else:
            float8_tensor = cast_to_float8_e4m3_dynamic(
                self._tensor, self._mm_config, reduce_amax=True
            )
        return (float8_tensor._data,), (float8_tensor._scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            assert isinstance(out, Float8Tensor), f"{type(out)}"
            out._scale = scale
            return
        return Float8Tensor(data, scale, param_dtype, self._mm_config), (data,)


class WeightWithDelayedFloat8CastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        amax_buffer: torch.Tensor,
        amax_history_buffer: torch.Tensor,
        scale_buffer: torch.Tensor,
        mm_config: ScaledMMConfig,
        is_amax_initialized: bool,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        amax_buffer: torch.Tensor,
        amax_history_buffer: torch.Tensor,
        scale_buffer: torch.Tensor,
        mm_config: ScaledMMConfig,
        is_amax_initialized: bool,
    ):
        self._tensor = tensor
        self._amax_buffer = amax_buffer
        self._amax_history_buffer = amax_history_buffer
        self._scale_buffer = scale_buffer
        self._mm_config = mm_config

        # Note: is_amax_initialized is not a buffer to avoid data dependent
        # control flow visible to dynamo
        # TODO(future PR): add serialization for this flag
        self.is_amax_initialized = is_amax_initialized

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDelayedFloat8CastTensor(
                args[0]._tensor,
                args[0]._amax_buffer,
                args[0]._amax_history_buffer,
                args[0]._scale_buffer,
                args[0]._mm_config,
                args[0].is_amax_initialized,
            )
        mm_config: Optional[ScaledMMConfig] = None
        amax_buffer: Optional[torch.Tensor] = None
        amax_history_buffer: Optional[torch.Tensor] = None
        scale_buffer: Optional[torch.Tensor] = None
        is_amax_initialized: Optional[bool] = None

        def unwrap(t):
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._mm_config
            else:
                mm_config = merge_mm_configs(mm_config, t._mm_config)
            nonlocal amax_buffer
            if amax_buffer is None:
                amax_buffer = t._amax_buffer
            nonlocal amax_history_buffer
            if amax_history_buffer is None:
                amax_history_buffer = t._amax_history_buffer
            nonlocal scale_buffer
            if scale_buffer is None:
                scale_buffer = t._scale_buffer
            nonlocal is_amax_initialized
            if is_amax_initialized is None:
                is_amax_initialized = t.is_amax_initialized
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithDelayedFloat8CastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: WeightWithDelayedFloat8CastTensor(
                x,
                amax_buffer,
                amax_history_buffer,
                scale_buffer,
                mm_config,
                is_amax_initialized,
            ),
            out,
        )

    def __tensor_flatten__(self):
        return (
            [
                "_tensor",
                "_amax_buffer",
                "_amax_history_buffer",
                "_scale_buffer",
            ],
            {
                "mm_config": self._mm_config,
                "is_amax_initialized": self.is_amax_initialized,
            },
        )

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return WeightWithDelayedFloat8CastTensor(
            inner_tensors["_tensor"],
            inner_tensors["_amax_buffer"],
            inner_tensors["_amax_history_buffer"],
            inner_tensors["_scale_buffer"],
            metadata["mm_config"],
            metadata["is_amax_initialized"],
        )

    def __repr__(self):
        return f"WeightWithDelayedFloat8CastTensor(tensor={self._tensor}, amax_buffer={self._amax_buffer}, scale_buffer={self._scale_buffer}, mm_config={self._mm_config})"

    def fsdp_pre_all_gather(self, mesh):
        # initialize if needed
        # TODO(before land): ensure settings are consistent between Float8Linear and here
        if not self.is_amax_initialized:
            from float8_experimental.float8_linear import (
                _maybe_initialize_amaxes_scales_for_float8_cast,
            )

            _maybe_initialize_amaxes_scales_for_float8_cast(
                self._tensor,
                self._amax_buffer,
                self._amax_history_buffer,
                self._scale_buffer,
                "max",  # TODO(before land): read this from parent
                e4m3_dtype,
                self.is_amax_initialized,
                reduce_amax=True,
            )
            self.is_amax_initialized = True

        # this will:
        # 1. cast the tensor to float8 using `_scale_buffer`
        # 2. populate `_amax_buffer` inplace
        # TODO(future PR): clean up all the casting functions and clearly
        # separate dynamic vs delayed, tech debt has accumulated
        float8_tensor = Float8Tensor.to_float8(
            self._tensor,
            self._scale_buffer,
            e4m3_dtype,
            self._amax_buffer,
            self._mm_config,
        )
        return (float8_tensor._data,), (float8_tensor._scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            assert isinstance(out, Float8Tensor), f"{type(out)}"
            out._scale = scale
            return
        return Float8Tensor(data, scale, param_dtype, self._mm_config), (data,)
