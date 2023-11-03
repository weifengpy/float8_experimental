import torch

from float8_experimental.float8_utils import (
    amax_history_to_scale,
    amax_to_scale,
    tensor_to_amax,
)


def _maybe_initialize_amaxes_scales_for_float8_cast(
    x,
    cur_amax,
    amax_history,
    scale,
    scale_fn_name,
    float8_dtype,
    is_initialized,
):
    """
    If x is about to be cast to `float8` and the amax buffers are not initialized,
    initializes them inplace.
    """
    if is_initialized:
        return
    with torch.no_grad():
        # Note: we need to enable distributed reduction here in order
        # to match numerics between single GPU and multi GPU code
        new_amax = tensor_to_amax(x, distributed_reduction=True)
        cur_amax.fill_(new_amax)
        amax_history[0] = new_amax
        new_scale = amax_history_to_scale(
            amax_history, float8_dtype, x.dtype, scale_fn_name
        )
        scale.copy_(new_scale)


def _update_history_with_new_amax(new_amax, amax_history):
    """
    Updates `amax_history` (the last N cur_amax values) inplace with the value
    of `new_amax`.
    """
    new_amax_history = torch.roll(amax_history, 1)
    new_amax_history[0] = new_amax
    amax_history.copy_(new_amax_history)


def swap_linear_with_float8_linear(
    model,
    module,
    emulate=False,
    skip_fqn_list=None,
    cur_fqn="",
):
    """
    Replaces all instances of torch.nn.Linear in the given model with module.

    Args:
        model (torch.nn.Module): The model to modify.
        module (Float8Linear): The Float8Linear module to use.
        emulate (bool, optional): Whether to emulate the fp8 matmul logic in float32.
        skip_fqn_list (List[str], optional): If specified, a list of FQNs to skip
        cur_fqn (str, optional): Current fqn, used to implement skip_fqn_list
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        new_fqn = name if cur_fqn == "" else f"{cur_fqn}.{name}"
        if ((skip_fqn_list is None) or (new_fqn not in skip_fqn_list)) and isinstance(
            child, torch.nn.Linear
        ):
            new_child = module.from_float(child, emulate)
            setattr(model, name, new_child)
        else:
            swap_linear_with_float8_linear(child, module, emulate)