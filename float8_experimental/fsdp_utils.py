import math
import warnings
from typing import List

import torch
import torch.nn as nn
from float8_experimental.float8_dynamic_utils import WeightWithDynamicFloat8CastTensor
from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_linear_utils import linear_requires_sync
from float8_experimental.float8_utils import EPS


def precompute_float8_amax_for_fsdp(module: nn.Module) -> None:
    """
    Calculate amax for all float8 parameters after optimizer step
    It performs a single all-reduce instead of many all-reduces for each parameter
    Exmaple usage:
        model(input).sum().backward()
        optim.step()
        precompute_float8_amax_for_fsdp(model)
    """
    from torch.distributed._tensor import DTensor

    if any(
        isinstance(m, Float8Linear)
        and linear_requires_sync(
            m.scaling_type_x, m.scaling_type_w, m.scaling_type_dL_dY
        )
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

    def compute_amaxes(weights: List[DTensor]):
        # inf-norm is equivalent to max(abs(w))
        max_weights = torch._foreach_norm(weights, ord=math.inf)  # Partial
        amax_tensor = torch.vstack(max_weights)  # Partial
        # clamp is dispatched through DTensor
        # it will issue a single all-reduce
        amax_tensor = torch.clamp(amax_tensor, EPS)  # Replicate
        amaxes = torch.split(amax_tensor, 1)  # Replicate
        return amaxes

    if weights:
        amaxes = compute_amaxes(weights)
        for amax, float8_linear in zip(amaxes, float8_linears):
            float8_linear.weight._local_tensor._precomputed_amax = amax._local_tensor
    else:
        warnings.warn(
            "Calling precompute_float8_weights without any weights using FSDP fp8 all-gather!"
        )
