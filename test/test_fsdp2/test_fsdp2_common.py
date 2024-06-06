import contextlib
from typing import List, Optional, Type, Union

import float8_experimental.config as config

import torch
import torch.distributed as dist
import torch.nn as nn
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear import Float8Linear
from float8_experimental.float8_linear_utils import (
    precompute_float8_amax,
    precompute_float8_amax_fused,
    precompute_float8_weights,
    sync_float8_amax_and_scale_history,
)


def check_parity_no_mp(
    test_cls,
    ref_model: nn.Module,
    ref_optim: torch.optim.Optimizer,
    fsdp_model: nn.Module,
    fsdp_optim: torch.optim.Optimizer,
    local_inp: torch.Tensor,
    module_cls: Type,
    pre_compute: Optional[Union[str, None]],
):
    for iter_idx in range(10):
        losses: List[torch.Tensor] = []
        for model, optim in ((ref_model, ref_optim), (fsdp_model, fsdp_optim)):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            losses.append(model(local_inp).sum())
            losses[-1].backward()
            if model is ref_model:
                for param in model.parameters():
                    dist.all_reduce(param.grad)
                    param.grad.div_(dist.get_world_size())
            if module_cls is Float8Linear:
                sync_float8_amax_and_scale_history(model)
            optim.step()
            if module_cls is Float8DynamicLinear and model is fsdp_model:
                if pre_compute is None:
                    pass
                elif pre_compute == "cast":
                    precompute_float8_weights(model)
                elif pre_compute == "amax":
                    precompute_float8_amax(model)
                elif pre_compute == "amax_fused":
                    precompute_float8_amax_fused(model)
        test_cls.assertEqual(losses[0], losses[1])


def check_parity_bf16_mp(
    test_cls,
    ref_model: nn.Module,
    ref_model_bf16: nn.Module,
    ref_optim: torch.optim.Optimizer,
    fsdp_model: nn.Module,
    fsdp_optim: torch.optim.Optimizer,
    local_inp: torch.Tensor,
    module_cls: Type,
):
    for iter_idx in range(10):
        losses: List[torch.Tensor] = []
        for model, optim in (
            (ref_model_bf16, ref_optim),
            (fsdp_model, fsdp_optim),
        ):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            losses.append(model(local_inp).sum())
            losses[-1].backward()
            if model is ref_model_bf16:
                for param_bf16, param_fp32 in zip(
                    ref_model_bf16.parameters(), ref_model.parameters()
                ):
                    dist.all_reduce(param_bf16.grad)
                    param_bf16.grad.div_(dist.get_world_size())
                    param_fp32.grad = param_bf16.grad.float()
                    param_bf16.grad = None
            if module_cls is Float8Linear:
                sync_float8_amax_and_scale_history(model)
            optim.step()
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_bf16.detach().copy_(param_fp32)
        test_cls.assertEqual(losses[0], losses[1])


@contextlib.contextmanager
def set_enable_fsdp_fp8_all_gather(enable_fsdp_fp8_all_gather: bool):
    prev = config.enable_fsdp_fp8_all_gather
    dist.barrier()
    config.enable_fsdp_fp8_all_gather = enable_fsdp_fp8_all_gather
    try:
        yield
    finally:
        dist.barrier()
        config.enable_fsdp_fp8_all_gather = prev
