"""
Test numerics of single GPU vs FSDP of toy model. At a high level:
1. start with reference input and state dict for a single GPU model
2. run fw+bw+optim on single GPU, save the results
3. run fw+bw+optim with FSDP, save the results
4. verify that the outputs and state dicts after optim update match

later 1-4 can be repeated for fp16, various combinations of fp8, etc.
"""

import os

import fire

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

# set up float8 path
import context

from float8_linear import swap_linear_with_float8_linear

torch.manual_seed(0)

# assumes user is running the script from /data/users/{user}/float8_playground
data_dir = os.path.join(os.getcwd(), 'tmp')
input_fname = os.path.join(data_dir, 'input.pt')
sd_in_fname = os.path.join(data_dir, 'sd_in.pt')
sd_out_single_gpu_fname = os.path.join(data_dir, 'sd_out_single_gpu.pt')
sd_out_fsdp_fname = os.path.join(data_dir, 'sd_out_fsdp.pt')
output_single_gpu_fname = os.path.join(data_dir, 'output_single_gpu.pt')
output_fsdp_fname = os.path.join(data_dir, 'output_fsdp.pt')

B, M, K, N = 8, 1, 4, 4
lr = 0.01


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_model(K, N, is_fp8):
    m = nn.Sequential(
        nn.Linear(K, N),
        # torch.float8_e4m3fn is not serializeable yet, for now
        # force model output to be float so we can serialize it
        # TODO revert this once serialization works
        nn.ReLU(),
    )
    if is_fp8:
        swap_linear_with_float8_linear(m)
    return m

# taken from https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
# and modified
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    is_fp8, = args
    model = get_model(K, N, is_fp8=is_fp8).to(rank)
    model.load_state_dict(torch.load(sd_in_fname))
    model = FSDP(model)
    # Note: we need to multiply by world_size here to match single GPU 
    # optimizer update
    optimizer = torch.optim.SGD(model.parameters(), lr=lr * world_size)
    optimizer.zero_grad()

    ref_input_global = torch.load(input_fname)

    # basic distributed data sampling
    bsz_global = ref_input_global.shape[0]
    assert B % world_size == 0
    bsz_local_start = int(rank / world_size * B)
    bsz_local_end = int((rank + 1) / world_size * B)
    ref_input_local = ref_input_global[bsz_local_start:bsz_local_end].to(rank)

    y_local = model(ref_input_local)
    y_local.sum().backward()
    optimizer.step()

    # get global y
    y_global = [torch.zeros(*y_local.shape).to(rank) for r in range(world_size)]
    dist.all_gather(y_global, y_local)
    y_global = torch.cat(y_global, dim=0)
    if rank == 0:
        torch.save(y_global.cpu(), output_fsdp_fname)

    # get global state dict
    # https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
    dist.barrier()
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        cpu_state = model.state_dict()
    if rank == 0:
        torch.save(cpu_state, sd_out_fsdp_fname)

    cleanup()

def run(mode: str, is_fp8: bool):

    if mode == 'generate':
        # generate reference input
        ref_input = torch.randn(B, M, K)
        model = get_model(K, N, is_fp8=is_fp8)
        torch.save(ref_input, input_fname)
        torch.save(model.state_dict(), sd_in_fname)

    elif mode == 'single_gpu':
        ref_input = torch.load(input_fname)
        model = get_model(K, N, is_fp8=is_fp8)
        model.load_state_dict(torch.load(sd_in_fname))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer.zero_grad()

        y = model(ref_input)
        y.sum().backward()
        optimizer.step()
        torch.save(y, output_single_gpu_fname)
        torch.save(model.state_dict(), sd_out_single_gpu_fname)

    elif mode == 'fsdp':
        WORLD_SIZE = torch.cuda.device_count()
        args = (is_fp8,)
        mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)

    elif mode == 'analyze':
        y_single_gpu = torch.load(output_single_gpu_fname)
        y_fsdp = torch.load(output_fsdp_fname)
        torch.testing.assert_close(y_single_gpu, y_fsdp)
        print('output testing single_gpu vs FSDP success')

        sd_in = torch.load(sd_in_fname)
        sd_out_single_gpu = torch.load(sd_out_single_gpu_fname)
        sd_out_fsdp = torch.load(sd_out_fsdp_fname)
        for k, v1 in sd_out_single_gpu.items():

            v2 = sd_out_fsdp[k]
            if is_fp8:
                # Note: for fp8 single-node vs FSDP, we are not expected
                # to match the scale of the weight gradient. Because of this,
                # we also cannot match the weight gradient, and therefore
                # after the first optimizer update we won't be able to match
                # anything.
                #
                # Reasoning is the order of operations of calculating dL/dW:
                # a. single node:
                #    1. calculate dL_dW and s_dL_dW
                #    2. you're done
                # b. FSDP:
                #    1. calculate dL_dW and s_dL_dW of each slice
                #    2. reduce using summation
                #
                # a and b cannot always match because calculating the scale 
                # involves taking max(dL_dW), FSDP reduces the gradients, and 
                # max(abs(a), abs(b)) != max(abs(a + b))
                if ('weight' in k)  or ('dL_dW' in k):
                    pass
                else:
                    torch.testing.assert_close(v1, v2)
            else:
                torch.testing.assert_close(v1, v2)
        print('state dict testing single_gpu vs FSDP success')


if __name__ == '__main__':
    fire.Fire(run)