"""
This file defines the aten functions for float8. Today, all of these functions
are emulated. In the future, they should be calling NVIDIA's float8 kernels.
"""

import torch

from float8_experimental.float8_utils import (
    E4M3_MAX_POS,
    E5M2_MAX_POS,
    tensor_to_amax,
    to_fp8_saturated,
)
from torch.library import Library


def mm_float8_emulated(
    m1,  # input 1 data
    s1,  # input 1 scale
    m2,  # input 2 data
    s2,  # input 2 scale
    dtype3,  # output dtype
):
    # naive implementation: dq -> op -> q
    m1_fp32 = m1.float() / s1
    m2_fp32 = m2.float() / s2
    m3_fp32 = torch.mm(m1_fp32, m2_fp32)

    return m3_fp32.to(dtype3), tensor_to_amax(m3_fp32)


#
# ATen op placeholders
#

# Register the aten level functions we need.
# These are mostly placeholder and might need to be implemented in c++ as needed
lib = Library("aten", "FRAGMENT")

lib.define(
    "mm_float8_emulated(Tensor m1, Tensor s1, Tensor m2, Tensor s2, ScalarType dtype3) -> (Tensor, Tensor)"
)
lib.impl("mm_float8_emulated", mm_float8_emulated, "CPU")
lib.impl("mm_float8_emulated", mm_float8_emulated, "CUDA")


@torch.library.impl(lib, "mm_float8_emulated", "Meta")
def _mm_float8_emulated_meta(m1, s1, m2, s2, dtype3):
    out = torch.mm(m1.float(), m2.float()).to(dtype3)
    return out, torch.empty(1, device="meta")
