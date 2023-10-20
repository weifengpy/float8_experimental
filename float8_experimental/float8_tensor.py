from typing import Any, Dict

import torch
from float8_experimental.float8_utils import tensor_to_amax, to_fp8_saturated
from torch._subclasses.fake_tensor import is_fake

aten = torch.ops.aten

FLOAT8_OPS_TABLE: Dict[Any, Any] = {}


def implements(aten_ops):
    """Register aten ops to the float8 op table"""

    def decorator(func):
        for op in aten_ops:
            FLOAT8_OPS_TABLE[op] = func
        return func

    return decorator


@implements(
    [
        aten.view.default,
        aten._unsafe_view.default,
        aten.t.default,
        aten.as_strided.default,
        aten.clone.default,
        aten.detach.default
    ]
)
def float8_desugar_op(aten_op, args, kwargs=None):
    assert is_fake(args[0]), "Float8Tensor.__torch_dispatch__ for user code is not supported"
    new_data = aten_op(args[0]._data, *args[1:], **kwargs)
    return Float8Tensor(new_data, args[0]._scale, args[0]._orig_dtype)

@implements(
    [aten.is_same_size.default]
)
def float8_is_same_size(aten_op, args, kwargs=None):
    return args[0].shape == args[1].shape

class ToFloat8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion to fp8
    """

    @staticmethod
    def forward(
        ctx,
        tensor,
        scale: float = None,
        float8_dtype=torch.float8_e4m3fn,
        amax_buffer=None,
    ):
        # In TransformerEngine, the casts to float8 are fused with calculating
        # the new amax value. In this codebase, the eager mode code for those
        # two things is colocated in this function. We expect PT2.0 to fuse it
        # for us.
        if amax_buffer is not None:
            amax_buffer.fill_(tensor_to_amax(tensor))

        tensor_scaled = tensor * scale
        bits_fp8 = to_fp8_saturated(tensor_scaled, float8_dtype)
        return Float8Tensor(bits_fp8, scale, tensor.dtype)

    @staticmethod
    def backward(ctx, g):
        if isinstance(g, Float8Tensor):
            return g.to_original_precision(), None, None, None
        else:
            return g, None, None, None


class FromFloat8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion from fp8
    """

    @staticmethod
    def forward(ctx, tensor):
        return tensor._data.to(tensor._orig_dtype) / tensor._scale

    @staticmethod
    def backward(ctx, g):
        return Float8Tensor.to_float8(g), None, None


class Float8Tensor(torch.Tensor):
    """
    A Python-only Float8 tensor subclass.  Contains:
    * `_data`: the underlying e4m3 or e5m2 data
    * `_scale`: the scale used to scale the original fp32 tensor. We multiply
      by scale to go from fp32 range to fp8 range, and divide by scale to go
      from fp8 range to fp32 range.
    * `_orig_dtype`: the original dtype of the tensor used to create this
      tensor.

    Intended usage of this abstraction:
    1. to bundle raw data + fp8 metadata together for easy passing through
       Python PyTorch systems.
    2. Float8-aware user code can use the private fields on these tensors
       to call into float8 operations.
    3. Float8-agnostic user code can use these tensors as is - they will
       convert to original precision in `__torch_dispatch__`.
    """
    _data: torch.Tensor
    _scale: torch.Tensor
    _orig_dtype: torch.dtype
    __slots__ = ["_data", "_scale", "_orig_dtype"]

    def __new__(cls, data: torch.Tensor, scale: torch.Tensor, orig_dtype: torch.dtype):
        assert scale.numel() == 1

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=orig_dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data
        self._scale = scale
        self._orig_dtype = orig_dtype

        return self

    def __repr__(self):
        return f"Float8Tensor(dtype={self._data.dtype}, scale={self._scale}, as_orig_prec={self.to_original_precision()}"

    def __tensor_flatten__(self):
        ctx = {
            "_scale": self._scale,
            "_orig_dtype": self._orig_dtype,
        }
        # return ("_data", "_scale"), (self._orig_dtype)
        return ["_data"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, metadata):
        assert len(inner_tensors) == 1
        # return Float8Tensor(tensors["_data"], tensors["_scale"], metadatas[0])
        return Float8Tensor(inner_tensors["_data"], metadata["_scale"], metadata["_orig_dtype"])

    def to_original_precision(self):
        return FromFloat8ConstrFunc.apply(self)

    @classmethod
    def to_float8(cls, tensor, scale, float8_dtype, amax_buffer=None):
        return ToFloat8ConstrFunc.apply(tensor, scale, float8_dtype, amax_buffer)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # 1. tracing through __torch_function__ logic is not supported yet in
        # PT2.0, so we explicitly disallow it here for callsites from user code.
        # 2. We do need to handle a couple of ops in order for
        # TorchDynamo tracing to succeed.
        if func in FLOAT8_OPS_TABLE:
            return FLOAT8_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(
            f"attempting to run {func}, this is not supported")

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl


# In order for dynamo to successfuly trace our tensor subclass, we need
# to be able to represent it in the graph.
torch._dynamo.allow_in_graph(Float8Tensor)
