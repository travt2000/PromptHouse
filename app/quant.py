"""Weight-only FP8 linear layers for "scaled FP8" safetensors checkpoints.

Checkpoint format handled here (verified by inspecting the shipped files):
    <layer>.weight        float8_e4m3fn (or float8_e5m2), shape [out, in]
    <layer>.weight_scale  float32, scalar (per-tensor) or [out] / [out, 1] (per-output-channel)
    real_weight = weight.to(float32) * weight_scale

Execution: the FP8 weight stays FP8 in memory (CPU *and* GPU - 1 byte/parameter). At matmul time the
layer's weight is upcast to the activation dtype (BF16; FP8 E4M3/E5M2 values are exactly representable
in BF16) and the scale is applied to the matmul output:  y = (x @ W_fp8^T) * scale + bias.
This is mathematically identical to dequantize-then-matmul, needs only one layer's BF16 weight as a
transient buffer, and does not quantize activations (so "full_precision_matrix_mult" layers are honored
for every layer). FP8 tensor-core matmuls are NOT used.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}


class FP8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Real tensors are attached by load_state_dict(assign=True).
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device="meta"))
        self.register_buffer("weight_scale", torch.empty((), dtype=torch.float32, device="meta"))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.bfloat16, device="meta"))
        else:
            self.bias = None
        # Per-tensor scale cached as a Python float (set by finalize_fp8) so applying it never syncs the GPU.
        self.scale_value: float | None = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Scale may be scalar, [1], [out] or [out, 1]; adopt the checkpoint's shape before the size check.
        k = prefix + "weight_scale"
        if k in state_dict and state_dict[k].shape != self.weight_scale.shape:
            self.weight_scale = torch.empty(state_dict[k].shape, dtype=torch.float32, device="meta")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype)
        out = F.linear(x, w)
        del w
        if self.scale_value is not None:
            out = out * self.scale_value  # Python scalar -> applied in fp32 opmath, result stays x.dtype
        else:
            out = out * self.weight_scale.reshape(-1).to(out.dtype)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, weight=fp8(scaled), compute=activation dtype"


def finalize_fp8(root: nn.Module) -> None:
    """After weights are loaded on CPU: validate dtypes and cache per-tensor scales as Python floats."""
    for name, m in root.named_modules():
        if isinstance(m, FP8Linear):
            if m.weight.dtype not in FP8_DTYPES:
                raise TypeError(f"{name}.weight has dtype {m.weight.dtype}, expected FP8")
            if m.weight.device.type != "cpu":
                raise RuntimeError("finalize_fp8 must run while weights are on CPU")
            s = m.weight_scale
            if s.numel() == 1:
                m.scale_value = float(s.reshape(()).item())
            elif s.numel() == m.out_features:
                m.scale_value = None
            else:
                raise ValueError(f"{name}.weight_scale has unsupported shape {tuple(s.shape)}")


def replace_linears_with_fp8(root: nn.Module, quantized_module_names: set[str]) -> int:
    """Swap nn.Linear modules named in `quantized_module_names` for FP8Linear. Returns the count replaced."""
    replaced = 0
    for name in sorted(quantized_module_names):
        parent_name, _, child = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        old = getattr(parent, child)
        if not isinstance(old, nn.Linear):
            raise TypeError(f"Checkpoint marks {name!r} as FP8-quantized but the model has {type(old).__name__} there")
        setattr(parent, child, FP8Linear(old.in_features, old.out_features, bias=old.bias is not None))
        replaced += 1
    return replaced
