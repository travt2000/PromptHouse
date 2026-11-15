import torch

from app.quant import FP8Linear, finalize_fp8


def _make(out_f=64, in_f=32, per_channel=False):
    w = torch.randn(out_f, in_f) * 0.05
    scale = (w.abs().amax(dim=1) / 448.0) if per_channel else (w.abs().max() / 448.0)
    q = (w / (scale[:, None] if per_channel else scale)).to(torch.float8_e4m3fn)
    m = FP8Linear(in_f, out_f)
    m.load_state_dict({"weight": q, "weight_scale": scale.float()}, assign=True)
    finalize_fp8(m)
    deq = q.float() * (scale[:, None] if per_channel else scale)
    return m, deq


def test_fp8_linear_matches_dequantized_matmul_per_tensor():
    m, deq = _make()
    x = torch.randn(5, 32, dtype=torch.bfloat16)
    ref = (x.float() @ deq.T)
    out = m(x).float()
    assert m.weight.dtype == torch.float8_e4m3fn  # weights stay FP8
    assert torch.allclose(out, ref, rtol=2e-2, atol=2e-2)


def test_fp8_linear_per_channel():
    m, deq = _make(per_channel=True)
    assert m.scale_value is None
    x = torch.randn(3, 32, dtype=torch.bfloat16)
    assert torch.allclose(m(x).float(), x.float() @ deq.T, rtol=3e-2, atol=3e-2)
