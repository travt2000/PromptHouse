"""Developer tool: derive a BF16 single-file checkpoint from a scaled-FP8 one (weight = fp8 * weight_scale).

Used to exercise the BF16 loading path and to cross-check FP8 execution numerically. The output is NOT an
official BF16 release (it carries the FP8 quantization error); name it accordingly.

    .venv\\Scripts\\python.exe scripts\\make_bf16_reference.py IN.safetensors OUT.safetensors
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)


def main(src: str, dst: str) -> None:
    out = {}
    with safe_open(src, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            if k.endswith((".weight_scale", ".comfy_quant", ".input_scale")):
                continue
            t = f.get_tensor(k)
            if t.dtype in FP8:
                s = f.get_tensor(k[: -len(".weight")] + ".weight_scale").float()
                s = s.reshape(-1, 1) if s.numel() > 1 else s.reshape(())
                t = (t.float() * s).to(torch.bfloat16)
            elif t.is_floating_point():
                t = t.to(torch.bfloat16)
            out[k] = t.contiguous()
    save_file(out, dst, metadata={"format": "pt", "derived_from": Path(src).name, "note": "BF16 dequantized from scaled FP8; not an official BF16 release"})
    print(f"wrote {dst}: {len(out)} tensors, {sum(t.numel() * t.element_size() for t in out.values()) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
