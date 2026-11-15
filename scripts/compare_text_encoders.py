"""Encode one prompt with two text encoder checkpoints (e.g. scaled-FP8 vs BF16) through the app's own
encoding path; report format detection, measured VRAM and how close the conditioning tensors are.

    .venv\\Scripts\\python.exe scripts\\compare_text_encoders.py A.safetensors B.safetensors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402,F401  (configures the CPU allocator before torch loads)
import torch  # noqa: E402

from app.conditioning import encode_prompt  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.model_manager import Component, ModelManager  # noqa: E402

GiB = 2**30


def main(a: str, b: str, prompt: str = "a red fox walking through fresh snow in a birch forest, morning light, photograph") -> None:
    s = load_settings()
    mgr = ModelManager()
    results = []
    for path in (a, b):
        mgr.configure({Component.TRANSFORMER: s.transformer_path, Component.TEXT_ENCODER: path, Component.VAE: s.vae_path},
                      s.text_encoder_config_dir, {Component.TEXT_ENCODER: False})
        info = mgr.inspect(Component.TEXT_ENCODER)
        cond = encode_prompt(mgr, prompt)
        st = mgr.status[Component.TEXT_ENCODER]
        print(f"{info.filename}: format={info.format} stored={info.stored_precision} exec={info.execution_precision} "
              f"dequantized_at_load={info.dequantized_at_load} | RAM {info.weight_bytes_to_load / GiB:.2f} GiB | measured VRAM weights "
              f"{st.last_vram_weight_bytes / GiB:.2f} GiB / peak allocated {st.last_peak_allocated_bytes / GiB:.2f} GiB")
        results.append(cond)
    ea, eb = results[0].embeds.float(), results[1].embeds.float()
    m = results[0].mask[0]
    ea, eb = ea[0, m], eb[0, m]  # real tokens only
    cos = torch.nn.functional.cosine_similarity(ea.reshape(-1, ea.shape[-1]), eb.reshape(-1, eb.shape[-1]), dim=-1)
    rel = (ea - eb).norm() / ea.norm()
    print(f"conditioning: per-vector cosine min {cos.min():.5f} mean {cos.mean():.5f}; relative L2 diff {rel:.5f}; masks equal {torch.equal(results[0].mask, results[1].mask)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
