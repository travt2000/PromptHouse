"""End-to-end CLI smoke test using the exact modules the web app uses.

    .venv\\Scripts\\python.exe scripts\\smoke_test.py --prompt "a fox walking in the snow" --seed 42

Verifies, after every stage, that no component's tensors remain on the GPU, and that generation reuses
the cached conditioning (the text encoder is not invoked again for --repeat > 1).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402,F401  (configures the CPU allocator before torch loads)
import torch  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.conditioning import PromptCache, encode_prompt  # noqa: E402
from app.decoding import decode_latents  # noqa: E402
from app.inference import MU, SAMPLER, STEPS, sample_latents  # noqa: E402
from app.model_manager import Component, ModelManager, cuda_tensors_of  # noqa: E402
from app.outputs import save_image  # noqa: E402

GiB = 1024**3


def vram(tag: str, mgr: ModelManager) -> None:
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated() / GiB
    on_gpu = {c.value: len(cuda_tensors_of(m.module)) for c, m in mgr._models.items() if m is not None}
    print(f"[{tag:>22}] device used {(total - free) / GiB:5.2f}/{total / GiB:.2f} GiB | torch allocated {alloc:5.2f} GiB | "
          f"GPU owner={mgr.gpu_owner} | tensors on GPU per component={on_gpu}", flush=True)
    assert mgr.gpu_owner is None and all(v == 0 for v in on_gpu.values()), "component left on GPU!"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red fox walking through fresh snow in a birch forest, morning light, photograph")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--repeat", type=int, default=1, help="generate N times from ONE encode")
    ap.add_argument("--transformer", help="override transformer checkpoint path")
    ap.add_argument("--text-encoder", help="override text encoder checkpoint path")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    s = load_settings()
    mgr = ModelManager()
    mgr.stage_cb = lambda st: print(f"  -> stage: {st}", flush=True)
    mgr.configure(
        {Component.TRANSFORMER: args.transformer or s.transformer_path, Component.TEXT_ENCODER: args.text_encoder or s.text_encoder_path, Component.VAE: s.vae_path},
        s.text_encoder_config_dir,
        {Component(k): v for k, v in s.keep_in_ram.items()},
    )
    for c in Component:
        info = mgr.inspect(c)
        print(f"{c.value}: {info.filename} format={info.format} stored={info.stored_precision} exec={info.execution_precision} compatible={info.compatible} {info.problems}")
    vram("start", mgr)

    cache = PromptCache()
    token = cache.invalidate()
    t0 = time.perf_counter()
    cond = encode_prompt(mgr, args.prompt)
    assert cache.install(cond, token)
    print(f"encode: {time.perf_counter() - t0:.1f}s  {cond.describe()}")
    vram("after encode", mgr)

    for r in range(args.repeat):
        seed = args.seed + r
        te_calls_before = mgr.status[Component.TEXT_ENCODER].last_upload_s
        t0 = time.perf_counter()
        latents = sample_latents(mgr, cache.get(), args.width, args.height, seed, progress=lambda i, n: print(f"  step {i}/{n}", flush=True))
        t_sample = time.perf_counter() - t0
        print(f"sample: {t_sample:.1f}s latents {tuple(latents.shape)} {latents.dtype} on {latents.device}; plan="
              f"{ {k: v for k, v in mgr.status[Component.TRANSFORMER].streaming.items() if not k.endswith('_set')} }"
              f" peak allocated {mgr.status[Component.TRANSFORMER].last_peak_allocated_bytes / GiB:.2f} GiB")
        vram("after sample", mgr)
        t0 = time.perf_counter()
        image, dinfo = decode_latents(mgr, latents)
        print(f"decode: {time.perf_counter() - t0:.1f}s image {image.size} mode={dinfo['mode']} peak {mgr.status[Component.VAE].last_peak_allocated_bytes / GiB:.2f} GiB")
        vram("after decode", mgr)
        assert mgr.status[Component.TEXT_ENCODER].last_upload_s == te_calls_before, "text encoder was re-invoked!"
        meta = {
            "prompt": cond.prompt, "seed": seed, "width": args.width, "height": args.height, "steps": STEPS, "guidance": 0.0, "mu": MU,
            "sampler": SAMPLER, "transformer": mgr.inspect(Component.TRANSFORMER).filename,
            "text_encoder": cond.encoder_filename, "vae": mgr.inspect(Component.VAE).filename, "vae_decode": dinfo["mode"], "source": "smoke_test",
        }
        image_id, path = save_image(image, meta)
        print(f"saved {path}")
    print(json.dumps(mgr.snapshot(), indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
