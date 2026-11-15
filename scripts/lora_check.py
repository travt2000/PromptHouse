"""Verify LoRA support against the REAL Krea 2 checkpoint (synthetic adapters; no LoRA download needed).

    .venv\\Scripts\\python.exe scripts\\lora_check.py            # low-RAM: real FP8 layers of 2 blocks on the GPU
    .venv\\Scripts\\python.exe scripts\\lora_check.py --full     # end to end: encode + 4 samplings (needs ~16 GB free RAM)

Low-RAM mode loads the scaled-FP8 weights of transformer blocks 0 and 27 from the configured checkpoint into
FP8Linear layers on the GPU, writes the same random LoRA in three key layouts (Krea original / Diffusers /
kohya with alpha), and checks for every patched layer that hook output == x @ (dequant(W) + s*alpha/r*B@A)^T,
that all three layouts give identical results, that strength 0 is bit-identical to no LoRA, and that the
base weights are unchanged afterwards.

--full additionally runs real sampling through the app's ModelManager (block streaming + VRAM planning):
no LoRA vs. strength 0 vs. disabled must be bit-identical latents, strength 1 must differ, and the transformer
must be read from disk only once for all four runs. Images are saved to outputs/ for inspection.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402,F401
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from torch import nn  # noqa: E402

from app import checkpoints as ck  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.lora import LoraLibrary, LoraPatch, _to_diffusers_name, krea2_targets  # noqa: E402
from app.quant import FP8Linear  # noqa: E402

RANK, ALPHA = 16, 8.0


def write_loras(out_dir: Path, modules: list[str]) -> dict[str, str]:
    """Same random weights, three file layouts."""
    t = krea2_targets()
    g = torch.Generator().manual_seed(1234)
    w = {}
    for m in modules:
        fin, fout = t[m]
        w[m] = (torch.randn(RANK, fin, generator=g) / fin**0.5, torch.randn(fout, RANK, generator=g) * 0.02)
    paths = {}
    sd = {}
    for m, (a, b) in w.items():
        sd[f"diffusion_model.{m}.lora_A.weight"], sd[f"diffusion_model.{m}.lora_B.weight"] = a.bfloat16(), b.bfloat16()
        sd[f"diffusion_model.{m}.alpha"] = torch.tensor(ALPHA)
    paths["original"] = str(out_dir / "synthetic_krea2_original.safetensors")
    save_file(sd, paths["original"])
    sd = {}
    for m, (a, b) in w.items():
        d = _to_diffusers_name(m)
        sd[f"transformer.{d}.lora_A.weight"], sd[f"transformer.{d}.lora_B.weight"] = a.bfloat16(), b.bfloat16()
        sd[f"transformer.{d}.alpha"] = torch.tensor(ALPHA)
    paths["diffusers"] = str(out_dir / "synthetic_krea2_diffusers.safetensors")
    save_file(sd, paths["diffusers"])
    sd = {}
    for m, (a, b) in w.items():
        k = "lora_unet_" + m.replace(".", "_")
        sd[f"{k}.lora_down.weight"], sd[f"{k}.lora_up.weight"] = a.half(), b.half()  # fp16 storage path too
        sd[f"{k}.alpha"] = torch.tensor(ALPHA)
    paths["kohya"] = str(out_dir / "synthetic_krea2_kohya.safetensors")
    save_file(sd, paths["kohya"])
    return paths


class Holder(nn.Module):
    """Minimal module tree with the real names (blocks.N.attn.wq ...) holding real checkpoint layers."""

    def __init__(self, info: ck.CheckpointInfo, blocks: list[int]):
        super().__init__()
        from app.krea2_mmdit import KREA2_CONFIG, SingleStreamBlock

        c = KREA2_CONFIG
        with torch.device("meta"):
            self.blocks = nn.ModuleDict({str(b): SingleStreamBlock(c.features, c.heads, c.multiplier, c.bias, c.kvheads) for b in blocks})
        tensors, _ = ck.read_header(info.path)
        prefix = info.details.get("key_prefix", "")
        want = {k: k[len(prefix):] for k in tensors if any(k[len(prefix):].startswith(f"blocks.{b}.") for b in blocks)}
        qnames = {want[k][: -len(".weight")] for k in want if tensors[k]["dtype"] in ck.FP8_TAGS}
        from app.quant import finalize_fp8, replace_linears_with_fp8

        replace_linears_with_fp8(self, qnames)
        sd = ck.load_tensors(info.path, want)
        missing, unexpected = self.load_state_dict(sd, strict=False, assign=True)
        assert not unexpected and not missing, (missing[:3], unexpected[:3])
        finalize_fp8(self)
        self.linears = [n for n, m in self.named_modules() if isinstance(m, (nn.Linear, FP8Linear)) and any(n.startswith(f"blocks.{b}.") for b in blocks)]
        self.requires_grad_(False)


def dequant(mod) -> torch.Tensor:
    if isinstance(mod, FP8Linear):
        s = mod.scale_value if mod.scale_value is not None else mod.weight_scale.reshape(-1, 1).float()
        return mod.weight.float() * s
    return mod.weight.float()


def layer_check(settings) -> bool:
    info = ck.inspect_checkpoint(settings.transformer_path, ck.KIND_TRANSFORMER)
    print(f"checkpoint: {info.filename} ({info.format})")
    blocks = [0, 27]
    holder = Holder(info, blocks)
    modules = holder.linears
    dev = torch.device("cuda")
    holder.to(dev)
    ok = True
    with tempfile.TemporaryDirectory() as td:
        paths = write_loras(Path(td), modules)
        lib = LoraLibrary()
        for style, p in paths.items():
            li = lib.inspect(p)
            print(f"  {style:9s}: compatible={li.compatible} style={li.key_style!r} rank={li.rank} modules={li.modules} alpha={li.has_alpha} "
                  f"dtype={li.stored_dtype} problems={li.problems}")
            ok &= li.compatible and li.modules == len(modules)
        g = torch.Generator(device=dev).manual_seed(0)
        xs = {n: torch.randn(512, holder.get_submodule(n).in_features, device=dev, dtype=torch.bfloat16, generator=g) for n in modules}
        base_w = {n: dequant(holder.get_submodule(n)).clone() for n in modules}
        with torch.inference_mode():
            base = {n: holder.get_submodule(n)(xs[n]).clone() for n in modules}
        outs = {}
        for style, p in paths.items():
            strength = 0.9
            adapters = lib.prepare([{"path": p, "strength": strength, "enabled": True}], list(paths.values()))
            patch = LoraPatch(holder, adapters, dev)
            with torch.inference_mode():
                outs[style] = {n: holder.get_submodule(n)(xs[n]).float() for n in modules}
            patch.close()
            worst = 0.0
            lora = adapters[0][0]
            for n in modules:
                a, b, k = lora.modules[n]
                w = base_w[n] + strength * k * (b.float().to(dev) @ a.float().to(dev))
                ref = xs[n].float() @ w.T
                mod = holder.get_submodule(n)
                if getattr(mod, "bias", None) is not None:
                    ref = ref + mod.bias.float()
                rel = ((outs[style][n] - ref).norm() / ref.norm()).item()
                worst = max(worst, rel)
            delta = max(((outs[style][n] - base[n].float()).norm() / base[n].float().norm()).item() for n in modules)
            print(f"  {style:9s}: worst relative error vs merged-weight reference {worst:.2e}; LoRA changed outputs by up to {delta:.2%}")
            ok &= worst < 2e-2 and delta > 1e-3
        # same weights in three layouts -> same outputs (kohya file stored fp16 -> tiny rounding difference)
        same = max(((outs["original"][n] - outs["diffusers"][n]).abs().max().item()) for n in modules)
        close = max(((outs["original"][n] - outs["kohya"][n]).norm() / outs["original"][n].norm()).item() for n in modules)
        print(f"  original vs diffusers max |diff| = {same:.3g} (expect 0); original vs kohya(fp16) rel = {close:.2e}")
        ok &= same == 0.0 and close < 1e-2
        adapters = lib.prepare([{"path": paths["original"], "strength": 0.0, "enabled": True}], [])
        patch = LoraPatch(holder, adapters, dev)
        with torch.inference_mode():
            zero_equal = all(torch.equal(holder.get_submodule(n)(xs[n]), base[n]) for n in modules)
        patch.close()
        with torch.inference_mode():
            restored = all(torch.equal(holder.get_submodule(n)(xs[n]), base[n]) for n in modules)
        unchanged = all(torch.equal(dequant(holder.get_submodule(n)), base_w[n]) for n in modules)
        hooks = sum(len(m._forward_hooks) for m in holder.modules())
        print(f"  strength 0 bit-identical: {zero_equal}; after close bit-identical: {restored}; base weights unchanged: {unchanged}; hooks left: {hooks}")
        ok &= zero_equal and restored and unchanged and hooks == 0
        lib.release_all()
    del holder
    torch.cuda.empty_cache()
    return ok


def full_check(settings, prompt: str, size: int) -> bool:
    from app.conditioning import encode_prompt
    from app.decoding import decode_latents
    from app.inference import sample_latents
    from app.model_manager import Component, ModelManager
    from app.outputs import save_image

    mgr = ModelManager()
    mgr.stage_cb = lambda st: print(f"  -> {st}", flush=True)
    mgr.configure({Component.TRANSFORMER: settings.transformer_path, Component.TEXT_ENCODER: settings.text_encoder_path, Component.VAE: settings.vae_path},
                  settings.text_encoder_config_dir, {Component.TEXT_ENCODER: False, Component.TRANSFORMER: True, Component.VAE: False})
    loads = []
    real_ensure = mgr._ensure_cpu

    def counting(c):
        before = mgr._models[c]
        lm = real_ensure(c)
        if lm is not before:
            loads.append(c)
        return lm

    mgr._ensure_cpu = counting
    cond = encode_prompt(mgr, prompt)
    lib = LoraLibrary()
    with tempfile.TemporaryDirectory() as td:
        paths = write_loras(Path(td), sorted(krea2_targets()))
        p = paths["original"]
        runs = {
            "no LoRA": [],
            "strength 0": [{"path": p, "strength": 0.0, "enabled": True}],
            "disabled": [{"path": p, "strength": 1.0, "enabled": False}],
            "strength 1": [{"path": p, "strength": 1.0, "enabled": True}],
        }
        lat = {}
        for name, stack in runs.items():
            t0 = time.time()
            adapters = lib.prepare(stack, [p])
            lat[name] = sample_latents(mgr, cond, size, size, 42, adapters=adapters)
            st = mgr.status[Component.TRANSFORMER]
            print(f"  {name:10s}: {time.time() - t0:5.1f}s  plan {st.streaming['resident_blocks']}/28 resident, lora {st.streaming['lora_bytes'] / 2**20:.0f} MiB; "
                  f"adapters={st.adapters}")
        ok = torch.equal(lat["no LoRA"], lat["strength 0"]) and torch.equal(lat["no LoRA"], lat["disabled"])
        diff = ((lat["strength 1"] - lat["no LoRA"]).float().norm() / lat["no LoRA"].float().norm()).item()
        print(f"  no-LoRA == strength0 == disabled (bit-exact): {ok}; strength 1 latent change {diff:.2%}")
        ok &= diff > 1e-3 and bool(torch.isfinite(lat["strength 1"]).all())
        n_tf = loads.count(Component.TRANSFORMER)
        print(f"  transformer read from disk {n_tf}x for 4 runs (expect 1)")
        ok &= n_tf == 1
        for name in ("no LoRA", "strength 1"):
            img, _ = decode_latents(mgr, lat[name])
            iid, path = save_image(img, {"prompt": prompt, "seed": 42, "lora_check": name, "lora": "synthetic random (not a trained LoRA)"})
            print(f"  saved {name}: {path}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--prompt", default="a red fox walking through fresh snow in a birch forest, morning light, photograph")
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()
    s = load_settings()
    ok = layer_check(s)
    print("LAYER CHECK:", "PASS" if ok else "FAIL")
    if args.full:
        f = full_check(s, args.prompt, args.size)
        print("FULL CHECK:", "PASS" if f else "FAIL")
        ok &= f
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
