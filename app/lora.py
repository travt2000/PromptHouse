"""LoRA adapters for the Krea 2 transformer: inspection, a RAM cache, and runtime (unmerged) application.

Design
  * Adapters are never merged into the base weights. The base checkpoint (BF16 or scaled FP8) stays exactly as
    loaded; changing, re-weighting, disabling or reordering LoRAs never reloads or modifies it.
  * During sampling each targeted linear layer gets a forward hook that adds the low-rank update:
        y = base(x) + sum_i  strength_i * (alpha_i / rank_i) * B_i (A_i x)
    All enabled adapters for one layer are concatenated into one A [sum r, in] and one B [out, sum r] (scales
    folded into B), so a stack costs two extra skinny matmuls per patched layer regardless of its length.
    The adapter tensors live on the GPU only while the transformer is (they are counted in the VRAM plan) and
    are released with it.
  * Because the updates are additive, stack order does not change the result (beyond floating-point rounding).

Accepted file layouts (single .safetensors, header inspected, nothing downloaded):
  * original Krea 2 module names (`blocks.3.attn.wq`), optional prefix `diffusion_model.` /
    `model.diffusion_model.` / `base_model.model.` (Krea reference trainer, AI-Toolkit, ComfyUI-style)
  * Diffusers `Krea2Transformer2DModel` names (`transformer.transformer_blocks.3.attn.to_q`)
  * kohya-style flattened names (`lora_unet_blocks_3_attn_wq` / `lora_transformer_...`)
  with `lora_A`/`lora_B` (PEFT) or `lora_down`/`lora_up` (kohya) tensors and optional per-module `alpha`.
Rejected with a message: LoHa / LoKr / DoRA / OFT / LoCon-mid tensors, unknown modules, shape mismatches.
Text-encoder tensors in the file are reported and not applied (this app conditions the prompt separately).
"""

from __future__ import annotations

import functools
import gc
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from torch import nn

from . import checkpoints as ck

log = logging.getLogger("krea.lora")
GiB = 1024**3
MAX_STACK = 16
STRENGTH_LIMIT = 5.0

_PREFIXES = ("base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.")
_KOHYA_PREFIXES = ("lora_unet_", "lora_transformer_")
_TE_PREFIXES = ("lora_te", "text_encoder", "text_encoders.", "te.", "te_", "cond_stage_model.", "conditioner.")
_A_SUFFIXES = (".lora_A.weight", ".lora_A.default.weight", ".lora_down.weight", ".lora.down.weight")
_B_SUFFIXES = (".lora_B.weight", ".lora_B.default.weight", ".lora_up.weight", ".lora.up.weight")
_ALPHA_SUFFIXES = (".alpha", ".lora_alpha")
_UNSUPPORTED_MARKERS = {
    "hada_": "LoHa", "lokr_": "LoKr", "dora_scale": "DoRA", "lora_magnitude_vector": "DoRA", "oft_": "OFT", "lora_mid": "LoCon (mid)",
}
_FLOAT_TAGS = {"BF16", "F16", "F32"}


# ---------------------------------------------------------------------------------------------- targets


@functools.lru_cache(maxsize=1)
def krea2_targets() -> dict[str, tuple[int, int]]:
    """Every nn.Linear in the Krea 2 transformer -> (in_features, out_features). Built on the meta device."""
    from .krea2_mmdit import KREA2_CONFIG, SingleStreamDiT

    with torch.device("meta"):
        model = SingleStreamDiT(KREA2_CONFIG)
    return {n: (m.in_features, m.out_features) for n, m in model.named_modules() if isinstance(m, nn.Linear)}


_ATTN = {"wq": "to_q", "wk": "to_k", "wv": "to_v", "wo": "to_out.0", "gate": "to_gate"}
_FF = {"gate": "ff.gate", "up": "ff.up", "down": "ff.down"}
_STANDALONE = {
    "first": "img_in", "last.linear": "final_layer.linear", "tmlp.0": "time_embed.linear_1", "tmlp.2": "time_embed.linear_2",
    "tproj.1": "time_mod_proj", "txtmlp.1": "txt_in.linear_1", "txtmlp.3": "txt_in.linear_2", "txtfusion.projector": "text_fusion.projector",
}


def _to_diffusers_name(name: str) -> str | None:
    """Original Krea 2 module name -> Diffusers Krea2Transformer2DModel name (same rules as Diffusers' converter)."""
    m = re.match(r"blocks\.(\d+)\.(attn|mlp)\.(\w+)$", name)
    if m:
        i, kind, sub = m.groups()
        return f"transformer_blocks.{i}.attn.{_ATTN[sub]}" if kind == "attn" and sub in _ATTN else (
            f"transformer_blocks.{i}.{_FF[sub]}" if kind == "mlp" and sub in _FF else None)
    m = re.match(r"txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(attn|mlp)\.(\w+)$", name)
    if m:
        blk, i, kind, sub = m.groups()
        return f"text_fusion.{blk}.{i}.attn.{_ATTN[sub]}" if kind == "attn" and sub in _ATTN else (
            f"text_fusion.{blk}.{i}.{_FF[sub]}" if kind == "mlp" and sub in _FF else None)
    return _STANDALONE.get(name)


@functools.lru_cache(maxsize=4)
def _name_maps(target_names: tuple[str, ...]) -> tuple[dict, dict]:
    diffusers = {}
    for n in target_names:
        d = _to_diffusers_name(n)
        if d:
            diffusers[d] = n
    flat = {n.replace(".", "_"): n for n in target_names}
    flat.update({d.replace(".", "_"): n for d, n in diffusers.items()})
    return diffusers, flat


# ---------------------------------------------------------------------------------------------- inspection


@dataclass
class LoraInfo:
    path: str
    filename: str
    size_bytes: int = 0
    fingerprint: str = ""
    compatible: bool = False
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    key_style: str = ""  # "krea2 original" | "diffusers" | "kohya"
    tensor_style: str = ""  # "lora_A/lora_B" | "lora_down/lora_up"
    stored_dtype: str = ""
    rank: str = ""
    has_alpha: bool = False
    modules: int = 0
    blocks_covered: int = 0
    other_modules: int = 0
    gpu_bytes: int = 0  # BF16 bytes of all A/B tensors (what goes to the GPU)
    ignored_text_encoder_tensors: int = 0
    title: str = ""
    trigger: str = ""
    # module -> {"A": key, "B": key, "alpha": key|None, "rank": r}
    plan: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("plan")
        return d


_inspect_cache: dict[tuple, LoraInfo] = {}
_inspect_lock = threading.Lock()


def inspect_lora(path: str, targets: dict[str, tuple[int, int]] | None = None) -> LoraInfo:
    """Validate a LoRA file from its safetensors header only (cached by path/size/mtime)."""
    p = Path(str(path).strip().strip('"'))
    info = LoraInfo(path=str(p), filename=p.name)
    if not str(path).strip():
        info.problems.append("No path given")
        return info
    if not p.is_file():
        info.problems.append(f"File not found: {p}")
        return info
    if p.suffix.lower() != ".safetensors":
        info.problems.append("Only .safetensors LoRA files are supported")
        return info
    st = p.stat()
    custom = targets is not None
    key = (os.path.normcase(str(p.resolve())), st.st_size, st.st_mtime_ns, custom)
    if not custom:
        with _inspect_lock:
            hit = _inspect_cache.get(key)
        if hit is not None:
            return hit
    targets = targets if custom else krea2_targets()
    info.size_bytes = st.st_size
    info.fingerprint = f"{key[0]}|{st.st_size}|{st.st_mtime_ns}"
    try:
        header, meta = ck.read_header(p)
    except (ck.CheckpointError, OSError) as e:
        info.problems.append(str(e))
        return info
    _analyse(info, header, meta, targets)
    if not custom:
        with _inspect_lock:
            if len(_inspect_cache) > 256:
                _inspect_cache.clear()
            _inspect_cache[key] = info
    return info


def _split_suffix(k: str) -> tuple[str, str] | None:
    for role, sufs in (("A", _A_SUFFIXES), ("B", _B_SUFFIXES), ("alpha", _ALPHA_SUFFIXES)):
        for s in sufs:
            if k.endswith(s):
                return k[: -len(s)], role
    return None


def _analyse(info: LoraInfo, header: dict, meta: dict, targets: dict[str, tuple[int, int]]) -> None:
    diffusers_map, flat_map = _name_maps(tuple(sorted(targets)))
    styles, tstyles, dtypes = set(), set(), set()
    unknown, ignored, unsupported = [], [], {}
    plan: dict[str, dict] = {}
    for k, v in header.items():
        low = k.lower()
        bad = next((t for m, t in _UNSUPPORTED_MARKERS.items() if m in low), None)
        if bad:
            unsupported.setdefault(bad, 0)
            unsupported[bad] += 1
            continue
        if low.startswith(_TE_PREFIXES):
            info.ignored_text_encoder_tensors += 1
            continue
        split = _split_suffix(k)
        if split is None:
            ignored.append(k)
            continue
        mod_key, role = split
        module, style = _resolve_module(mod_key, targets, diffusers_map, flat_map)
        if module is None:
            unknown.append(mod_key)
            continue
        styles.add(style)
        if role in ("A", "B"):
            tstyles.add("lora_down/lora_up" if ("down" in k[len(mod_key):] or "up" in k[len(mod_key):]) else "lora_A/lora_B")
            dtypes.add(v["dtype"])
        entry = plan.setdefault(module, {"A": None, "B": None, "alpha": None})
        if entry[role] is not None:
            info.problems.append(f"Module {module} appears twice ({entry[role]} and {k})")
            continue
        entry[role] = k

    if unsupported:
        info.problems.append("Unsupported adapter type: " + ", ".join(f"{t} ({n} tensors)" for t, n in unsupported.items())
                             + ". Only standard LoRA (A/B or down/up matrices) is supported.")
    if unknown:
        uniq = sorted(set(unknown))
        info.problems.append(f"{len(uniq)} module(s) do not exist in the Krea 2 transformer (e.g. {uniq[0]}); "
                             "this LoRA was probably trained for a different model.")
    if len(styles) > 1:
        info.problems.append(f"Mixed key naming styles in one file: {sorted(styles)}")
    bad_dtypes = dtypes - _FLOAT_TAGS
    if bad_dtypes:
        info.problems.append(f"Unsupported LoRA tensor dtype(s) {sorted(bad_dtypes)} (BF16/FP16/FP32 supported)")

    ranks, gpu_bytes = set(), 0
    for module, e in sorted(plan.items()):
        if e["A"] is None or e["B"] is None:
            info.problems.append(f"{module}: has {'B' if e['A'] is None else 'A'} matrix but no {'A' if e['A'] is None else 'B'}")
            continue
        a, b = header[e["A"]]["shape"], header[e["B"]]["shape"]
        fin, fout = targets[module]
        if len(a) == 4 and a[2:] == [1, 1]:
            a = a[:2]
        if len(b) == 4 and b[2:] == [1, 1]:
            b = b[:2]
        if len(a) != 2 or len(b) != 2 or a[0] != b[1] or a[1] != fin or b[0] != fout:
            info.problems.append(f"{module}: LoRA shapes A{a} B{b} do not fit the layer ({fin} -> {fout})")
            continue
        if e["alpha"] is not None and header[e["alpha"]]["shape"] not in ([], [1]):
            info.problems.append(f"{module}: alpha must be a scalar, got shape {header[e['alpha']]['shape']}")
            continue
        e["rank"] = a[0]
        ranks.add(a[0])
        gpu_bytes += 2 * a[0] * (fin + fout)
    if ignored:
        info.warnings.append(f"{len(ignored)} tensor(s) that are not LoRA matrices are NOT applied (e.g. {ignored[0]}).")
    if info.ignored_text_encoder_tensors:
        info.warnings.append(f"{info.ignored_text_encoder_tensors} text-encoder tensor(s) are NOT applied: only the transformer part "
                             "of a LoRA is used (prompt conditioning is computed separately).")
    complete = {m: e for m, e in plan.items() if e.get("rank")}
    if not complete and not info.problems:
        info.problems.append("No transformer LoRA weights found in this file")

    info.plan = complete
    info.modules = len(complete)
    info.blocks_covered = len({int(m.split(".")[1]) for m in complete if m.startswith("blocks.")})
    info.other_modules = sum(1 for m in complete if not m.startswith("blocks."))
    info.key_style = next(iter(styles), "")
    info.tensor_style = "/".join(sorted(tstyles))
    info.stored_dtype = "/".join(sorted(dtypes))
    info.rank = (str(min(ranks)) if len(ranks) == 1 else f"{min(ranks)}-{max(ranks)}") if ranks else ""
    info.has_alpha = any(e["alpha"] for e in complete.values())
    info.gpu_bytes = gpu_bytes
    info.title = str(meta.get("modelspec.title", ""))[:200]
    info.trigger = str(meta.get("modelspec.trigger_phrase", "") or meta.get("trigger_phrase", ""))[:300]
    info.compatible = not info.problems


def _resolve_module(mod_key: str, targets, diffusers_map, flat_map) -> tuple[str | None, str]:
    k = mod_key
    for pre in _KOHYA_PREFIXES:
        if k.startswith(pre):
            return flat_map.get(k[len(pre):]), "kohya"
    stripped = True
    while stripped:
        stripped = False
        for pre in _PREFIXES:
            if k.startswith(pre):
                k = k[len(pre):]
                stripped = True
    if k in targets:
        return k, "krea2 original"
    if k in diffusers_map:
        return diffusers_map[k], "diffusers"
    return None, ""


# ---------------------------------------------------------------------------------------------- RAM cache


@dataclass
class LoadedLora:
    info: LoraInfo
    # module -> (A [r, in] bf16, B [out, r] bf16, alpha/rank)
    modules: dict[str, tuple[torch.Tensor, torch.Tensor, float]]
    ram_bytes: int
    load_s: float


class LoraLibrary:
    """Keeps loaded LoRA files in system RAM, separately from the base checkpoint, keyed by file fingerprint.

    Must be driven from the GPU worker thread (frees are then returned to the OS; see ModelManager)."""

    def __init__(self, targets: dict[str, tuple[int, int]] | None = None):
        self._targets = targets
        self._loaded: dict[str, LoadedLora] = {}
        self._lock = threading.RLock()

    def inspect(self, path: str) -> LoraInfo:
        return inspect_lora(path, self._targets)

    def prepare(self, entries: list[dict], keep_paths: list[str], stage_cb=lambda s: None) -> list[tuple[LoadedLora, float]]:
        """Load every enabled entry (from cache when unchanged); evict cached files not in `keep_paths`."""
        self.evict_except(keep_paths + [e["path"] for e in entries])
        out = []
        for e in entries:
            if not e.get("enabled", True):
                continue
            info = self.inspect(e["path"])
            if not info.compatible:
                raise LoraError(f"LoRA {info.filename}: " + "; ".join(info.problems))
            lora = self._loaded.get(info.fingerprint)
            if lora is None:
                stage_cb(f"Loading LoRA {info.filename}")
                lora = self._load(info)
            out.append((lora, float(e.get("strength", 1.0))))
        return out

    def _load(self, info: LoraInfo) -> LoadedLora:
        need = info.gpu_bytes
        avail = psutil.virtual_memory().available
        if need + GiB > avail:
            raise LoraError(f"Not enough free system RAM to load LoRA {info.filename}: needs {need / GiB:.2f} GiB, "
                            f"{avail / GiB:.1f} GiB available.")
        t0 = time.perf_counter()
        keys = {}
        for m, e in info.plan.items():
            keys[e["A"]] = e["A"]
            keys[e["B"]] = e["B"]
            if e["alpha"]:
                keys[e["alpha"]] = e["alpha"]
        raw = ck.load_tensors(info.path, keys, to_bf16=False)  # alpha must keep full precision
        modules = {}
        for m, e in info.plan.items():
            a, b = raw.pop(e["A"]), raw.pop(e["B"])
            a = a.reshape(a.shape[0], a.shape[1]).to(torch.bfloat16).contiguous()
            b = b.reshape(b.shape[0], b.shape[1]).to(torch.bfloat16).contiguous()
            r = a.shape[0]
            alpha_scale = float(raw.pop(e["alpha"]).float().reshape(-1)[0]) / r if e["alpha"] else 1.0
            modules[m] = (a, b, alpha_scale)
        del raw
        ram = sum(a.numel() * 2 + b.numel() * 2 for a, b, _ in modules.values())
        lora = LoadedLora(info=info, modules=modules, ram_bytes=ram, load_s=round(time.perf_counter() - t0, 2))
        with self._lock:
            self._loaded[info.fingerprint] = lora
        log.info("LoRA %s loaded: %d modules, rank %s, %.1f MiB (%.2fs)", info.filename, len(modules), info.rank, ram / 2**20, lora.load_s)
        return lora

    def evict_except(self, keep_paths: list[str]) -> list[str]:
        keep = {os.path.normcase(os.path.abspath(p.strip().strip('"'))) for p in keep_paths if p}
        evicted = []
        with self._lock:
            for fp, lora in list(self._loaded.items()):
                if os.path.normcase(os.path.abspath(lora.info.path)) not in keep or lora.info.fingerprint != self.inspect(lora.info.path).fingerprint:
                    del self._loaded[fp]
                    evicted.append(lora.info.filename)
                    lora.modules.clear()
        if evicted:
            gc.collect()
            log.info("LoRA RAM cache released: %s", evicted)
        return evicted

    def release_all(self) -> None:
        self.evict_except([])

    def snapshot(self) -> dict:
        with self._lock:
            items = [{"filename": l.info.filename, "path": l.info.path, "ram_bytes": l.ram_bytes, "load_s": l.load_s} for l in self._loaded.values()]
        return {"loaded": items, "ram_bytes": sum(i["ram_bytes"] for i in items)}


class LoraError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------- runtime patch


def adapters_gpu_bytes(adapters: list[tuple[LoadedLora, float]]) -> int:
    return sum(l.ram_bytes for l, s in adapters if s != 0.0)


class LoraPatch:
    """Attaches a LoRA stack to `model` via forward hooks for the duration of one GPU session.

    Base parameters are not touched. close() removes every hook and frees every adapter tensor."""

    def __init__(self, model: nn.Module, adapters: list[tuple[LoadedLora, float]], device: torch.device):
        self.handles = []
        self.tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.applied = []
        per_module: dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]] = {}
        for lora, strength in adapters:
            if strength == 0.0:
                continue
            self.applied.append({"file": lora.info.filename, "strength": strength, "modules": len(lora.modules)})
            for m, (a, b, alpha_scale) in lora.modules.items():
                per_module.setdefault(m, []).append((a, b, strength * alpha_scale))
        try:
            for m, parts in per_module.items():
                mod = model.get_submodule(m)
                R = sum(a.shape[0] for a, _, _ in parts)
                A = torch.empty(R, parts[0][0].shape[1], dtype=torch.bfloat16, device=device)
                B = torch.empty(parts[0][1].shape[0], R, dtype=torch.bfloat16, device=device)
                o = 0
                for a, b, scale in parts:
                    r = a.shape[0]
                    A[o: o + r].copy_(a, non_blocking=True)
                    B[:, o: o + r].copy_((b.to(device, non_blocking=True).float() * scale).to(torch.bfloat16))
                    o += r
                self.tensors[m] = (A, B)
                self.handles.append(mod.register_forward_hook(_make_hook(A, B)))
        except BaseException:
            self.close()
            raise
        self.patched_modules = len(self.tensors)
        self.gpu_bytes = sum(A.numel() * 2 + B.numel() * 2 for A, B in self.tensors.values())

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.tensors.clear()


def _make_hook(A: torch.Tensor, B: torch.Tensor):
    def hook(_mod, args, out):
        x = args[0]
        if x.dtype != A.dtype:
            return out + F.linear(F.linear(x.to(A.dtype), A), B).to(out.dtype)
        return out + F.linear(F.linear(x, A), B)

    return hook
