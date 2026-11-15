"""Single-file .safetensors inspection and loading for the three Krea 2 components.

Everything here reads local files only. Nothing is downloaded.

Supported checkpoint layouts (verified against the shipped files):
  * Krea 2 transformer, original Krea key names (`blocks.N.attn.wq.weight`, `txtfusion.*`, ...),
    optionally prefixed with `model.diffusion_model.` / `diffusion_model.`;
    BF16/FP16/FP32 or scaled FP8 (per-tensor `weight_scale`).
  * Qwen3-VL-4B text encoder, `model.layers.N.*` / `model.embed_tokens.*` (or HF `model.language_model.*`)
    names; BF16 or scaled FP8 (per-tensor `weight_scale`, optional `comfy_quant` JSON marker tensors).
    Tokenizer + config.json come from a separate local directory (not embedded in the .safetensors).
  * Qwen-Image VAE, original Wan-2.1-style key names (`decoder.middle.0...`) or Diffusers
    `AutoencoderKLQwenImage` key names.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

SAFETENSORS_DTYPES = {
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E5M2": (torch.float8_e5m2, 1),
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "F32": (torch.float32, 4),
    "F64": (torch.float64, 8),
    "U8": (torch.uint8, 1),
    "I8": (torch.int8, 1),
    "I32": (torch.int32, 4),
    "I64": (torch.int64, 8),
    "BOOL": (torch.bool, 1),
}
FP8_TAGS = {"F8_E4M3", "F8_E5M2"}
SUPPORTED_QUANT_FORMATS = {"float8_e4m3fn", "float8_e5m2"}

KIND_TRANSFORMER = "krea2_transformer"
KIND_TEXT_ENCODER = "qwen3vl_text_encoder"
KIND_VAE = "qwen_image_vae"

TRANSFORMER_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


class CheckpointError(ValueError):
    pass


def read_header(path: str | os.PathLike) -> tuple[dict, dict]:
    """Return (tensor_table, metadata) from a .safetensors header without reading tensor data."""
    p = Path(path)
    with open(p, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise CheckpointError(f"{p.name}: file too small to be safetensors")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 200 * 1024 * 1024:
            raise CheckpointError(f"{p.name}: invalid safetensors header length {n}")
        try:
            hdr = json.loads(f.read(n))
        except json.JSONDecodeError as e:
            raise CheckpointError(f"{p.name}: corrupt safetensors header ({e})") from e
    meta = hdr.pop("__metadata__", {}) or {}
    return hdr, meta


@dataclass
class CheckpointInfo:
    path: str
    filename: str
    size_bytes: int
    kind: str
    expected_kind: str
    compatible: bool
    format: str  # "fp8_scaled" | "bf16" | "fp16" | "fp32" | "mixed" | "unknown"
    stored_precision: str
    execution_precision: str
    dequantized_at_load: bool
    dequant_note: str
    stored_bytes_by_dtype: dict = field(default_factory=dict)
    quantized_layers: int = 0
    full_precision_layers: int = 0
    weight_bytes_to_load: int = 0
    problems: list = field(default_factory=list)
    details: dict = field(default_factory=dict)
    mtime: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return f"{os.path.normcase(os.path.abspath(self.path))}|{self.size_bytes}|{self.mtime}"


def _strip_prefix(keys, prefixes) -> tuple[str, dict]:
    for p in prefixes:
        if all(k.startswith(p) for k in keys):
            return p, {k[len(p):]: k for k in keys}
    return "", {k: k for k in keys}


def _detect_kind(tensors: dict) -> str:
    keys = set(tensors)
    _, stripped = _strip_prefix(keys, TRANSFORMER_PREFIXES)
    if {"blocks.0.attn.wq.weight", "txtfusion.projector.weight", "first.weight"} <= set(stripped):
        return KIND_TRANSFORMER
    if any(k.startswith(("model.layers.0.self_attn.q_norm", "model.language_model.layers.0.self_attn.q_norm")) for k in keys):
        return KIND_TEXT_ENCODER
    if ("decoder.conv1.weight" in keys and "decoder.head.0.gamma" in keys) or "decoder.conv_in.weight" in keys:
        return KIND_VAE
    if any(k.startswith("transformer_blocks.") for k in keys):
        return "diffusers_transformer"
    return "unknown"


def _quant_layers(tensors: dict, meta: dict) -> tuple[dict, list[str]]:
    """Map quantized layer name -> {"fp8": tag, "scale_shape": [...], "full_precision": bool}."""
    problems = []
    layers = {}
    for k, v in tensors.items():
        if v["dtype"] in FP8_TAGS:
            if not k.endswith(".weight"):
                problems.append(f"FP8 tensor {k} is not a layer weight; unsupported layout")
                continue
            base = k[: -len(".weight")]
            sk = base + ".weight_scale"
            if sk not in tensors:
                problems.append(f"FP8 weight {k} has no {sk}; unscaled FP8 is not supported")
                continue
            sdt = tensors[sk]["dtype"]
            if sdt not in ("F32", "BF16", "F16"):
                problems.append(f"{sk} has dtype {sdt}; expected a float scale")
            shape = tensors[sk]["shape"]
            out = v["shape"][0]
            if not (shape in ([], [1], [out], [out, 1])):
                problems.append(f"{sk} has shape {shape}; only per-tensor or per-output-channel scales are supported")
            layers[base] = {"fp8": v["dtype"], "scale_shape": shape, "full_precision": False}
        elif k.endswith(".input_scale"):
            problems.append(f"{k}: activation-quantized (input_scale) checkpoints are not supported")
    # Optional quantization metadata (header JSON) - validate formats rather than ignore them.
    qm = meta.get("_quantization_metadata")
    if qm:
        try:
            qlayers = json.loads(qm).get("layers", {})
        except json.JSONDecodeError:
            qlayers = {}
            problems.append("unparseable _quantization_metadata")
        for name, spec in qlayers.items():
            fmt = spec.get("format")
            if fmt not in SUPPORTED_QUANT_FORMATS:
                problems.append(f"layer {name}: unsupported quantization format {fmt!r}")
            if name in layers:
                layers[name]["full_precision"] = bool(spec.get("full_precision_matrix_mult"))
            elif _strip_any(name, layers) is None:
                problems.append(f"metadata lists quantized layer {name} with no FP8 weight")
    return layers, problems


def _strip_any(name, layers):
    for p in TRANSFORMER_PREFIXES:
        if p + name in layers:
            return p + name
    return None


def _read_marker_specs(path: Path, tensors: dict, base_offset: int) -> tuple[dict, list[str]]:
    """Decode per-layer `*.comfy_quant` uint8 JSON marker tensors (a plain JSON byte string; no ComfyUI code involved)."""
    specs, problems = {}, []
    markers = [(k, v) for k, v in tensors.items() if k.endswith(".comfy_quant")]
    if not markers:
        return specs, problems
    with open(path, "rb") as f:
        for k, v in markers:
            a, b = v["data_offsets"]
            f.seek(base_offset + a)
            try:
                spec = json.loads(f.read(b - a).decode("utf-8"))
            except Exception:
                problems.append(f"{k}: unreadable quantization marker")
                continue
            specs[k[: -len(".comfy_quant")]] = spec
            if spec.get("format") not in SUPPORTED_QUANT_FORMATS:
                problems.append(f"{k}: unsupported quantization format {spec.get('format')!r}")
    return specs, problems


def inspect_checkpoint(path: str | os.PathLike, expected_kind: str) -> CheckpointInfo:
    p = Path(path)
    if not str(path).strip():
        raise CheckpointError("No path set")
    if not p.exists():
        raise CheckpointError(f"File not found: {p}")
    if p.is_dir():
        raise CheckpointError(f"{p} is a directory; select a single .safetensors file")
    if p.suffix.lower() != ".safetensors":
        raise CheckpointError(f"{p.name}: only .safetensors files are supported")

    tensors, meta = read_header(p)
    st = p.stat()
    with open(p, "rb") as f:
        base_offset = 8 + struct.unpack("<Q", f.read(8))[0]

    kind = _detect_kind(tensors)
    by_dtype: dict[str, int] = {}
    for v in tensors.values():
        a, b = v["data_offsets"]
        by_dtype[v["dtype"]] = by_dtype.get(v["dtype"], 0) + (b - a)

    qlayers, problems = _quant_layers(tensors, meta)
    marker_specs, mprob = _read_marker_specs(p, tensors, base_offset)
    problems += mprob

    float_bytes = {k: v for k, v in by_dtype.items() if k in ("BF16", "F16", "F32", "F64") or k in FP8_TAGS}
    if qlayers:
        fmt = "fp8_scaled"
        tags = sorted({q["fp8"] for q in qlayers.values()})
        stored = " + ".join({"F8_E4M3": "FP8 E4M3", "F8_E5M2": "FP8 E5M2"}[t] for t in tags) + " (scaled, per-tensor) weights; other tensors " + "/".join(
            sorted(k for k in float_bytes if k not in FP8_TAGS)
        )
    else:
        dominant = max(float_bytes, key=float_bytes.get) if float_bytes else "unknown"
        fmt = {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}.get(dominant, "unknown")
        if len([k for k in float_bytes if float_bytes[k] > 0.05 * sum(float_bytes.values())]) > 1:
            fmt = "mixed"
        stored = "/".join(sorted(float_bytes))

    if fmt == "fp8_scaled":
        note = "FP8 weights stay FP8 in RAM and VRAM; each layer is upcast to BF16 at matmul time (weight-only FP8, no FP8 tensor-core matmul)."
    elif fmt in ("fp16", "fp32", "mixed"):
        note = "Non-BF16 floating tensors are converted to BF16 at load time."
    else:
        note = "Loaded as stored (BF16)."

    info = CheckpointInfo(
        path=str(p),
        filename=p.name,
        size_bytes=st.st_size,
        mtime=st.st_mtime,
        kind=kind,
        expected_kind=expected_kind,
        compatible=False,
        format=fmt,
        stored_precision=stored,
        execution_precision="BF16",
        dequantized_at_load=fmt in ("fp16", "fp32", "mixed"),
        dequant_note=note,
        stored_bytes_by_dtype=by_dtype,
        quantized_layers=len(qlayers),
        full_precision_layers=sum(1 for q in qlayers.values() if q["full_precision"]),
        problems=problems,
    )

    if kind != expected_kind:
        info.problems.insert(0, f"Detected component type '{kind}', expected '{expected_kind}'")
    else:
        _validate_kind(info, tensors, qlayers, marker_specs)
    info.compatible = not info.problems
    return info


def _shape(tensors, key):
    return tensors[key]["shape"] if key in tensors else None


def _validate_kind(info: CheckpointInfo, tensors: dict, qlayers: dict, marker_specs: dict) -> None:
    if info.kind == KIND_TRANSFORMER:
        prefix, stripped = _strip_prefix(set(tensors), TRANSFORMER_PREFIXES)
        t = {k: tensors[v] for k, v in stripped.items()}
        nblocks = 1 + max(int(k.split(".")[1]) for k in t if k.startswith("blocks."))
        proj = _shape(t, "txtfusion.projector.weight")
        txtin = _shape(t, "txtmlp.1.weight")
        first = _shape(t, "first.weight")
        info.details = {
            "key_prefix": prefix,
            "blocks": nblocks,
            "hidden": first[0] if first else None,
            "text_layers_expected": proj[1] if proj else None,
            "text_dim_expected": txtin[1] if txtin else None,
        }
        if (nblocks, first, proj, txtin) != (28, [6144, 64], [1, 12], [6144, 2560]):
            info.problems.append(
                f"Transformer shape mismatch vs Krea 2 config (blocks={nblocks}, first={first}, projector={proj}, txtmlp={txtin})"
            )
        # Bytes that will be loaded (all tensors).
        info.weight_bytes_to_load = sum(b - a for v in tensors.values() for a, b in [v["data_offsets"]])
    elif info.kind == KIND_TEXT_ENCODER:
        lm = "model.language_model." if any(k.startswith("model.language_model.") for k in tensors) else "model."
        layers = 1 + max(int(k.split(".")[len(lm.split(".")) ]) for k in tensors if k.startswith(lm + "layers."))
        emb = _shape(tensors, lm + "embed_tokens.weight")
        q = _shape(tensors, lm + "layers.0.self_attn.q_proj.weight")
        kv = _shape(tensors, lm + "layers.0.self_attn.k_proj.weight")
        info.details = {"key_prefix": lm, "layers": layers, "hidden": emb[1] if emb else None, "vocab": emb[0] if emb else None, "has_vision_tower": any(k.startswith("model.visual.") for k in tensors)}
        if (layers, emb, q, kv) != (36, [151936, 2560], [4096, 2560], [1024, 2560]):
            info.problems.append(f"Not a Qwen3-VL-4B text encoder (layers={layers}, embed={emb}, q_proj={q}, k_proj={kv})")
        for name, spec in marker_specs.items():
            if name not in qlayers:
                info.problems.append(f"marker for {name} but no scaled FP8 weight")
        info.weight_bytes_to_load = sum(
            b - a for k, v in tensors.items() if not k.startswith("model.visual.") and not k.endswith(".comfy_quant") for a, b in [v["data_offsets"]]
        )
        info.details["vision_tower_loaded"] = False
    elif info.kind == KIND_VAE:
        diffusers_fmt = "decoder.conv_in.weight" in tensors
        zin = _shape(tensors, "decoder.conv_in.weight" if diffusers_fmt else "decoder.conv1.weight")
        info.details = {"key_format": "diffusers" if diffusers_fmt else "original (Wan/Qwen-Image)", "z_dim": zin[1] if zin else None}
        if not zin or zin[1] != 16 or zin[0] != 384:
            info.problems.append(f"VAE latent/channel shape {zin} does not match the Qwen-Image VAE (16 latent channels)")
        info.details["note"] = "Architecture matches the Qwen-Image VAE (shared with Wan 2.1); weights cannot be told apart by structure alone."
        info.weight_bytes_to_load = sum(b - a for v in tensors.values() for a, b in [v["data_offsets"]])


def validate_tokenizer_dir(path: str | os.PathLike) -> tuple[bool, list[str], dict]:
    """The Qwen3-VL config + tokenizer assets that are not embedded in a single-file checkpoint."""
    problems = []
    p = Path(path) if str(path).strip() else None
    if p is None or not p.is_dir():
        return False, [f"Config/tokenizer directory not found: {path}"], {}
    required = ["config.json", "tokenizer_config.json"]
    for r in required:
        if not (p / r).is_file():
            problems.append(f"missing {r}")
    if not (p / "tokenizer.json").is_file() and not ((p / "vocab.json").is_file() and (p / "merges.txt").is_file()):
        problems.append("missing tokenizer.json (or vocab.json + merges.txt)")
    details = {}
    if (p / "config.json").is_file():
        try:
            cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
            details = {"model_type": cfg.get("model_type"), "layers": cfg.get("text_config", {}).get("num_hidden_layers"), "hidden": cfg.get("text_config", {}).get("hidden_size")}
            if cfg.get("model_type") != "qwen3_vl" or details["layers"] != 36 or details["hidden"] != 2560:
                problems.append(f"config.json is not Qwen3-VL-4B ({details})")
        except Exception as e:
            problems.append(f"config.json unreadable: {e}")
    big = [f.name for f in p.glob("*.safetensors")] + [f.name for f in p.glob("*.bin")]
    if big:
        details["ignored_weight_files"] = big
    return not problems, problems, details


def dir_fingerprint(path: str) -> str:
    p = Path(path)
    parts = [os.path.normcase(os.path.abspath(p))]
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        f = p / name
        if f.is_file():
            s = f.stat()
            parts.append(f"{name}:{s.st_size}:{s.st_mtime}")
    return "|".join(parts)


# --------------------------------------------------------------------------------------------- loading


def load_tensors(path: str, key_map: dict[str, str], to_bf16: bool = True) -> dict[str, torch.Tensor]:
    """Read selected tensors (file key -> model key) into process RAM. FP8 tensors keep their dtype; other
    float tensors are converted to BF16 except FP8 scales (kept FP32).

    safetensors returns memory-mapped tensors whose bytes are only read from disk on first access; every
    tensor is copied into an owned allocation here so "loaded in RAM" is true (and the disk read happens
    during the loading stage, not later during the GPU upload)."""
    from safetensors import safe_open

    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for fk, mk in key_map.items():
            t = f.get_tensor(fk)
            if fk.endswith("weight_scale"):
                t = t.to(torch.float32, copy=True)
            elif to_bf16 and t.is_floating_point() and t.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) and t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)
            else:
                t = t.clone()
            out[mk] = t
    return out
