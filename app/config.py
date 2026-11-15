"""Settings persisted in settings.json at the project root, plus validation of model paths."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import checkpoints as ck

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "settings.json"
MAX_SEED = 2**53 - 1  # exactly representable in JavaScript

_DEFAULT_DIR = ROOT / "models" / "krea2turbo"
LORA_DIR = ROOT / "models" / "loras"
# User data (character database + document folders). PROMPTHOUSE_DATA_DIR moves it, e.g. for tests.
DATA_DIR = Path(os.environ.get("PROMPTHOUSE_DATA_DIR") or ROOT / "data")
DB_FILE = DATA_DIR / "prompthouse.db"  # characters (SQLite)
CHARACTER_DOCS_DIR = DATA_DIR / "characters"  # one Markdown folder per character
MAX_LORAS = 16
LORA_STRENGTH_LIMIT = 5.0


def _default(name: str) -> str:
    return str(_DEFAULT_DIR / name)


@dataclass
class Settings:
    transformer_path: str = field(default_factory=lambda: _default("krea2_turbo_fp8_scaled.safetensors"))
    text_encoder_path: str = field(default_factory=lambda: _default("qwen3vl_4b_fp8_scaled.safetensors"))
    text_encoder_config_dir: str = field(default_factory=lambda: _default("qwen3vl_4b_config"))
    vae_path: str = field(default_factory=lambda: _default("qwen_image_vae.safetensors"))
    seed: int = 42
    randomize_seed: bool = False
    keep_in_ram: dict = field(default_factory=lambda: {"text_encoder": False, "transformer": True, "vae": True})
    # Ordered LoRA stack: [{"path": str, "strength": float, "enabled": bool}]
    loras: list = field(default_factory=list)

    def model_fields(self) -> dict:
        return {k: getattr(self, k) for k in ("transformer_path", "text_encoder_path", "text_encoder_config_dir", "vae_path")}


class SettingsError(ValueError):
    pass


_lock = threading.Lock()


def _coerce(data: dict, base: Settings) -> Settings:
    s = Settings(**asdict(base))
    for key in ("transformer_path", "text_encoder_path", "text_encoder_config_dir", "vae_path"):
        if key in data:
            v = data[key]
            if not isinstance(v, str):
                raise SettingsError(f"{key} must be a string")
            setattr(s, key, v.strip().strip('"'))
    if "seed" in data:
        v = data["seed"]
        if isinstance(v, bool) or not isinstance(v, int) and not (isinstance(v, str) and v.strip().isdigit()):
            raise SettingsError("seed must be a non-negative integer")
        v = int(v)
        if not 0 <= v <= MAX_SEED:
            raise SettingsError(f"seed must be between 0 and {MAX_SEED}")
        s.seed = v
    if "randomize_seed" in data:
        if not isinstance(data["randomize_seed"], bool):
            raise SettingsError("randomize_seed must be true/false")
        s.randomize_seed = data["randomize_seed"]
    if "keep_in_ram" in data:
        k = data["keep_in_ram"]
        if not isinstance(k, dict) or not all(c in ("text_encoder", "transformer", "vae") and isinstance(b, bool) for c, b in k.items()):
            raise SettingsError("keep_in_ram must map text_encoder/transformer/vae to booleans")
        s.keep_in_ram = {**s.keep_in_ram, **k}
    if "loras" in data:
        s.loras = coerce_lora_stack(data["loras"])
    return s


def coerce_lora_stack(v) -> list[dict]:
    if not isinstance(v, list):
        raise SettingsError("loras must be a list")
    if len(v) > MAX_LORAS:
        raise SettingsError(f"At most {MAX_LORAS} LoRAs can be stacked")
    out, seen = [], set()
    for i, e in enumerate(v):
        if not isinstance(e, dict) or set(e) - {"path", "strength", "enabled"}:
            raise SettingsError(f"LoRA #{i + 1}: expected an object with path, strength, enabled")
        path = e.get("path")
        if not isinstance(path, str) or not path.strip().strip('"'):
            raise SettingsError(f"LoRA #{i + 1}: path must be a non-empty string")
        path = path.strip().strip('"')
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            raise SettingsError(f"LoRA {os.path.basename(path)} is already in the stack (raise its strength instead)")
        seen.add(key)
        strength = e.get("strength", 1.0)
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength):
            raise SettingsError(f"LoRA {os.path.basename(path)}: strength must be a number")
        if abs(strength) > LORA_STRENGTH_LIMIT:
            raise SettingsError(f"LoRA {os.path.basename(path)}: strength must be between -{LORA_STRENGTH_LIMIT:g} and {LORA_STRENGTH_LIMIT:g}")
        enabled = e.get("enabled", True)
        if not isinstance(enabled, bool):
            raise SettingsError(f"LoRA {os.path.basename(path)}: enabled must be true/false")
        out.append({"path": path, "strength": round(float(strength), 4), "enabled": enabled})
    return out


def load_settings(path: Path | None = None) -> Settings:
    path = path or SETTINGS_FILE
    with _lock:
        if not path.exists():
            return Settings()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise SettingsError(f"{path.name} is unreadable: {e}") from e
        return _coerce(data, Settings())


def save_settings(s: Settings, path: Path | None = None) -> None:
    path = path or SETTINGS_FILE
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".settings-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(s), f, indent=2)
        os.replace(tmp, path)  # atomic


def update_settings(current: Settings, patch: dict) -> Settings:
    unknown = set(patch) - set(asdict(current))
    if unknown:
        raise SettingsError(f"Unknown settings: {sorted(unknown)}")
    return _coerce(patch, current)


def validate_models(s: Settings) -> dict:
    """Inspect every configured model path. Never loads weights; reads headers only."""
    out = {}
    for key, kind in (("transformer_path", ck.KIND_TRANSFORMER), ("text_encoder_path", ck.KIND_TEXT_ENCODER), ("vae_path", ck.KIND_VAE)):
        try:
            out[key] = ck.inspect_checkpoint(getattr(s, key), kind).to_dict()
        except (ck.CheckpointError, OSError) as e:
            out[key] = {"compatible": False, "problems": [str(e)], "path": getattr(s, key)}
    ok, problems, details = ck.validate_tokenizer_dir(s.text_encoder_config_dir)
    out["text_encoder_config_dir"] = {"compatible": ok, "problems": problems, "details": details, "path": s.text_encoder_config_dir}
    return out
