"""Vision-model settings for style analysis (the ``vision`` section of settings.json).

The app never downloads anything: the llama-server executable, the model GGUF and its vision
projector (mmproj) are paths the user supplies. Alternatively an already-running OpenAI-compatible
endpoint (LM Studio, Ollama, ...) can be used, in which case no process is started.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

STRUCTURED_MODES = ("json_schema", "json_object", "none")


class VisionSettingsError(ValueError):
    pass


@dataclass
class VisionSettings:
    # --- managed llama-server (default) ---
    llama_server_path: str = ""      # llama-server(.exe) from a llama.cpp release
    model_path: str = ""             # vision-capable model GGUF
    mmproj_path: str = ""            # matching multimodal projector GGUF (required for vision)
    n_ctx: int = 8192
    n_gpu_layers: int = 99
    extra_args: str = ""             # passed through verbatim, e.g. "--n-cpu-moe 20"
    port: int = 0                    # 0 = pick a free port
    startup_timeout_s: int = 900     # loading from a slow HDD can take minutes
    # --- or an already-running OpenAI-compatible endpoint ---
    external_url: str = ""           # e.g. http://127.0.0.1:1234 (if set, no process is started)
    external_model: str = ""
    # --- requests ---
    request_timeout_s: int = 300
    image_max_edge: int = 1024       # images are downscaled before sending
    jpeg_quality: int = 90
    temperature: float = 0.2
    max_tokens: int = 2048
    structured_mode: str = "json_schema"   # falls back automatically on HTTP 400
    chat_template_kwargs: str = '{"enable_thinking": false}'
    # --- guide writing (text-only request to the same model) ---
    guide_max_tokens: int = 3072
    guide_temperature: float = 0.5
    # --- prompt writing from the guides (text-only; the server starts without the mmproj) ---
    prompt_max_tokens: int = 1536
    prompt_temperature: float = 0.9
    # --- library / memory ---
    recursive: bool = True           # default for "include subfolders" when scanning
    release_krea_ram: bool = True    # free the Krea weights from system RAM before starting the vision model

    def model_label(self) -> str:
        """Short label stored with every analysis so results can be attributed to a model."""
        if self.external_url:
            return self.external_model or "external"
        return Path(self.model_path).name if self.model_path else "unset"


def _coerce(name: str, value, default):
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        raise VisionSettingsError(f"{name} must be true or false")
    if isinstance(default, int):
        if isinstance(value, bool):
            raise VisionSettingsError(f"{name} must be an integer")
        try:
            return int(value)
        except (TypeError, ValueError):
            raise VisionSettingsError(f"{name} must be an integer") from None
    if isinstance(default, float):
        if isinstance(value, bool):
            raise VisionSettingsError(f"{name} must be a number")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise VisionSettingsError(f"{name} must be a number") from None
    if not isinstance(value, str):
        raise VisionSettingsError(f"{name} must be text")
    return value.strip().strip('"') if name.endswith(("_path", "_url")) else value


def validate(s: VisionSettings) -> None:
    """Value-range checks that do not touch the filesystem."""
    if s.structured_mode not in STRUCTURED_MODES:
        raise VisionSettingsError(f"structured_mode must be one of {', '.join(STRUCTURED_MODES)}")
    if not 64 <= s.image_max_edge <= 4096:
        raise VisionSettingsError("image_max_edge must be between 64 and 4096")
    if not 1 <= s.jpeg_quality <= 100:
        raise VisionSettingsError("jpeg_quality must be between 1 and 100")
    if not 0 <= s.port <= 65535:
        raise VisionSettingsError("port must be between 0 and 65535")
    if s.n_ctx < 2048:
        raise VisionSettingsError("n_ctx (context size) must be at least 2048")
    if not 256 <= s.max_tokens <= s.n_ctx:
        raise VisionSettingsError("max_tokens must be between 256 and the context size")
    if not 256 <= s.guide_max_tokens <= s.n_ctx - 1024:
        raise VisionSettingsError(f"guide_max_tokens must be between 256 and the context size minus 1024 ({s.n_ctx - 1024}), "
                                  "leaving room for the data")
    if not 256 <= s.prompt_max_tokens <= s.n_ctx - 1024:
        raise VisionSettingsError(f"prompt_max_tokens must be between 256 and the context size minus 1024 ({s.n_ctx - 1024})")
    for name in ("temperature", "guide_temperature", "prompt_temperature"):
        if not 0 <= getattr(s, name) <= 2:
            raise VisionSettingsError(f"{name} must be between 0 and 2")
    for name in ("startup_timeout_s", "request_timeout_s"):
        if not 10 <= getattr(s, name) <= 86400:
            raise VisionSettingsError(f"{name} must be between 10 and 86400 seconds")
    if s.external_url and not s.external_url.startswith(("http://", "https://")):
        raise VisionSettingsError("The endpoint URL must start with http:// or https://")
    try:
        parsed = json.loads(s.chat_template_kwargs or "{}")
    except json.JSONDecodeError as exc:
        raise VisionSettingsError(f"chat_template_kwargs is not valid JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise VisionSettingsError("chat_template_kwargs must be a JSON object")


def merge(base: dict | None, patch, strict: bool = True) -> dict:
    """``base`` (a stored dict) updated with ``patch``; returns a plain dict for settings.json.

    ``strict=False`` (loading a file) skips unknown keys and bad values instead of raising, and falls
    back to the defaults if the result is invalid as a whole."""
    if not isinstance(patch, dict):
        if strict:
            raise VisionSettingsError("vision settings must be an object")
        patch = {}
    known = {f.name for f in fields(VisionSettings)}
    s = VisionSettings()
    for name, value in (base or {}).items():  # base was validated when stored; be lenient anyway
        if name in known:
            try:
                setattr(s, name, _coerce(name, value, getattr(s, name)))
            except VisionSettingsError:
                pass
    for name, value in patch.items():
        if name not in known:
            if strict:
                raise VisionSettingsError(f"Unknown vision setting: {name}")
            continue
        try:
            setattr(s, name, _coerce(name, value, getattr(s, name)))
        except VisionSettingsError:
            if strict:
                raise
    try:
        validate(s)
    except VisionSettingsError:
        if strict:
            raise
        s = VisionSettings()
    return asdict(s)


def from_dict(d: dict | None) -> VisionSettings:
    return VisionSettings(**merge(None, d or {}, strict=False))


def check_ready(s: VisionSettings, vision: bool = True) -> list[str]:
    """Problems that would stop a job from starting (filesystem checks). ``vision=False`` (text-only jobs such as
    prompt writing) does not need the vision projector."""
    if s.external_url:
        return []
    problems: list[str] = []
    needed = [("llama_server_path", "llama-server executable"), ("model_path", "vision model GGUF")]
    if vision:
        needed.append(("mmproj_path", "vision projector (mmproj) GGUF"))
    for attr, label in needed:
        value = getattr(s, attr)
        if not value:
            problems.append(f"Set the {label} in Settings → Style analysis.")
        elif not Path(value).is_file():
            problems.append(f"{label[0].upper() + label[1:]} not found: {value}")
    return problems
