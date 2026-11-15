"""FastAPI application. Run with:  .venv\\Scripts\\python.exe run.py   (binds 127.0.0.1, single worker)."""

from __future__ import annotations

import logging
import os
import string
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import checkpoints as ck
from . import resolution as resmod
from .conditioning import PromptCache
from .config import LORA_DIR, MAX_LORAS, MAX_SEED, Settings, SettingsError, load_settings, save_settings, update_settings, validate_models
from .jobs import Busy, JobRunner, new_random_seed
from .lora import STRENGTH_LIMIT, LoraLibrary
from .model_manager import Component, ModelError, ModelManager
from .monitoring import Monitor
from .outputs import image_path

log = logging.getLogger("krea.app")
STATIC = Path(__file__).resolve().parent.parent / "static"


class State:
    settings: Settings
    manager: ModelManager | None = None
    runner: JobRunner | None = None
    cache: PromptCache
    monitor: Monitor
    loras: LoraLibrary
    startup_error: str | None = None


S = State()


def _apply_settings_to_manager(s: Settings) -> list[Component]:
    assert S.manager is not None
    return S.manager.configure(
        {Component.TRANSFORMER: s.transformer_path, Component.TEXT_ENCODER: s.text_encoder_path, Component.VAE: s.vae_path},
        s.text_encoder_config_dir,
        {Component(k): bool(v) for k, v in s.keep_in_ram.items()},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    S.cache = PromptCache()
    S.monitor = Monitor()
    S.loras = LoraLibrary()
    try:
        S.settings = load_settings()
    except SettingsError as e:
        S.settings = Settings()
        S.startup_error = f"settings.json could not be read ({e}); defaults in use"
    try:
        S.manager = ModelManager()
        S.runner = JobRunner(S.manager, S.cache, S.loras)
        S.runner.run_on_worker(lambda: _apply_settings_to_manager(S.settings))
    except ModelError as e:
        S.startup_error = str(e)
        log.error("GPU initialisation failed: %s", e)
    yield


app = FastAPI(title="Krea 2 Turbo", lifespan=lifespan)


def _runner() -> JobRunner:
    if S.runner is None:
        raise HTTPException(503, S.startup_error or "GPU backend unavailable")
    return S.runner


# ------------------------------------------------------------------------------------------ status


@app.get("/api/status")
def status():
    runner = S.runner
    cond = S.cache.get()
    return {
        "startup_error": S.startup_error,
        "busy": bool(runner and runner.busy),
        "active_job": runner.active.to_dict() if runner and runner.active else None,
        "conditioning": cond.describe() if cond else None,
        "conditioning_valid_for_current_encoder": bool(cond and S.manager and _safe_fp() == cond.encoder_fingerprint),
        "models": S.manager.snapshot() if S.manager else None,
        "settings": {"seed": S.settings.seed, "randomize_seed": S.settings.randomize_seed},
        "lora_cache": S.loras.snapshot(),
    }


def _safe_fp() -> str | None:
    try:
        return S.manager.text_encoder_fingerprint()
    except Exception:  # noqa: BLE001
        return None


@app.get("/api/system")
def system():
    return S.monitor.sample()


@app.get("/api/resolutions")
def resolutions():
    return resmod.table()


# ------------------------------------------------------------------------------------------ settings


@app.get("/api/settings")
def get_settings():
    return {"settings": S.settings.__dict__, "validation": validate_models(S.settings), "max_seed": MAX_SEED}


@app.put("/api/settings")
def put_settings(patch: dict):
    try:
        new = update_settings(S.settings, patch)
    except SettingsError as e:
        raise HTTPException(400, str(e)) from e
    models_changed = new.model_fields() != S.settings.model_fields() or new.keep_in_ram != S.settings.keep_in_ram
    if models_changed and S.runner and S.runner.busy:
        raise HTTPException(409, "Model settings cannot change while a GPU job is running. Try again when it finishes.")
    changed: list[Component] = []
    if models_changed and S.manager is not None:
        try:
            changed = S.runner.run_on_worker(lambda: _apply_settings_to_manager(new)) if S.runner else _apply_settings_to_manager(new)
        except (ModelError, Busy) as e:
            raise HTTPException(409, str(e)) from e
        if Component.TEXT_ENCODER in changed:
            S.cache.invalidate()  # conditioning came from a different text encoder
    save_settings(new)
    S.settings = new
    return {"settings": new.__dict__, "validation": validate_models(new), "invalidated": [c.value for c in changed], "max_seed": MAX_SEED}


# ------------------------------------------------------------------------------------------ LoRA stack


def _lora_view() -> dict:
    stack = [{**e, "info": S.loras.inspect(e["path"]).to_dict()} for e in S.settings.loras]
    return {"stack": stack, "default_dir": str(LORA_DIR), "max_loras": MAX_LORAS, "strength_limit": STRENGTH_LIMIT,
            "cache": S.loras.snapshot(), "busy": bool(S.runner and S.runner.busy)}


@app.get("/api/loras")
def get_loras():
    return _lora_view()


class LoraStack(BaseModel):
    stack: list[dict]


@app.put("/api/loras")
def put_loras(body: LoraStack):
    """Replace the ordered LoRA stack (add / remove / reorder / enable / strength are all edits of this list).

    Allowed while a job runs: a generation uses the stack captured when it was submitted. Only adapter files
    are affected; the base transformer is never reloaded for a stack change."""
    try:
        new = update_settings(S.settings, {"loras": body.stack})
    except SettingsError as e:
        raise HTTPException(400, str(e)) from e
    bad = []
    old_paths = {e["path"] for e in S.settings.loras}
    for e in new.loras:
        info = S.loras.inspect(e["path"])
        if not info.compatible and e["path"] not in old_paths:  # entries already in the stack may be kept/edited (and show their error)
            bad.append(f"{info.filename}: " + "; ".join(info.problems))
    if bad:
        raise HTTPException(400, "Incompatible LoRA: " + " | ".join(bad))
    save_settings(new)
    S.settings = new
    if S.runner:  # drop removed adapters from RAM on the worker thread (after any running job; never blocks this request)
        keep = [e["path"] for e in new.loras]
        S.runner.defer_to_worker(lambda: S.loras.evict_except(keep))
    return _lora_view()


class PathCheck(BaseModel):
    kind: str
    path: str


@app.post("/api/validate-path")
def validate_path(body: PathCheck):
    kinds = {"transformer_path": ck.KIND_TRANSFORMER, "text_encoder_path": ck.KIND_TEXT_ENCODER, "vae_path": ck.KIND_VAE}
    if body.kind == "lora":
        return S.loras.inspect(body.path).to_dict()
    if body.kind == "text_encoder_config_dir":
        ok, problems, details = ck.validate_tokenizer_dir(body.path.strip().strip('"'))
        return {"compatible": ok, "problems": problems, "details": details}
    if body.kind not in kinds:
        raise HTTPException(400, "unknown kind")
    try:
        return ck.inspect_checkpoint(body.path.strip().strip('"'), kinds[body.kind]).to_dict()
    except (ck.CheckpointError, OSError) as e:
        return {"compatible": False, "problems": [str(e)]}


@app.get("/api/browse")
def browse(path: str = Query("")):
    """Minimal server-side file browser for picking local model paths (directories + .safetensors only)."""
    if not path:
        if os.name == "nt":
            drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
            return {"path": "", "parent": None, "entries": [{"name": d, "path": d, "type": "dir"} for d in drives]}
        path = "/"
    p = Path(path)
    if not p.is_dir():
        raise HTTPException(400, f"Not a directory: {path}")
    entries = []
    try:
        for c in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            try:
                if c.is_dir():
                    entries.append({"name": c.name, "path": str(c), "type": "dir"})
                elif c.suffix.lower() == ".safetensors":
                    entries.append({"name": c.name, "path": str(c), "type": "file", "size": c.stat().st_size})
            except OSError:
                continue
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    parent = str(p.parent) if p.parent != p else ""
    return {"path": str(p), "parent": parent, "entries": entries}


# ------------------------------------------------------------------------------------------ GPU jobs


class EncodeRequest(BaseModel):
    prompt: str = Field(..., max_length=20000)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=20000)
    aspect: str
    megapixels: float
    multiple: int
    width: int
    height: int


@app.post("/api/encode")
def encode(req: EncodeRequest):
    runner = _runner()
    try:
        job = runner.submit_encode(req.prompt)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Busy as e:
        raise HTTPException(409, str(e)) from e
    return job.to_dict()


@app.post("/api/generate")
def generate(req: GenerateRequest):
    runner = _runner()
    try:
        res = resmod.resolve(req.aspect, req.megapixels, req.multiple)
    except resmod.ResolutionError as e:
        raise HTTPException(400, str(e)) from e
    if (res.width, res.height) != (req.width, req.height):
        raise HTTPException(400, f"Displayed size {req.width}x{req.height} disagrees with the server's {res.width}x{res.height}; reload the page.")
    seed = new_random_seed() if S.settings.randomize_seed else S.settings.seed
    try:
        job = runner.submit_generate(req.prompt, res, seed, S.settings.loras)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Busy as e:
        raise HTTPException(409, str(e)) from e
    d = job.to_dict()
    d["warnings"] = list(res.warnings)
    return d


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    runner = _runner()
    job = runner.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job.to_dict()


@app.get("/api/images/{image_id}")
def get_image(image_id: str, download: bool = False):
    p = image_path(image_id)
    if p is None:
        raise HTTPException(404, "Image not found")
    return FileResponse(p, media_type="image/png", filename=p.name if download else None,
                        content_disposition_type="attachment" if download else "inline")


@app.exception_handler(Exception)
async def unhandled(_req, exc):  # pragma: no cover
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
