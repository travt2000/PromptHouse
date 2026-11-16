"""FastAPI application. Run with:  .venv\\Scripts\\python.exe run.py   (binds 127.0.0.1, single worker)."""

from __future__ import annotations

import io
import logging
import os
import string
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from . import checkpoints as ck
from . import config as cfg
from . import resolution as resmod
from . import character_docs as cdocs
from .characters import CharacterError, CharacterStore, Duplicate, NotFound, compose_prompt
from .conditioning import PromptCache
from .config import LORA_DIR, MAX_LORAS, MAX_SEED, Settings, SettingsError, load_settings, save_settings, update_settings, validate_models
from .jobs import Busy, JobRunner, new_random_seed
from .lora import STRENGTH_LIMIT, LoraLibrary
from .model_manager import Component, ModelError, ModelManager
from .monitoring import Monitor
from .outputs import image_path
from .style import vsettings as vs
from .style.character_types import CharacterTypesError, load_character_types
from .style.kinds import KINDS as STYLE_KINDS, get_kind
from .prompt_files import PromptFileError
from .style import prompts as sprompts
from .style.llama_server import ServerManager
from .style.runner import StyleRunner
from .style.scanner import scan_folder
from .style.store import StyleStore

log = logging.getLogger("krea.app")
STATIC = Path(__file__).resolve().parent.parent / "static"


class State:
    settings: Settings
    manager: ModelManager | None = None
    runner: JobRunner | None = None
    cache: PromptCache
    monitor: Monitor
    loras: LoraLibrary
    characters: CharacterStore | None = None
    # None = landing page; {"character_id": int | None} = generation workspace (None id = no character)
    workspace: dict | None = None
    docs_root: Path
    doc_index: cdocs.DocIndex
    style_store: StyleStore | None = None
    style_server: ServerManager | None = None
    style: StyleRunner | None = None
    style_thumbs: Path
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
    S.workspace = None
    S.docs_root = Path(cfg.CHARACTER_DOCS_DIR)
    S.doc_index = cdocs.DocIndex()
    S.style_thumbs = Path(cfg.STYLE_THUMBS_DIR)
    S.style_server = ServerManager(Path(cfg.LLAMA_LOG_FILE))
    try:
        S.characters = CharacterStore(cfg.DB_FILE)
        for c in S.characters.list():  # characters created before document folders existed
            if not c["folder"]:
                try:
                    _provision_folder(c)
                except OSError as e:
                    log.warning("could not create the document folder for %s: %s", c["name"], e)
        S.style_store = StyleStore(cfg.DB_FILE)  # tables created by the character store's migrations
        S.style = StyleRunner(S.style_store, S.style_server, _launch_style, _prepare_for_vision)
    except Exception as e:  # noqa: BLE001 - a broken DB must not take down generation
        S.characters = None
        S.style_store = S.style = None
        S.startup_error = f"Character database unavailable ({e})"
        log.error("character DB: %s", e)
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
    if S.style is not None:  # never leave a vision model server running after the app exits
        S.style.cancel()
    if S.style_server is not None:
        S.style_server.stop()
    if S.style is not None:
        S.style.join(15)


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
        "workspace": _workspace_view(),
        "style": S.style.status() if S.style else None,
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
    return {"settings": S.settings.__dict__, "validation": validate_models(S.settings), "max_seed": MAX_SEED,
            "vision_problems": vs.check_ready(S.settings.vision_settings())}


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
    return {"settings": new.__dict__, "validation": validate_models(new), "invalidated": [c.value for c in changed], "max_seed": MAX_SEED,
            "vision_problems": vs.check_ready(new.vision_settings())}


# ------------------------------------------------------------------------------------------ LoRA stack


def _lora_view() -> dict:
    stack = [{**e, "info": S.loras.inspect(e["path"]).to_dict()} for e in S.settings.loras]
    c = _active_character()
    locked = None
    if c and c["lora_path"]:
        key = _pathkey(c["lora_path"])
        for e in stack:
            e["shadowed"] = _pathkey(e["path"]) == key  # the character's LoRA replaces this entry while active
        locked = {"path": c["lora_path"], "strength": c["lora_strength"], "enabled": True, "character": c["name"],
                  "info": S.loras.inspect(c["lora_path"]).to_dict()}
    return {"stack": stack, "locked": locked, "default_dir": str(LORA_DIR), "max_loras": MAX_LORAS, "strength_limit": STRENGTH_LIMIT,
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
        _evict_unused()
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


BROWSE_EXTS = {".safetensors", ".gguf", ".exe"}


@app.get("/api/browse")
def browse(path: str = Query(""), ext: str = Query(".safetensors")):
    """Minimal server-side file browser for picking local model paths: directories plus files with the
    requested extensions (comma-separated; .safetensors, .gguf or .exe)."""
    exts = {e.strip().lower() for e in ext.split(",") if e.strip()}
    if not exts or not exts <= BROWSE_EXTS:
        raise HTTPException(400, f"ext must be among {sorted(BROWSE_EXTS)}")
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
                elif c.suffix.lower() in exts:
                    entries.append({"name": c.name, "path": str(c), "type": "file", "size": c.stat().st_size})
            except OSError:
                continue
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    parent = str(p.parent) if p.parent != p else ""
    return {"path": str(p), "parent": parent, "entries": entries}


# ------------------------------------------------------------------------------------------ characters


def _pathkey(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _store() -> CharacterStore:
    if S.characters is None:
        raise HTTPException(503, S.startup_error or "Character database unavailable")
    return S.characters


def _active_character() -> dict | None:
    if not S.workspace or S.workspace.get("character_id") is None or S.characters is None:
        return None
    try:
        return S.characters.get(S.workspace["character_id"])
    except NotFound:
        S.workspace = None  # deleted elsewhere
        return None


def _workspace_view() -> dict | None:
    if S.workspace is None:
        return None
    return {"character": _active_character()}


def _effective_stack() -> list[dict]:
    """The stack a generation uses: the active character's LoRA first (locked), then the user's stack
    without any entry for the same file."""
    user = [dict(e) for e in S.settings.loras]
    c = _active_character()
    if not c or not c["lora_path"]:
        return user
    key = _pathkey(c["lora_path"])
    return [{"path": c["lora_path"], "strength": c["lora_strength"], "enabled": True}] + [e for e in user if _pathkey(e["path"]) != key]


def _evict_unused() -> None:
    """Free cached adapters that neither the user's stack nor the active character uses (on the GPU worker)."""
    if S.runner:
        keep = [e["path"] for e in _effective_stack()]
        S.runner.defer_to_worker(lambda: S.loras.evict_except(keep))


def _provision_folder(c: dict) -> dict:
    taken = {x["folder"] for x in S.characters.list() if x["folder"]}
    folder = cdocs.allocate_folder(S.docs_root, c["name"], taken)
    cdocs.provision(S.docs_root, folder, c["name"])
    return S.characters.set_folder(c["id"], folder)


def _character_view(c: dict) -> dict:
    out = dict(c)
    out["docs_dir"] = str(S.docs_root / c["folder"]) if c["folder"] else None
    out["lora_info"] = S.loras.inspect(c["lora_path"]).to_dict() if c["lora_path"] else None
    out["character_type_info"] = _character_type(c["character_type"])
    return out


def _character_type(type_id: str):
    """The configured type for an id: its dict, None if unset, or {"id", "missing": True} if the id is no
    longer in config/character_types.json (or the file is broken)."""
    if not type_id:
        return None
    try:
        t = next((t for t in load_character_types() if t.id == type_id), None)
    except CharacterTypesError:
        t = None
    return t.to_dict() if t else {"id": type_id, "label": type_id, "missing": True}


def _check_character_type(body: dict, current: str = "") -> None:
    """A new type must exist in the types file; keeping the current one is always allowed."""
    t = body.get("character_type") if isinstance(body, dict) else None
    if not isinstance(t, str) or not t.strip() or t.strip() == current:
        return
    try:
        known = {x.id for x in load_character_types()}
    except CharacterTypesError as e:
        raise HTTPException(400, str(e)) from e
    if t.strip() not in known:
        raise HTTPException(400, f"Unknown character type: {t}")


def _check_character_lora(path) -> None:
    if isinstance(path, str) and path.strip().strip('"'):
        info = S.loras.inspect(path.strip().strip('"'))
        if not info.compatible:
            raise HTTPException(400, f"LoRA {info.filename}: " + "; ".join(info.problems))


def _character_call(fn):
    try:
        return fn()
    except NotFound as e:
        raise HTTPException(404, str(e)) from e
    except Duplicate as e:
        raise HTTPException(409, str(e)) from e
    except CharacterError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/characters")
def list_characters():
    return {"characters": [_character_view(c) for c in _store().list()], "workspace": _workspace_view()}


@app.post("/api/characters")
def create_character(body: dict):
    store = _store()
    _check_character_lora(body.get("lora_path"))
    _check_character_type(body)
    c = _character_call(lambda: store.create(body))
    try:
        c = _provision_folder(c)
    except (OSError, cdocs.DocError) as e:
        store.delete(c["id"])  # do not leave a character without its folder
        raise HTTPException(500, f"Could not create the character's document folder: {e}") from e
    return _character_view(c)


@app.get("/api/characters/{cid}")
def get_character(cid: int):
    store = _store()
    return _character_view(_character_call(lambda: store.get(cid)))


@app.put("/api/characters/{cid}")
def update_character(cid: int, body: dict):
    store = _store()
    _check_character_lora(body.get("lora_path"))
    _check_character_type(body, _character_call(lambda: store.get(cid))["character_type"])
    c = _character_call(lambda: store.update(cid, body))
    if S.workspace and S.workspace.get("character_id") == cid:
        _evict_unused()  # the active character's LoRA may have changed
    return _character_view(c)


@app.delete("/api/characters/{cid}")
def delete_character(cid: int):
    store = _store()
    c = _character_call(lambda: store.get(cid))
    if S.style and S.style.active and S.style.status()["character_id"] == cid:
        raise HTTPException(409, "A style job is running for this character; cancel it or wait for it to finish.")
    _character_call(lambda: store.delete(cid))  # never touches the LoRA file, the document folder or any image file
    S.doc_index.forget(cid)
    if S.workspace and S.workspace.get("character_id") == cid:
        S.workspace = None
        _evict_unused()
    kept = str(S.docs_root / c["folder"]) if c["folder"] and (S.docs_root / c["folder"]).exists() else None
    return {"deleted": cid, "docs_kept": kept, "workspace": _workspace_view()}


# ------------------------------------------------------------------------------------------ character documents


def _docs_folder(cid: int) -> tuple[dict, Path]:
    store = _store()
    c = _character_call(lambda: store.get(cid))
    if not c["folder"]:
        try:
            c = _provision_folder(c)
        except (OSError, cdocs.DocError) as e:
            raise HTTPException(500, f"Could not create the character's document folder: {e}") from e
    d = S.docs_root / c["folder"]
    if not d.is_dir():  # removed outside the app: recreate it empty (templates are only for new characters)
        d.mkdir(parents=True, exist_ok=True)
    return c, d


def _doc_call(fn):
    try:
        return fn()
    except cdocs.DocConflict as e:
        raise HTTPException(409, str(e)) from e
    except cdocs.DocError as e:
        raise HTTPException(400, str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(404, f"Not found: {e}") from e
    except OSError as e:
        raise HTTPException(500, f"File error: {e}") from e


@app.get("/api/characters/{cid}/docs")
def list_character_docs(cid: int):
    c, d = _docs_folder(cid)
    return {"character_id": cid, "folder": str(d), "files": _doc_call(lambda: cdocs.list_docs(d))}


@app.get("/api/characters/{cid}/docs/file")
def read_character_doc(cid: int, path: str = Query(...)):
    _, d = _docs_folder(cid)
    return _doc_call(lambda: cdocs.read_doc(d, path))


class DocWrite(BaseModel):
    path: str
    content: str = Field(..., max_length=cdocs.MAX_DOC_BYTES)
    expected_mtime: str | None = None  # from the last read/write; None = create a new file (must not exist)


@app.put("/api/characters/{cid}/docs/file")
def write_character_doc(cid: int, body: DocWrite):
    _, d = _docs_folder(cid)
    return _doc_call(lambda: cdocs.write_doc(d, body.path, body.content, body.expected_mtime))


@app.delete("/api/characters/{cid}/docs/file")
def delete_character_doc(cid: int, path: str = Query(...)):
    _, d = _docs_folder(cid)
    _doc_call(lambda: cdocs.delete_doc(d, path))
    return {"deleted": path}


@app.post("/api/characters/{cid}/docs/open")
def open_character_folder(cid: int):
    """Open the folder in the OS file manager (this is a local, single-user app bound to 127.0.0.1)."""
    _, d = _docs_folder(cid)
    if not hasattr(os, "startfile"):
        raise HTTPException(501, f"Opening folders is only supported on Windows. Folder: {d}")
    os.startfile(d)  # noqa: S606 - fixed path under the documents root
    return {"opened": str(d)}


@app.get("/api/characters/{cid}/context")
def character_context(cid: int, q: str = Query("", max_length=2000), k: int = Query(5, ge=1, le=50)):
    """Retrieval groundwork: the character's documents as chunks, and the best matches for `q`.
    Lexical (BM25) for now; this is the call future LLM features will use to fetch context."""
    c, _ = _docs_folder(cid)
    chunks, warnings = S.doc_index.chunks(cid, S.docs_root, c["folder"])
    hits = cdocs.DocIndex.search(chunks, q, k) if q.strip() else []
    return {"character_id": cid, "retriever": "bm25", "chunks_total": len(chunks), "warnings": warnings,
            "results": [{**ch.to_dict(), "score": score} for ch, score in hits]}


# ------------------------------------------------------------------------------------------ style analysis
# Per-character image libraries read by a local vision model (llama-server), and guides written from them.
# Jobs run on the GPU worker (see app/style/runner.py), so they never overlap encode/generate.

StyleKind = Literal["character", "inspiration"]


def _launch_style(request: dict, work) -> None:
    runner = _runner()
    runner.submit_task("vision", request, lambda job: work(job.set_stage))


def _prepare_for_vision(v: vs.VisionSettings) -> None:
    """On the GPU worker, before the vision model starts: hand back cached VRAM and (by default) free the
    Krea weights and LoRA adapters from system RAM. They are re-read from disk by the next generation."""
    if S.manager is not None:
        S.manager.make_room_for_external(v.release_krea_ram)
    if v.release_krea_ram:
        S.loras.evict_except([])


def _style() -> StyleRunner:
    if S.style is None or S.style_store is None:
        raise HTTPException(503, S.startup_error or "Character database unavailable")
    return S.style


def _style_kind(key: str):
    try:
        return get_kind(key)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _style_counts(cid: int, kind_key: str, v: vs.VisionSettings | None = None) -> dict:
    kind = get_kind(kind_key)
    return S.style_store.counts(cid, kind.key, kind.schema_version, (v or S.settings.vision_settings()).model_label())


def _style_image(image_id: int) -> dict:
    _style()
    image = S.style_store.get_image(image_id)
    if image is None:
        raise HTTPException(404, "No such image.")
    return image


def _gpu_free_for_style(style: StyleRunner) -> None:
    if style.active:
        raise HTTPException(409, "A style job is already running; only one runs at a time.")
    if S.runner is not None and S.runner.busy:
        raise HTTPException(409, f"A {S.runner.active.kind} job is using the GPU; try again when it finishes.")


@app.get("/api/style/character-types")
def style_character_types():
    """The configured character types. The file is re-read each time; a broken file is reported, not raised."""
    try:
        return {"types": [t.to_dict() for t in load_character_types()], "error": None}
    except CharacterTypesError as e:
        return {"types": [], "error": str(e)}


@app.get("/api/style/status")
def style_status():
    return _style().status()


@app.post("/api/style/cancel")
def style_cancel():
    _style().cancel()
    return {"ok": True}


@app.get("/api/style/server-log")
def style_server_log():
    return {"log": S.style_server.log_tail(80) if S.style_server else ""}


@app.get("/api/characters/{cid}/style")
def character_style(cid: int):
    style = _style()
    store = _store()
    c = _character_call(lambda: store.get(cid))
    v = S.settings.vision_settings()
    latest = {}
    for k in STYLE_KINDS:
        items = S.style_store.list_guides(cid, k)
        latest[k] = items[0] if items else None
    return {"character": _character_view(c), "model_label": v.model_label(), "problems": vs.check_ready(v),
            "recursive": v.recursive, "counts": {k: _style_counts(cid, k, v) for k in STYLE_KINDS},
            "latest_guides": latest, "status": style.status()}


@app.get("/api/characters/{cid}/style/images")
def character_style_images(cid: int, kind: StyleKind, status: str | None = Query(None, pattern="^(ok|failed|pending)$"),
                           limit: int = Query(60, ge=1, le=500), offset: int = Query(0, ge=0)):
    _style()
    store = _store()
    _character_call(lambda: store.get(cid))
    k = get_kind(kind)
    label = S.settings.vision_settings().model_label()
    items, total = S.style_store.list_images(cid, k.key, status, k.schema_version, label, limit, offset)
    return {"items": items, "total": total, "counts": _style_counts(cid, k.key)}


class StyleScan(BaseModel):
    kind: StyleKind
    folder: str = Field(..., max_length=4096)
    recursive: bool | None = None
    include_removed: bool = False


@app.post("/api/characters/{cid}/style/scan")
def character_style_scan(cid: int, req: StyleScan):
    """Register a folder's images in one of the character's libraries (files are only read)."""
    _style()
    store = _store()
    _character_call(lambda: store.get(cid))
    recursive = S.settings.vision_settings().recursive if req.recursive is None else req.recursive
    try:
        r = scan_folder(S.style_store, cid, req.kind, req.folder, recursive, req.include_removed)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e)) from e
    return {**r.__dict__, "counts": _style_counts(cid, req.kind)}


class StyleRemove(BaseModel):
    image_ids: list[int] = Field(..., max_length=100_000)


@app.post("/api/characters/{cid}/style/remove")
def character_style_remove(cid: int, req: StyleRemove):
    """Remove images (and their analyses) from the character's libraries. Files on disk are never touched."""
    style = _style()
    if style.active and style.status()["character_id"] == cid:
        raise HTTPException(409, "Wait for the running style job to finish before removing images.")
    removed, orphaned = S.style_store.remove_images(cid, req.image_ids)
    for sha in orphaned:
        (S.style_thumbs / f"{sha}.jpg").unlink(missing_ok=True)
    return {"removed": removed}


class StyleAnalyze(BaseModel):
    kind: StyleKind
    reanalyze: bool = False
    image_ids: list[int] | None = None


@app.post("/api/characters/{cid}/style/analyze")
def character_style_analyze(cid: int, req: StyleAnalyze):
    """Analyze a library's pending images (or the given ones, or all with `reanalyze`) with the vision model.
    A pending image whose file was already analyzed in another character's library of the same kind (same
    model and schema) reuses that analysis instead of being sent to the model again."""
    style = _style()
    store = _store()
    _character_call(lambda: store.get(cid))
    v = S.settings.vision_settings()
    problems = vs.check_ready(v)
    if problems:
        raise HTTPException(400, " ".join(problems))
    kind = _style_kind(req.kind)
    label = v.model_label()
    if req.image_ids is not None:
        lib = S.style_store.library_ids(cid, kind.key)
        ids = [i for i in dict.fromkeys(req.image_ids) if i in lib]
    else:
        ids = S.style_store.pending_ids(cid, kind.key, kind.schema_version, label, req.reanalyze)
    if not ids:
        raise HTTPException(400, "Nothing to analyze.")
    _gpu_free_for_style(style)
    reused = [] if (req.reanalyze or req.image_ids is not None) else S.style_store.reuse_analyses(ids, kind.schema_version, label)
    rest = [i for i in ids if i not in set(reused)]
    if rest:
        try:
            style.start(v, cid, kind, rest, len(reused))
        except RuntimeError as e:  # includes Busy
            raise HTTPException(409, str(e)) from e
    return {"queued": len(rest), "reused": len(reused)}


class StyleGuideRequest(BaseModel):
    kind: StyleKind


@app.post("/api/characters/{cid}/style/guides")
def character_style_write_guide(cid: int, req: StyleGuideRequest):
    """Write a Self-Style Guide (character library) or Content Guide (inspiration) into the character's documents."""
    style = _style()
    v = S.settings.vision_settings()
    problems = vs.check_ready(v)
    if problems:
        raise HTTPException(400, " ".join(problems))
    kind = _style_kind(req.kind)
    c, folder = _docs_folder(cid)
    n = _style_counts(cid, kind.key, v)["ok"]
    if not n:
        raise HTTPException(400, "Analyze some images in this library first; the guide is written from their analyses.")
    ctype = _ctype_for(c)
    _gpu_free_for_style(style)
    try:
        style.start_guide(v, cid, kind, ctype, c["name"], folder)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"images": n}


def _ctype_for(c: dict):
    if not c["character_type"]:
        return None
    try:
        return next((t for t in load_character_types() if t.id == c["character_type"]), None)
    except CharacterTypesError as e:
        raise HTTPException(400, f"Fix the character types file first: {e}") from e


class PromptIdeasRequest(BaseModel):
    idea: str = Field("", max_length=1000)  # optional direction, e.g. "rainy city night"
    count: int = Field(4, ge=1, le=sprompts.MAX_COUNT)


@app.post("/api/characters/{cid}/prompts")
def character_write_prompts(cid: int, req: PromptIdeasRequest):
    """Write image prompts with the local model from the character's guides (guides/ in its documents).
    Runs on the GPU worker like a style job; the result appears in /api/style/status as ``prompts``."""
    style = _style()
    v = S.settings.vision_settings()
    problems = vs.check_ready(v, vision=False)
    if problems:
        raise HTTPException(400, " ".join(problems))
    c, folder = _docs_folder(cid)
    try:
        guides = sprompts.load_guides(folder)
    except OSError as e:
        raise HTTPException(500, f"Could not read the guides: {e}") from e
    if not guides:
        raise HTTPException(400, "This character has no guides yet. Write a Self-Style Guide or Content Guide in the "
                                 "Style window first (they are saved to guides/ in the character's documents).")
    try:
        messages = sprompts.build_messages(guides, _ctype_for(c), req.idea, req.count, sprompts.budget_chars(v))
    except PromptFileError as e:
        raise HTTPException(500, f"Prompt file problem: {e}") from e
    _gpu_free_for_style(style)
    try:
        style.start_prompts(v, cid, messages, req.count, [g.path for g in guides], req.idea.strip())
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"guides": [g.path for g in guides], "count": req.count}


@app.get("/api/characters/{cid}/style/guides")
def character_style_guides(cid: int, kind: StyleKind):
    _style()
    return {"items": S.style_store.list_guides(cid, kind)}


@app.get("/api/style/guides/{guide_id}")
def style_guide(guide_id: int):
    _style()
    g = S.style_store.get_guide(guide_id)
    if g is None:
        raise HTTPException(404, "No such guide.")
    return g


@app.get("/api/style/images/{image_id}")
def style_image_detail(image_id: int):
    image = _style_image(image_id)
    kind = get_kind(image["kind"])
    return {"image": image, "analyses": S.style_store.get_analyses(image_id),
            "current_model": S.settings.vision_settings().model_label(), "current_schema_version": kind.schema_version}


@app.get("/api/style/images/{image_id}/file")
def style_image_file(image_id: int):
    image = _style_image(image_id)  # only registered images are ever served
    if not Path(image["path"]).is_file():
        raise HTTPException(404, "The image file is no longer on disk.")
    return FileResponse(image["path"])


STYLE_THUMB_EDGE = 360


@app.get("/api/style/images/{image_id}/thumb")
def style_image_thumb(image_id: int):
    image = _style_image(image_id)
    cached = S.style_thumbs / f"{image['sha256']}.jpg"
    if not cached.is_file():
        if not Path(image["path"]).is_file():
            raise HTTPException(404, "The image file is no longer on disk.")
        try:
            with Image.open(image["path"]) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((STYLE_THUMB_EDGE, STYLE_THUMB_EDGE), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=80)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(415, f"Could not read the image: {e}") from e
        S.style_thumbs.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(f".{cached.name}.tmp")
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, cached)
    return Response(cached.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


class WorkspaceRequest(BaseModel):
    character_id: int | None = None  # None = generate without a character


@app.post("/api/workspace")
def enter_workspace(body: WorkspaceRequest):
    """Load a character (or none) into the generation workspace. Allowed while a job runs: that job keeps
    the prompt and LoRA stack it was submitted with."""
    if body.character_id is not None:
        store = _store()
        c = _character_call(lambda: store.get(body.character_id))
        _check_character_lora(c["lora_path"])
    S.workspace = {"character_id": body.character_id}
    _evict_unused()
    return {"workspace": _workspace_view()}


@app.delete("/api/workspace")
def leave_workspace():
    S.workspace = None
    _evict_unused()
    return {"workspace": None}


# ------------------------------------------------------------------------------------------ GPU jobs


class EncodeRequest(BaseModel):
    prompt: str = Field(..., max_length=20000)
    prepend_trigger: bool = True  # only used while a character with a trigger is active


class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=20000)
    aspect: str
    megapixels: float
    multiple: int
    width: int
    height: int
    prepend_trigger: bool = True


def _effective_prompt(prompt: str, prepend: bool) -> tuple[str, dict | None, bool]:
    """(prompt to encode, active character, whether its trigger was prepended)."""
    c = _active_character()
    if not c or not prompt.strip():
        return prompt, c, False
    p = compose_prompt(prompt, c["trigger"], prepend)
    return p, c, p != prompt


@app.post("/api/encode")
def encode(req: EncodeRequest):
    runner = _runner()
    prompt, _, _ = _effective_prompt(req.prompt, req.prepend_trigger)
    try:
        job = runner.submit_encode(prompt)
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
    prompt, c, prepended = _effective_prompt(req.prompt, req.prepend_trigger)
    extra = {"character": {"id": c["id"], "name": c["name"]} if c else None, "user_prompt": req.prompt, "trigger_prepended": prepended}
    try:
        job = runner.submit_generate(prompt, res, seed, _effective_stack(), extra_meta=extra)
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
