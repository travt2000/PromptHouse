"""Style jobs (image analysis, guide writing) on PromptHouse's GPU worker.

One job at a time, and never alongside an encode/generate: the job is submitted to the same single
GPU worker thread (``JobRunner.submit_task``), so generation requests are refused while it runs, and
the Krea models are never on the GPU at the same time as the vision model. Lifecycle of a job:

    prepare (free Krea weights from RAM if configured) -> start llama-server -> work -> always stop it

Stopping the subprocess returns all of its VRAM and RAM to the OS.
"""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from .analyzer import AnalysisError, OpenAICompatAnalyzer
from .character_types import CharacterType
from .guides import GuideCancelled, GuideError, GuideWriter, produce_guide
from .kinds import Kind
from .llama_server import ServerError, ServerManager
from .prompts import parse_prompts
from .publish import publish_guide
from .store import StyleStore
from .vsettings import VisionSettings

AnalyzerFactory = Callable[[str, VisionSettings, str], object]
WriterFactory = Callable[[str, VisionSettings, str], GuideWriter]
Stage = Callable[[str], None]
# launch(request, work): run work(stage) on the GPU worker; raises if the GPU is busy
Launcher = Callable[[dict, Callable[[Stage], None]], None]


def default_factory(base_url: str, settings: VisionSettings, model_name: str):
    return OpenAICompatAnalyzer(base_url, settings, model_name)


class StyleRunner:
    def __init__(self, store: StyleStore, server: ServerManager, launch: Launcher,
                 prepare: Callable[[VisionSettings], None] | None = None,
                 analyzer_factory: AnalyzerFactory = default_factory, writer_factory: WriterFactory = GuideWriter):
        self.store = store
        self.server = server
        self.launch = launch
        self.prepare = prepare
        self.factory = analyzer_factory
        self.writer_factory = writer_factory
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._active = False
        self._stage: Stage = lambda m: None
        self._done = threading.Event()
        self._done.set()
        self._status = self._idle()

    @staticmethod
    def _idle() -> dict:
        return {"state": "idle", "job": None, "kind": None, "character_id": None, "message": "", "total": 0, "done": 0,
                "failed": 0, "reused": 0, "current_image": None, "guide_id": None, "guide_path": None,
                "archived_path": None, "error": None, "prompts": None, "prompt_sources": None, "idea": None}

    @property
    def active(self) -> bool:
        return self._active

    def status(self) -> dict:
        with self._lock:
            out = dict(self._status)
        out["active"] = self._active
        out["server_running"] = self.server.running
        return out

    def _set(self, **kw) -> None:
        with self._lock:
            self._status.update(kw)
        if "message" in kw and kw["message"]:
            try:
                self._stage(kw["message"])
            except Exception:  # noqa: BLE001 - the stage mirror is cosmetic
                pass

    def _start(self, job: str, kind: Kind | None, cid: int, total: int, settings: VisionSettings,
               work: Callable[[VisionSettings, str, str], dict], reused: int = 0, vision: bool = True,
               extra: dict | None = None) -> None:
        key = kind.key if kind else None
        with self._lock:
            if self._active:
                raise RuntimeError("A style job is already running.")
            self._active = True
            self._done.clear()
            self._cancel.clear()
            self._status = {**self._idle(), "state": "queued", "job": job, "kind": key, "character_id": cid,
                            "message": "Waiting for the GPU…", "total": total, "reused": reused, **(extra or {})}
        try:
            self.launch({"job": job, "kind": key, "character_id": cid, "images": total},
                        lambda stage: self._run(settings, work, stage, vision))
        except Exception:
            with self._lock:
                self._active = False
                self._status = self._idle()
            self._done.set()
            raise

    def start(self, settings: VisionSettings, cid: int, kind: Kind, image_ids: list[int], reused: int = 0) -> None:
        """Analyze the given images of one library of character ``cid``."""
        self._start("analyze", kind, cid, len(image_ids), settings,
                    lambda s, url, model: self._analyze(s, kind, image_ids, url, model), reused)

    def start_guide(self, settings: VisionSettings, cid: int, kind: Kind, ctype: CharacterType | None,
                    subject: str, folder: Path) -> None:
        """Write a guide from the library's current (schema, model) records into the character's documents."""
        self._start("guide", kind, cid, 1, settings,
                    lambda s, url, model: self._guide(s, cid, kind, ctype, subject, folder, url, model))

    def start_prompts(self, settings: VisionSettings, cid: int, messages: list[dict], count: int,
                      sources: list[str], idea: str = "") -> None:
        """Write ``count`` image prompts from the character's guides (text-only: the projector is not loaded).
        The prompts end up in ``status()["prompts"]``."""
        self._start("prompts", None, cid, count, settings,
                    lambda s, url, model: self._prompts(s, messages, count, url, model), vision=False,
                    extra={"prompt_sources": sources, "idea": idea})

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    # ---- jobs ----
    def _analyze(self, s: VisionSettings, kind: Kind, image_ids: list[int], base_url: str, model_name: str) -> dict:
        label = s.model_label()
        analyzer = self.factory(base_url, s, model_name)
        try:
            self._set(state="running", message="Analyzing images…")
            for image_id in image_ids:
                if self._cancel.is_set():
                    break
                image = self.store.get_image(image_id)
                if image is None:  # removed meanwhile
                    continue
                self._set(current_image=image_id)
                try:
                    res = analyzer.analyze(image["path"], kind)
                    self.store.save_analysis(image_id, kind.schema_version, label, "ok", res.record, res.raw_text, None, res.seconds)
                    failed = False
                except AnalysisError as exc:
                    self.store.save_analysis(image_id, kind.schema_version, label, "failed", None, exc.raw_text, str(exc), None)
                    failed = True
                with self._lock:
                    self._status["done"] += 1
                    self._status["failed"] += int(failed)
                    done, total = self._status["done"], self._status["total"]
                self._set(message=f"Analyzing images {done}/{total}")
        finally:
            if hasattr(analyzer, "close"):
                analyzer.close()
        return {}

    def _guide(self, s: VisionSettings, cid: int, kind: Kind, ctype: CharacterType | None, subject: str, folder: Path,
               base_url: str, model_name: str) -> dict:
        label = s.model_label()
        records = self.store.ok_records(cid, kind.key, kind.schema_version, label)
        self._set(state="running", message=f"Writing the guide from {len(records)} analyzed images…")
        created = datetime.now()
        try:
            result = produce_guide(kind, records, self.writer_factory(base_url, s, model_name), s, ctype, created,
                                   self._cancel, subject)
        except GuideCancelled:
            return {}
        except GuideError as exc:
            with self._lock:
                self._status["failed"] = 1
            return {"state": "error", "message": "The guide could not be written.", "error": str(exc)}
        doc_path = archived = None
        error = None
        try:
            doc_path, archived = publish_guide(folder, kind.key, result.markdown, created, len(records), label)
        except Exception as exc:  # noqa: BLE001 - keep the guide in the database even if the file write fails
            error = f"The guide was saved in the database but could not be written to the documents folder: {exc}"
        guide_id = self.store.save_guide(cid, kind.key, label, kind.schema_version, ctype.id if ctype else None, len(records),
                                         result.truncated, result.seconds, doc_path, archived, result.markdown, result.stats)
        with self._lock:
            self._status["done"] = 1
        msg = f"Guide written to {doc_path}." if doc_path else "Guide written (database only)."
        if result.truncated:
            msg += " It was cut short: raise Guide max tokens."
        return {"guide_id": guide_id, "guide_path": doc_path, "archived_path": archived, "message": msg, "error": error}

    def _prompts(self, s: VisionSettings, messages: list[dict], count: int, base_url: str, model_name: str) -> dict:
        self._set(state="running", message=f"Writing {count} prompt{'s' if count != 1 else ''} from the guides…")
        try:
            text, truncated = self.writer_factory(base_url, s, model_name).write(
                messages, self._cancel, s.prompt_temperature, s.prompt_max_tokens)
        except GuideCancelled:
            return {}
        except GuideError as exc:
            return {"state": "error", "message": "The prompts could not be written.", "error": str(exc)}
        prompts = parse_prompts(text)[:count]
        if not prompts:
            return {"state": "error", "message": "The prompts could not be written.",
                    "error": "The model's reply contained no prompts:\n" + text[:500]}
        with self._lock:
            self._status["done"] = len(prompts)
        msg = f"Wrote {len(prompts)} prompt{'s' if len(prompts) != 1 else ''}."
        if len(prompts) < count:
            msg += f" (asked for {count}{'; the reply was cut short: raise Prompt max tokens' if truncated else ''})"
        return {"prompts": prompts, "message": msg}

    # ---- lifecycle (runs on the GPU worker thread) ----
    def _run(self, s: VisionSettings, work: Callable[[VisionSettings, str, str], dict], stage: Stage,
             vision: bool = True) -> None:
        self._stage = stage
        outcome: dict = {}  # published only after the server is down, so "done" never overlaps a live process
        try:
            if self._cancel.is_set():
                raise ServerError("cancelled")
            if self.prepare is not None:
                self._set(state="preparing", message="Freeing GPU memory for the local model…")
                self.prepare(s)
            if s.external_url:
                base_url, model_name = s.external_url, s.external_model or "local"
            else:
                self._set(state="loading_model", message="Loading the local model (this can take a while from a slow disk)…")
                base_url, model_name = self.server.start(s, self._cancel, vision), "local"
            result = work(s, base_url, model_name) or {}
            if self._cancel.is_set():
                outcome = {"state": "cancelled", "message": "Cancelled.", "current_image": None}
            else:
                outcome = {"state": "done", "message": "Finished.", "current_image": None, **result}
        except ServerError as exc:
            if self._cancel.is_set():  # cancelled before or while the model was loading
                outcome = {"state": "cancelled", "message": "Cancelled.", "current_image": None}
            else:
                outcome = {"state": "error", "message": "The local model server could not be started.", "error": str(exc),
                           "current_image": None}
        except Exception as exc:  # noqa: BLE001 - never leave the status stuck on "running"
            outcome = {"state": "error", "message": "Unexpected error.", "error": f"{type(exc).__name__}: {exc}", "current_image": None}
        finally:
            self._set(message="Stopping the local model…")
            self.server.stop()
            with self._lock:
                self._status.update(outcome)
                self._active = False
            self._stage = lambda m: None
            self._done.set()
