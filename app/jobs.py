"""Single-worker GPU job execution. One GPU job at a time; a second request while busy is rejected."""

from __future__ import annotations

import logging
import secrets
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable

import torch

from .conditioning import PromptCache, encode_prompt
from .decoding import decode_latents
from .inference import GUIDANCE, MU, SAMPLER, STEPS, sample_latents
from .lora import LoraError, LoraLibrary
from .model_manager import Component, CudaUnrecoverable, InsufficientVRAM, ModelError, ModelManager
from .outputs import save_image
from .resolution import Resolution

log = logging.getLogger("krea.jobs")


class Busy(RuntimeError):
    pass


@dataclass
class Job:
    kind: str  # "encode" | "generate"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"  # queued | running | complete | error
    stage: str = "Queued"
    stages: list = field(default_factory=list)
    progress: dict | None = None  # {"step": i, "total": n}
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    request: dict = field(default_factory=dict)
    result: dict | None = None
    error: str | None = None
    restart_required: bool = False

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.stages.append({"stage": stage, "t": round(time.time() - (self.started or self.created), 2)})

    def to_dict(self) -> dict:
        return asdict(self)


def new_random_seed() -> int:
    return secrets.randbelow(2**32)


class JobRunner:
    def __init__(self, manager: ModelManager, cache: PromptCache, loras: LoraLibrary | None = None):
        self.manager = manager
        self.cache = cache
        self.loras = loras or LoraLibrary()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-worker")
        self._lock = threading.Lock()
        self._active: Job | None = None
        self.jobs: dict[str, Job] = {}

    # ------------------------------------------------------------------ public
    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active is not None

    @property
    def active(self) -> Job | None:
        return self._active

    def submit_encode(self, prompt: str) -> Job:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is empty")
        job = Job(kind="encode", request={"prompt": prompt})
        self._claim(job)
        token = self.cache.invalidate()  # previous conditioning becomes unavailable immediately
        self._executor.submit(self._run, job, lambda: self._encode(job, prompt, token))
        return job

    def submit_generate(self, prompt: str, res: Resolution, seed: int, lora_stack: list[dict] | None = None,
                        extra_meta: dict | None = None) -> Job:
        """`lora_stack` is a snapshot of the stack at submit time (later edits apply to the next image).
        `extra_meta` (e.g. the active character) is recorded in the request and the image metadata."""
        cond = self.cache.valid_for(prompt, self.manager.text_encoder_fingerprint())
        if cond is None:
            raise ValueError("The current prompt has not been encoded with the current text encoder. Click Encode Prompt first.")
        stack = [dict(e) for e in (lora_stack or [])]
        active = [e for e in stack if e.get("enabled", True)]
        for e in active:  # fail fast, before claiming the GPU
            info = self.loras.inspect(e["path"])
            if not info.compatible:
                raise ValueError(f"LoRA {info.filename}: " + "; ".join(info.problems))
        job = Job(kind="generate", request={"prompt": prompt, "width": res.width, "height": res.height, "seed": seed, "conditioning_id": cond.id,
                                            "loras": [{"file": self.loras.inspect(e["path"]).filename, "strength": e.get("strength", 1.0)} for e in active],
                                            **(extra_meta or {})})
        self._claim(job)
        keep = [e["path"] for e in stack]
        self._executor.submit(self._run, job, lambda: self._generate(job, cond, res, seed, active, keep, extra_meta))
        return job

    def run_on_worker(self, fn: Callable[[], object]) -> object:
        """Run a model-management call (e.g. reconfiguration/release) on the GPU worker thread.

        Serialises it with GPU jobs, and frees model memory on the thread that allocated it: PyTorch's
        Windows CPU allocator (mimalloc) defers frees made from other threads, so releasing weights from a
        request thread would not return RAM until the worker allocates again."""
        if self.busy:
            raise Busy(f"A {self._active.kind} job is running; try again when it finishes")
        return self._executor.submit(fn).result()

    def defer_to_worker(self, fn: Callable[[], object]) -> None:
        """Queue a housekeeping call on the GPU worker thread without waiting (runs after any active job)."""
        def safe():
            try:
                fn()
            except Exception:  # noqa: BLE001
                log.exception("deferred worker task failed")
        self._executor.submit(safe)

    # ------------------------------------------------------------------ internals
    def _claim(self, job: Job) -> None:
        if self.manager.cuda_unrecoverable:
            raise Busy("The CUDA context is unrecoverable. Restart the application.")
        with self._lock:
            if self._active is not None:
                raise Busy(f"A {self._active.kind} job is already running ({self._active.stage})")
            self._active = job
            self.jobs[job.id] = job
            if len(self.jobs) > 50:
                for old in list(self.jobs)[:-50]:
                    self.jobs.pop(old, None)

    def _run(self, job: Job, fn: Callable[[], None]) -> None:
        job.status = "running"
        job.started = time.time()
        self.manager.stage_cb = job.set_stage
        failed = False
        try:
            fn()
            job.status = "complete"
            job.set_stage("Complete")
        except torch.OutOfMemoryError as e:
            failed = True
            job.error = (
                f"CUDA out of memory during '{job.stage}'. The operation was stopped and its partial results discarded. "
                f"Close other GPU applications or choose a smaller resolution, then retry. ({str(e).splitlines()[0]})"
            )
        except InsufficientVRAM as e:
            failed = True
            job.error = str(e)
        except CudaUnrecoverable as e:
            failed = True
            job.error = str(e)
            job.restart_required = True
        except (ModelError, LoraError, ValueError) as e:
            failed = True
            job.error = str(e)
        except Exception as e:  # noqa: BLE001
            failed = True
            job.error = f"{type(e).__name__}: {e}"
            log.error("job %s failed:\n%s", job.id, traceback.format_exc())
        finally:
            self.manager.stage_cb = lambda s: None
        if failed:
            # The exception and its traceback frames are gone now, so their GPU tensors can be freed.
            hint = self.manager.post_failure_cleanup()
            if hint:
                job.restart_required = True
                job.error = f"{job.error} {hint}"
            job.status = "error"
            job.set_stage("Error")
        job.finished = time.time()
        with self._lock:
            self._active = None

    def _encode(self, job: Job, prompt: str, token: int) -> None:
        cond = encode_prompt(self.manager, prompt)
        installed = self.cache.install(cond, token)
        if not installed:
            raise ValueError("Encoding finished after the configuration changed; result discarded. Encode again.")
        job.result = {"conditioning": cond.describe(), "seconds": round(time.time() - job.started, 2)}
        job.set_stage("Ready to generate")

    def _generate(self, job: Job, cond, res: Resolution, seed: int, loras: list[dict] | None = None, keep: list[str] | None = None,
                  extra_meta: dict | None = None) -> None:
        t0 = time.time()

        def progress(i: int, n: int) -> None:
            job.progress = {"step": i, "total": n}

        job.progress = {"step": 0, "total": STEPS}
        adapters = self.loras.prepare(loras or [], keep or [], self.manager.stage_cb)
        lora_meta = [{"file": l.info.filename, "strength": s, "rank": l.info.rank, "key_style": l.info.key_style, "size_bytes": l.info.size_bytes}
                     for l, s in adapters]
        latents = sample_latents(self.manager, cond, res.width, res.height, seed, progress=progress, adapters=adapters)
        del adapters
        t_sample = time.time() - t0
        tstatus = self.manager.status.get(Component.TRANSFORMER)
        plan = (tstatus.streaming if tstatus else None) or {}
        image, decode_info = decode_latents(self.manager, latents)
        del latents
        job.set_stage("Saving image")
        tinfo = self.manager.inspect(Component.TRANSFORMER)
        meta = {
            "prompt": cond.prompt,
            "seed": seed,
            "width": res.width,
            "height": res.height,
            "aspect": res.aspect,
            "megapixels_target": res.megapixels_target,
            "dimension_multiple": res.multiple,
            "steps": STEPS,
            "guidance": GUIDANCE,
            "mu": MU,
            "sampler": SAMPLER,
            "model": "Krea 2 Turbo",
            "transformer": tinfo.filename,
            "transformer_format": tinfo.format,
            "text_encoder": cond.encoder_filename,
            "vae": self.manager.inspect(Component.VAE).filename,
            "conditioning_id": cond.id,
            "loras": lora_meta,
            "resident_blocks": plan.get("resident_blocks"),
            "streamed_blocks": len(plan.get("streamed_blocks") or []),
            "vae_decode": decode_info["mode"],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **(extra_meta or {}),
        }
        image_id, _ = save_image(image, meta)
        notes = [s.note for s in self.manager.status.values() if getattr(s, "note", "") and "exceeded" in s.note]
        job.result = {"image_id": image_id, "width": res.width, "height": res.height, "seed": seed, "metadata": meta,
                      "seconds": round(time.time() - t0, 2), "sampling_seconds": round(t_sample, 2),
                      "decode": decode_info, "warnings": notes}
