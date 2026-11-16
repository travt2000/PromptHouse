"""Explicit model loading, GPU residency and VRAM lifecycle for the three Krea 2 components.

Rules enforced here:
  * At most ONE component is GPU-resident at any time (`use()` refuses if another one is).
  * A component is reported unloaded only after none of its parameters/buffers live on CUDA, its
    temporary GPU tensors are released, gc has run and the CUDA caching allocator has been emptied.
  * Weights are kept in system RAM between operations only if the per-component policy says so
    (`keep_in_ram`); otherwise the CPU copy is dropped as well and status says "not_loaded".

Transformer VRAM strategy ("block streaming"):
  The Krea 2 Turbo transformer is ~13.1 GB even as scaled FP8, which does not fit on a 12 GB card next to
  the activations. For each generation the manager measures device-wide free VRAM (cudaMemGetInfo),
  estimates activation memory for the requested resolution, keeps every non-block weight plus as many
  of the 28 transformer blocks as fit resident on the GPU, and streams the remaining blocks from pinned
  system RAM on a dedicated CUDA copy stream (block i+1 copies while block i computes; at most two
  streamed blocks are on the GPU at once). All of it is released when sampling ends.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Iterator

import psutil
import torch
from torch import nn

from . import checkpoints as ck
from .lora import LoraPatch, adapters_gpu_bytes
from .quant import FP8Linear, finalize_fp8, replace_linears_with_fp8

log = logging.getLogger("krea.models")
GiB = 1024**3
MiB = 1024**2


class Component(str, Enum):
    TEXT_ENCODER = "text_encoder"
    TRANSFORMER = "transformer"
    VAE = "vae"


class Residency(str, Enum):
    NOT_LOADED = "not_loaded"  # no weights in RAM or VRAM
    LOADING = "loading_from_disk"
    CPU = "cpu"  # weights held in system RAM, nothing on GPU
    TO_GPU = "loading_to_gpu"
    GPU = "gpu"
    UNLOADING = "unloading_from_gpu"


class ModelError(RuntimeError):
    pass


class InsufficientVRAM(ModelError):
    pass


class CudaUnrecoverable(ModelError):
    pass


@dataclass
class ComponentStatus:
    residency: str = Residency.NOT_LOADED.value
    checkpoint: dict | None = None
    ram_bytes: int = 0
    vram_weight_bytes: int = 0  # current: torch allocated delta after upload (0 when not on GPU)
    last_vram_weight_bytes: int = 0  # measured on the most recent GPU load
    vram_estimated_bytes: int = 0
    last_peak_allocated_bytes: int = 0
    last_peak_reserved_bytes: int = 0  # incl. caching-allocator overhead
    last_load_disk_s: float | None = None
    last_upload_s: float | None = None
    streaming: dict | None = None
    adapters: dict | None = None  # LoRA stack applied in the most recent GPU session
    note: str = ""


@dataclass
class LoadedModel:
    module: nn.Module
    info: ck.CheckpointInfo
    fingerprint: str
    # name -> CPU master tensor for every tensor that can move to the GPU
    cpu_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def _tensor_owner(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    owner_name, _, attr = name.rpartition(".")
    return (module.get_submodule(owner_name) if owner_name else module), attr


def _set_tensor(module: nn.Module, name: str, t: torch.Tensor) -> None:
    owner, attr = _tensor_owner(module, name)
    if attr in owner._parameters:
        owner._parameters[attr].data = t
    elif attr in owner._buffers:
        owner._buffers[attr] = t
    else:
        raise KeyError(name)


def _collect_tensors(module: nn.Module, include=lambda n: True) -> dict[str, torch.Tensor]:
    out = {}
    for n, p in module.named_parameters():
        if include(n):
            out[n] = p.data
    for n, b in module.named_buffers():
        if include(n) and b is not None:
            out[n] = b
    return out


def tensor_bytes(ts) -> int:
    return sum(t.numel() * t.element_size() for t in ts)


def cuda_tensors_of(module: nn.Module) -> list[str]:
    bad = [n for n, p in module.named_parameters() if p.device.type == "cuda"]
    bad += [n for n, b in module.named_buffers() if b is not None and b.device.type == "cuda"]
    return bad


def estimate_transformer_activation_bytes(width: int, height: int, txt_len: int = 512) -> int:
    """Conservative peak activation estimate for one SingleStreamDiT forward (batch 1, BF16)."""
    img = (width // 16) * (height // 16)
    L = txt_len + img
    L += (-L) % 256
    per_token = 6144 * 2 * 6 + 6144 * 4 * 3 + 16384 * 2 * 4  # residual/attn bf16 copies, fp32 norm/rope temps, MLP
    return int(L * per_token + 3 * L * L + 16384 * 6144 * 2 + 384 * MiB)


class BlockStreamer:
    """Streams transformer blocks host->device through two fixed GPU slots (a ring buffer).

    Each streamed block lives in RAM as one packed (page-locked) buffer. Block k+1 is copied into the free
    slot on a dedicated copy stream while block k computes from the other slot; the block's parameters are
    views into its slot. Events order the hand-off: compute waits for "copy done", the copy into a slot waits
    for "compute finished with the previous occupant". VRAM cost is exactly 2 x block size, with no
    allocator churn (allocating/freeing per block across two streams fragmented the caching allocator)."""

    def __init__(self, module: nn.Module, streamed: list[int], packed: dict, cpu_tensors: dict[str, torch.Tensor], device: torch.device):
        self.module = module
        self.device = device
        self.streamed = streamed
        self.next_of = {b: streamed[(i + 1) % len(streamed)] for i, b in enumerate(streamed)}
        self.packed = packed  # b -> {"buf": host uint8 tensor, "layout": [(name, offset, nbytes, dtype, shape)]}
        self.cpu = cpu_tensors
        slot_bytes = max(packed[b]["buf"].numel() for b in streamed)
        self.slots = [torch.empty(slot_bytes, dtype=torch.uint8, device=device) for _ in range(2)]
        self.slot_free = [torch.cuda.Event(), torch.cuda.Event()]
        self.next_slot = 0
        self.copy_stream = torch.cuda.Stream(device=device)
        self.pending: dict[int, tuple[torch.cuda.Event, int]] = {}  # b -> (copy-done event, slot)
        self.active_slot: dict[int, int] = {}
        self.handles = []
        self.transfers = 0
        for b in streamed:
            blk = module.blocks[b]
            self.handles.append(blk.register_forward_pre_hook(self._make_pre(b)))
            self.handles.append(blk.register_forward_hook(self._make_post(b)))

    def prefetch(self, b: int) -> None:
        if b in self.pending:
            return
        s = self.next_slot
        self.next_slot = 1 - s
        buf = self.packed[b]["buf"]
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(self.slot_free[s])  # previous occupant's compute must be finished
            self.slots[s][: buf.numel()].copy_(buf, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.copy_stream)
        self.pending[b] = (done, s)
        self.transfers += 1

    def _make_pre(self, b: int):
        def pre(_mod, _args):
            self.prefetch(b)
            done, s = self.pending.pop(b)
            torch.cuda.current_stream(self.device).wait_event(done)
            slot = self.slots[s]
            for name, off, nb, dtype, shape in self.packed[b]["layout"]:
                _set_tensor(self.module, name, slot[off: off + nb].view(dtype).view(shape))
            self.active_slot[b] = s
            self.prefetch(self.next_of[b])  # overlaps with this block's compute (goes to the other slot)

        return pre

    def _make_post(self, b: int):
        def post(_mod, _args, _out):
            for name, *_ in self.packed[b]["layout"]:
                _set_tensor(self.module, name, self.cpu[name])
            self.slot_free[self.active_slot.pop(b)].record(torch.cuda.current_stream(self.device))

        return post

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        try:
            self.copy_stream.synchronize()
            torch.cuda.current_stream(self.device).synchronize()
        except Exception:  # noqa: BLE001 - best effort during failure cleanup
            pass
        self.pending.clear()
        for b in self.streamed:
            for name, *_ in self.packed[b]["layout"]:
                _set_tensor(self.module, name, self.cpu[name])
        self.slots.clear()


class ModelManager:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self._lock = threading.RLock()
        self._gpu_owner: Component | None = None
        self._models: dict[Component, LoadedModel | None] = {c: None for c in Component}
        self._paths: dict[Component, str] = {c: "" for c in Component}
        self._te_config_dir: str = ""
        self.keep_in_ram: dict[Component, bool] = {Component.TEXT_ENCODER: False, Component.TRANSFORMER: True, Component.VAE: True}
        self.status: dict[Component, ComponentStatus] = {c: ComponentStatus() for c in Component}
        self.cuda_unrecoverable = False
        self.stage_cb: Callable[[str], None] = lambda s: None
        self.check_cuda()

    # ------------------------------------------------------------------ configuration
    def check_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise ModelError("CUDA is not available; this application requires an NVIDIA GPU.")
        if not torch.cuda.is_bf16_supported():
            raise ModelError("This GPU does not support BF16, which Krea 2 inference requires.")

    @property
    def gpu_owner(self) -> Component | None:
        return self._gpu_owner

    def configure(self, paths: dict[Component, str], te_config_dir: str, keep_in_ram: dict[Component, bool] | None = None) -> list[Component]:
        """Set checkpoint paths. Components whose source changed are released (RAM and VRAM). Returns them."""
        with self._lock:
            if self._gpu_owner is not None:
                raise ModelError("Cannot change model configuration while a component is on the GPU")
            changed = []
            for c in Component:
                new = paths.get(c, self._paths[c])
                src_changed = new != self._paths[c] or (c == Component.TEXT_ENCODER and te_config_dir != self._te_config_dir)
                if src_changed:
                    changed.append(c)
                    self._paths[c] = new
                    self.release_cpu(c)
            self._te_config_dir = te_config_dir
            if keep_in_ram:
                self.keep_in_ram.update(keep_in_ram)
                for c, keep in keep_in_ram.items():
                    if not keep:
                        self.release_cpu(c)
            return changed

    def release_cpu(self, c: Component) -> None:
        with self._lock:
            if self._gpu_owner == c:
                raise ModelError(f"{c.value} is on the GPU; release it first")
            lm = self._models[c]
            self._models[c] = None
            if lm is not None:
                # Detach every weight from the module itself (not just our references): a caller that still
                # holds the module object (a finished `with use() as lm` frame, a traceback) must not keep
                # gigabytes alive after we report the component as released.
                self._unpin_all(lm)  # must unregister page-locked buffers before their memory is freed
                _strip_module(lm.module)
                lm.cpu_tensors.clear()
                lm.extra.clear()
                del lm
                gc.collect()
                # Freed pinned host blocks are cached by PyTorch's host allocator; return them to the OS.
                if hasattr(torch._C, "_host_emptyCache"):
                    torch._C._host_emptyCache()
            st = self.status[c]
            st.residency = Residency.NOT_LOADED.value
            st.ram_bytes = 0
            st.streaming = None

    def inspect(self, c: Component) -> ck.CheckpointInfo:
        kind = {Component.TRANSFORMER: ck.KIND_TRANSFORMER, Component.TEXT_ENCODER: ck.KIND_TEXT_ENCODER, Component.VAE: ck.KIND_VAE}[c]
        return ck.inspect_checkpoint(self._paths[c], kind)

    def text_encoder_fingerprint(self) -> str:
        info = self.inspect(Component.TEXT_ENCODER)
        return info.fingerprint + "||" + ck.dir_fingerprint(self._te_config_dir)

    # ------------------------------------------------------------------ CPU loading
    def _ensure_cpu(self, c: Component) -> LoadedModel:
        info = self.inspect(c)
        if not info.compatible:
            raise ModelError(f"{info.filename}: incompatible {c.value} checkpoint: " + "; ".join(info.problems))
        fp = info.fingerprint + ("||" + ck.dir_fingerprint(self._te_config_dir) if c == Component.TEXT_ENCODER else "")
        lm = self._models[c]
        if lm is not None and lm.fingerprint == fp:
            return lm
        if lm is not None:  # file changed on disk
            self.release_cpu(c)
        need = info.weight_bytes_to_load
        if info.format == "fp32":  # converted to BF16 on load
            need //= 2
        headroom = int(1.5 * GiB)
        avail = psutil.virtual_memory().available
        if need + headroom > avail:
            # Memory pressure: drop other components' RAM caches (largest first) before giving up.
            for other in sorted((o for o in Component if o != c and self._models[o] is not None and self._gpu_owner != o),
                                key=lambda o: self.status[o].ram_bytes, reverse=True):
                self.stage_cb(f"Releasing {other.value.replace('_', ' ')} from RAM to make room")
                self.release_cpu(other)
                self.status[other].note = f"Released from RAM at {time.strftime('%H:%M:%S')} to make room for {c.value}; reloads on next use."
                log.info("evicted %s from RAM to load %s", other.value, c.value)
                avail = psutil.virtual_memory().available
                if need + headroom <= avail:
                    break
        if need + headroom > avail:
            raise ModelError(
                f"Not enough free system RAM to load {info.filename}: needs about {need / GiB:.1f} GiB "
                f"(+{headroom / GiB:.1f} GiB headroom), {avail / GiB:.1f} GiB available. Close other applications."
            )
        st = self.status[c]
        st.note = ""
        st.residency = Residency.LOADING.value
        st.checkpoint = info.to_dict()
        self.stage_cb(f"Reading {c.value.replace('_', ' ')} from disk")
        t0 = time.perf_counter()
        try:
            builder = {Component.TRANSFORMER: _build_transformer, Component.TEXT_ENCODER: _build_text_encoder, Component.VAE: _build_vae}[c]
            module, cpu_tensors, extra = builder(info, self._te_config_dir)
        except Exception:
            st.residency = Residency.NOT_LOADED.value
            gc.collect()
            raise
        lm = LoadedModel(module=module, info=info, fingerprint=fp, cpu_tensors=cpu_tensors, extra=extra)
        self._models[c] = lm
        st.residency = Residency.CPU.value
        st.ram_bytes = tensor_bytes(cpu_tensors.values())
        st.vram_estimated_bytes = st.ram_bytes
        st.last_load_disk_s = round(time.perf_counter() - t0, 2)
        log.info("%s loaded to RAM in %.1fs (%.2f GiB)", c.value, st.last_load_disk_s, st.ram_bytes / GiB)
        return lm

    # ------------------------------------------------------------------ GPU residency
    @contextmanager
    def use(self, c: Component, *, width: int | None = None, height: int | None = None, adapters: list | None = None) -> Iterator[LoadedModel]:
        """Acquire component `c` on the GPU for the duration of the block, then fully release it.

        `adapters` (transformer only): [(LoadedLora, strength)] attached as runtime hooks for this session;
        the base weights are never modified, so a different stack next time needs no reload."""
        if adapters and c != Component.TRANSFORMER:
            raise ModelError("LoRA adapters can only be applied to the transformer")
        if self.cuda_unrecoverable:
            raise CudaUnrecoverable("The CUDA context is in an unrecoverable state. Restart the application.")
        with self._lock:
            if self._gpu_owner is not None:
                raise ModelError(f"Refusing to load {c.value}: {self._gpu_owner.value} is still resident on the GPU")
            leftovers = [o for o in Component if self._models[o] is not None and cuda_tensors_of(self._models[o].module)]
            if leftovers:
                raise ModelError(f"Refusing to load {c.value}: tensors of {[o.value for o in leftovers]} are still on the GPU")
            self._gpu_owner = c

        label = c.value.replace("_", " ")
        st = self.status[c]
        streamer: BlockStreamer | None = None
        patch: LoraPatch | None = None
        lm: LoadedModel | None = None
        uploaded: list[str] = []
        try:
            lm = self._ensure_cpu(c)
            if "exceeded" in st.note:
                st.note = ""
            st.residency = Residency.TO_GPU.value
            self.stage_cb(f"Loading {label} to GPU")
            torch.cuda.synchronize(self.device)
            base_alloc = torch.cuda.memory_allocated(self.device)
            t0 = time.perf_counter()

            if c == Component.TRANSFORMER:
                plan = self._plan_transformer(lm, width or 1024, height or 1024, extra_resident_bytes=adapters_gpu_bytes(adapters or []))
                st.streaming = plan
                resident_names = [n for n in lm.cpu_tensors if not n.startswith("blocks.") or int(n.split(".")[1]) in plan["resident_set"]]
                if plan["streamed_blocks"]:
                    self._pin_blocks(lm, plan["streamed_blocks"])
                    plan["pinned_bytes"] = lm.extra.get("pinned_bytes", 0)
                    plan["pin_error"] = lm.extra.get("pin_error")
            else:
                resident_names = list(lm.cpu_tensors)

            for n in resident_names:
                _set_tensor(lm.module, n, lm.cpu_tensors[n].to(self.device, non_blocking=True))
                uploaded.append(n)
            torch.cuda.synchronize(self.device)
            if c == Component.TRANSFORMER and st.streaming["streamed_blocks"]:
                streamer = BlockStreamer(lm.module, st.streaming["streamed_blocks"], lm.extra["packed_blocks"], lm.cpu_tensors, self.device)
            if c == Component.TRANSFORMER:
                st.adapters = None
                if adapters:
                    self.stage_cb("Attaching LoRA adapters")
                    patch = LoraPatch(lm.module, adapters, self.device)
                    torch.cuda.synchronize(self.device)
                    st.adapters = {"applied": patch.applied, "patched_modules": patch.patched_modules, "vram_bytes": patch.gpu_bytes}
            st.last_upload_s = round(time.perf_counter() - t0, 2)
            st.vram_weight_bytes = torch.cuda.memory_allocated(self.device) - base_alloc
            st.last_vram_weight_bytes = st.vram_weight_bytes
            st.residency = Residency.GPU.value
            torch.cuda.reset_peak_memory_stats(self.device)
            log.info("%s on GPU: %.2f GiB weights resident (%.1fs)", c.value, st.vram_weight_bytes / GiB, st.last_upload_s)
            yield lm
            if streamer is not None:
                st.streaming["transfers"] = streamer.transfers
        finally:
            if uploaded:
                st.residency = Residency.UNLOADING.value
                self.stage_cb(f"Unloading {label} from GPU")
            try:
                st.last_peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
                st.last_peak_reserved_bytes = torch.cuda.max_memory_reserved(self.device)
                total = torch.cuda.get_device_properties(self.device).total_memory
                if st.last_peak_allocated_bytes > total:
                    st.note = (f"Peak allocation {st.last_peak_allocated_bytes / GiB:.1f} GiB exceeded the GPU's "
                               f"{total / GiB:.1f} GiB: the driver spilled into shared system memory (very slow).")
                    log.warning("%s: %s", c.value, st.note)
            except Exception:  # noqa: BLE001
                pass
            if patch is not None:
                patch.close()
                patch = None
            if streamer is not None:
                streamer.close()
            if lm is not None:
                for n in uploaded:
                    _set_tensor(lm.module, n, lm.cpu_tensors[n])
            uploaded.clear()
            self._cleanup_cuda()
            still = cuda_tensors_of(lm.module) if lm is not None else []
            if still:
                self.cuda_unrecoverable = True
                raise CudaUnrecoverable(f"{c.value}: {len(still)} tensors could not be removed from the GPU (e.g. {still[0]}). Restart the application.")
            st.vram_weight_bytes = 0
            with self._lock:
                self._gpu_owner = None
            if lm is not None and not self.keep_in_ram.get(c, True):
                self.release_cpu(c)
                lm = None
            elif lm is not None:
                st.residency = Residency.CPU.value
            else:
                st.residency = Residency.NOT_LOADED.value

    def _plan_transformer(self, lm: LoadedModel, width: int, height: int, extra_resident_bytes: int = 0) -> dict:
        nblocks = len(lm.module.blocks)
        block_bytes = [tensor_bytes(t for n, t in lm.cpu_tensors.items() if n.startswith(f"blocks.{b}.")) for b in range(nblocks)]
        # Non-block weights plus LoRA adapter tensors (always fully resident while the transformer is on the GPU).
        base_bytes = tensor_bytes(t for n, t in lm.cpu_tensors.items() if not n.startswith("blocks.")) + int(extra_resident_bytes)
        free, total = torch.cuda.mem_get_info(self.device)
        act = estimate_transformer_activation_bytes(width, height)
        # Measured on an RTX 5070 at 1 MP: the caching allocator's reserved-but-unallocated memory (streamed
        # blocks of one size interleaved with activations) consumed ~1 GiB beyond the allocated peak, so keep
        # a margin of 1 GiB + 8% of the card.
        safety = GiB + int(0.08 * total)
        bmax = max(block_bytes)
        budget_all = free - safety - act - base_bytes
        if budget_all >= sum(block_bytes):
            resident = list(range(nblocks))
        else:
            budget = budget_all - 2 * bmax  # two in-flight streamed blocks
            if budget < 0:
                raise InsufficientVRAM(
                    f"Not enough free VRAM for {width}x{height}: {free / GiB:.2f} GiB free on the device, need about "
                    f"{(safety + act + base_bytes + 2 * bmax) / GiB:.2f} GiB (activations {act / GiB:.2f} GiB + resident weights "
                    f"{(base_bytes + 2 * bmax) / GiB:.2f} GiB). Close other GPU applications or choose a smaller resolution."
                )
            resident, used = [], 0
            for b in range(nblocks):
                if used + block_bytes[b] <= budget:
                    resident.append(b)
                    used += block_bytes[b]
                else:
                    break
        streamed = [b for b in range(nblocks) if b not in resident]
        return {
            "free_vram_at_plan_bytes": int(free),
            "total_vram_bytes": int(total),
            "activation_estimate_bytes": int(act),
            "base_weight_bytes": int(base_bytes - extra_resident_bytes),
            "lora_bytes": int(extra_resident_bytes),
            "block_bytes": int(bmax),
            "resident_blocks": len(resident),
            "streamed_blocks": streamed,
            "resident_set": set(resident),
            "streamed_set": set(streamed),
            "estimated_weight_vram_bytes": int(base_bytes + sum(block_bytes[b] for b in resident) + (2 * bmax if streamed else 0)),
        }

    def _pin_blocks(self, lm: LoadedModel, blocks: list[int]) -> None:
        """Page-lock streamed blocks for fast async host->device copies.

        Each block's tensors are packed into ONE exact-size host buffer that is registered in place with
        cudaHostRegister. (torch's pin_memory() rounds every tensor up to a power of two - ~40% extra RAM
        for these weight shapes - and caches freed pinned blocks.) Buffers are unregistered in release_cpu.
        """
        packed: dict[int, dict] = lm.extra.setdefault("packed_blocks", {})
        todo = [b for b in blocks if b not in packed]
        if not todo:
            return
        self.stage_cb("Pinning streamed transformer blocks in RAM")
        cr = torch.cuda.cudart()
        for b in todo:
            names = [n for n in lm.cpu_tensors if n.startswith(f"blocks.{b}.")]
            layout, total = [], 0
            for n in names:
                t = lm.cpu_tensors[n]
                nb = t.numel() * t.element_size()
                layout.append((n, total, nb, t.dtype, tuple(t.shape)))
                total += -(-nb // 256) * 256
            buf = torch.empty(total, dtype=torch.uint8)
            for n, off, nb, dtype, shape in layout:
                view = buf[off: off + nb].view(dtype).view(shape)
                view.copy_(lm.cpu_tensors[n])
                lm.cpu_tensors[n] = view
                _set_tensor(lm.module, n, view)
            entry = {"buf": buf, "layout": layout, "fin": None}
            if not lm.extra.get("pin_error"):
                rc = cr.cudaHostRegister(buf.data_ptr(), total, 0)
                if int(rc) == 0:
                    # Safety net: never let registered memory be freed while still registered.
                    entry["fin"] = weakref.finalize(buf, _host_unregister, buf.data_ptr())
                else:  # packed pageable buffer still works, copies are just synchronous
                    lm.extra["pin_error"] = f"cudaHostRegister failed: {rc}"
                    log.warning("cudaHostRegister failed (%s); streaming from pageable memory", rc)
            packed[b] = entry
        lm.extra["pinned_bytes"] = sum(e["buf"].numel() for e in packed.values() if e["fin"] is not None)

    @staticmethod
    def _unpin_all(lm: LoadedModel) -> None:
        packed = lm.extra.get("packed_blocks") or {}
        if packed:
            torch.cuda.synchronize()
            for e in packed.values():
                if e["fin"] is not None:
                    e["fin"]()  # unregisters exactly once
            packed.clear()

    def _cleanup_cuda(self) -> None:
        gc.collect()
        try:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "out of memory" not in msg.lower():
                self.cuda_unrecoverable = True
            log.error("CUDA cleanup failed: %s", msg)

    def make_room_for_external(self, release_ram: bool) -> None:
        """Before another process (the style-analysis vision model) uses the GPU: return PyTorch's cached
        VRAM to the driver and, if `release_ram`, free every component's weights from system RAM too (they
        are re-read from disk on next use). Call on the GPU worker thread, with no component on the GPU."""
        with self._lock:
            if self._gpu_owner is not None:
                raise ModelError(f"{self._gpu_owner.value} is on the GPU")
            if release_ram:
                for c in Component:
                    self.release_cpu(c)
        self._cleanup_cuda()

    def post_failure_cleanup(self) -> str | None:
        """Call after a failed stage once the exception (and its traceback frames) has been dropped."""
        self._cleanup_cuda()
        if self.cuda_unrecoverable:
            return "CUDA context is unrecoverable; restart the application."
        try:
            probe = torch.ones(1, device=self.device)
            del probe
            torch.cuda.synchronize(self.device)
        except Exception as e:  # noqa: BLE001
            self.cuda_unrecoverable = True
            return f"CUDA probe failed after error ({e}); restart the application."
        return None

    # ------------------------------------------------------------------ reporting
    def snapshot(self) -> dict:
        out = {}
        for c in Component:
            d = asdict(self.status[c])
            if d.get("streaming"):
                d["streaming"] = {k: v for k, v in d["streaming"].items() if not k.endswith("_set")}
            d["keep_in_ram"] = self.keep_in_ram.get(c, True)
            d["path"] = self._paths[c]
            out[c.value] = d
        alloc = reserved = None
        try:
            alloc = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
        except Exception:  # noqa: BLE001
            pass
        return {
            "components": out,
            "gpu_owner": self._gpu_owner.value if self._gpu_owner else None,
            "torch_allocated_bytes": alloc,
            "torch_reserved_bytes": reserved,
            "cuda_unrecoverable": self.cuda_unrecoverable,
        }


# ====================================================================== component builders (CPU)


def _build_transformer(info: ck.CheckpointInfo, _cfg_dir: str):
    from .krea2_mmdit import KREA2_CONFIG, SingleStreamDiT

    tensors, _ = ck.read_header(info.path)
    prefix = info.details.get("key_prefix", "")
    key_map = {k: k[len(prefix):] for k in tensors}
    qnames = {k[len(prefix): -len(".weight")] for k, v in tensors.items() if v["dtype"] in ck.FP8_TAGS}
    with torch.device("meta"):
        model = SingleStreamDiT(KREA2_CONFIG)
    replace_linears_with_fp8(model, qnames)
    sd = ck.load_tensors(info.path, key_map)
    model.load_state_dict(sd, strict=True, assign=True)
    del sd
    finalize_fp8(model)
    model.eval().requires_grad_(False)
    _assert_no_meta(model, "transformer")
    return model, _collect_tensors(model), {"fp8_layers": len(qnames)}


def _build_text_encoder(info: ck.CheckpointInfo, cfg_dir: str):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoTokenizer, Qwen3VLModel

    ok, problems, _ = ck.validate_tokenizer_dir(cfg_dir)
    if not ok:
        raise ModelError("Qwen3-VL config/tokenizer directory invalid: " + "; ".join(problems))
    config = AutoConfig.from_pretrained(cfg_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg_dir, local_files_only=True)
    with init_empty_weights():  # parameters on meta, buffers (rotary inv_freq) real on CPU
        model = Qwen3VLModel._from_config(config, attn_implementation="sdpa")

    tensors, _ = ck.read_header(info.path)
    src = info.details["key_prefix"]  # "model." or "model.language_model."
    key_map = {}
    for k in tensors:
        if k.startswith("model.visual.") or k.endswith(".comfy_quant") or k.startswith("lm_head."):
            continue  # vision tower is not used for text-only prompt encoding
        if not k.startswith(src):
            raise ModelError(f"Unexpected text encoder key {k}")
        key_map[k] = "language_model." + k[len(src):]
    qnames = {key_map[k][: -len(".weight")] for k, v in tensors.items() if k in key_map and v["dtype"] in ck.FP8_TAGS}
    replace_linears_with_fp8(model, qnames)
    sd = ck.load_tensors(info.path, key_map)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    del sd
    missing = [m for m in missing if not m.startswith("visual.")]
    if missing or unexpected:
        raise ModelError(f"Text encoder state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    finalize_fp8(model)
    model.eval().requires_grad_(False)
    _assert_no_meta(model.language_model, "text encoder")
    lm_tensors = _collect_tensors(model, include=lambda n: n.startswith("language_model."))
    return model, lm_tensors, {"tokenizer": tokenizer, "fp8_layers": len(qnames)}


def _build_vae(info: ck.CheckpointInfo, _cfg_dir: str):
    from diffusers import AutoencoderKLQwenImage
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
    from safetensors.torch import load_file

    # Default AutoencoderKLQwenImage config == the Qwen-Image VAE (base_dim 96, z_dim 16, latents_mean/std).
    vae = AutoencoderKLQwenImage()
    raw = load_file(info.path)
    sd = raw if "decoder.conv_in.weight" in raw else convert_wan_vae_to_diffusers(raw)
    del raw
    vae.load_state_dict(sd, strict=True)
    del sd
    nn.Module.to(vae, dtype=torch.bfloat16)  # official Krea code runs the VAE in BF16; plain Module.to avoids a Diffusers warning
    vae.eval().requires_grad_(False)
    return vae, _collect_tensors(vae), {}


def _host_unregister(ptr: int) -> None:
    try:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaHostUnregister(ptr)
    except Exception:  # noqa: BLE001 - interpreter shutdown / broken context
        pass


def _strip_module(module: nn.Module) -> None:
    for sub in module.modules():
        for k, p in list(sub._parameters.items()):
            if p is not None and p.device.type != "meta":
                sub._parameters[k] = nn.Parameter(torch.empty(0, dtype=p.dtype, device="meta"), requires_grad=False)
        for k, b in list(sub._buffers.items()):
            if b is not None and b.device.type != "meta":
                sub._buffers[k] = torch.empty(0, dtype=b.dtype, device="meta")


def _assert_no_meta(module: nn.Module, label: str) -> None:
    meta = [n for n, t in list(module.named_parameters()) + list(module.named_buffers()) if t is not None and t.device.type == "meta"]
    if meta:
        raise ModelError(f"{label}: {len(meta)} tensors were not loaded from the checkpoint (e.g. {meta[:3]})")


__all__ = ["Component", "ModelManager", "ModelError", "InsufficientVRAM", "CudaUnrecoverable", "FP8Linear"]
