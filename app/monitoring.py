"""Hardware monitoring: NVML (nvidia-ml-py) for device-wide GPU stats, psutil for CPU/RAM.

No CUDA/PyTorch GPU work happens here; NVML queries are cheap driver calls, safe during inference.
"""

from __future__ import annotations

import threading
import time

import psutil

try:
    import pynvml  # provided by the nvidia-ml-py package
except ImportError:  # pragma: no cover
    pynvml = None


class Monitor:
    def __init__(self, gpu_index: int = 0):
        self._lock = threading.Lock()
        self._handle = None
        self.error: str | None = None
        self.gpu_index = gpu_index
        if pynvml is None:
            self.error = "nvidia-ml-py is not installed"
        else:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception as e:  # noqa: BLE001
                self.error = f"NVML unavailable: {e}"
        psutil.cpu_percent(interval=None)  # prime the CPU counter (first call always returns 0.0)
        self._proc = psutil.Process()

    def sample(self) -> dict:
        vm = psutil.virtual_memory()
        out = {
            "time": time.time(),
            "cpu": {"percent": psutil.cpu_percent(interval=None), "count": psutil.cpu_count()},
            "ram": {"total": vm.total, "used": vm.total - vm.available, "percent": vm.percent, "process_rss": self._proc.memory_info().rss},
            "gpu": None,
            "gpu_error": self.error,
        }
        if self._handle is not None:
            with self._lock:
                try:
                    name = pynvml.nvmlDeviceGetName(self._handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    out["gpu"] = {
                        "name": name.decode() if isinstance(name, bytes) else name,
                        "vram_total": mem.total,
                        "vram_used": mem.used,  # device-wide (all processes), not just PyTorch tensors
                        "util_percent": util.gpu,
                    }
                except Exception as e:  # noqa: BLE001
                    out["gpu_error"] = f"NVML query failed: {e}"
        return out
