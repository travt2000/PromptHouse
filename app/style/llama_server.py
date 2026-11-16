"""Hidden ``llama-server`` subprocess: started for a batch, stopped when it ends.

Stopping the process returns all VRAM and RAM to the OS, which is the same residency
discipline PromptHouse uses for its own models.
"""
from __future__ import annotations

import atexit
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from .vsettings import VisionSettings as Settings


class ServerError(RuntimeError):
    pass


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_command(s: Settings, port: int, vision: bool = True) -> list[str]:
    """``vision=False`` leaves out the projector: text-only jobs load less and use less VRAM."""
    cmd = [s.llama_server_path, "-m", s.model_path]
    if vision:
        cmd += ["--mmproj", s.mmproj_path]
    cmd += [
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(s.n_ctx),
        "-ngl", str(s.n_gpu_layers),
    ]
    if s.extra_args.strip():
        cmd += shlex.split(s.extra_args, posix=(sys.platform != "win32"))
    return cmd


class ServerManager:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.base_url: str | None = None
        atexit.register(self.stop)

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def log_tail(self, lines: int = 40) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def start(self, s: Settings, cancel: threading.Event | None = None, vision: bool = True) -> str:
        """Launch the server and block until ``/health`` reports ready. Returns the base URL."""
        with self._lock:
            if self.running:
                return self.base_url  # type: ignore[return-value]
            port = s.port or free_port()
            cmd = build_command(s, port, vision)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(self.log_path, "w", encoding="utf-8")
            log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
            log.flush()
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
            try:
                self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
            except OSError as exc:
                log.close()
                raise ServerError(f"Could not start llama-server: {exc}") from None
            finally:
                log.close()  # the child keeps its own handle
            self.base_url = f"http://127.0.0.1:{port}"

        deadline = time.monotonic() + s.startup_timeout_s
        while True:
            if cancel is not None and cancel.is_set():
                self.stop()
                raise ServerError("Cancelled while the model was loading.")
            code = self._proc.poll() if self._proc else -1
            if code is not None:
                tail = self.log_tail(15)
                self._proc = None
                raise ServerError(f"llama-server exited with code {code} while loading.\n{tail}")
            try:
                if httpx.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                    return self.base_url  # type: ignore[return-value]
            except httpx.HTTPError:
                pass  # not listening yet, or still loading (503)
            if time.monotonic() > deadline:
                tail = self.log_tail(15)
                self.stop()
                raise ServerError(f"llama-server was not ready after {s.startup_timeout_s}s.\n{tail}")
            time.sleep(0.5)

    def stop(self) -> None:
        with self._lock:
            proc, self._proc, self.base_url = self._proc, None, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
