"""Image -> structured record, through any OpenAI-compatible chat endpoint.

The prompt and the schema come from the analysis :class:`~app.style.kinds.Kind`.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
import jsonschema
from PIL import Image, ImageOps

from .vsettings import VisionSettings as Settings
from .kinds import Kind


class AnalysisError(RuntimeError):
    def __init__(self, message: str, raw_text: str | None = None):
        super().__init__(message)
        self.raw_text = raw_text


@dataclass
class AnalysisResult:
    record: dict
    raw_text: str
    seconds: float


class Analyzer(Protocol):
    def analyze(self, image_path: str, kind: Kind) -> AnalysisResult: ...


def prepare_image(path: str, max_edge: int, quality: int) -> str:
    """Downscale and encode as a JPEG data URL (image size drives per-image token cost and speed)."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[3])
            img = background
        else:
            img = img.convert("RGB")
        img.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def parse_record(text: str, schema: dict) -> dict:
    """Extract the JSON object from a model reply and validate it against ``schema``."""
    cleaned = _FENCE.sub("", _THINK.sub("", text).strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise AnalysisError("The reply contained no JSON object.", text)
    try:
        record = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"The reply was not valid JSON: {exc}", text) from None
    try:
        jsonschema.validate(record, schema)
    except jsonschema.ValidationError as exc:
        where = "/".join(str(p) for p in exc.absolute_path) or "(root)"
        raise AnalysisError(f"The record does not match the schema at {where}: {exc.message}", text) from None
    return record


_FALLBACK = {"json_schema": "json_object", "json_object": "none"}


class OpenAICompatAnalyzer:
    def __init__(self, base_url: str, settings: Settings, model_name: str = "local"):
        self.base_url = base_url.rstrip("/")
        self.s = settings
        self.model_name = model_name
        self.mode = settings.structured_mode
        self._client = httpx.Client(timeout=settings.request_timeout_s)
        self._kwargs = json.loads(settings.chat_template_kwargs or "{}")

    def close(self) -> None:
        self._client.close()

    def _payload(self, data_url: str, mode: str, kind: Kind) -> dict:
        payload: dict = {
            "model": self.model_name,
            "temperature": self.s.temperature,
            "max_tokens": self.s.max_tokens,
        }
        if self._kwargs:
            payload["chat_template_kwargs"] = self._kwargs
        if mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": f"{kind.key}_record", "strict": True, "schema": kind.schema_copy()},
            }
        elif mode == "json_object":
            payload["response_format"] = {"type": "json_object", "schema": kind.schema_copy()}
        # without server-side enforcement the schema goes into the prompt (prompts/analysis/schema-in-prompt.md)
        user_text = kind.user_prompt(json.dumps(kind.schema) if mode not in ("json_schema", "json_object") else "")
        payload["messages"] = [
            {"role": "system", "content": kind.system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]
        return payload

    def _chat(self, data_url: str, kind: Kind) -> str:
        while True:
            resp = self._client.post(f"{self.base_url}/v1/chat/completions", json=self._payload(data_url, self.mode, kind))
            if resp.status_code == 400 and self.mode in _FALLBACK:
                # This server build does not accept that structured-output mode; step down once and remember.
                self.mode = _FALLBACK[self.mode]
                continue
            if resp.status_code != 200:
                raise AnalysisError(f"Server returned HTTP {resp.status_code}: {resp.text[:300]}")
            try:
                return resp.json()["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, ValueError):
                raise AnalysisError("Unexpected response shape from the server.", resp.text[:500]) from None

    def analyze(self, image_path: str, kind: Kind) -> AnalysisResult:
        if not Path(image_path).is_file():
            raise AnalysisError(f"Image file is missing: {image_path}")
        started = time.monotonic()
        try:
            data_url = prepare_image(image_path, self.s.image_max_edge, self.s.jpeg_quality)
        except Exception as exc:
            raise AnalysisError(f"Could not read the image: {exc}") from None
        last: AnalysisError | None = None
        for _ in range(2):  # one retry on an unparseable or invalid reply
            try:
                text = self._chat(data_url, kind)
            except httpx.HTTPError as exc:
                raise AnalysisError(f"Request failed: {exc}") from None
            try:
                record = parse_record(text, kind.schema)
            except AnalysisError as exc:
                last = exc
                continue
            return AnalysisResult(record, text, time.monotonic() - started)
        assert last is not None
        raise last
