"""Aggregated analyses -> a Markdown guide written by the model.

The prompts are files in ``prompts/guides/`` (see ``app/prompt_files.py``). The model writes the sections;
code adds the header and the data appendix (so the numbers in the appendix are exact). The result is
published into the character's document folder by ``app.style.publish``; the raw tally is kept in the
database for later stages (content planning, prompt generation) that want the data rather than prose.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import httpx

from ..prompt_files import render
from .aggregate import aggregate, prompt_summary, stats_markdown
from .character_types import CharacterType
from .vsettings import VisionSettings as Settings
from .kinds import Kind

_SECTION_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.M)


@dataclass(frozen=True)
class GuideSpec:
    """The prompt is ``prompts/guides/<key>.system.md``; its "## " headings are the guide's sections."""
    key: str
    title: str

    @property
    def system_prompt(self) -> str:
        return render(f"guides/{self.key}.system")

    @property
    def sections(self) -> tuple[str, ...]:
        return tuple(_SECTION_HEADING.findall(self.system_prompt))


GUIDES: dict[str, GuideSpec] = {
    "character": GuideSpec("character", "Self-Style Guide"),
    "inspiration": GuideSpec("inspiration", "Content Guide"),
}


class GuideError(RuntimeError):
    pass


class GuideCancelled(GuideError):
    pass


def type_vars(ctype: CharacterType | None) -> dict:
    """Variables for ``prompts/shared/character-type.md``."""
    return {"type_label": ctype.label if ctype else "", "type_description": ctype.description if ctype else "",
            "type_guidance": ctype.guidance if ctype else ""}


def build_messages(kind: Kind, stats: dict, ctype: CharacterType | None, budget_chars: int) -> list[dict]:
    """System + user messages, with free-text samples trimmed until the prompt fits ``budget_chars``."""
    spec = GUIDES[kind.key]
    system = spec.system_prompt
    for samples in (8, 5, 3, 2, 1, 0):
        user = render("guides/user", title=spec.title, image_count=stats["image_count"],
                      weighted=bool(stats.get("weighted")), data=prompt_summary(stats, samples), **type_vars(ctype))
        if len(system) + len(user) <= budget_chars:
            break
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def prompt_budget_chars(s: Settings) -> int:
    """Rough character budget for the prompt: context minus the reply, at ~3 characters per token."""
    return max(3000, (s.n_ctx - s.guide_max_tokens - 512) * 3)


_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean_body(text: str) -> str:
    body = _THINK.sub("", text).strip()
    body = re.sub(r"^```(?:markdown|md)?\s*\n|\n```\s*$", "", body).strip()
    lines = body.splitlines()
    if lines and re.match(r"^#\s", lines[0]):  # the document title is added by code
        body = "\n".join(lines[1:]).strip()
    return body


class GuideWriter:
    def __init__(self, base_url: str, settings: Settings, model_name: str = "local"):
        self.base_url = base_url.rstrip("/")
        self.s = settings
        self.model_name = model_name
        self._kwargs = json.loads(settings.chat_template_kwargs or "{}")

    def write(self, messages: list[dict], cancel: threading.Event | None = None,
              temperature: float | None = None, max_tokens: int | None = None) -> tuple[str, bool]:
        """Returns ``(text, truncated)``. Waits on a worker thread so a cancel is noticed promptly.
        ``temperature``/``max_tokens`` default to the guide settings."""
        payload: dict = {"model": self.model_name, "messages": messages,
                         "temperature": self.s.guide_temperature if temperature is None else temperature,
                         "max_tokens": self.s.guide_max_tokens if max_tokens is None else max_tokens}
        if self._kwargs:
            payload["chat_template_kwargs"] = self._kwargs
        box: dict = {}

        def call():
            try:
                with httpx.Client(timeout=max(self.s.request_timeout_s, 600)) as client:
                    box["resp"] = client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            except Exception as exc:  # reported on the waiting thread
                box["exc"] = exc

        worker = threading.Thread(target=call, daemon=True, name="guide-request")
        worker.start()
        while worker.is_alive():
            if cancel is not None and cancel.is_set():
                raise GuideCancelled("Cancelled.")
            worker.join(0.25)
        if "exc" in box:
            raise GuideError(f"Request failed: {box['exc']}")
        resp = box["resp"]
        if resp.status_code != 200:
            raise GuideError(f"Server returned HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            choice = resp.json()["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, ValueError):
            raise GuideError(f"Unexpected response shape from the server: {resp.text[:300]}") from None
        if not clean_body(text):
            raise GuideError("The model returned an empty reply.")
        return text, choice.get("finish_reason") == "length"


def compose_document(kind: Kind, body: str, stats: dict, model_label: str, ctype: CharacterType | None,
                     created: datetime, truncated: bool, subject: str = "") -> str:
    """``subject`` (the character's name) is added to the title."""
    spec = GUIDES[kind.key]
    meta = [
        f"Generated {created.strftime('%Y-%m-%d %H:%M')}",
        f"{stats['image_count']} images",
        f"model {model_label}",
        f"schema v{kind.schema_version}",
        f"character type: {ctype.label if ctype else 'not set'}",
    ]
    missing = [name for name in spec.sections if not re.search(rf"^##\s+{re.escape(name)}\s*$", body, re.M | re.I)]
    notes = []
    if truncated:
        notes.append("> **Note:** the model hit its output limit, so this guide is cut short. "
                     "Raise *Guide max tokens* in Settings and write it again.")
    if missing:
        notes.append(f"> **Note:** the model left out these sections: {', '.join(missing)}.")
    title = f"{spec.title} — {subject}" if subject else spec.title
    parts = [f"# {title}", "", f"*{' · '.join(meta)}*", ""]
    if notes:
        parts += ["\n>\n".join(notes), ""]
    parts += [body.strip(), "", "---", "", stats_markdown(stats)]
    return "\n".join(parts)


@dataclass
class GuideResult:
    markdown: str
    stats: dict
    truncated: bool
    seconds: float


def produce_guide(kind: Kind, records: list[dict], writer: GuideWriter, settings: Settings,
                  ctype: CharacterType | None, created: datetime,
                  cancel: threading.Event | None = None, subject: str = "") -> GuideResult:
    if not records:
        raise GuideError("There are no analyzed images to write a guide from.")
    started = time.monotonic()
    stats = aggregate(kind, records)
    messages = build_messages(kind, stats, ctype, prompt_budget_chars(settings))
    text, truncated = writer.write(messages, cancel)
    md = compose_document(kind, clean_body(text), stats, settings.model_label(), ctype, created, truncated, subject)
    return GuideResult(md, stats, truncated, time.monotonic() - started)

