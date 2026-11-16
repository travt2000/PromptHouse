"""Image prompts written by the local model from the character's guides.

The guides are read from the character's document folder (``guides/self-style-guide.md`` and
``guides/content-guide.md``), so edits made in Docs or an external editor are what the model sees.
The data appendix and the "Gaps" sections are left out (they describe the analysis, not the
character), and the prompt-keyword sections are moved to the front so they survive trimming.
The request itself is ``prompts/image-prompts/system.md`` and ``prompts/image-prompts/user.md``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .. import character_docs as cdocs
from .character_types import CharacterType
from ..prompt_files import render
from .guides import _THINK, type_vars
from .publish import GUIDE_FILES
from .vsettings import VisionSettings

MAX_COUNT = 8
_SKIP = re.compile(r"^(appendix\b|gaps\b)", re.I)
_FIRST = re.compile(r"prompt", re.I)
_H2 = re.compile(r"^##\s+(.*?)\s*#*\s*$")
_COMMENT = re.compile(r"<!--.*?-->", re.S)

class PromptsError(ValueError):
    pass


@dataclass
class GuideSource:
    kind: str
    path: str
    text: str


def _sections(body: str) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = [("", [])]
    for line in body.split("\n"):
        m = _H2.match(line)
        if m:
            out.append((m.group(1), [line]))
        else:
            out[-1][1].append(line)
    return out


def guide_text(raw: str, path: str) -> str:
    """The parts of a guide that are useful for writing prompts, keyword sections first."""
    body = _COMMENT.sub("", cdocs.parse_document(raw, path).body)
    sections = _sections(body)
    if len(sections) == 1:  # no "## " sections (a hand-written guide): use it as it is
        return body.strip()
    title, first, rest = [], [], []
    for heading, lines in sections:
        if not heading:  # title, generated-by line, notes: only the title is worth keeping
            title = [l for l in lines if l.startswith("# ")][:1]
            continue
        if _SKIP.match(heading):
            continue
        text = re.sub(r"\n\s*-{3,}\s*$", "", "\n".join(lines).strip()).strip()
        (first if _FIRST.search(heading) else rest).append(text)
    return "\n\n".join(title + first + rest).strip()


def load_guides(folder: Path) -> list[GuideSource]:
    """The character's current guides, in the order Self-Style Guide, Content Guide. Missing files are skipped."""
    out = []
    for kind, rel in GUIDE_FILES.items():
        p = folder / rel
        if not p.is_file():
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        text = guide_text(raw, rel)
        if text:
            out.append(GuideSource(kind, rel, text))
    return out


def budget_chars(s: VisionSettings) -> int:
    """Rough character budget for the request: context minus the reply, at ~3 characters per token."""
    return max(3000, (s.n_ctx - s.prompt_max_tokens - 512) * 3)


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    nl = cut.rfind("\n")
    return (cut[:nl] if nl > limit // 2 else cut).rstrip() + "\n[…trimmed to fit the context size]"


def system_prompt() -> str:
    return render("image-prompts/system")


def build_messages(guides: list[GuideSource], ctype: CharacterType | None, idea: str, count: int,
                   budget: int) -> list[dict]:
    if not guides:
        raise PromptsError("No guides to write from.")
    system = system_prompt()
    by_kind = {g.kind: g.text for g in guides}
    variables = {"idea": idea.strip(), "count": count, "plural": count != 1,
                 "has_content_guide": "inspiration" in by_kind, **type_vars(ctype)}
    # everything but the guides is never trimmed; the guides share what is left of the budget
    tail = render("image-prompts/user", self_style_guide="", content_guide="", **variables)
    share = max(1000, budget - len(system) - len(tail) - 200) // len(guides)
    user = render("image-prompts/user", self_style_guide=_trim(by_kind.get("character", ""), share),
                  content_guide=_trim(by_kind.get("inspiration", ""), share), **variables)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_ITEM = re.compile(r"^\s*(?:\d{1,2}[.):]|[-*•])\s+(.*)$")
_LABEL = re.compile(r"^(?:\*\*|__)[^*_]{1,80}(?:\*\*|__)\s*[:—–-]?\s*")


def _clean(item: str) -> str:
    s = item.strip()
    rest = _LABEL.sub("", s)
    if len(rest) >= 20:  # a bold title in front of the prompt ("**Golden hour:** ...")
        s = rest
    s = re.sub(r"^(?:prompt\s*\d*\s*[:—–-])\s*", "", s, flags=re.I)
    s = s.replace("**", "").replace("__", "").strip()
    if len(s) >= 2 and s[0] in "\"'“" and s[-1] in "\"'”":
        s = s[1:-1].strip()
    return re.sub(r"\s+", " ", s)


def parse_prompts(text: str) -> list[str]:
    """The prompts in a model reply: numbered or bulleted items (continuation lines joined), or paragraphs."""
    body = _THINK.sub("", text).strip()
    body = re.sub(r"^```[a-z]*\s*\n|\n```\s*$", "", body).strip()
    items: list[str] = []
    cur: str | None = None
    for line in body.split("\n"):
        m = _ITEM.match(line)
        if m:
            if cur is not None:
                items.append(cur)
            cur = m.group(1)
        elif line.strip() and cur is not None and not line.lstrip().startswith("#"):
            cur += " " + line.strip()
        elif not line.strip() and cur is not None:
            items.append(cur)
            cur = None
    if cur is not None:
        items.append(cur)
    if not items:
        items = [p for p in re.split(r"\n\s*\n", body) if p.strip() and not p.lstrip().startswith("#")]
    out: list[str] = []
    for item in items:
        s = _clean(item)
        if len(s) >= 15 and s not in out:
            out.append(s)
    return out
