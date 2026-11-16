"""Text sent to the local language model lives in ``prompts/`` as Markdown templates, not in code.

Files are re-read when they change (checked by modification time on every use), so a tweak takes effect on
the next request without restarting the app. ``PROMPTHOUSE_PROMPTS_DIR`` points the app at another folder.

Template syntax (a small subset of Mustache; see ``prompts/README.md``):

* ``{{name}}``                      the value of a variable (inserted as is, never re-parsed)
* ``{{#name}} … {{/name}}``         the text only if the variable is non-empty / true
* ``{{^name}} … {{/name}}``         the text only if the variable is empty / false
* ``{{> folder/file}}``             another template (``prompts/folder/file.md``), with the same variables
* ``<!-- … -->``                    a comment for whoever edits the file; never sent to the model

A line holding only a section tag is removed with its line break, so sections can sit on their own lines.
Trailing blank lines of a template are ignored. An unknown variable is an error, not an empty string, so a
typo in a template is reported instead of silently sending a broken prompt.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path

# Deliberately not in app.config: this module is also imported by light-weight tools (the test fake server).
PROMPTS_DIR = Path(os.environ.get("PROMPTHOUSE_PROMPTS_DIR") or Path(__file__).resolve().parent.parent / "prompts")


class PromptFileError(RuntimeError):
    pass


_NAME = re.compile(r"^[a-z0-9_-]+(?:/[a-z0-9_.-]+)*$")
_COMMENT = re.compile(r"[ \t]*<!--.*?-->[ \t]*(?:\n|\Z)?", re.S)
_STANDALONE = re.compile(r"^[ \t]*(\{\{[#^/]\s*\w+\s*\}\})[ \t]*(?:\n|\Z)", re.M)
_SECTION = re.compile(r"\{\{([#^])\s*(\w+)\s*\}\}(.*?)\{\{/\s*\2\s*\}\}", re.S)
_TAG = re.compile(r"\{\{\s*(>)?\s*([\w./-]+)\s*\}\}")
_LEFTOVER = re.compile(r"\{\{[#^/]\s*\w*\s*\}\}")

_cache: dict[str, tuple[tuple[int, int], str]] = {}
_lock = threading.Lock()


def prompts_dir() -> Path:
    return Path(PROMPTS_DIR)


def path_of(name: str) -> Path:
    if not _NAME.match(name) or ".." in name:
        raise PromptFileError(f"Invalid prompt name: {name!r}")
    return prompts_dir() / f"{name}.md"


def source(name: str) -> str:
    """The template text of ``prompts/<name>.md`` with comments and standalone-tag lines removed."""
    p = path_of(name)
    try:
        st = p.stat()
    except OSError:
        raise PromptFileError(f"Prompt file not found: {p}") from None
    key = (st.st_mtime_ns, st.st_size)
    with _lock:
        hit = _cache.get(str(p))
        if hit and hit[0] == key:
            return hit[1]
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise PromptFileError(f"Could not read prompt file {p}: {e}") from None
    text = _COMMENT.sub("", text.replace("\r\n", "\n"))
    text = _STANDALONE.sub(r"\1", text).strip("\n")
    with _lock:
        _cache[str(p)] = (key, text)
    return text


def render(name: str, /, **variables) -> str:
    """``prompts/<name>.md`` with its sections, variables and includes filled in. (``name`` is positional-only,
    so a template may use any variable name.)"""
    return _render(name, (), variables)


def _render(name: str, _stack: tuple[str, ...], variables: dict) -> str:
    if name in _stack:
        raise PromptFileError(f"Prompt files include each other in a loop: {' -> '.join(_stack + (name,))}")
    text = source(name)

    def missing(key: str):
        known = ", ".join(sorted(variables)) or "none"
        return PromptFileError(f"prompts/{name}.md uses {{{{{key}}}}}, which is not available here (available: {known})")

    def sections(t: str) -> str:
        def one(m: re.Match) -> str:
            neg, key, inner = m.group(1) == "^", m.group(2), m.group(3)
            if key not in variables:
                raise missing(key)
            return sections(inner) if bool(variables[key]) != neg else ""
        return _SECTION.sub(one, t)

    text = sections(text)
    stray = _LEFTOVER.search(text)
    if stray:
        raise PromptFileError(f"prompts/{name}.md has an unmatched section tag {stray.group(0)}")

    def tag(m: re.Match) -> str:
        if m.group(1):
            return _render(m.group(2), _stack + (name,), variables)
        key = m.group(2)
        if key not in variables:
            raise missing(key)
        return str(variables[key])

    return _TAG.sub(tag, text).rstrip("\n")


def clear_cache() -> None:
    with _lock:
        _cache.clear()
