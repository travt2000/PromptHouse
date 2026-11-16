"""Write a generated guide into the character's document folder, where the Docs window, external editors
and the retrieval index (``/api/characters/{id}/context``) all see it.

The current guide lives at a fixed path (``guides/self-style-guide.md``, ``guides/content-guide.md``).
Writing a new one first archives the file that is there, including any edits made to it since, as
``guides/archive/<name>-<timestamp>.md`` with ``index: false`` so old versions don't compete in search.
Nothing is overwritten without a copy, and every version also stays in the database.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .. import character_docs as cdocs

GUIDE_FILES = {"character": "guides/self-style-guide.md", "inspiration": "guides/content-guide.md"}
GUIDE_TYPES = {"character": "self-style-guide", "inspiration": "content-guide"}
_FRONT = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)


def front_matter(doc_type: str, created: datetime, image_count: int, model_label: str) -> str:
    return ("---\n"
            f"type: {doc_type}\n"
            "tags: [guide, generated]\n"
            f"generated: {created.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            f"source: style analysis of {image_count} images ({model_label})\n"
            "---\n")


def mark_unindexed(content: str) -> str:
    """The same document with ``index: false`` in its front matter (added if it has none)."""
    m = _FRONT.match(content)
    if not m:
        return "---\nindex: false\n---\n" + content
    lines = [l for l in m.group(1).split("\n") if not re.match(r"\s*index\s*:", l)]
    return "---\n" + "\n".join(lines + ["index: false"]).lstrip("\n") + "\n---\n" + content[m.end():]


def publish_guide(folder: Path, kind: str, markdown: str, created: datetime, image_count: int,
                  model_label: str) -> tuple[str, str | None]:
    """Write the guide; returns ``(path, archived_path or None)`` relative to the folder."""
    rel = GUIDE_FILES[kind]
    archived = None
    expected = None
    target, _ = cdocs.safe_path(folder, rel)
    if target.exists():
        old = cdocs.read_doc(folder, rel)
        expected = old["mtime"]
        stem = Path(rel).stem
        stamp = created.strftime("%Y%m%d-%H%M%S")
        candidate, n = f"guides/archive/{stem}-{stamp}.md", 2
        while cdocs.safe_path(folder, candidate)[0].exists():
            candidate, n = f"guides/archive/{stem}-{stamp}-{n}.md", n + 1
        cdocs.write_doc(folder, candidate, mark_unindexed(old["content"]), None)
        archived = candidate
    content = front_matter(GUIDE_TYPES[kind], created, image_count, model_label) + markdown
    cdocs.write_doc(folder, rel, content, expected)  # refused if the file changed since it was archived
    return rel, archived
