"""Per-character document folders (Markdown) and the retrieval groundwork for later LLM features.

Layout (one folder per character under `CHARACTER_DOCS_DIR`, default `data/characters/`):

    data/characters/<folder>/
        README.md         what this folder is for (written once, at creation)
        bible.md          character bible: identity, appearance, personality, backstory
        style-guide.md    visual style: look, wardrobe, photography, things to avoid
        voice.md          voice and tone for captions / posts
        notes/            anything else (any depth of subfolders)

The folder name is stored in the database (`characters.folder`) and does not change when the character is
renamed. Templates are written only when the folder is first created, so files you delete stay deleted.
The files are plain Markdown: edit them here or in any editor (VS Code, Obsidian, ...).

Retrieval groundwork (no LLM yet):
  * `parse_document` reads optional YAML front matter (`type`, `tags`, `title`, ...).
  * `chunk_document` splits a document by headings (ignoring `<!-- comments -->` and files with `index: false`) (then by paragraphs when a section is long) into chunks
    with stable ids, the heading path, line numbers and a content hash (for a future embedding cache).
  * `DocIndex` caches chunks per file (by size + mtime) and offers `search()`: a small BM25 lexical
    retriever. It is the placeholder that an embedding retriever will later replace behind the same call.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

import yaml

MAX_DOC_BYTES = 1_000_000
MAX_CHUNK_CHARS = 1500
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-()]*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_WORD_RE = re.compile(r"[0-9a-z]+(?:'[a-z]+)?")


class DocError(ValueError):
    pass


class DocConflict(DocError):
    """The file changed on disk since the client loaded it."""


# ------------------------------------------------------------------------------------------ folders


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s[:60].strip("-") or "character"


def allocate_folder(root: Path, name: str, taken: set[str]) -> str:
    """A folder name for a new character: its slug, made unique against existing folders and DB entries."""
    base = slugify(name)
    taken_l = {t.lower() for t in taken}
    for i in range(1, 10_000):
        cand = base if i == 1 else f"{base}-{i}"
        if cand.lower() not in taken_l and not (root / cand).exists():
            return cand
    raise DocError("Could not allocate a folder name")


def _templates(name: str) -> dict[str, str]:
    today = time.strftime("%Y-%m-%d")
    return {
        "README.md": f"""---
index: false
---
# {name} — documents

This folder holds the reference documents for **{name}** (created {today}).
Everything here is plain Markdown; edit it in PromptHouse or in any editor.

* `bible.md` — who the character is: identity, appearance, personality, backstory.
* `style-guide.md` — how the character is shown: look, wardrobe, photography, things to avoid.
* `voice.md` — how the character speaks: tone and vocabulary for captions and posts.
* `notes/` — anything else. Add files and subfolders freely.

Optional front matter at the top of a file (between `---` lines) describes it for later search
and planning features, e.g. `type: bible` or `tags: [wardrobe, summer]`.
Headings (`#`, `##`, ...) split documents into sections; keep one topic per section.
Text inside `<!-- ... -->` comments (like the hints in the templates) is not searched, and a file with
`index: false` in its front matter (like this one) is left out of search entirely.
""",
        "bible.md": f"""---
type: bible
---
# {name} — character bible

## Identity
<!-- Name, age, where they live, what they do. -->

## Appearance
<!-- Face, hair, eyes, build, distinguishing features. -->

## Personality
<!-- Traits, interests, values, quirks. -->

## Backstory
<!-- Key history that explains who they are now. -->
""",
        "style-guide.md": f"""---
type: style-guide
---
# {name} — style guide

## Look
<!-- Overall aesthetic and mood. -->

## Wardrobe
<!-- Signature pieces, colours, what they never wear. -->

## Photography
<!-- Settings, lighting, framing and camera styles that suit the character. -->

## Avoid
<!-- Things that break the character's look. -->
""",
        "voice.md": f"""---
type: voice
---
# {name} — voice

## Tone
<!-- How they sound: e.g. warm, dry, playful. -->

## Vocabulary
<!-- Words and phrases they use, and ones they never would. -->

## Examples
<!-- A few sample captions or lines in their voice. -->
""",
        "notes/.gitkeep": "",
    }


def provision(root: Path, folder: str, name: str) -> Path:
    """Create the folder with starter templates. Existing folders are left exactly as they are."""
    d = root / folder
    if d.exists():
        return d
    d.mkdir(parents=True)
    for rel, text in _templates(name).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    return d


# ------------------------------------------------------------------------------------------ files


def safe_path(folder: Path, rel: str) -> tuple[Path, str]:
    """Resolve a client-supplied relative path inside `folder`. Only `.md` files; no traversal."""
    if not isinstance(rel, str) or not rel.strip():
        raise DocError("A file name is required")
    rel = rel.strip().replace("\\", "/")
    parts = PurePosixPath(rel).parts
    if rel.startswith("/") or ":" in rel or not parts or any(p in ("", ".", "..") for p in parts):
        raise DocError("Use a relative path inside the character's folder (no '..', no drive letters)")
    if any(not _NAME_RE.match(p) for p in parts):
        raise DocError("File and folder names may contain letters, digits, spaces and _ . - ( ), and must not start with a dot")
    if len(parts) > 6 or len(rel) > 200:
        raise DocError("Path is too long")
    if not parts[-1].lower().endswith(".md"):
        raise DocError("Only Markdown (.md) files can be edited here")
    p = (folder / Path(*parts)).resolve()
    if not p.is_relative_to(folder.resolve()):
        raise DocError("Path escapes the character's folder")
    return p, "/".join(parts)


def list_docs(folder: Path) -> list[dict]:
    out = []
    if not folder.is_dir():
        return out
    base = folder.resolve()
    for p in sorted(folder.rglob("*.md"), key=lambda x: str(x.relative_to(folder)).lower()):
        rel = p.relative_to(folder)
        if any(part.startswith(".") for part in rel.parts) or not p.is_file():
            continue
        if not p.resolve().is_relative_to(base):  # symlink pointing elsewhere
            continue
        st = p.stat()
        doc = parse_document(_read(p) if st.st_size <= MAX_DOC_BYTES else "", rel.as_posix())
        out.append({"path": rel.as_posix(), "title": doc.title, "type": doc.type, "tags": doc.tags,
                    "size": st.st_size, "mtime": str(st.st_mtime_ns), "too_large": st.st_size > MAX_DOC_BYTES})
    return out


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def read_doc(folder: Path, rel: str) -> dict:
    p, rel = safe_path(folder, rel)
    if not p.is_file():
        raise FileNotFoundError(rel)
    st = p.stat()
    if st.st_size > MAX_DOC_BYTES:
        raise DocError(f"{rel} is larger than {MAX_DOC_BYTES // 1000} KB; edit it in an external editor")
    return {"path": rel, "content": _read(p), "mtime": str(st.st_mtime_ns)}  # a string: nanosecond ints exceed JavaScript number precision


def write_doc(folder: Path, rel: str, content: str, expected_mtime: str | None) -> dict:
    """Create (expected_mtime None: the file must not exist) or overwrite (expected_mtime must match disk).
    Written atomically via a temporary file."""
    p, rel = safe_path(folder, rel)
    if not isinstance(content, str):
        raise DocError("content must be text")
    data = content.replace("\r\n", "\n").encode("utf-8")
    if len(data) > MAX_DOC_BYTES:
        raise DocError(f"Documents are limited to {MAX_DOC_BYTES // 1000} KB")
    if expected_mtime is None:
        if p.exists():
            raise DocConflict(f"{rel} already exists")
    else:
        if not p.exists():
            raise DocConflict(f"{rel} was deleted on disk")
        if str(p.stat().st_mtime_ns) != str(expected_mtime):
            raise DocConflict(f"{rel} changed on disk since you opened it (edited elsewhere?). Reload it, then save again.")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    return {"path": rel, "mtime": str(p.stat().st_mtime_ns), "size": len(data)}


def delete_doc(folder: Path, rel: str) -> None:
    p, rel = safe_path(folder, rel)
    if not p.is_file():
        raise FileNotFoundError(rel)
    p.unlink()


# ------------------------------------------------------------------------------------------ parsing + chunking


@dataclass
class Document:
    path: str
    title: str
    type: str
    tags: list[str]
    meta: dict
    body: str
    body_line: int  # 1-based line number where the body starts (after front matter)
    warnings: list[str] = field(default_factory=list)


def parse_document(text: str, path: str) -> Document:
    meta: dict = {}
    warnings: list[str] = []
    body, body_line = text, 1
    if text.startswith("---"):
        lines = text.split("\n")
        end = next((i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None)
        if end is not None:
            try:
                loaded = yaml.safe_load("\n".join(lines[1:end])) or {}
                if isinstance(loaded, dict):
                    meta = loaded
                else:
                    warnings.append("front matter is not a key/value mapping; ignored")
            except yaml.YAMLError as e:
                warnings.append(f"front matter is not valid YAML; ignored ({str(e).splitlines()[0]})")
            body, body_line = "\n".join(lines[end + 1:]), end + 2
    stem = PurePosixPath(path).stem
    title = str(meta.get("title") or "")
    if not title:
        m = next((_HEADING_RE.match(l) for l in body.split("\n") if _HEADING_RE.match(l)), None)
        title = m.group(2) if m else stem
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    tags = [str(t) for t in tags] if isinstance(tags, list) else []
    dtype = str(meta.get("type") or stem).strip().lower()
    return Document(path=path, title=title, type=dtype, tags=tags, meta=meta, body=body, body_line=body_line, warnings=warnings)


@dataclass
class Chunk:
    id: str             # stable for (character folder, path, ordinal)
    character_id: int
    path: str
    doc_type: str
    doc_title: str
    tags: list[str]
    headings: list[str]  # heading path, e.g. ["Ada — style guide", "Wardrobe"]
    text: str
    start_line: int
    end_line: int
    sha256: str         # of `text`; lets a future embedding cache skip unchanged chunks

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long(lines: list[tuple[int, str]], limit: int) -> list[list[tuple[int, str]]]:
    """Split a section's lines at blank lines so each piece stays under `limit` characters when possible."""
    paras, cur = [], []
    for ln in lines:
        cur.append(ln)
        if not ln[1].strip():
            paras.append(cur)
            cur = []
    if cur:
        paras.append(cur)
    out, piece, size = [], [], 0
    for para in paras:
        n = sum(len(t) + 1 for _, t in para)
        if piece and size + n > limit:
            out.append(piece)
            piece, size = [], 0
        piece.extend(para)
        size += n
    if piece:
        out.append(piece)
    return out


def chunk_document(doc: Document, character_id: int, folder: str, limit: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Heading-based chunks. HTML comments are dropped (line numbers kept), sections that are only headings
    are skipped, and `index: false` in the front matter excludes the whole document."""
    if doc.meta.get("index") is False:
        return []
    body = _COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), doc.body)
    sections: list[tuple[list[str], list[tuple[int, str]]]] = []
    stack: list[tuple[int, str]] = []
    cur: list[tuple[int, str]] = []
    in_fence = False

    def flush():
        if any(t.strip() for _, t in cur):
            sections.append(([h for _, h in stack], list(cur)))

    for i, line in enumerate(body.split("\n")):
        lineno = doc.body_line + i
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            flush()
            cur = [(lineno, line)]
            level = len(m.group(1))
            stack = [(lv, h) for lv, h in stack if lv < level] + [(level, m.group(2))]
        else:
            cur.append((lineno, line))
    flush()

    chunks: list[Chunk] = []
    for headings, lines in sections:
        for piece in _split_long(lines, limit):
            text = "\n".join(t for _, t in piece).strip()
            if not text or all(_HEADING_RE.match(t) for t in text.split("\n") if t.strip()):
                continue
            n = len(chunks)
            cid = hashlib.sha1(f"{folder}/{doc.path}#{n}".encode()).hexdigest()[:16]
            chunks.append(Chunk(id=cid, character_id=character_id, path=doc.path, doc_type=doc.type, doc_title=doc.title,
                                tags=doc.tags, headings=headings, text=text, start_line=piece[0][0], end_line=piece[-1][0],
                                sha256=hashlib.sha256(text.encode()).hexdigest()))
    return chunks


# ------------------------------------------------------------------------------------------ index + lexical search


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class DocIndex:
    """Chunks of every character's documents, cached per file by (size, mtime). Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._files: dict[tuple[int, str], tuple[tuple[int, int], list[Chunk], list[str]]] = {}

    def chunks(self, character_id: int, folder_root: Path, folder: str) -> tuple[list[Chunk], list[dict]]:
        """All chunks for a character (re-parsing only changed files) and per-file warnings."""
        d = folder_root / folder
        files = list_docs(d)
        seen, out, warnings = set(), [], []
        with self._lock:
            for f in files:
                key = (character_id, f["path"])
                seen.add(key)
                if f["too_large"]:
                    warnings.append({"path": f["path"], "warning": "too large to index"})
                    continue
                sig = (f["size"], f["mtime"])
                cached = self._files.get(key)
                if cached is None or cached[0] != sig:
                    doc = parse_document(_read(d / f["path"]), f["path"])
                    cached = (sig, chunk_document(doc, character_id, folder), doc.warnings)
                    self._files[key] = cached
                out.extend(cached[1])
                warnings.extend({"path": f["path"], "warning": w} for w in cached[2])
            for key in [k for k in self._files if k[0] == character_id and k not in seen]:
                del self._files[key]
        return out, warnings

    def forget(self, character_id: int) -> None:
        with self._lock:
            for key in [k for k in self._files if k[0] == character_id]:
                del self._files[key]

    @staticmethod
    def search(chunks: list[Chunk], query: str, k: int = 5, k1: float = 1.5, b: float = 0.75) -> list[tuple[Chunk, float]]:
        """BM25 over chunk text + headings + doc type/tags. Placeholder for embedding retrieval."""
        q = list(dict.fromkeys(tokenize(query)))
        if not q or not chunks:
            return []
        docs = [Counter(tokenize(" ".join([c.text, *c.headings, c.doc_type, *c.tags]))) for c in chunks]
        lens = [sum(d.values()) for d in docs]
        avg = sum(lens) / len(lens) or 1.0
        n = len(docs)
        df = {t: sum(1 for d in docs if t in d) for t in q}
        scored = []
        for c, d, ln in zip(chunks, docs, lens):
            s = 0.0
            for t in q:
                tf = d.get(t, 0)
                if tf:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * ln / avg))
            if s > 0:
                scored.append((c, round(s, 4)))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
