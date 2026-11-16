"""Folder scanning: hash, size and perceptual-hash every readable image into a character's library."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from .store import StyleStore

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ScanResult:
    found: int = 0
    added: int = 0
    duplicates: int = 0
    near_duplicates: int = 0
    skipped_removed: int = 0   # previously removed from this library, so not re-added
    restored: int = 0          # previously removed, re-added because include_removed was set
    unreadable: list[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dhash(img: Image.Image) -> str:
    """64-bit difference hash as 16 hex chars."""
    small = ImageOps.exif_transpose(img).convert("L").resize((9, 8), Image.LANCZOS)
    px = list(small.get_flattened_data() if hasattr(small, "get_flattened_data") else small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return f"{bits:016x}"


def iter_image_files(folder: Path, recursive: bool):
    if recursive:
        for root, dirs, files in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(files):
                if Path(name).suffix.lower() in IMAGE_EXTENSIONS and not name.startswith("."):
                    yield Path(root) / name
    else:
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."):
                yield p


def scan_folder(store: StyleStore, cid: int, kind: str, folder: str, recursive: bool = True,
                include_removed: bool = False) -> ScanResult:
    """Register every readable image under ``folder`` in character ``cid``'s ``kind`` library.

    Images the user removed from this library are skipped unless ``include_removed`` is set. Files are
    only read, never modified."""
    root = Path(folder.strip().strip('"')).expanduser()
    if not folder.strip() or not root.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    result = ScanResult()
    removed = store.removed_hashes(cid, kind)
    for path in iter_image_files(root, recursive):
        result.found += 1
        try:
            with Image.open(path) as img:
                img.load()
                w, h = ImageOps.exif_transpose(img).size
                dh = dhash(img)
            digest = sha256_file(path)
        except Exception:  # corrupt or unsupported file: report it and carry on
            result.unreadable.append(str(path))
            continue
        if digest in removed:
            if not include_removed:
                result.skipped_removed += 1
                continue
            store.forget_removed(cid, kind, digest)
            removed.discard(digest)
            result.restored += 1
        _, created, near = store.add_image(cid, kind, str(path.resolve()), digest, w, h, dh)
        if created:
            result.added += 1
            if near is not None:
                result.near_duplicates += 1
        else:
            result.duplicates += 1
    return result
