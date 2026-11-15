"""Saving generated images with JSON metadata (PNG text chunk + sidecar .json)."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$")


def save_image(image: Image.Image, metadata: dict, out_dir: Path = OUTPUT_DIR) -> tuple[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    png_path = out_dir / f"{image_id}.png"
    info = PngInfo()
    info.add_text("krea2_metadata", json.dumps(metadata, ensure_ascii=False))
    image.save(png_path, format="PNG", pnginfo=info)
    (out_dir / f"{image_id}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return image_id, png_path


def image_path(image_id: str, out_dir: Path = OUTPUT_DIR) -> Path | None:
    if not _ID_RE.match(image_id):
        return None
    p = out_dir / f"{image_id}.png"
    return p if p.is_file() else None
