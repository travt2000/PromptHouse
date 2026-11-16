"""Shared fixtures for the style-analysis tests (fake llama-server, sample image folders)."""
import stat
import sys
from pathlib import Path

import pytest
from PIL import Image

FAKE = Path(__file__).resolve().parent / "fake_llama_server.py"


@pytest.fixture
def fake_llama(tmp_path) -> str:
    """An executable wrapper that runs the fake server with the current Python."""
    if sys.platform == "win32":
        path = tmp_path / "llama-server.bat"
        path.write_text(f'@"{sys.executable}" "{FAKE}" %*\r\n')
    else:
        path = tmp_path / "llama-server"
        path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.fixture
def model_files(tmp_path):
    model, mmproj = tmp_path / "gemma-4-test-Q4.gguf", tmp_path / "mmproj-test.gguf"
    model.write_bytes(b"x")
    mmproj.write_bytes(b"x")
    return str(model), str(mmproj)


def make_image(path: Path, color, size=(64, 48), pattern=False):
    img = Image.new("RGB", size, color)
    if pattern:  # a gradient so perceptual hashes differ between images
        for x in range(size[0]):
            for y in range(size[1]):
                img.putpixel((x, y), (x * 4 % 256, y * 5 % 256, (x * y) % 256))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


@pytest.fixture
def folder(tmp_path):
    """3 distinct images (a, b, sub/c), an exact duplicate of c, a broken file and a non-image."""
    root = tmp_path / "refs"
    make_image(root / "a.png", (200, 30, 30), pattern=True)
    make_image(root / "b.jpg", (30, 200, 30))
    make_image(root / "sub" / "c.png", (30, 30, 200))
    (root / "sub" / "d.png").write_bytes((root / "sub" / "c.png").read_bytes())  # exact duplicate of c
    (root / "broken.png").write_bytes(b"not an image")
    (root / "notes.txt").write_text("ignore me")
    return root
