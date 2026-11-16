"""Character types (Influencer, Fashion Model, Cosplay Character, ...), loaded from JSON.

The list lives in ``config/character_types.json`` at the project root (``PROMPTHOUSE_CHARACTER_TYPES``
points elsewhere) and is read on every request, so edits apply on a page reload. Each character stores
one type id; per-image analysis does not use it, guide writing passes the type's ``guidance`` to the model.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "character_types.json"
_ID = re.compile(r"^[a-z0-9_]{1,64}$")


class CharacterTypesError(ValueError):
    pass


@dataclass(frozen=True)
class CharacterType:
    id: str
    label: str
    description: str = ""
    guidance: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def config_path() -> Path:
    override = os.environ.get("PROMPTHOUSE_CHARACTER_TYPES")
    return Path(override) if override else DEFAULT_PATH


def load_character_types(path: Path | None = None) -> list[CharacterType]:
    path = path or config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CharacterTypesError(f"Character types file not found: {path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise CharacterTypesError(f"Could not read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise CharacterTypesError(f"{path.name} is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}") from None
    entries = raw.get("character_types") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise CharacterTypesError(f'{path.name} must be an object with a "character_types" list.')
    out: list[CharacterType] = []
    seen: set[str] = set()
    for n, entry in enumerate(entries, 1):
        where = f"{path.name}, entry {n}"
        if not isinstance(entry, dict):
            raise CharacterTypesError(f"{where}: must be an object.")
        type_id, label = entry.get("id"), entry.get("label")
        if not isinstance(type_id, str) or not _ID.match(type_id):
            raise CharacterTypesError(f'{where}: "id" must be lowercase letters, digits or underscores (got {type_id!r}).')
        if type_id in seen:
            raise CharacterTypesError(f'{where}: duplicate id "{type_id}".')
        if not isinstance(label, str) or not label.strip():
            raise CharacterTypesError(f'{where}: "label" is required.')
        for key in ("description", "guidance"):
            if not isinstance(entry.get(key, ""), str):
                raise CharacterTypesError(f'{where}: "{key}" must be text.')
        seen.add(type_id)
        out.append(CharacterType(type_id, label.strip(), entry.get("description", "").strip(),
                                 entry.get("guidance", "").strip()))
    return out
