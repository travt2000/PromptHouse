"""Character records in a local SQLite database (stdlib sqlite3, one file, no server).

A character is the unit the rest of the app will grow around. For now it holds:
  name (unique, case-insensitive), description, an optional LoRA (path + default strength),
  a trigger phrase, whether that trigger is prepended to prompts by default, the name of its
  Markdown document folder (see app/character_docs.py) and its character type (config/character_types.json).
Schema v3 also holds the style libraries (images, vision-model analyses, guide history; see app/style/store.py).

Schema changes are applied in order by `_MIGRATIONS`, tracked with SQLite's `PRAGMA user_version`.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

NAME_MAX = 100
TRIGGER_MAX = 200
DESCRIPTION_MAX = 10_000
PATH_MAX = 1024
STRENGTH_LIMIT = 5.0

FIELDS = ("name", "description", "lora_path", "lora_strength", "trigger", "prepend_trigger", "character_type")
TYPE_ID_RE = re.compile(r"^[a-z0-9_]{0,64}$")

_MIGRATIONS = [
    # v1
    """
    CREATE TABLE characters (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        name            TEXT    NOT NULL UNIQUE COLLATE NOCASE,
        description     TEXT    NOT NULL DEFAULT '',
        lora_path       TEXT    NOT NULL DEFAULT '',
        lora_strength   REAL    NOT NULL DEFAULT 1.0,
        trigger         TEXT    NOT NULL DEFAULT '',
        prepend_trigger INTEGER NOT NULL DEFAULT 1,
        created_at      TEXT    NOT NULL,
        updated_at      TEXT    NOT NULL
    );
    """,
    # v2: per-character document folder (name under the documents root; '' = not provisioned yet)
    """
    ALTER TABLE characters ADD COLUMN folder TEXT NOT NULL DEFAULT '';
    """,
    # v3: character type + style libraries (per character: images of the character and inspiration images,
    # their vision-model analyses, and the history of guides written from them)
    """
    ALTER TABLE characters ADD COLUMN character_type TEXT NOT NULL DEFAULT '';
    CREATE TABLE style_images (
        id           INTEGER PRIMARY KEY,
        character_id INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
        kind         TEXT    NOT NULL CHECK (kind IN ('character', 'inspiration')),
        path         TEXT    NOT NULL,
        sha256       TEXT    NOT NULL,
        width        INTEGER NOT NULL,
        height       INTEGER NOT NULL,
        dhash        TEXT    NOT NULL,
        near_dup_of  INTEGER REFERENCES style_images(id) ON DELETE SET NULL,
        added_at     REAL    NOT NULL,
        UNIQUE (character_id, kind, sha256)
    );
    CREATE INDEX style_images_sha ON style_images(sha256);
    CREATE TABLE style_analyses (
        id             INTEGER PRIMARY KEY,
        image_id       INTEGER NOT NULL REFERENCES style_images(id) ON DELETE CASCADE,
        schema_version INTEGER NOT NULL,
        model_label    TEXT    NOT NULL,
        status         TEXT    NOT NULL CHECK (status IN ('ok', 'failed')),
        record_json    TEXT,
        raw_text       TEXT,
        error          TEXT,
        seconds        REAL,
        reused_from    INTEGER,
        created_at     REAL    NOT NULL,
        UNIQUE (image_id, schema_version, model_label)
    );
    CREATE TABLE style_removed (
        character_id INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
        kind         TEXT    NOT NULL,
        sha256       TEXT    NOT NULL,
        path         TEXT    NOT NULL,
        removed_at   REAL    NOT NULL,
        PRIMARY KEY (character_id, kind, sha256)
    );
    CREATE TABLE style_guides (
        id             INTEGER PRIMARY KEY,
        character_id   INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
        kind           TEXT    NOT NULL,
        created_at     REAL    NOT NULL,
        model_label    TEXT    NOT NULL,
        schema_version INTEGER NOT NULL,
        character_type TEXT,
        image_count    INTEGER NOT NULL,
        truncated      INTEGER NOT NULL DEFAULT 0,
        seconds        REAL,
        doc_path       TEXT,
        archived_path  TEXT,
        markdown       TEXT    NOT NULL,
        stats_json     TEXT    NOT NULL
    );
    CREATE INDEX style_guides_lib ON style_guides(character_id, kind, created_at);
    """,
]


class CharacterError(ValueError):
    pass


class NotFound(CharacterError):
    pass


class Duplicate(CharacterError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["prepend_trigger"] = bool(d["prepend_trigger"])
    return d


class CharacterStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # serialise writes from FastAPI's thread pool
        with self._connect() as db:
            self._migrate(db)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def _migrate(self, db: sqlite3.Connection) -> None:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version > len(_MIGRATIONS):
            raise CharacterError(f"{self.path.name} was created by a newer version of the app (schema v{version})")
        for i in range(version, len(_MIGRATIONS)):
            db.executescript(_MIGRATIONS[i])
            db.execute(f"PRAGMA user_version = {i + 1}")
        db.commit()

    # ------------------------------------------------------------------ queries
    def list(self) -> list[dict]:
        with closing(self._connect()) as db:
            return [_row(r) for r in db.execute("SELECT * FROM characters ORDER BY name COLLATE NOCASE")]

    def get(self, cid: int) -> dict:
        with closing(self._connect()) as db:
            r = db.execute("SELECT * FROM characters WHERE id = ?", (cid,)).fetchone()
        if r is None:
            raise NotFound(f"Character {cid} not found")
        return _row(r)

    def create(self, data: dict) -> dict:
        v = validate(data, partial=False)
        now = _now()
        with self._lock, closing(self._connect()) as db:
            try:
                cur = db.execute(
                    "INSERT INTO characters (name, description, lora_path, lora_strength, trigger, prepend_trigger, character_type, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (v["name"], v.get("description", ""), v.get("lora_path", ""), v.get("lora_strength", 1.0),
                     v.get("trigger", ""), int(v.get("prepend_trigger", True)), v.get("character_type", ""), now, now),
                )
                db.commit()
            except sqlite3.IntegrityError as e:
                raise Duplicate(f"A character named '{v['name']}' already exists") from e
            cid = cur.lastrowid
        return self.get(cid)

    def update(self, cid: int, data: dict) -> dict:
        v = validate(data, partial=True)
        if not v:
            return self.get(cid)
        cols = ", ".join(f"{k} = ?" for k in v) + ", updated_at = ?"
        vals = [int(x) if isinstance(x, bool) else x for x in v.values()] + [_now(), cid]
        with self._lock, closing(self._connect()) as db:
            try:
                cur = db.execute(f"UPDATE characters SET {cols} WHERE id = ?", vals)  # column names come from FIELDS only
                db.commit()
            except sqlite3.IntegrityError as e:
                raise Duplicate(f"A character named '{v['name']}' already exists") from e
            if cur.rowcount == 0:
                raise NotFound(f"Character {cid} not found")
        return self.get(cid)

    def set_folder(self, cid: int, folder: str) -> dict:
        """Not a user-editable field: assigned once when the character's document folder is created."""
        with self._lock, closing(self._connect()) as db:
            cur = db.execute("UPDATE characters SET folder = ? WHERE id = ?", (folder, cid))
            db.commit()
        if cur.rowcount == 0:
            raise NotFound(f"Character {cid} not found")
        return self.get(cid)

    def delete(self, cid: int) -> None:
        """Deletes the record and (by cascade) its style libraries and guide history; never any file."""
        with self._lock, closing(self._connect()) as db:
            cur = db.execute("DELETE FROM characters WHERE id = ?", (cid,))
            db.commit()
        if cur.rowcount == 0:
            raise NotFound(f"Character {cid} not found")


def validate(data: dict, partial: bool) -> dict:
    """Check and normalise a create (partial=False) or update (partial=True) payload."""
    if not isinstance(data, dict):
        raise CharacterError("Expected an object")
    unknown = set(data) - set(FIELDS)
    if unknown:
        raise CharacterError(f"Unknown fields: {sorted(unknown)}")
    out: dict = {}
    if "name" in data or not partial:
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise CharacterError("Name is required")
        name = " ".join(name.split())
        if len(name) > NAME_MAX:
            raise CharacterError(f"Name must be at most {NAME_MAX} characters")
        out["name"] = name
    for key, limit in (("description", DESCRIPTION_MAX), ("trigger", TRIGGER_MAX), ("lora_path", PATH_MAX)):
        if key in data:
            val = data[key]
            if val is None:
                val = ""
            if not isinstance(val, str):
                raise CharacterError(f"{key} must be text")
            val = val.strip()
            if key == "lora_path":
                val = val.strip('"')
            elif key == "trigger":
                val = " ".join(val.split())
            if len(val) > limit:
                raise CharacterError(f"{key} must be at most {limit} characters")
            out[key] = val
    if "lora_strength" in data:
        s = data["lora_strength"]
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s):
            raise CharacterError("LoRA strength must be a number")
        if abs(s) > STRENGTH_LIMIT:
            raise CharacterError(f"LoRA strength must be between -{STRENGTH_LIMIT:g} and {STRENGTH_LIMIT:g}")
        out["lora_strength"] = round(float(s), 4)
    if "prepend_trigger" in data:
        if not isinstance(data["prepend_trigger"], bool):
            raise CharacterError("prepend_trigger must be true/false")
        out["prepend_trigger"] = data["prepend_trigger"]
    if "character_type" in data:  # an id from config/character_types.json ("" = not set); existence is checked by the API
        t = data["character_type"]
        if t is None:
            t = ""
        if not isinstance(t, str) or not TYPE_ID_RE.match(t.strip()):
            raise CharacterError("character_type must be a type id (lowercase letters, digits, underscores)")
        out["character_type"] = t.strip()
    return out


def compose_prompt(prompt: str, trigger: str, prepend: bool) -> str:
    """The prompt actually encoded: `"<trigger>, <prompt>"`, unless prepending is off, there is no trigger,
    or the prompt already starts with the trigger (as a whole word, case-insensitive).

    Mirrored by `composePrompt` in static/app.js (the UI compares it with the cached conditioning)."""
    trigger = (trigger or "").strip()
    if not prepend or not trigger:
        return prompt
    body = prompt.lstrip()
    if re.match(re.escape(trigger) + r"(?![0-9A-Za-z_])", body, flags=re.IGNORECASE):
        return prompt
    return f"{trigger}, {body}"
