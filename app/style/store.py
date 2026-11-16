"""Style libraries in the PromptHouse database (tables created by schema v3 in app/characters.py).

Every character has two image libraries (``kind``): ``character`` (images of the character itself) and
``inspiration`` (content they should make). An image belongs to one library of one character; the
same file may be registered in several. Exact and near duplicates are detected within a library.
An analysis belongs to its image and is tagged with the schema version and model that produced it.
Deleting a character cascades to its libraries, analyses and guide history (never to files on disk).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

NEAR_DUP_BITS = 4  # dHash Hamming distance at or below which two images are "similar"


class StyleStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    # ------------------------------------------------------------------ images
    def add_image(self, cid: int, kind: str, path: str, sha256: str, width: int, height: int, dhash: str) -> tuple[int, bool, int | None]:
        """Returns ``(id, created, near_dup_of)``. A hash already in that library is an exact duplicate."""
        with self._lock, closing(self._connect()) as db:
            row = db.execute("SELECT id FROM style_images WHERE character_id = ? AND kind = ? AND sha256 = ?",
                             (cid, kind, sha256)).fetchone()
            if row:
                return row["id"], False, None
            near = None
            value = int(dhash, 16)
            for other in db.execute("SELECT id, dhash FROM style_images WHERE character_id = ? AND kind = ? ORDER BY id", (cid, kind)):
                if bin(value ^ int(other["dhash"], 16)).count("1") <= NEAR_DUP_BITS:
                    near = other["id"]
                    break
            cur = db.execute(
                "INSERT INTO style_images (character_id, kind, path, sha256, width, height, dhash, near_dup_of, added_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)", (cid, kind, path, sha256, width, height, dhash, near, time.time()))
            db.commit()
            return cur.lastrowid, True, near

    def remove_images(self, cid: int, image_ids: list[int]) -> tuple[int, list[str]]:
        """Remove images of character ``cid`` (ids of other characters are ignored) and their analyses.

        Each removed hash is remembered per library so a rescan skips it. Returns ``(removed, orphaned)``
        where ``orphaned`` are hashes no image uses any more (their thumbnails can be deleted)."""
        ids = sorted({int(i) for i in image_ids})
        if not ids:
            return 0, []
        marks = ",".join("?" * len(ids))
        with self._lock, closing(self._connect()) as db:
            rows = db.execute(f"SELECT id, kind, sha256, path FROM style_images WHERE character_id = ? AND id IN ({marks})",
                              [cid, *ids]).fetchall()
            if not rows:
                return 0, []
            now = time.time()
            db.executemany(
                "INSERT INTO style_removed (character_id, kind, sha256, path, removed_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(character_id, kind, sha256) DO UPDATE SET path=excluded.path, removed_at=excluded.removed_at",
                [(cid, r["kind"], r["sha256"], r["path"], now) for r in rows])
            found = [r["id"] for r in rows]
            db.execute(f"DELETE FROM style_images WHERE id IN ({','.join('?' * len(found))})", found)  # analyses cascade
            hashes = sorted({r["sha256"] for r in rows})
            still = {r[0] for r in db.execute(f"SELECT sha256 FROM style_images WHERE sha256 IN ({','.join('?' * len(hashes))})", hashes)}
            db.commit()
            return len(rows), [h for h in hashes if h not in still]

    def removed_hashes(self, cid: int, kind: str) -> set[str]:
        with closing(self._connect()) as db:
            return {r[0] for r in db.execute("SELECT sha256 FROM style_removed WHERE character_id = ? AND kind = ?", (cid, kind))}

    def forget_removed(self, cid: int, kind: str, sha256: str) -> None:
        with self._lock, closing(self._connect()) as db:
            db.execute("DELETE FROM style_removed WHERE character_id = ? AND kind = ? AND sha256 = ?", (cid, kind, sha256))
            db.commit()

    def get_image(self, image_id: int) -> dict | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM style_images WHERE id = ?", (image_id,)).fetchone()
            return dict(row) if row else None

    def list_images(self, cid: int, kind: str, status: str | None, schema_version: int, model_label: str,
                    limit: int, offset: int) -> tuple[list[dict], int]:
        """A library's images with their analysis status for (schema, model). ``status``: None, ok, failed, pending."""
        where, params = "WHERE i.character_id = ? AND i.kind = ?", [schema_version, model_label, cid, kind]
        if status == "pending":
            where += " AND a.id IS NULL"
        elif status in ("ok", "failed"):
            where += " AND a.status = ?"
            params.append(status)
        base = ("FROM style_images i LEFT JOIN style_analyses a "
                "ON a.image_id = i.id AND a.schema_version = ? AND a.model_label = ? " + where)
        with closing(self._connect()) as db:
            total = db.execute(f"SELECT COUNT(*) {base}", params).fetchone()[0]
            rows = db.execute(f"SELECT i.id, i.kind, i.path, i.width, i.height, i.near_dup_of, a.status AS analysis_status {base} "
                              "ORDER BY i.id LIMIT ? OFFSET ?", params + [limit, offset]).fetchall()
            return [dict(r) for r in rows], total

    def library_ids(self, cid: int, kind: str) -> set[int]:
        with closing(self._connect()) as db:
            return {r[0] for r in db.execute("SELECT id FROM style_images WHERE character_id = ? AND kind = ?", (cid, kind))}

    def pending_ids(self, cid: int, kind: str, schema_version: int, model_label: str, reanalyze: bool) -> list[int]:
        with closing(self._connect()) as db:
            if reanalyze:
                rows = db.execute("SELECT id FROM style_images WHERE character_id = ? AND kind = ? ORDER BY id", (cid, kind))
            else:
                rows = db.execute(
                    "SELECT i.id FROM style_images i LEFT JOIN style_analyses a "
                    "ON a.image_id = i.id AND a.schema_version = ? AND a.model_label = ? AND a.status = 'ok' "
                    "WHERE i.character_id = ? AND i.kind = ? AND a.id IS NULL ORDER BY i.id",
                    (schema_version, model_label, cid, kind))
            return [r[0] for r in rows]

    def counts(self, cid: int, kind: str, schema_version: int, model_label: str) -> dict:
        with closing(self._connect()) as db:
            total = db.execute("SELECT COUNT(*) FROM style_images WHERE character_id = ? AND kind = ?", (cid, kind)).fetchone()[0]
            by = {r["status"]: r["n"] for r in db.execute(
                "SELECT a.status, COUNT(*) AS n FROM style_analyses a JOIN style_images i ON i.id = a.image_id "
                "WHERE i.character_id = ? AND i.kind = ? AND a.schema_version = ? AND a.model_label = ? GROUP BY a.status",
                (cid, kind, schema_version, model_label))}
        ok, failed = by.get("ok", 0), by.get("failed", 0)
        return {"images": total, "ok": ok, "failed": failed, "pending": total - ok - failed}

    # ------------------------------------------------------------------ analyses
    def save_analysis(self, image_id: int, schema_version: int, model_label: str, status: str, record: dict | None,
                      raw_text: str | None, error: str | None, seconds: float | None, reused_from: int | None = None) -> None:
        with self._lock, closing(self._connect()) as db:
            db.execute(
                "INSERT INTO style_analyses (image_id, schema_version, model_label, status, record_json, raw_text, error, "
                "seconds, reused_from, created_at) VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(image_id, schema_version, model_label) DO UPDATE SET status=excluded.status, "
                "record_json=excluded.record_json, raw_text=excluded.raw_text, error=excluded.error, seconds=excluded.seconds, "
                "reused_from=excluded.reused_from, created_at=excluded.created_at",
                (image_id, schema_version, model_label, status, json.dumps(record) if record is not None else None,
                 raw_text, error, seconds, reused_from, time.time()))
            db.commit()

    def reuse_analyses(self, image_ids: list[int], schema_version: int, model_label: str) -> list[int]:
        """Copy an existing successful analysis of the same file (same hash and library kind, any character)
        onto each image that lacks one. Analysis does not depend on the character, so re-running the model
        would only repeat the work. Returns the ids that were filled this way."""
        filled = []
        with self._lock, closing(self._connect()) as db:
            for image_id in image_ids:
                src = db.execute(
                    "SELECT a.* FROM style_images me JOIN style_images other ON other.sha256 = me.sha256 AND other.kind = me.kind "
                    "AND other.id != me.id JOIN style_analyses a ON a.image_id = other.id AND a.schema_version = ? "
                    "AND a.model_label = ? AND a.status = 'ok' WHERE me.id = ? ORDER BY a.created_at DESC LIMIT 1",
                    (schema_version, model_label, image_id)).fetchone()
                if src is None:
                    continue
                db.execute(
                    "INSERT INTO style_analyses (image_id, schema_version, model_label, status, record_json, raw_text, error, "
                    "seconds, reused_from, created_at) VALUES (?,?,?,'ok',?,?,NULL,?,?,?) "
                    "ON CONFLICT(image_id, schema_version, model_label) DO UPDATE SET status='ok', record_json=excluded.record_json, "
                    "raw_text=excluded.raw_text, error=NULL, seconds=excluded.seconds, reused_from=excluded.reused_from, "
                    "created_at=excluded.created_at",
                    (image_id, schema_version, model_label, src["record_json"], src["raw_text"], src["seconds"],
                     src["image_id"], time.time()))
                filled.append(image_id)
            db.commit()
        return filled

    def ok_records(self, cid: int, kind: str, schema_version: int, model_label: str) -> list[dict]:
        """The successful records of a library for one (schema, model), oldest image first."""
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT a.record_json FROM style_analyses a JOIN style_images i ON i.id = a.image_id "
                "WHERE i.character_id = ? AND i.kind = ? AND a.schema_version = ? AND a.model_label = ? AND a.status = 'ok' "
                "ORDER BY i.id", (cid, kind, schema_version, model_label)).fetchall()
            return [json.loads(r[0]) for r in rows]

    def get_analyses(self, image_id: int) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT schema_version, model_label, status, record_json, raw_text, error, seconds, reused_from, created_at "
                "FROM style_analyses WHERE image_id = ? ORDER BY created_at DESC", (image_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["record"] = json.loads(d.pop("record_json")) if d["record_json"] else None
            out.append(d)
        return out

    # ------------------------------------------------------------------ guides
    _GUIDE_COLUMNS = ("id, character_id, kind, created_at, model_label, schema_version, character_type, image_count, "
                      "truncated, seconds, doc_path, archived_path")

    def save_guide(self, cid: int, kind: str, model_label: str, schema_version: int, character_type: str | None,
                   image_count: int, truncated: bool, seconds: float, doc_path: str | None, archived_path: str | None,
                   markdown: str, stats: dict) -> int:
        with self._lock, closing(self._connect()) as db:
            cur = db.execute(
                "INSERT INTO style_guides (character_id, kind, created_at, model_label, schema_version, character_type, "
                "image_count, truncated, seconds, doc_path, archived_path, markdown, stats_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (cid, kind, time.time(), model_label, schema_version, character_type, image_count, int(truncated), seconds,
                 doc_path, archived_path, markdown, json.dumps(stats)))
            db.commit()
            return cur.lastrowid

    def list_guides(self, cid: int, kind: str) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute(f"SELECT {self._GUIDE_COLUMNS} FROM style_guides WHERE character_id = ? AND kind = ? "
                              "ORDER BY created_at DESC, id DESC", (cid, kind)).fetchall()
        return [{**dict(r), "truncated": bool(r["truncated"])} for r in rows]

    def get_guide(self, guide_id: int) -> dict | None:
        with closing(self._connect()) as db:
            row = db.execute(f"SELECT {self._GUIDE_COLUMNS}, markdown, stats_json FROM style_guides WHERE id = ?",
                             (guide_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["stats"] = json.loads(d.pop("stats_json"))
        d["truncated"] = bool(d["truncated"])
        return d
