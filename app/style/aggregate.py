"""Many per-image records -> one statistical summary (deterministic, no model involved).

The schema is walked generically, so both kinds (and future schema versions) aggregate the
same way:

* enum field        -> weighted share of each value among images where it was known
* list of enums     -> weighted share of images containing each value
* free-text field   -> distinct values, most frequent first
* list of text      -> distinct items, most frequent first
* list of objects   -> enum sub-fields as a list of enums; text sub-fields joined into one item

Character records are weighted by their own ``assessment_confidence`` so a heavily filtered
image counts for less. The result is a plain dict (stored as JSON next to every guide).
"""
from __future__ import annotations

import re

from .kinds import Kind
from .schema import UNKNOWN

SIGNATURE, COMMON = 0.6, 0.25
CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.6, "low": 0.3}
_BLANK = {"", UNKNOWN, "none", "n/a", "na", "-", "null"}
_MAX_TEXT_ITEMS = 40


def tier(share: float) -> str:
    return "signature" if share >= SIGNATURE else "common" if share >= COMMON else "occasional"


def record_weight(kind: Kind, record: dict) -> float:
    if kind.key != "character":
        return 1.0
    level = (record.get("image_context") or {}).get("assessment_confidence", UNKNOWN)
    return CONFIDENCE_WEIGHTS.get(level, 0.5)


def _walk(node: dict, path: tuple[str, ...]):
    t = node.get("type")
    if t == "object":
        for key, child in node["properties"].items():
            yield from _walk(child, path + (key,))
    elif t == "array":
        item = node["items"]
        if item.get("type") == "object":
            yield "objects", path, item
        elif "enum" in item:
            yield "multi", path, item
        else:
            yield "list", path, item
    elif "enum" in node:
        yield "enum", path, node
    else:
        yield "text", path, node


def _get(record: dict, path: tuple[str, ...]):
    node = record
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _clean(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip() if text is not None else ""


class _Texts:
    """Case-insensitive distinct-value counter that keeps the first spelling seen."""

    def __init__(self):
        self.counts: dict[str, list] = {}

    def add(self, text) -> None:
        value = _clean(text)
        if value.lower() in _BLANK:
            return
        entry = self.counts.setdefault(value.lower(), [value, 0])
        entry[1] += 1

    def top(self) -> list[list]:
        items = sorted(self.counts.values(), key=lambda e: -e[1])  # stable: ties keep first-seen order
        return [[text, n] for text, n in items[:_MAX_TEXT_ITEMS]]


def _shares(weights: dict[str, float], denominator: float) -> list[list]:
    if denominator <= 0:
        return []
    out = [[v, round(w / denominator, 3)] for v, w in weights.items() if w > 0]
    return sorted(out, key=lambda e: -e[1])


def aggregate(kind: Kind, records: list[dict]) -> dict:
    weights = [record_weight(kind, r) for r in records]
    total = sum(weights)
    fields: list[dict] = []

    for shape, path, node in _walk(kind.schema, ()):
        name = ".".join(path)
        if shape == "enum":
            counts: dict[str, float] = {}
            known = 0.0
            for rec, w in zip(records, weights):
                value = _get(rec, path)
                if isinstance(value, str) and value != UNKNOWN:
                    counts[value] = counts.get(value, 0.0) + w
                    known += w
            fields.append({"path": name, "type": "enum", "coverage": round(known / total, 3) if total else 0,
                           "values": _shares(counts, known)})
        elif shape == "multi":
            counts = {}
            for rec, w in zip(records, weights):
                for value in set(_get(rec, path) or []):
                    counts[value] = counts.get(value, 0.0) + w
            fields.append({"path": name, "type": "multi", "values": _shares(counts, total)})
        elif shape in ("text", "list"):
            texts = _Texts()
            for rec in records:
                value = _get(rec, path)
                for item in (value if isinstance(value, list) else [value]):
                    texts.add(item)
            fields.append({"path": name, "type": "text", "items": texts.top()})
        else:  # list of objects, e.g. garments
            props = node["properties"]
            enum_keys = [k for k, v in props.items() if "enum" in v]
            text_keys = [k for k, v in props.items() if "enum" not in v]
            for key in enum_keys:
                counts = {}
                for rec, w in zip(records, weights):
                    present = {o.get(key) for o in (_get(rec, path) or []) if isinstance(o, dict)} - {None, UNKNOWN}
                    for value in present:
                        counts[value] = counts.get(value, 0.0) + w
                fields.append({"path": f"{name}.{key}", "type": "multi", "values": _shares(counts, total)})
            texts = _Texts()
            for rec in records:
                for obj in _get(rec, path) or []:
                    if isinstance(obj, dict):
                        texts.add(" ".join(_clean(obj.get(k)) for k in text_keys if _clean(obj.get(k)).lower() not in _BLANK))
            fields.append({"path": name, "type": "text", "items": texts.top()})

    return {"kind": kind.key, "schema_version": kind.schema_version, "image_count": len(records),
            "total_weight": round(total, 3), "weighted": kind.key == "character", "fields": fields}


# ---------- rendering ----------

def _label(path: str) -> str:
    return path.split(".", 1)[-1].replace(".", " ").replace("_", " ")


def _pretty(value: str) -> str:
    return str(value).replace("_", " ")


def _sections(stats: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in stats["fields"]:
        out.setdefault(f["path"].split(".", 1)[0], []).append(f)
    return out


def _enum_line(f: dict, limit: int = 6, floor: float = 0.05) -> str:
    vals = [v for v in f["values"] if v[1] >= floor][:limit]
    if not vals:
        return ""
    parts = []
    for value, share in vals:
        t = tier(share)
        parts.append(f"{_pretty(value)} {round(share * 100)}%" + (f" [{t}]" if t != "occasional" else ""))
    return ", ".join(parts)


def prompt_summary(stats: dict, text_items: int) -> str:
    """Compact plain-text summary for the model. ``text_items`` caps samples per free-text field."""
    lines: list[str] = []
    for section, fields in _sections(stats).items():
        lines.append(f"[{_pretty(section)}]")
        for f in fields:
            if f["type"] in ("enum", "multi"):
                line = _enum_line(f)
                if not line:
                    continue
                known = f" (known in {round(f['coverage'] * 100)}% of images)" if f["type"] == "enum" and f["coverage"] < 0.95 else ""
                lines.append(f"- {_label(f['path'])}{known}: {line}")
            elif text_items > 0 and f["items"]:
                items = f["items"][:text_items]
                rendered = " | ".join(f"{t} (x{n})" if n > 1 else t for t, n in items)
                lines.append(f"- {_label(f['path'])}: {rendered}")
        lines.append("")
    return "\n".join(lines).strip()


def stats_markdown(stats: dict) -> str:
    """The appendix: the numbers the guide was written from, generated by code."""
    note = (" Shares are weighted by each image's assessment confidence." if stats.get("weighted") else "")
    out = [
        "## Appendix: Data Summary",
        "",
        f"From {stats['image_count']} analyzed images. Tiers: **signature** ≥{round(SIGNATURE * 100)}%, "
        f"**common** ≥{round(COMMON * 100)}%.{note} Generated by code, not by the model.",
    ]
    for section, fields in _sections(stats).items():
        rows = []
        for f in fields:
            if f["type"] in ("enum", "multi"):
                cell = _enum_line(f, limit=8, floor=0.0)
            else:
                cell = " · ".join(f"{t} ({n})" if n > 1 else t for t, n in f["items"][:6])
            if cell:
                rows.append(f"| {_label(f['path'])} | {cell.replace('|', '/')} |")
        if rows:
            out += ["", f"### {_pretty(section).title()}", "", "| Field | Values |", "|---|---|", *rows]
    return "\n".join(out) + "\n"
