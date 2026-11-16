"""Style analysis: schemas, record parsing, vision settings, scanning and the per-character style store."""
import json
import sqlite3

import jsonschema
import pytest

from app import config
from app.characters import _MIGRATIONS, CharacterStore
from app.style import schema
from app.style import vsettings as vs
from app.style.analyzer import AnalysisError, parse_record
from app.style.kinds import CHARACTER, INSPIRATION, KINDS, get_kind
from app.style.llama_server import build_command
from app.style.publish import mark_unindexed, publish_guide
from app.style.scanner import scan_folder
from app.style.store import StyleStore

APPEARANCE_FIELDS = {"hair_color", "hair_length", "hair_texture", "eye_color", "skin_tone", "skin_undertone",
                     "face_shape", "build", "distinctive_features"}
CSV, ISV = CHARACTER.schema_version, INSPIRATION.schema_version


def _keys(node) -> set[str]:
    out = set()
    if node.get("type") == "object":
        for k, v in node["properties"].items():
            out |= {k} | _keys(v)
    elif node.get("type") == "array":
        out |= _keys(node["items"])
    return out


@pytest.fixture
def stores(tmp_path):
    chars = CharacterStore(tmp_path / "t.db")
    a, b = chars.create({"name": "Ann"})["id"], chars.create({"name": "Bea"})["id"]
    return chars, StyleStore(tmp_path / "t.db"), a, b


# ------------------------------------------------------------------ schemas and parsing
@pytest.mark.parametrize("kind", KINDS.values(), ids=list(KINDS))
def test_schema_is_valid_and_sample_validates(kind):
    jsonschema.Draft202012Validator.check_schema(kind.schema)
    jsonschema.validate(schema.sample_record(kind.schema), kind.schema)


def test_vocabulary_is_clean():
    for name, values in schema.VOCAB.items():
        assert len(values) == len(set(values)), name
        assert schema.UNKNOWN not in values, name  # added automatically


def test_inspiration_schema_has_no_physical_traits():
    assert not (_keys(INSPIRATION.schema) & APPEARANCE_FIELDS)
    assert APPEARANCE_FIELDS <= _keys(CHARACTER.schema)


def test_kinds_have_distinct_prompts():
    assert CHARACTER.system_prompt != INSPIRATION.system_prompt
    assert "Never describe the subject's physical traits" in INSPIRATION.system_prompt
    assert get_kind("character") is CHARACTER
    with pytest.raises(ValueError):
        get_kind("nope")


def test_parse_record_handles_fences_and_think_blocks():
    rec = schema.sample_record(INSPIRATION.schema)
    text = "<think>reasoning</think>\n```json\n" + json.dumps(rec) + "\n```"
    assert parse_record(text, INSPIRATION.schema) == rec


@pytest.mark.parametrize("bad", ["no json here", "{not json}", "{}"])
def test_parse_record_rejects_bad_output(bad):
    with pytest.raises(AnalysisError) as exc:
        parse_record(bad, INSPIRATION.schema)
    assert exc.value.raw_text == bad


def test_parse_record_rejects_value_outside_vocabulary_and_other_kind():
    rec = schema.sample_record(INSPIRATION.schema)
    rec["photography"]["shot_type"] = "selfie"
    with pytest.raises(AnalysisError, match="photography/shot_type"):
        parse_record(json.dumps(rec), INSPIRATION.schema)
    with pytest.raises(AnalysisError):
        parse_record(json.dumps(schema.sample_record(CHARACTER.schema)), INSPIRATION.schema)


# ------------------------------------------------------------------ vision settings
def test_vision_settings_merge_and_validation():
    d = vs.merge(None, {"n_ctx": "4096", "extra_args": "--n-cpu-moe 20", "model_path": ' "D:\\m.gguf" '})
    assert d["n_ctx"] == 4096 and d["extra_args"] == "--n-cpu-moe 20" and d["model_path"] == "D:\\m.gguf"
    assert vs.merge(d, {"temperature": 0.4})["n_ctx"] == 4096  # partial update keeps the rest
    for bad in ({"nope": 1}, {"structured_mode": "x"}, {"image_max_edge": 5}, {"chat_template_kwargs": "[1]"},
                {"n_ctx": "abc"}, {"release_krea_ram": "yes"}, {"guide_max_tokens": 8000}, {"external_url": "localhost:1"}):
        with pytest.raises(vs.VisionSettingsError):
            vs.merge(d, bad)
    # a damaged file never stops the app: bad values fall back
    assert vs.merge(None, {"n_ctx": "abc", "nope": 1, "temperature": 0.3}, strict=False)["temperature"] == 0.3
    assert vs.merge(None, "garbage", strict=False) == vs.merge(None, {})


def test_vision_settings_in_app_settings(tmp_path):
    s = config.update_settings(config.Settings(), {"vision": {"model_path": "x.gguf", "n_ctx": 4096}})
    assert s.vision_settings().model_label() == "x.gguf" and s.vision["n_ctx"] == 4096
    with pytest.raises(config.SettingsError):
        config.update_settings(s, {"vision": {"n_ctx": 10}})
    path = tmp_path / "settings.json"
    config.save_settings(s, path)
    assert config.load_settings(path).vision["n_ctx"] == 4096
    raw = json.loads(path.read_text())
    raw["vision"]["structured_mode"] = "broken"
    path.write_text(json.dumps(raw))
    assert config.load_settings(path).vision["structured_mode"] == "json_schema"  # lenient load


def test_check_ready_reports_missing_paths(tmp_path):
    assert len(vs.check_ready(vs.VisionSettings())) == 3
    assert vs.check_ready(vs.VisionSettings(external_url="http://127.0.0.1:1")) == []
    assert "not found" in vs.check_ready(vs.VisionSettings(llama_server_path=str(tmp_path / "x.exe")))[0]


def test_build_command_includes_vision_and_extra_args():
    s = vs.VisionSettings(llama_server_path="/x/llama-server", model_path="/m.gguf", mmproj_path="/p.gguf",
                          n_ctx=4096, n_gpu_layers=20, extra_args="--n-cpu-moe 12 --no-mmproj-offload")
    cmd = build_command(s, 9999)
    assert cmd[:5] == ["/x/llama-server", "-m", "/m.gguf", "--mmproj", "/p.gguf"]
    assert cmd[cmd.index("--port") + 1] == "9999" and cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[-3:] == ["--n-cpu-moe", "12", "--no-mmproj-offload"]


# ------------------------------------------------------------------ scanning and the store
def test_scan_dedupes_and_skips_junk(stores, folder):
    _, st, a, _ = stores
    result = scan_folder(st, a, "inspiration", str(folder), recursive=True)
    assert result.found == 5 and result.added == 3 and result.duplicates == 1
    assert [p.endswith("broken.png") for p in result.unreadable] == [True]
    again = scan_folder(st, a, "inspiration", str(folder), recursive=True)
    assert again.added == 0 and again.duplicates == 4
    assert scan_folder(st, a, "character", str(folder), recursive=False).added == 2
    with pytest.raises(FileNotFoundError):
        scan_folder(st, a, "inspiration", str(folder / "nope"))
    with pytest.raises(FileNotFoundError):
        scan_folder(st, a, "inspiration", "   ")


def test_libraries_are_separate_per_kind_and_character(stores, folder):
    _, st, a, b = stores
    scan_folder(st, a, "inspiration", str(folder))
    assert scan_folder(st, a, "character", str(folder)).added == 3  # same files, other library
    assert scan_folder(st, b, "inspiration", str(folder)).added == 3  # same files, other character
    assert st.counts(a, "character", CSV, "m")["images"] == 3 and st.counts(b, "character", CSV, "m")["images"] == 0
    ids_a = set(st.pending_ids(a, "inspiration", ISV, "m", False))
    ids_b = set(st.pending_ids(b, "inspiration", ISV, "m", False))
    assert len(ids_a) == 3 and not ids_a & ids_b
    items, total = st.list_images(a, "character", None, CSV, "m", 10, 0)
    assert total == 3 and {i["kind"] for i in items} == {"character"}
    assert st.library_ids(b, "inspiration") == ids_b


def test_pending_and_upsert(stores, folder):
    _, st, a, _ = stores
    scan_folder(st, a, "inspiration", str(folder))
    rec = schema.sample_record(INSPIRATION.schema)
    ids = st.pending_ids(a, "inspiration", ISV, "m", reanalyze=False)
    st.save_analysis(ids[0], ISV, "m", "ok", rec, "raw", None, 1.0)
    st.save_analysis(ids[1], ISV, "m", "failed", None, "raw", "bad", None)
    assert st.pending_ids(a, "inspiration", ISV, "m", False) == [ids[1], ids[2]]  # failed ones are retried
    assert st.pending_ids(a, "inspiration", ISV, "other-model", False) == ids     # another model starts over
    assert st.pending_ids(a, "inspiration", ISV + 1, "m", False) == ids           # so does a new schema version
    assert st.counts(a, "inspiration", ISV, "m") == {"images": 3, "ok": 1, "failed": 1, "pending": 1}
    st.save_analysis(ids[1], ISV, "m", "ok", rec, "raw", None, 2.0)  # upsert
    assert st.counts(a, "inspiration", ISV, "m")["failed"] == 0
    items, total = st.list_images(a, "inspiration", "pending", ISV, "m", 10, 0)
    assert total == 1 and items[0]["id"] == ids[2]
    assert st.ok_records(a, "inspiration", ISV, "m") == [rec, rec]


def test_reuse_analysis_of_the_same_file(stores, folder):
    _, st, a, b = stores
    scan_folder(st, a, "inspiration", str(folder))
    scan_folder(st, b, "inspiration", str(folder))
    scan_folder(st, b, "character", str(folder))
    rec = schema.sample_record(INSPIRATION.schema)
    a_ids = st.pending_ids(a, "inspiration", ISV, "m", False)
    st.save_analysis(a_ids[0], ISV, "m", "ok", rec, "raw", None, 3.0)
    st.save_analysis(a_ids[1], ISV, "m", "failed", None, "x", "bad", None)  # failures are never copied
    b_ids = st.pending_ids(b, "inspiration", ISV, "m", False)
    assert st.reuse_analyses(b_ids, ISV, "m") == [b_ids[0]]
    copied = st.get_analyses(b_ids[0])[0]
    assert copied["record"] == rec and copied["reused_from"] == a_ids[0]
    assert st.reuse_analyses(b_ids, ISV, "other-model") == []  # only the same model and schema
    c_ids = st.pending_ids(b, "character", CSV, "m", False)
    assert st.reuse_analyses(c_ids, CSV, "m") == []  # never across library kinds (different prompt and schema)


def test_remove_images_and_rescan(stores, folder):
    _, st, a, b = stores
    scan_folder(st, a, "inspiration", str(folder))
    scan_folder(st, b, "inspiration", str(folder))
    ids = st.pending_ids(a, "inspiration", ISV, "m", False)
    st.save_analysis(ids[0], ISV, "m", "ok", schema.sample_record(INSPIRATION.schema), "raw", None, 1.0)
    target = st.get_image(ids[0])
    b_ids = st.pending_ids(b, "inspiration", ISV, "m", False)

    assert st.remove_images(b, ids) == (0, [])  # another character's ids are ignored
    removed, orphaned = st.remove_images(a, [ids[0], ids[0], 9999])
    assert removed == 1 and orphaned == []  # character b still holds the file, so its thumbnail stays
    assert st.get_image(ids[0]) is None and st.get_analyses(ids[0]) == []
    assert st.counts(a, "inspiration", ISV, "m")["images"] == 2 and st.counts(b, "inspiration", ISV, "m")["images"] == 3
    assert (folder / "a.png").is_file()  # files are never deleted

    again = scan_folder(st, a, "inspiration", str(folder))
    assert again.added == 0 and again.skipped_removed == 1
    assert scan_folder(st, b, "inspiration", str(folder)).skipped_removed == 0  # per character
    back = scan_folder(st, a, "inspiration", str(folder), include_removed=True)
    assert back.restored == 1 and back.added == 1 and st.removed_hashes(a, "inspiration") == set()

    same = [i for i in st.library_ids(a, "inspiration") | set(b_ids) if st.get_image(i)["sha256"] == target["sha256"]]
    st.remove_images(a, same)
    assert st.remove_images(b, same) == (1, [target["sha256"]])  # no image uses the file any more


def test_near_duplicate_flag_cleared_when_original_removed(stores, tmp_path):
    from PIL import Image

    from tests.conftest import make_image
    _, st, a, _ = stores
    make_image(tmp_path / "n" / "x.png", (10, 10, 10), pattern=True)
    with Image.open(tmp_path / "n" / "x.png") as img:  # the same picture re-encoded: different bytes, same look
        img.convert("RGB").save(tmp_path / "n" / "y.jpg", quality=92)
    r = scan_folder(st, a, "character", str(tmp_path / "n"))
    assert r.added == 2 and r.near_duplicates == 1
    first, second = sorted(st.library_ids(a, "character"))
    assert st.get_image(second)["near_dup_of"] == first
    st.remove_images(a, [first])
    assert st.get_image(second)["near_dup_of"] is None


def test_deleting_a_character_cascades_to_its_libraries(stores, folder):
    chars, st, a, b = stores
    scan_folder(st, a, "inspiration", str(folder))
    scan_folder(st, b, "inspiration", str(folder))
    ids = st.pending_ids(a, "inspiration", ISV, "m", False)
    st.save_analysis(ids[0], ISV, "m", "ok", schema.sample_record(INSPIRATION.schema), "raw", None, 1.0)
    st.save_guide(a, "inspiration", "m", ISV, None, 1, False, 1.0, "guides/content-guide.md", None, "# G", {"x": 1})
    st.remove_images(a, [ids[1]])
    chars.delete(a)
    with sqlite3.connect(st.path) as db:
        for table in ("style_images", "style_removed", "style_guides"):
            assert db.execute(f"SELECT COUNT(*) FROM {table} WHERE character_id = ?", (a,)).fetchone()[0] == 0, table
        assert db.execute("SELECT COUNT(*) FROM style_analyses").fetchone()[0] == 0
    assert st.counts(b, "inspiration", ISV, "m")["images"] == 3


def test_guide_history(stores):
    _, st, a, _ = stores
    g1 = st.save_guide(a, "character", "m", CSV, "influencer", 4, False, 2.0, "guides/self-style-guide.md", None, "# One", {"n": 1})
    g2 = st.save_guide(a, "character", "m", CSV, None, 5, True, 2.0, "guides/self-style-guide.md", "guides/archive/x.md", "# Two", {})
    listed = st.list_guides(a, "character")
    assert [g["id"] for g in listed] == [g2, g1] and listed[0]["truncated"] is True and "markdown" not in listed[0]
    assert st.list_guides(a, "inspiration") == []
    full = st.get_guide(g1)
    assert full["markdown"] == "# One" and full["stats"] == {"n": 1} and full["character_type"] == "influencer"
    assert st.get_guide(999) is None


def test_migration_from_v2_keeps_characters(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        for m in _MIGRATIONS[:2]:
            db.executescript(m)
        db.execute("PRAGMA user_version = 2")
        db.execute("INSERT INTO characters (name, folder, created_at, updated_at) VALUES ('Bailey Bell', 'bailey-bell', 't', 't')")
    chars = CharacterStore(path)
    [c] = chars.list()
    assert c["character_type"] == "" and c["folder"] == "bailey-bell"
    assert chars.update(c["id"], {"character_type": "influencer"})["character_type"] == "influencer"
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == len(_MIGRATIONS) == 3


def test_character_type_validation(stores):
    chars, _, a, _ = stores
    from app.characters import CharacterError
    for bad in ("Has Space", "x" * 65, 3):
        with pytest.raises(CharacterError):
            chars.update(a, {"character_type": bad})
    assert chars.update(a, {"character_type": None})["character_type"] == ""


# ------------------------------------------------------------------ publishing guides into the documents folder
def test_mark_unindexed():
    assert mark_unindexed("# T\n") == "---\nindex: false\n---\n# T\n"
    assert mark_unindexed("---\ntype: x\nindex: true\n---\nbody") == "---\ntype: x\nindex: false\n---\nbody"


def test_publish_guide_archives_previous_version(tmp_path):
    from datetime import datetime

    from app import character_docs as cdocs
    folder = tmp_path / "doc"
    folder.mkdir()
    rel, archived = publish_guide(folder, "character", "# Self-Style Guide\n\n## Overview\nFirst.\n", datetime(2026, 11, 15, 9, 0), 3, "m.gguf")
    assert rel == "guides/self-style-guide.md" and archived is None
    first = cdocs.read_doc(folder, rel)["content"]
    assert first.startswith("---\ntype: self-style-guide\n") and "## Overview\nFirst." in first
    doc = cdocs.parse_document(first, rel)
    assert doc.meta["type"] == "self-style-guide" and doc.meta["tags"] == ["guide", "generated"]

    # edited by hand, then a new guide is written: the edited version is archived, not lost
    cdocs.write_doc(folder, rel, first + "\nMy note.\n", cdocs.read_doc(folder, rel)["mtime"])
    rel2, archived = publish_guide(folder, "character", "# Self-Style Guide\n\n## Overview\nSecond.\n", datetime(2026, 11, 16, 9, 0), 4, "m.gguf")
    assert archived == "guides/archive/self-style-guide-20261116-090000.md"
    old = cdocs.read_doc(folder, archived)["content"]
    assert "My note." in old and cdocs.parse_document(old, archived).meta["index"] is False
    assert "Second." in cdocs.read_doc(folder, rel2)["content"]
    # only the current guide is searchable
    idx = cdocs.DocIndex()
    chunks, _ = idx.chunks(1, tmp_path, "doc")
    assert {c.path for c in chunks} == {"guides/self-style-guide.md"}
    _, again = publish_guide(folder, "character", "# x\n", datetime(2026, 11, 16, 9, 0), 4, "m")
    assert again.endswith("-090000-2.md")  # same second: no collision
