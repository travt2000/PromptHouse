"""Character document folders: provisioning, safe file access, parsing/chunking, lexical retrieval, API."""

import os
import sqlite3
import time

import pytest
import torch

from app import character_docs as cd


# ------------------------------------------------------------------------------------------ folders


def test_slugify_and_allocate(tmp_path):
    assert cd.slugify("Bailey Bell") == "bailey-bell"
    assert cd.slugify("  Zoë  O'Neil!! ") == "zoe-o-neil"
    assert cd.slugify("???") == "character"
    (tmp_path / "ada").mkdir()
    assert cd.allocate_folder(tmp_path, "Ada", set()) == "ada-2"            # existing directory
    assert cd.allocate_folder(tmp_path, "Ada", {"ADA-2"}) == "ada-3"       # taken by another character (case-insensitive)


def test_provision_writes_templates_once(tmp_path):
    d = cd.provision(tmp_path, "ada", "Ada")
    files = sorted(f["path"] for f in cd.list_docs(d))
    assert files == ["README.md", "bible.md", "style-guide.md", "voice.md"]
    assert (d / "notes").is_dir()
    assert "# Ada — character bible" in (d / "bible.md").read_text(encoding="utf-8")
    (d / "voice.md").unlink()
    (d / "bible.md").write_text("mine", encoding="utf-8")
    cd.provision(tmp_path, "ada", "Ada")  # existing folder: untouched
    assert not (d / "voice.md").exists() and (d / "bible.md").read_text(encoding="utf-8") == "mine"


# ------------------------------------------------------------------------------------------ files


@pytest.mark.parametrize("rel", ["../x.md", "a/../../x.md", "/etc/x.md", "C:/x.md", "C:\\x.md", "x.txt", "", ".hidden.md",
                                 "notes/.secret/x.md", "x\x00.md", "a/" * 7 + "x.md"])
def test_safe_path_rejects(tmp_path, rel):
    with pytest.raises(cd.DocError):
        cd.safe_path(tmp_path, rel)


def test_safe_path_accepts_nested_and_backslashes(tmp_path):
    p, rel = cd.safe_path(tmp_path, "notes\\Summer (2026)\\wardrobe.md")
    assert rel == "notes/Summer (2026)/wardrobe.md" and p == (tmp_path / "notes" / "Summer (2026)" / "wardrobe.md").resolve()


def test_write_read_conflict_delete(tmp_path):
    r = cd.write_doc(tmp_path, "notes/a.md", "# A\r\nline\r\n", None)
    assert (tmp_path / "notes" / "a.md").read_bytes() == b"# A\nline\n"  # normalised to LF
    with pytest.raises(cd.DocConflict, match="already exists"):
        cd.write_doc(tmp_path, "notes/a.md", "x", None)
    doc = cd.read_doc(tmp_path, "notes/a.md")
    assert doc["content"] == "# A\nline\n" and doc["mtime"] == r["mtime"]
    assert isinstance(doc["mtime"], str)  # nanosecond ints would lose precision as JavaScript numbers
    # changed elsewhere -> stale save refused, nothing overwritten
    time.sleep(0.02)
    (tmp_path / "notes" / "a.md").write_text("edited in VS Code", encoding="utf-8")
    os.utime(tmp_path / "notes" / "a.md", ns=(int(r["mtime"]) + 10**9, int(r["mtime"]) + 10**9))
    with pytest.raises(cd.DocConflict, match="changed on disk"):
        cd.write_doc(tmp_path, "notes/a.md", "mine", r["mtime"])
    assert (tmp_path / "notes" / "a.md").read_text(encoding="utf-8") == "edited in VS Code"
    fresh = cd.read_doc(tmp_path, "notes/a.md")
    cd.write_doc(tmp_path, "notes/a.md", "mine", fresh["mtime"])
    assert cd.read_doc(tmp_path, "notes/a.md")["content"] == "mine"
    assert not list(tmp_path.rglob(".*.tmp"))
    cd.delete_doc(tmp_path, "notes/a.md")
    with pytest.raises(FileNotFoundError):
        cd.read_doc(tmp_path, "notes/a.md")
    with pytest.raises(cd.DocConflict, match="deleted"):
        cd.write_doc(tmp_path, "notes/a.md", "x", fresh["mtime"])
    with pytest.raises(cd.DocError, match="limited"):
        cd.write_doc(tmp_path, "big.md", "x" * (cd.MAX_DOC_BYTES + 1), None)


def test_list_docs_skips_hidden_and_non_markdown(tmp_path):
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "x.md").write_text("x")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "Bible.md").write_text("---\ntype: Bible\ntags: [core, look]\n---\n# The Bible\n")
    (tmp_path / "plain.md").write_text("no heading")
    files = {f["path"]: f for f in cd.list_docs(tmp_path)}
    assert set(files) == {"Bible.md", "plain.md"}
    assert (files["Bible.md"]["title"], files["Bible.md"]["type"], files["Bible.md"]["tags"]) == ("The Bible", "bible", ["core", "look"])
    assert (files["plain.md"]["title"], files["plain.md"]["type"]) == ("plain", "plain")


# ------------------------------------------------------------------------------------------ parsing + chunking


def test_front_matter_variants():
    d = cd.parse_document("---\ntitle: T\ntags: a, b\n---\nbody\n", "x/style-guide.md")
    assert (d.title, d.tags, d.type, d.body, d.body_line) == ("T", ["a", "b"], "style-guide", "body\n", 5)
    bad = cd.parse_document("---\n: : :\n  - [\n---\n# H\n", "n.md")
    assert bad.warnings and bad.title == "H" and bad.meta == {}
    lst = cd.parse_document("---\n- a\n- b\n---\nx", "n.md")
    assert "mapping" in lst.warnings[0]
    unterminated = cd.parse_document("---\ntype: x\n# H", "n.md")
    assert unterminated.meta == {} and unterminated.body.startswith("---")


DOC = """---
type: style-guide
tags: [look]
---
# Ada — style guide

Intro paragraph.

## Wardrobe
Green linen dress in summer.

```
# not a heading inside a code fence
```

### Shoes
White sneakers.

## Avoid
Neon colours.
"""


def test_chunking_follows_headings_with_line_numbers():
    doc = cd.parse_document(DOC, "style-guide.md")
    chunks = cd.chunk_document(doc, 7, "ada")
    assert [c.headings for c in chunks] == [["Ada — style guide"], ["Ada — style guide", "Wardrobe"],
                                            ["Ada — style guide", "Wardrobe", "Shoes"], ["Ada — style guide", "Avoid"]]
    wardrobe = chunks[1]
    assert "# not a heading inside a code fence" in wardrobe.text and wardrobe.text.startswith("## Wardrobe")
    lines = DOC.split("\n")
    assert lines[wardrobe.start_line - 1] == "## Wardrobe"
    assert lines[chunks[3].end_line - 1] in ("Neon colours.", "")
    assert all(c.doc_type == "style-guide" and c.tags == ["look"] and c.character_id == 7 for c in chunks)
    again = cd.chunk_document(cd.parse_document(DOC, "style-guide.md"), 7, "ada")
    assert [c.id for c in again] == [c.id for c in chunks] and len({c.id for c in chunks}) == 4  # stable + unique
    assert [c.sha256 for c in again] == [c.sha256 for c in chunks]


def test_long_sections_split_at_paragraphs():
    paras = [f"Paragraph {i} " + "word " * 60 for i in range(10)]
    doc = cd.parse_document("# Long\n\n" + "\n\n".join(paras), "long.md")
    chunks = cd.chunk_document(doc, 1, "f", limit=800)
    assert len(chunks) > 1 and all(len(c.text) <= 800 for c in chunks)
    assert all(c.headings == ["Long"] for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert all(f"Paragraph {i} " in joined for i in range(10))


def test_index_caches_per_file_and_bm25_ranks(tmp_path):
    root = tmp_path
    cd.provision(root, "ada", "Ada")
    (root / "ada" / "style-guide.md").write_text(DOC, encoding="utf-8")
    idx = cd.DocIndex()
    calls = []
    real = cd.chunk_document
    cd.chunk_document = lambda *a, **k: calls.append(a[0].path) or real(*a, **k)
    try:
        chunks, warns = idx.chunks(1, root, "ada")
        assert not warns and len(calls) == 4
        idx.chunks(1, root, "ada")
        assert len(calls) == 4  # nothing changed: nothing re-parsed
        p = root / "ada" / "voice.md"
        p.write_text("# Voice\nDry humour.\n", encoding="utf-8")
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        chunks, _ = idx.chunks(1, root, "ada")
        assert calls[4:] == ["voice.md"]
    finally:
        cd.chunk_document = real
    hits = cd.DocIndex.search(chunks, "what dress does she wear in summer", k=3)
    assert hits and hits[0][0].headings[-1] == "Wardrobe"
    assert cd.DocIndex.search(chunks, "humour")[0][0].path == "voice.md"
    assert cd.DocIndex.search(chunks, "zzzz") == [] and cd.DocIndex.search(chunks, "   ") == []
    (root / "ada" / "voice.md").unlink()
    chunks, _ = idx.chunks(1, root, "ada")
    assert all(c.path != "voice.md" for c in chunks)


# ------------------------------------------------------------------------------------------ API

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (app startup)")


def _client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import config

    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DB_FILE", tmp_path / "chars.db")
    monkeypatch.setattr(config, "CHARACTER_DOCS_DIR", tmp_path / "characters")
    monkeypatch.setattr(config, "STYLE_THUMBS_DIR", tmp_path / "style_thumbs")
    monkeypatch.setattr(config, "LLAMA_LOG_FILE", tmp_path / "logs" / "llama-server.log")
    from app.main import app

    return TestClient(app)


@needs_cuda
def test_existing_v1_characters_get_folders_on_startup(tmp_path, monkeypatch):
    with sqlite3.connect(tmp_path / "chars.db") as db:  # a database from the previous version
        db.executescript(__import__("app.characters", fromlist=["x"])._MIGRATIONS[0])
        db.execute("PRAGMA user_version = 1")
        db.execute("INSERT INTO characters (name, created_at, updated_at) VALUES ('Bailey Bell', 't', 't')")
    with _client(tmp_path, monkeypatch) as c:
        ch = c.get("/api/characters").json()["characters"][0]
        assert ch["folder"] == "bailey-bell" and ch["docs_dir"] == str(tmp_path / "characters" / "bailey-bell")
        assert (tmp_path / "characters" / "bailey-bell" / "bible.md").is_file()
    with sqlite3.connect(tmp_path / "chars.db") as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == len(__import__("app.characters", fromlist=["x"])._MIGRATIONS)


@needs_cuda
def test_docs_api(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as c:
        ada = c.post("/api/characters", json={"name": "Ada"}).json()
        assert ada["folder"] == "ada"
        r = c.get(f"/api/characters/{ada['id']}/docs").json()
        assert sorted(f["path"] for f in r["files"]) == ["README.md", "bible.md", "style-guide.md", "voice.md"]
        base = f"/api/characters/{ada['id']}/docs/file"

        doc = c.get(base, params={"path": "bible.md"}).json()
        assert "character bible" in doc["content"]
        w = c.put(base, json={"path": "bible.md", "content": "# Ada\n## Appearance\nRed hair, green eyes.\n", "expected_mtime": doc["mtime"]})
        assert w.status_code == 200, w.text
        assert c.put(base, json={"path": "bible.md", "content": "stale", "expected_mtime": doc["mtime"]}).status_code == 409
        assert c.put(base, json={"path": "bible.md", "content": "dup", "expected_mtime": None}).status_code == 409
        assert c.put(base, json={"path": "notes/summer.md", "content": "# Summer\nBeach days.\n"}).status_code == 200
        assert c.put(base, json={"path": "../escape.md", "content": "x"}).status_code == 400
        assert c.get(base, params={"path": "../../settings.json"}).status_code == 400
        assert c.get(base, params={"path": "missing.md"}).status_code == 404
        assert not (tmp_path / "characters" / "escape.md").exists()

        ctx = c.get(f"/api/characters/{ada['id']}/context", params={"q": "hair colour eyes"}).json()
        assert ctx["retriever"] == "bm25" and ctx["results"][0]["path"] == "bible.md"
        assert ctx["results"][0]["headings"] == ["Ada", "Appearance"]
        assert c.get(f"/api/characters/{ada['id']}/context").json()["results"] == []

        assert c.delete(base, params={"path": "notes/summer.md"}).status_code == 200
        assert c.delete(base, params={"path": "notes/summer.md"}).status_code == 404

        # a second "Ada" after deletion gets a new folder; the old one (and its files) is kept
        r = c.delete(f"/api/characters/{ada['id']}").json()
        assert r["docs_kept"] == str(tmp_path / "characters" / "ada")
        assert (tmp_path / "characters" / "ada" / "bible.md").read_text(encoding="utf-8").startswith("# Ada\n")
        ada2 = c.post("/api/characters", json={"name": "ada"}).json()
        assert ada2["folder"] == "ada-2"
        # renaming keeps the folder
        assert c.put(f"/api/characters/{ada2['id']}", json={"name": "Ada Two"}).json()["folder"] == "ada-2"
        assert c.get("/api/characters/999/docs").status_code == 404


def test_comments_heading_only_sections_and_index_false_are_not_indexed(tmp_path):
    cd.provision(tmp_path, "x", "X")
    chunks, _ = cd.DocIndex().chunks(1, tmp_path, "x")
    assert chunks == []  # untouched templates: only headings + hint comments
    doc = cd.parse_document("# T\n<!-- hidden\nhint -->\n## Real\nVisible text.\n", "n.md")
    (c,) = cd.chunk_document(doc, 1, "x")
    assert c.headings == ["T", "Real"] and "hidden" not in c.text and c.start_line == 4
    off = cd.parse_document("---\nindex: false\n---\n# T\nText.\n", "n.md")
    assert cd.chunk_document(off, 1, "x") == []
