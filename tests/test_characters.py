"""Character store (SQLite), prompt composition, and the character/workspace API."""

import sqlite3

import pytest
import torch

from app import characters as CH
from app.characters import CharacterError, CharacterStore, Duplicate, NotFound, compose_prompt


@pytest.fixture
def store(tmp_path):
    return CharacterStore(tmp_path / "db" / "chars.db")


def test_crud_roundtrip(store, tmp_path):
    a = store.create({"name": "  Ada   Lovelace ", "trigger": " ohwx  woman ", "lora_path": ' "D:\\l\\ada.safetensors" ', "lora_strength": 0.8})
    assert a["name"] == "Ada Lovelace" and a["trigger"] == "ohwx woman" and a["lora_path"] == "D:\\l\\ada.safetensors"
    assert a["prepend_trigger"] is True and a["lora_strength"] == 0.8 and a["description"] == ""
    b = store.create({"name": "bob", "prepend_trigger": False})
    assert [c["name"] for c in store.list()] == ["Ada Lovelace", "bob"]
    u = store.update(a["id"], {"description": "notes", "lora_strength": 1.25, "prepend_trigger": False})
    assert (u["description"], u["lora_strength"], u["prepend_trigger"], u["name"]) == ("notes", 1.25, False, "Ada Lovelace")
    assert u["updated_at"] >= a["updated_at"]
    # persisted: a fresh store on the same file sees the same rows
    again = CharacterStore(tmp_path / "db" / "chars.db")
    assert again.get(a["id"])["description"] == "notes" and again.get(b["id"])["prepend_trigger"] is False
    store.delete(b["id"])
    with pytest.raises(NotFound):
        store.get(b["id"])
    with pytest.raises(NotFound):
        store.delete(b["id"])
    with pytest.raises(NotFound):
        store.update(9999, {"name": "x"})


def test_names_are_unique_case_insensitively(store):
    a = store.create({"name": "Mira"})
    with pytest.raises(Duplicate):
        store.create({"name": "mira"})
    b = store.create({"name": "Other"})
    with pytest.raises(Duplicate):
        store.update(b["id"], {"name": "MIRA"})
    assert store.update(a["id"], {"name": "mira"})["name"] == "mira"  # renaming itself (case change) is fine


@pytest.mark.parametrize("data, needle", [
    ({}, "Name is required"),
    ({"name": "   "}, "Name is required"),
    ({"name": "x" * 101}, "at most 100"),
    ({"name": "a", "lora_strength": "1"}, "number"),
    ({"name": "a", "lora_strength": float("nan")}, "number"),
    ({"name": "a", "lora_strength": 5.5}, "between"),
    ({"name": "a", "prepend_trigger": "yes"}, "true/false"),
    ({"name": "a", "trigger": 5}, "text"),
    ({"name": "a", "color": "red"}, "Unknown fields"),
])
def test_validation(store, data, needle):
    with pytest.raises(CharacterError, match=needle):
        store.create(data)


def test_schema_version_and_newer_db_refused(tmp_path):
    p = tmp_path / "c.db"
    CharacterStore(p)
    with sqlite3.connect(p) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == len(CH._MIGRATIONS)
        db.execute("PRAGMA user_version = 99")
    with pytest.raises(CharacterError, match="newer version"):
        CharacterStore(p)


@pytest.mark.parametrize("prompt, trigger, prepend, expected", [
    ("a woman in a park", "ohwx", True, "ohwx, a woman in a park"),
    ("  a woman", "ohwx", True, "ohwx, a woman"),
    ("a woman", "ohwx", False, "a woman"),
    ("a woman", "", True, "a woman"),
    ("ohwx, a woman", "ohwx", True, "ohwx, a woman"),          # already there
    ("OHWX woman smiling", "ohwx", True, "OHWX woman smiling"),  # case-insensitive
    ("ohwxyz on a beach", "ohwx", True, "ohwx, ohwxyz on a beach"),  # not a whole-word match
    ("a (photo) of", "a (photo)", True, "a (photo) of"),        # regex characters in the trigger are literal
])
def test_compose_prompt(prompt, trigger, prepend, expected):
    assert compose_prompt(prompt, trigger, prepend) == expected


# ------------------------------------------------------------------------------------------ API

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (app startup)")


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import config

    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DB_FILE", tmp_path / "chars.db")
    monkeypatch.setattr(config, "CHARACTER_DOCS_DIR", tmp_path / "characters")
    monkeypatch.setattr(config, "STYLE_THUMBS_DIR", tmp_path / "style_thumbs")
    monkeypatch.setattr(config, "LLAMA_LOG_FILE", tmp_path / "logs" / "llama-server.log")
    from app.main import app

    with TestClient(app) as c:
        yield c


@needs_cuda
def test_character_crud_and_workspace_api(client, tmp_path):
    from tests.test_lora import D, _krea_lora

    lora = _krea_lora(tmp_path / "ada.safetensors", {"diffusion_model.blocks.0.attn.wq": ("peft", D, D)},
                      meta={"ss_tag_frequency": '{"x": {"ohwx woman": 3}}'})
    bad = _krea_lora(tmp_path / "bad.safetensors", {"diffusion_model.nope": ("peft", 4, 4)})

    assert client.get("/api/status").json()["workspace"] is None  # app starts on the landing page
    assert client.get("/api/characters").json()["characters"] == []
    r = client.post("/api/characters", json={"name": "Ada", "lora_path": lora, "lora_strength": 0.7, "trigger": "ohwx woman"})
    assert r.status_code == 200, r.text
    ada = r.json()
    assert ada["lora_info"]["compatible"] and ada["prepend_trigger"] is True
    assert client.post("/api/characters", json={"name": "ada"}).status_code == 409
    r = client.post("/api/characters", json={"name": "Broken", "lora_path": bad})
    assert r.status_code == 400 and "do not exist" in r.json()["detail"]
    assert client.post("/api/characters", json={"name": ""}).status_code == 400
    plain = client.post("/api/characters", json={"name": "Plain"}).json()
    assert plain["lora_info"] is None
    assert [c["name"] for c in client.get("/api/characters").json()["characters"]] == ["Ada", "Plain"]
    assert client.put(f"/api/characters/{plain['id']}", json={"lora_path": bad}).status_code == 400
    assert client.put("/api/characters/999", json={"name": "x"}).status_code == 404

    # load Ada: her LoRA is locked first in the effective stack; a user-stack entry for the same file is shadowed
    other = _krea_lora(tmp_path / "style.safetensors", {"diffusion_model.blocks.1.attn.wq": ("peft", D, D)})
    client.put("/api/loras", json={"stack": [{"path": other, "strength": 0.5, "enabled": True}, {"path": lora, "strength": 2.0, "enabled": True}]})
    r = client.post("/api/workspace", json={"character_id": ada["id"]})
    assert r.status_code == 200 and r.json()["workspace"]["character"]["name"] == "Ada"
    v = client.get("/api/loras").json()
    assert v["locked"]["path"] == lora and v["locked"]["strength"] == 0.7
    assert [e["shadowed"] for e in v["stack"]] == [False, True]

    from app import main

    assert main._effective_stack() == [{"path": lora, "strength": 0.7, "enabled": True}, {"path": other, "strength": 0.5, "enabled": True}]

    # encode/generate receive the trigger-prefixed prompt and the locked stack
    seen = {}
    real_encode, real_generate = main.S.runner.submit_encode, main.S.runner.submit_generate

    class FakeJob:
        def to_dict(self):
            return {"id": "j", "request": {}}

    def fake_encode(prompt):
        seen["encode"] = prompt
        return FakeJob()

    def fake_generate(prompt, res, seed, stack, extra_meta=None):
        seen["generate"] = (prompt, stack, extra_meta)
        return FakeJob()

    main.S.runner.submit_encode, main.S.runner.submit_generate = fake_encode, fake_generate
    try:
        client.post("/api/encode", json={"prompt": "reading in a cafe"})
        assert seen["encode"] == "ohwx woman, reading in a cafe"
        client.post("/api/encode", json={"prompt": "reading in a cafe", "prepend_trigger": False})
        assert seen["encode"] == "reading in a cafe"
        gen = {"prompt": "reading in a cafe", "aspect": "1:1", "megapixels": 1.0, "multiple": 16, "width": 1024, "height": 1024}
        assert client.post("/api/generate", json=gen).status_code == 200
        prompt, stack, extra = seen["generate"]
        assert prompt == "ohwx woman, reading in a cafe" and stack[0] == {"path": lora, "strength": 0.7, "enabled": True}
        assert extra == {"character": {"id": ada["id"], "name": "Ada"}, "user_prompt": "reading in a cafe", "trigger_prepended": True}

        # editing the active character applies immediately
        client.put(f"/api/characters/{ada['id']}", json={"lora_strength": 1.1, "trigger": "zxq"})
        client.post("/api/generate", json=gen)
        prompt, stack, _ = seen["generate"]
        assert prompt == "zxq, reading in a cafe" and stack[0]["strength"] == 1.1

        # free mode: no lock, no trigger; the user's stack is used as-is
        client.post("/api/workspace", json={"character_id": None})
        assert client.get("/api/status").json()["workspace"] == {"character": None}
        assert client.get("/api/loras").json()["locked"] is None
        client.post("/api/generate", json=gen)
        prompt, stack, extra = seen["generate"]
        assert prompt == "reading in a cafe" and [e["path"] for e in stack] == [other, lora] and extra["character"] is None
    finally:
        main.S.runner.submit_encode, main.S.runner.submit_generate = real_encode, real_generate

    # deleting the active character returns to the landing page; the LoRA file is untouched
    client.post("/api/workspace", json={"character_id": ada["id"]})
    r = client.delete(f"/api/characters/{ada['id']}")
    assert r.status_code == 200 and r.json()["workspace"] is None
    assert client.get("/api/status").json()["workspace"] is None
    import os
    assert os.path.exists(lora)
    assert client.post("/api/workspace", json={"character_id": ada["id"]}).status_code == 404
    assert client.delete("/api/workspace").json()["workspace"] is None
