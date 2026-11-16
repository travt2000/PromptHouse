"""HTTP API behaviour with the real app (CUDA backend initialised, but no model weights are loaded here)."""

import pytest
import torch
from fastapi.testclient import TestClient

from app import config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DB_FILE", tmp_path / "chars.db")
    monkeypatch.setattr(config, "CHARACTER_DOCS_DIR", tmp_path / "characters")
    monkeypatch.setattr(config, "STYLE_THUMBS_DIR", tmp_path / "style_thumbs")
    monkeypatch.setattr(config, "LLAMA_LOG_FILE", tmp_path / "logs" / "llama-server.log")
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_status_and_static(client):
    s = client.get("/api/status").json()
    assert s["busy"] is False and s["conditioning"] is None and s["models"]["gpu_owner"] is None
    assert client.get("/").status_code == 200
    sysinfo = client.get("/api/system").json()
    assert sysinfo["ram"]["total"] > 0 and "cpu" in sysinfo


def test_resolution_table_matches_backend_validation(client):
    t = client.get("/api/resolutions").json()
    e = t["entries"]["16:9|1|16"]
    assert (e["width"], e["height"]) == (1360, 768)


def test_generate_rejects_unencoded_prompt_and_mismatched_dims(client):
    body = {"prompt": "x", "aspect": "16:9", "megapixels": 1.0, "multiple": 16, "width": 1360, "height": 768}
    r = client.post("/api/generate", json=body)
    assert r.status_code == 400 and "Encode" in r.json()["detail"]
    r = client.post("/api/generate", json={**body, "width": 1344})
    assert r.status_code == 400 and "disagrees" in r.json()["detail"]
    r = client.post("/api/generate", json={**body, "aspect": "2:1"})
    assert r.status_code == 400


def test_encode_rejects_empty_prompt(client):
    assert client.post("/api/encode", json={"prompt": "   "}).status_code == 400


def test_settings_persist_and_validate(client, tmp_path):
    r = client.put("/api/settings", json={"seed": 7, "randomize_seed": True})
    assert r.status_code == 200 and r.json()["settings"]["seed"] == 7
    assert '"seed": 7' in (tmp_path / "settings.json").read_text()
    assert client.put("/api/settings", json={"seed": -5}).status_code == 400
    r = client.put("/api/settings", json={"vae_path": str(tmp_path / "missing.safetensors")})
    assert r.status_code == 200 and r.json()["validation"]["vae_path"]["compatible"] is False
    assert "vae" in r.json()["invalidated"]


def test_validate_path_reports_wrong_component(client):
    s = client.get("/api/settings").json()["settings"]
    r = client.post("/api/validate-path", json={"kind": "transformer_path", "path": s["vae_path"]}).json()
    if "Detected component type" not in " ".join(r["problems"]) and "not found" not in " ".join(r["problems"]):
        pytest.fail(r)


def test_lora_stack_endpoints(client, tmp_path):
    from tests.test_lora import D, _krea_lora

    a = _krea_lora(tmp_path / "a.safetensors", {"diffusion_model.blocks.0.attn.wq": ("peft", D, D)})
    b = _krea_lora(tmp_path / "b.safetensors", {"transformer.transformer_blocks.1.attn.to_q": ("peft", D, D)})
    bad = _krea_lora(tmp_path / "bad.safetensors", {"diffusion_model.nope": ("peft", 4, 4)})
    assert client.get("/api/loras").json()["stack"] == []
    r = client.put("/api/loras", json={"stack": [{"path": a, "strength": 1.0, "enabled": True}, {"path": b, "strength": 0.5, "enabled": True}]})
    assert r.status_code == 200, r.text
    assert [e["info"]["key_style"] for e in r.json()["stack"]] == ["krea2 original", "diffusers"]
    # reorder + disable + strength, persisted to settings.json
    r = client.put("/api/loras", json={"stack": [{"path": b, "strength": -0.25, "enabled": False}, {"path": a, "strength": 1.0, "enabled": True}]})
    assert [(e["path"], e["strength"], e["enabled"]) for e in r.json()["stack"]] == [(b, -0.25, False), (a, 1.0, True)]
    assert "-0.25" in (tmp_path / "settings.json").read_text()
    r = client.put("/api/loras", json={"stack": [{"path": bad, "strength": 1.0, "enabled": True}]})
    assert r.status_code == 400 and "do not exist" in r.json()["detail"]
    assert client.put("/api/loras", json={"stack": [{"path": a}, {"path": a}]}).status_code == 400
    v = client.post("/api/validate-path", json={"kind": "lora", "path": b}).json()
    assert v["compatible"] and v["modules"] == 1
    assert client.put("/api/loras", json={"stack": []}).json()["stack"] == []
