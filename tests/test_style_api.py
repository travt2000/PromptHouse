"""Style analysis end to end: the real app (GPU worker, llama-server subprocess manager, analyzer, guide
writer) against the fake llama-server. No real vision model is involved, so model output quality is not tested."""
import json
import time

import pytest
import torch
from fastapi.testclient import TestClient

from app import config
from app.style.kinds import CHARACTER, INSPIRATION
from app.style.llama_server import ServerError, ServerManager
from app.style.vsettings import VisionSettings

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (the app's GPU worker)")
INSP = {"kind": "inspiration"}
CHAR = {"kind": "character"}


def wait_idle(client, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get("/api/style/status").json()
        if not s["active"] and s["state"] in ("done", "error", "cancelled"):
            return s
        time.sleep(0.1)
    raise AssertionError("style job did not finish")


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DB_FILE", tmp_path / "chars.db")
    monkeypatch.setattr(config, "CHARACTER_DOCS_DIR", tmp_path / "characters")
    monkeypatch.setattr(config, "STYLE_THUMBS_DIR", tmp_path / "style_thumbs")
    monkeypatch.setattr(config, "LLAMA_LOG_FILE", tmp_path / "logs" / "llama-server.log")
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def client(app_client, fake_llama, model_files):
    r = app_client.put("/api/settings", json={"vision": {"llama_server_path": fake_llama, "model_path": model_files[0],
                                                         "mmproj_path": model_files[1], "startup_timeout_s": 30}})
    assert r.status_code == 200, r.text
    return app_client


@pytest.fixture
def cid(client):
    return client.post("/api/characters", json={"name": "Bailey Bell", "character_type": "influencer"}).json()["id"]


def url(cid, tail=""):
    return f"/api/characters/{cid}/style{tail}"


# ------------------------------------------------------------------ subprocess manager (no app needed)
def test_server_manager_start_stop(tmp_path, fake_llama, model_files):
    mgr = ServerManager(tmp_path / "log.txt")
    s = VisionSettings(llama_server_path=fake_llama, model_path=model_files[0], mmproj_path=model_files[1], startup_timeout_s=30)
    base = mgr.start(s)
    assert mgr.running and base.startswith("http://127.0.0.1:")
    proc = mgr._proc
    mgr.stop()
    assert not mgr.running and proc.poll() is not None


def test_server_manager_reports_crash_with_log(tmp_path, fake_llama, model_files, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "crash")
    mgr = ServerManager(tmp_path / "log.txt")
    s = VisionSettings(llama_server_path=fake_llama, model_path=model_files[0], mmproj_path=model_files[1], startup_timeout_s=30)
    with pytest.raises(ServerError, match="unknown model architecture"):
        mgr.start(s)
    assert not mgr.running


# ------------------------------------------------------------------ API
@needs_cuda
def test_full_batch_and_resume(client, cid, folder):
    r = client.post(url(cid, "/scan"), json={**INSP, "folder": str(folder)}).json()
    assert r["added"] == 3 and r["counts"]["pending"] == 3
    assert client.post(url(cid, "/analyze"), json=INSP).json() == {"queued": 3, "reused": 0}
    status = wait_idle(client)
    assert status["state"] == "done" and status["kind"] == "inspiration" and status["character_id"] == cid
    assert status["done"] == 3 and status["failed"] == 0
    assert not status["server_running"]  # the subprocess is gone after the batch
    assert client.get("/api/status").json()["busy"] is False  # the GPU worker is free again

    listing = client.get(url(cid, "/images"), params=INSP).json()
    assert listing["counts"] == {"images": 3, "ok": 3, "failed": 0, "pending": 0}
    detail = client.get(f"/api/style/images/{listing['items'][0]['id']}").json()
    assert detail["image"]["kind"] == "inspiration" and detail["current_schema_version"] == INSPIRATION.schema_version
    a = detail["analyses"][0]
    assert a["status"] == "ok" and a["record"]["content"]["format"] and "appearance" not in a["record"]
    thumb = client.get(f"/api/style/images/{listing['items'][0]['id']}/thumb")
    assert thumb.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/style/images/{listing['items'][0]['id']}/file").status_code == 200

    # nothing pending now, so a plain run is refused; a forced one redoes everything
    assert client.post(url(cid, "/analyze"), json=INSP).status_code == 400
    assert client.post(url(cid, "/analyze"), json={**INSP, "reanalyze": True}).json() == {"queued": 3, "reused": 0}
    assert wait_idle(client)["state"] == "done"


@needs_cuda
def test_vision_job_holds_the_gpu_worker(client, cid, folder, monkeypatch):
    """While a style job runs, encode/generate are refused (only one component on the GPU at a time)."""
    monkeypatch.setenv("FAKE_LLAMA_MODE", "guide_slow")  # analysis is fast; use a slow guide to hold the worker
    client.post(url(cid, "/scan"), json={**INSP, "folder": str(folder)})
    client.post(url(cid, "/analyze"), json=INSP)
    wait_idle(client)
    client.post(url(cid, "/guides"), json=INSP)
    end = time.time() + 20
    while client.get("/api/style/status").json()["state"] != "running" and time.time() < end:
        time.sleep(0.05)
    st = client.get("/api/status").json()
    assert st["busy"] and st["active_job"]["kind"] == "vision" and st["style"]["job"] == "guide"
    r = client.post("/api/encode", json={"prompt": "a cat"})
    assert r.status_code == 409 and "vision" in r.json()["detail"]
    assert client.post(url(cid, "/analyze"), json={**INSP, "reanalyze": True}).status_code == 409
    assert client.delete(f"/api/characters/{cid}").status_code == 409  # not while its job runs
    client.post("/api/style/cancel")
    assert wait_idle(client, 20)["state"] == "cancelled"
    assert client.get("/api/status").json()["busy"] is False


@needs_cuda
def test_character_batch_uses_character_prompt_and_schema(client, cid, folder, monkeypatch, tmp_path):
    seen = tmp_path / "seen.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_SEEN", str(seen))
    client.post(url(cid, "/scan"), json={**INSP, "folder": str(folder)})
    assert client.post(url(cid, "/scan"), json={**CHAR, "folder": str(folder), "recursive": False}).json()["added"] == 2
    assert client.post(url(cid, "/analyze"), json=CHAR).json() == {"queued": 2, "reused": 0}  # only that library
    status = wait_idle(client)
    assert status["state"] == "done" and status["kind"] == "character"
    assert {json.loads(l)["kind"] for l in seen.read_text().splitlines()} == {"character"}
    summary = client.get(url(cid)).json()
    assert summary["counts"]["character"] == {"images": 2, "ok": 2, "failed": 0, "pending": 0}
    assert summary["counts"]["inspiration"]["pending"] == 3  # untouched
    assert summary["problems"] == [] and summary["character"]["character_type_info"]["label"] == "Influencer"
    item = client.get(url(cid, "/images"), params=CHAR).json()["items"][0]
    rec = client.get(f"/api/style/images/{item['id']}").json()["analyses"][0]["record"]
    assert rec["appearance"]["hair_color"] and rec["recommendations"]["summary"]


@needs_cuda
def test_second_character_reuses_inspiration_analyses(client, cid, folder, monkeypatch, tmp_path):
    client.post(url(cid, "/scan"), json={**INSP, "folder": str(folder)})
    client.post(url(cid, "/analyze"), json=INSP)
    wait_idle(client)
    other = client.post("/api/characters", json={"name": "Other"}).json()["id"]
    client.post(url(other, "/scan"), json={**INSP, "folder": str(folder), "recursive": False})
    seen = tmp_path / "seen.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_SEEN", str(seen))
    assert client.post(url(other, "/analyze"), json=INSP).json() == {"queued": 0, "reused": 2}
    assert not seen.exists()  # the model was not started
    assert client.get(url(other, "/images"), params=INSP).json()["counts"]["ok"] == 2


@needs_cuda
def test_analyze_by_id_ignores_other_library_and_character(client, cid, folder):
    client.post(url(cid, "/scan"), json={**INSP, "folder": str(folder)})
    insp_id = client.get(url(cid, "/images"), params=INSP).json()["items"][0]["id"]
    other = client.post("/api/characters", json={"name": "Other"}).json()["id"]
    assert client.post(url(cid, "/analyze"), json={**CHAR, "image_ids": [insp_id]}).status_code == 400
    assert client.post(url(other, "/analyze"), json={**INSP, "image_ids": [insp_id]}).status_code == 400
    assert client.post(url(cid, "/analyze"), json={**INSP, "image_ids": [insp_id]}).json() == {"queued": 1, "reused": 0}
    assert wait_idle(client)["state"] == "done"


@needs_cuda
def test_remove_images_via_api(client, cid, folder, tmp_path):
    client.post(url(cid, "/scan"), json={**INSP, "folder": str(folder)})
    client.post(url(cid, "/analyze"), json=INSP)
    assert wait_idle(client)["state"] == "done"
    items = client.get(url(cid, "/images"), params=INSP).json()["items"]
    first = client.get(f"/api/style/images/{items[0]['id']}").json()["image"]
    client.get(f"/api/style/images/{items[0]['id']}/thumb")
    thumb = tmp_path / "style_thumbs" / f"{first['sha256']}.jpg"
    assert thumb.is_file()

    assert client.post(url(cid, "/remove"), json={"image_ids": [items[0]["id"], items[1]["id"]]}).json() == {"removed": 2}
    assert client.get(url(cid, "/images"), params=INSP).json()["counts"] == {"images": 1, "ok": 1, "failed": 0, "pending": 0}
    assert client.get(f"/api/style/images/{items[0]['id']}").status_code == 404
    assert not thumb.exists() and (folder / "a.png").is_file()  # the file on disk is untouched

    r = client.post(url(cid, "/scan"), json={**INSP, "folder": str(folder)}).json()
    assert r["added"] == 0 and r["skipped_removed"] == 2
    r = client.post(url(cid, "/scan"), json={**INSP, "folder": str(folder), "include_removed": True}).json()
    assert r["added"] == 2 and r["restored"] == 2


@needs_cuda
def test_bad_model_output_is_recorded_as_failed(client, cid, folder, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "bad_json")
    client.post(url(cid, "/scan"), json={**INSP, "folder": str(folder)})
    client.post(url(cid, "/analyze"), json=INSP)
    status = wait_idle(client)
    assert status["state"] == "done" and status["failed"] == 3
    first = client.get(url(cid, "/images"), params={**INSP, "status": "failed"}).json()["items"][0]["id"]
    a = client.get(f"/api/style/images/{first}").json()["analyses"][0]
    assert a["status"] == "failed" and a["raw_text"] == "this is not json" and "no JSON" in a["error"]


@needs_cuda
def test_falls_back_when_server_rejects_response_format(client, cid, folder, monkeypatch, tmp_path):
    seen = tmp_path / "seen.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_MODE", "no_schema")
    monkeypatch.setenv("FAKE_LLAMA_SEEN", str(seen))
    client.post(url(cid, "/scan"), json={**INSP, "folder": str(folder), "recursive": False})
    client.post(url(cid, "/analyze"), json=INSP)
    assert wait_idle(client)["state"] == "done"
    lines = [json.loads(l) for l in seen.read_text().splitlines()]
    formats = [l["format"] for l in lines]
    assert formats[:3] == ["json_schema", "json_object", "none"]  # stepped down once, then remembered
    assert set(formats[3:]) == {"none"}
    assert all(l["kwargs"] == {"enable_thinking": False} for l in lines)


@needs_cuda
def test_server_crash_surfaces_error_and_allows_retry(client, cid, folder, monkeypatch):
    client.post(url(cid, "/scan"), json={**INSP, "folder": str(folder)})
    monkeypatch.setenv("FAKE_LLAMA_MODE", "crash")
    client.post(url(cid, "/analyze"), json=INSP)
    status = wait_idle(client)
    assert status["state"] == "error" and "unknown model architecture" in status["error"]
    assert "unknown model architecture" in client.get("/api/style/server-log").json()["log"]
    monkeypatch.delenv("FAKE_LLAMA_MODE")
    client.post(url(cid, "/analyze"), json=INSP)
    assert wait_idle(client)["state"] == "done"


@needs_cuda
def test_requires_configuration_and_guards(app_client, folder):
    c = app_client.post("/api/characters", json={"name": "A"}).json()["id"]
    app_client.post(url(c, "/scan"), json={**INSP, "folder": str(folder)})
    r = app_client.post(url(c, "/analyze"), json=INSP)
    assert r.status_code == 400 and "llama-server" in r.json()["detail"]
    assert len(app_client.get(url(c)).json()["problems"]) == 3
    assert app_client.post(url(c, "/scan"), json={**INSP, "folder": "/definitely/not/here"}).status_code == 400
    assert app_client.post(url(c, "/scan"), json={"kind": "bogus", "folder": str(folder)}).status_code == 422
    assert app_client.get(url(999)).status_code == 404
    assert app_client.post(url(999, "/scan"), json={**INSP, "folder": str(folder)}).status_code == 404
    assert app_client.get("/api/style/images/9999").status_code == 404
    assert app_client.get("/api/style/images/9999/thumb").status_code == 404
    assert app_client.get(url(c, "/images"), params={**INSP, "status": "bogus"}).status_code == 422
    assert app_client.put("/api/settings", json={"vision": {"structured_mode": "nope"}}).status_code == 400
    assert app_client.get("/api/browse", params={"path": str(folder), "ext": ".png"}).status_code == 400
    assert app_client.get("/api/style/guides/999").status_code == 404


@needs_cuda
def test_browse_lists_requested_extensions(app_client, tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "llama-server.exe").write_bytes(b"x")
    (tmp_path / "w.safetensors").write_bytes(b"x")
    names = lambda ext: {e["name"] for e in app_client.get("/api/browse", params={"path": str(tmp_path), "ext": ext}).json()["entries"] if e["type"] == "file"}
    assert names(".gguf") == {"m.gguf"}
    assert names(".exe,.gguf") == {"m.gguf", "llama-server.exe"}
    assert "w.safetensors" in {e["name"] for e in app_client.get("/api/browse", params={"path": str(tmp_path)}).json()["entries"]}


@needs_cuda
def test_character_type_on_characters(app_client):
    types = app_client.get("/api/style/character-types").json()
    assert types["error"] is None and any(t["id"] == "influencer" for t in types["types"])
    c = app_client.post("/api/characters", json={"name": "A", "character_type": "cosplay"}).json()
    assert c["character_type"] == "cosplay" and c["character_type_info"]["label"] == "Cosplay Character"
    assert app_client.put(f"/api/characters/{c['id']}", json={"character_type": "nope"}).status_code == 400
    assert app_client.post("/api/characters", json={"name": "B", "character_type": "nope"}).status_code == 400
    assert app_client.put(f"/api/characters/{c['id']}", json={"character_type": ""}).json()["character_type_info"] is None


@needs_cuda
def test_character_type_file_is_reread(app_client, tmp_path, monkeypatch):
    c = app_client.post("/api/characters", json={"name": "A", "character_type": "cosplay"}).json()
    custom = tmp_path / "types.json"
    custom.write_text('{"character_types": [{"id": "streamer", "label": "Streamer"}]}', encoding="utf-8")
    monkeypatch.setenv("PROMPTHOUSE_CHARACTER_TYPES", str(custom))
    assert [t["id"] for t in app_client.get("/api/style/character-types").json()["types"]] == ["streamer"]
    # the stored id is kept (and shown as missing) rather than silently cleared; saving other fields still works
    info = app_client.get(f"/api/characters/{c['id']}").json()["character_type_info"]
    assert info == {"id": "cosplay", "label": "cosplay", "missing": True}
    assert app_client.put(f"/api/characters/{c['id']}", json={"character_type": "cosplay", "trigger": "x"}).status_code == 200
    custom.write_text("{broken", encoding="utf-8")
    broken = app_client.get("/api/style/character-types").json()
    assert broken["types"] == [] and "not valid JSON" in broken["error"]


def analyzed(client, cid, folder, kind=INSP):
    client.post(url(cid, "/scan"), json={**kind, "folder": str(folder)})
    client.post(url(cid, "/analyze"), json=kind)
    assert wait_idle(client)["state"] == "done"


@needs_cuda
def test_guide_written_into_character_documents(client, cid, folder, tmp_path, monkeypatch):
    seen = tmp_path / "seen.jsonl"
    assert client.post(url(cid, "/guides"), json=CHAR).status_code == 400  # nothing analyzed yet
    analyzed(client, cid, folder, CHAR)
    monkeypatch.setenv("FAKE_LLAMA_SEEN", str(seen))

    assert client.post(url(cid, "/guides"), json=CHAR).json() == {"images": 3}
    status = wait_idle(client)
    assert status["state"] == "done" and status["job"] == "guide" and status["kind"] == "character", status
    assert status["guide_path"] == "guides/self-style-guide.md" and status["archived_path"] is None
    assert not status["server_running"]
    request = json.loads(seen.read_text().splitlines()[-1])
    assert request["guide"] == "character" and request["max_tokens"] == 3072
    assert "Character type: Influencer" in request["user"] and "Data from 3 analyzed images" in request["user"]

    [listed] = client.get(url(cid, "/guides"), params=CHAR).json()["items"]
    assert listed["id"] == status["guide_id"] and listed["image_count"] == 3 and listed["character_type"] == "influencer"
    assert listed["doc_path"] == "guides/self-style-guide.md"
    assert client.get(url(cid, "/guides"), params=INSP).json()["items"] == []
    guide = client.get(f"/api/style/guides/{listed['id']}").json()
    md = guide["markdown"]
    assert md.startswith("# Self-Style Guide — Bailey Bell") and "A Title The Model Added" not in md and "<think>" not in md
    assert "## Prompt Keywords" in md and "## Appendix: Data Summary" in md and guide["stats"]["image_count"] == 3

    # the guide is a character document: listed, readable, and found by the retrieval endpoint
    doc = client.get(f"/api/characters/{cid}/docs/file", params={"path": "guides/self-style-guide.md"}).json()
    assert doc["content"].startswith("---\ntype: self-style-guide\n") and doc["content"].endswith(md)
    files = {f["path"]: f for f in client.get(f"/api/characters/{cid}/docs").json()["files"]}
    assert files["guides/self-style-guide.md"]["type"] == "self-style-guide"
    hits = client.get(f"/api/characters/{cid}/context", params={"q": "Prompt Keywords fake text"}).json()["results"]
    assert hits and hits[0]["path"] == "guides/self-style-guide.md"

    # a second guide archives the first (unindexed) instead of overwriting it
    client.post(url(cid, "/guides"), json=CHAR)
    second = wait_idle(client)
    assert second["archived_path"].startswith("guides/archive/self-style-guide-")
    assert len(client.get(url(cid, "/guides"), params=CHAR).json()["items"]) == 2
    hits = client.get(f"/api/characters/{cid}/context", params={"q": "Prompt Keywords fake text", "k": 50}).json()["results"]
    assert {h["path"] for h in hits} == {"guides/self-style-guide.md"}


@needs_cuda
def test_guide_truncated_and_failed(client, cid, folder, monkeypatch):
    analyzed(client, cid, folder)
    monkeypatch.setenv("FAKE_LLAMA_MODE", "guide_short")
    client.post(url(cid, "/guides"), json=INSP)
    status = wait_idle(client)
    assert status["state"] == "done" and "cut short" in status["message"] and status["guide_path"] == "guides/content-guide.md"
    md = client.get(f"/api/style/guides/{status['guide_id']}").json()["markdown"]
    assert "output limit" in md and "left out these sections: Gaps" in md

    monkeypatch.setenv("FAKE_LLAMA_MODE", "guide_error")
    client.post(url(cid, "/guides"), json=INSP)
    status = wait_idle(client)
    assert status["state"] == "error" and "HTTP 500" in status["error"] and status["guide_id"] is None
    assert len(client.get(url(cid, "/guides"), params=INSP).json()["items"]) == 1


@needs_cuda
def test_guide_cancel_is_prompt(client, cid, folder, monkeypatch):
    analyzed(client, cid, folder)
    monkeypatch.setenv("FAKE_LLAMA_MODE", "guide_slow")
    client.post(url(cid, "/guides"), json=INSP)
    end = time.time() + 20
    while client.get("/api/style/status").json()["state"] != "running" and time.time() < end:
        time.sleep(0.05)
    started = time.time()
    client.post("/api/style/cancel")
    status = wait_idle(client, timeout=20)
    assert status["state"] == "cancelled" and time.time() - started < 15
    assert not status["server_running"] and client.get(url(cid, "/guides"), params=INSP).json()["items"] == []


@needs_cuda
def test_cancel_stops_server(client, cid, folder):
    client.post(url(cid, "/scan"), json={**INSP, "folder": str(folder)})
    client.post(url(cid, "/analyze"), json=INSP)
    client.post("/api/style/cancel")
    status = wait_idle(client)
    assert status["state"] in ("cancelled", "done") and not status["server_running"]


@needs_cuda
def test_deleting_character_removes_style_data_but_not_files(client, cid, folder):
    analyzed(client, cid, folder)
    img = client.get(url(cid, "/images"), params=INSP).json()["items"][0]["id"]
    r = client.delete(f"/api/characters/{cid}")
    assert r.status_code == 200
    assert client.get(f"/api/style/images/{img}").status_code == 404
    assert (folder / "a.png").is_file()
