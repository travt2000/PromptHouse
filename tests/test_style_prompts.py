"""Prompts written from the character's guides: guide extraction, request building, reply parsing, and the
end-to-end job against the fake llama-server (text-only: started without the vision projector)."""
import json

import pytest

from app.style import prompts as sp
from app.style.character_types import CharacterType
from app.style.llama_server import build_command
from app.style.vsettings import VisionSettings, VisionSettingsError, check_ready, merge

from test_style_api import app_client, needs_cuda, wait_idle  # noqa: F401 - fixtures

SELF_GUIDE = """---
type: self-style-guide
tags: [guide, generated]
---
# Self-Style Guide — Bailey Bell

*Generated 2026-11-15 10:00 · 12 images · model x.gguf*

> **Note:** the model left out these sections: Mood.

## Overview
Bailey is a copper-haired, freckled creator. <!-- my private note -->

## Defining Features
Copper-red wavy hair, green eyes, light freckles.

## Prompt Keywords
- Appearance: copper-red wavy hair, green eyes, freckles

## Gaps & Uncertainties
Few full-body shots.

---

## Appendix: Data Summary
| hair | copper (80%) |
"""

CONTENT_GUIDE = """---
type: content-guide
---
# Content Guide — Bailey Bell

## Overview
Cozy lifestyle content.

## Content Ideas
1. Rainy café window, knit sweater, soft light.

## Prompt Building Blocks
- Light: golden hour, overcast window light

## Gaps
Not many night scenes.
"""


# ------------------------------------------------------------------ guide extraction
def test_guide_text_keeps_useful_sections_keywords_first():
    t = sp.guide_text(SELF_GUIDE, "guides/self-style-guide.md")
    assert t.startswith("# Self-Style Guide — Bailey Bell\n\n## Prompt Keywords")
    assert "## Defining Features" in t and "## Overview" in t
    for gone in ("Appendix", "copper (80%)", "Gaps", "Few full-body", "my private note", "Generated 2026", "Note:", "type:"):
        assert gone not in t
    assert not t.rstrip().endswith("---")


def test_guide_text_without_sections_is_used_whole():
    raw = "---\ntype: self-style-guide\n---\nRed hair, always in green. <!-- hidden -->\n"
    assert sp.guide_text(raw, "guides/self-style-guide.md") == "Red hair, always in green."


def test_load_guides_order_and_missing(tmp_path):
    assert sp.load_guides(tmp_path) == []
    (tmp_path / "guides").mkdir()
    (tmp_path / "guides" / "content-guide.md").write_text(CONTENT_GUIDE, encoding="utf-8")
    assert [g.kind for g in sp.load_guides(tmp_path)] == ["inspiration"]
    (tmp_path / "guides" / "self-style-guide.md").write_text(SELF_GUIDE, encoding="utf-8")
    gs = sp.load_guides(tmp_path)
    assert [(g.kind, g.path) for g in gs] == [("character", "guides/self-style-guide.md"),
                                              ("inspiration", "guides/content-guide.md")]


# ------------------------------------------------------------------ request
def _guides():
    return [sp.GuideSource("character", "guides/self-style-guide.md", sp.guide_text(SELF_GUIDE, "a.md")),
            sp.GuideSource("inspiration", "guides/content-guide.md", sp.guide_text(CONTENT_GUIDE, "b.md"))]


def test_build_messages():
    ctype = CharacterType("influencer", "Influencer", "Lifestyle creator.", "Keep it relatable.")
    system, user = sp.build_messages(_guides(), ctype, "  rainy city night ", 3, 20000)
    assert system == {"role": "system", "content": sp.system_prompt()}
    u = user["content"]
    assert "=== Self-Style Guide ===" in u and "=== Content Guide ===" in u and "copper-red" in u
    assert "Character type: Influencer." in u and "Keep it relatable." in u
    assert "Direction from the creator (follow it, within the guides): rainy city night" in u
    assert u.endswith("Write 3 prompts.")
    _, one = sp.build_messages(_guides()[:1], None, "", 1, 20000)
    assert "No specific direction" in one["content"] and "There is no Content Guide" in one["content"]
    assert one["content"].endswith("Write 1 prompt.") and "Character type: not set." in one["content"]
    with pytest.raises(sp.PromptsError):
        sp.build_messages([], None, "", 1, 20000)


def test_build_messages_trims_to_budget():
    big = [sp.GuideSource("character", "x", "## Prompt Keywords\nkeep me\n" + "\n".join(["filler line"] * 5000))]
    _, user = sp.build_messages(big, None, "", 2, 6000)
    assert len(user["content"]) < 6000 and "keep me" in user["content"] and "[…trimmed" in user["content"]


def test_budget_follows_settings():
    assert sp.budget_chars(VisionSettings(n_ctx=8192, prompt_max_tokens=1536)) == (8192 - 1536 - 512) * 3
    assert sp.budget_chars(VisionSettings(n_ctx=2048, prompt_max_tokens=1024)) == 3000


# ------------------------------------------------------------------ reply parsing
def test_parse_numbered_with_noise():
    text = ("<think>let me plan</think>Here are four prompts:\n\n"
            "1. **Rainy café:** A close-up of a woman with copper hair\n   by a rain-streaked window, soft light.\n"
            "2) \"A full-body shot on a city street at golden hour, warm grade.\"\n"
            "3. Prompt 3: An overhead shot of a cozy desk with a latte and notebook.\n"
            "4. A full-body shot on a city street at golden hour, warm grade.\n"
            "5. short\n\nEnjoy!")
    assert sp.parse_prompts(text) == [
        "A close-up of a woman with copper hair by a rain-streaked window, soft light.",
        "A full-body shot on a city street at golden hour, warm grade.",
        "An overhead shot of a cozy desk with a latte and notebook.",
    ]


def test_parse_bullets_and_paragraph_fallback():
    assert sp.parse_prompts("- First prompt about a beach at dawn.\n- Second prompt about a forest trail.") == [
        "First prompt about a beach at dawn.", "Second prompt about a forest trail."]
    assert sp.parse_prompts("```\nA single paragraph prompt about a rooftop.\n\nAnother one about a quiet library.\n```") == [
        "A single paragraph prompt about a rooftop.", "Another one about a quiet library."]
    assert sp.parse_prompts("<think>x</think>Sorry.") == []


# ------------------------------------------------------------------ text-only server
def test_text_only_server_skips_projector():
    s = VisionSettings(llama_server_path="srv", model_path="m.gguf", mmproj_path="p.gguf")
    assert "--mmproj" in build_command(s, 1234)
    cmd = build_command(s, 1234, vision=False)
    assert "--mmproj" not in cmd and "p.gguf" not in cmd and cmd[:3] == ["srv", "-m", "m.gguf"]


def test_check_ready_text_only(tmp_path):
    srv, model = tmp_path / "srv.exe", tmp_path / "m.gguf"
    srv.write_bytes(b"x")
    model.write_bytes(b"x")
    s = VisionSettings(llama_server_path=str(srv), model_path=str(model))
    assert check_ready(s, vision=False) == []
    assert any("mmproj" in p for p in check_ready(s))


def test_prompt_settings_validated():
    assert merge(None, {"prompt_max_tokens": 1024, "prompt_temperature": 1.1})["prompt_temperature"] == 1.1
    with pytest.raises(VisionSettingsError, match="prompt_max_tokens"):
        merge(None, {"prompt_max_tokens": 8000})
    with pytest.raises(VisionSettingsError, match="prompt_temperature"):
        merge(None, {"prompt_temperature": 3})


# ------------------------------------------------------------------ API, end to end against the fake server
@pytest.fixture
def pclient(app_client, fake_llama, model_files):  # noqa: F811
    # no mmproj: prompt writing must not need it
    r = app_client.put("/api/settings", json={"vision": {"llama_server_path": fake_llama, "model_path": model_files[0],
                                                         "startup_timeout_s": 30}})
    assert r.status_code == 200, r.text
    return app_client


def _put_doc(client, cid, path, content):
    r = client.put(f"/api/characters/{cid}/docs/file", json={"path": path, "content": content})
    assert r.status_code == 200, r.text


@needs_cuda
def test_prompts_end_to_end(pclient, tmp_path, monkeypatch):
    client = pclient
    cid = client.post("/api/characters", json={"name": "Bailey Bell", "character_type": "influencer"}).json()["id"]
    r = client.post(f"/api/characters/{cid}/prompts", json={"count": 2})
    assert r.status_code == 400 and "no guides yet" in r.json()["detail"]
    assert client.post(f"/api/characters/{cid}/prompts", json={"count": 99}).status_code == 422

    _put_doc(client, cid, "guides/self-style-guide.md", SELF_GUIDE)
    _put_doc(client, cid, "guides/content-guide.md", CONTENT_GUIDE)
    seen, argv = tmp_path / "seen.jsonl", tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_LLAMA_SEEN", str(seen))
    monkeypatch.setenv("FAKE_LLAMA_ARGV", str(argv))
    r = client.post(f"/api/characters/{cid}/prompts", json={"count": 3, "idea": "rainy city night"})
    assert r.json() == {"guides": ["guides/self-style-guide.md", "guides/content-guide.md"], "count": 3}
    s = wait_idle(client)
    assert s["state"] == "done" and s["job"] == "prompts" and s["kind"] is None and s["character_id"] == cid, s
    assert s["prompts"] == [f"A medium shot of the subject in scene number {i}, soft window light, warm tones." for i in (1, 2, 3)]
    assert s["message"] == "Wrote 3 prompts." and s["idea"] == "rainy city night"
    assert not s["server_running"] and client.get("/api/status").json()["busy"] is False
    assert "--mmproj" not in json.loads(argv.read_text())
    req = json.loads(seen.read_text().splitlines()[-1])
    assert req["prompts"] and req["max_tokens"] == 1536 and req["temperature"] == 0.9
    assert "rainy city night" in req["user"] and "copper-red wavy hair" in req["user"] and "Cozy lifestyle" in req["user"]
    assert "Appendix" not in req["user"] and "Character type: Influencer" in req["user"]

    # a reply without prompts is an error, not an empty success
    monkeypatch.setenv("FAKE_LLAMA_MODE", "prompts_empty")
    client.post(f"/api/characters/{cid}/prompts", json={"count": 1})
    s = wait_idle(client)
    assert s["state"] == "error" and "no prompts" in s["error"] and s["prompts"] is None


@needs_cuda
def test_prompts_cancel(pclient, monkeypatch):
    client = pclient
    cid = client.post("/api/characters", json={"name": "Cancel Me"}).json()["id"]
    _put_doc(client, cid, "guides/content-guide.md", CONTENT_GUIDE)
    monkeypatch.setenv("FAKE_LLAMA_MODE", "guide_slow")
    client.post(f"/api/characters/{cid}/prompts", json={"count": 1})
    import time
    end = time.time() + 20
    while client.get("/api/style/status").json()["state"] != "running" and time.time() < end:
        time.sleep(0.05)
    assert client.post("/api/encode", json={"prompt": "a cat"}).status_code == 409  # the GPU worker is held
    assert client.post(f"/api/characters/{cid}/prompts", json={"count": 1}).status_code == 409
    client.post("/api/style/cancel")
    assert wait_idle(client, 20)["state"] == "cancelled"
