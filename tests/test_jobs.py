"""JobRunner orchestration with the GPU stages replaced by fakes (no real inference here)."""

import threading
import time

import pytest
import torch

from app import jobs as J
from app.conditioning import Conditioning, PromptCache
from app.resolution import resolve


class FakeManager:
    def __init__(self):
        self.cuda_unrecoverable = False
        self.stage_cb = lambda s: None
        self.cleanups = 0
        self.status = {}

    def text_encoder_fingerprint(self):
        return "fp"

    def post_failure_cleanup(self):
        self.cleanups += 1
        return None

    def inspect(self, c):
        class I:
            filename = "x.safetensors"
            format = "fp8_scaled"
        return I()


def _cond(prompt):
    return Conditioning(prompt=prompt, embeds=torch.zeros(1, 512, 12, 2560, dtype=torch.bfloat16), mask=torch.ones(1, 512, dtype=torch.bool),
                        encoder_fingerprint="fp", encoder_filename="te", prompt_tokens=1, truncated=False)


def wait(job, timeout=10):
    t = time.time()
    while job.status not in ("complete", "error"):
        assert time.time() - t < timeout
        time.sleep(0.01)


@pytest.fixture
def env(monkeypatch, tmp_path):
    calls = []
    sampled_adapters = []
    gate = threading.Event()
    gate.set()

    def fake_encode(manager, prompt):
        calls.append("encode")
        manager.stage_cb("Encoding prompt")
        gate.wait(5)
        return _cond(prompt)

    def fake_sample(manager, cond, w, h, seed, progress=None, adapters=None):
        calls.append(("sample", cond.id, seed))
        sampled_adapters.append(adapters)
        gate.wait(5)
        for i in range(8):
            progress(i + 1, 8)
        return torch.zeros(1, 16, h // 8, w // 8)

    def fake_decode(manager, latents):
        calls.append("decode")
        from PIL import Image
        return Image.new("RGB", (latents.shape[3] * 8, latents.shape[2] * 8)), {"mode": "full frame", "tiled": False}

    monkeypatch.setattr(J, "encode_prompt", fake_encode)
    monkeypatch.setattr(J, "sample_latents", fake_sample)
    monkeypatch.setattr(J, "decode_latents", fake_decode)
    monkeypatch.setattr(J, "save_image", lambda img, meta: ("20260101-000000-deadbeef", tmp_path / "x.png"))
    mgr = FakeManager()
    runner = J.JobRunner(mgr, PromptCache())
    runner.sampled_adapters = sampled_adapters
    return runner, calls, gate, mgr


def test_generate_requires_encoded_prompt(env):
    runner, calls, _, _ = env
    with pytest.raises(ValueError):
        runner.submit_generate("a cat", resolve("1:1", 1.0, 16), 42)
    assert calls == []


def test_generate_reuses_cached_conditioning_without_reencoding(env):
    runner, calls, _, _ = env
    wait(runner.submit_encode("a cat"))
    for seed in (1, 2):
        j = runner.submit_generate("a cat", resolve("16:9", 1.0, 16), seed)
        wait(j)
        assert j.status == "complete" and j.result["seed"] == seed and j.result["width"] == 1360
    assert calls.count("encode") == 1
    cid = runner.cache.get().id
    assert [c for c in calls if isinstance(c, tuple)] == [("sample", cid, 1), ("sample", cid, 2)]
    # sample strictly before decode for each generation
    assert calls == ["encode", ("sample", cid, 1), "decode", ("sample", cid, 2), "decode"]


def test_edited_prompt_is_not_generated(env):
    runner, _, _, _ = env
    wait(runner.submit_encode("a cat"))
    with pytest.raises(ValueError):
        runner.submit_generate("a cat wearing a hat", resolve("1:1", 1.0, 16), 1)


def test_no_simultaneous_gpu_jobs(env):
    runner, _, gate, _ = env
    wait(runner.submit_encode("a cat"))
    gate.clear()
    j = runner.submit_generate("a cat", resolve("1:1", 1.0, 16), 1)
    with pytest.raises(J.Busy):
        runner.submit_encode("a dog")
    with pytest.raises(J.Busy):
        runner.submit_generate("a cat", resolve("1:1", 1.0, 16), 2)
    gate.set()
    wait(j)
    assert not runner.busy


def test_failure_reports_error_and_cleans_up(env, monkeypatch):
    runner, _, _, mgr = env
    wait(runner.submit_encode("a cat"))

    def oom(*a, **k):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")

    monkeypatch.setattr(J, "sample_latents", oom)
    j = runner.submit_generate("a cat", resolve("1:1", 1.0, 16), 1)
    wait(j)
    assert j.status == "error" and "out of memory" in j.error and j.result is None
    assert mgr.cleanups == 1 and not runner.busy
    assert runner.cache.get() is not None  # conditioning survives a failed generation


def test_encode_marks_previous_conditioning_unavailable(env):
    runner, _, gate, _ = env
    wait(runner.submit_encode("a cat"))
    gate.clear()
    j = runner.submit_encode("a dog")
    assert runner.cache.get() is None  # unavailable while the new encode runs
    gate.set()
    wait(j)
    assert runner.cache.get().prompt == "a dog"


def test_generate_uses_stack_snapshot_and_records_loras(env, tmp_path):
    from tests.test_lora import TINY_TARGETS, _tiny_lora
    from app.lora import LoraLibrary

    runner, _, gate, _ = env
    runner.loras = LoraLibrary(targets=TINY_TARGETS)
    p1, _ = _tiny_lora(tmp_path / "style.safetensors", 1)
    p2, _ = _tiny_lora(tmp_path / "off.safetensors", 2)
    wait(runner.submit_encode("a cat"))
    stack = [{"path": p1, "strength": 0.7, "enabled": True}, {"path": p2, "strength": 1.0, "enabled": False}]
    gate.clear()
    j = runner.submit_generate("a cat", resolve("1:1", 1.0, 16), 1, stack)
    stack[0]["strength"] = 2.0  # editing the stack after submit does not affect the running job
    gate.set()
    wait(j)
    assert j.status == "complete", j.error
    (adapters,) = runner.sampled_adapters
    assert [(l.info.filename, s) for l, s in adapters] == [("style.safetensors", 0.7)]
    assert [(x["file"], x["strength"]) for x in j.result["metadata"]["loras"]] == [("style.safetensors", 0.7)]


def test_generate_rejects_incompatible_enabled_lora_before_claiming_gpu(env, tmp_path):
    runner, calls, _, _ = env
    wait(runner.submit_encode("a cat"))
    with pytest.raises(ValueError, match="not found"):
        runner.submit_generate("a cat", resolve("1:1", 1.0, 16), 1, [{"path": str(tmp_path / "x.safetensors"), "strength": 1.0, "enabled": True}])
    assert not runner.busy and not any(isinstance(c, tuple) for c in calls)
