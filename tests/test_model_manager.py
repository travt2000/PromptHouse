"""Residency rules of ModelManager.use(), exercised with tiny stand-in modules on the real GPU.

These do not prove Krea 2 inference (see scripts/smoke_test.py); they test the lifecycle logic."""

import pytest
import torch
from torch import nn

from app import checkpoints as ck
from app.model_manager import Component, LoadedModel, ModelError, ModelManager, cuda_tensors_of

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _fake_info(name):
    return ck.CheckpointInfo(path=name, filename=name, size_bytes=0, kind="x", expected_kind="x", compatible=True, format="bf16",
                             stored_precision="BF16", execution_precision="BF16", dequantized_at_load=False, dequant_note="")


class TinyDiT(nn.Module):
    """Stand-in with the same `blocks` structure the streaming planner uses."""

    def __init__(self):
        super().__init__()
        self.first = nn.Linear(256, 256)
        self.blocks = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)])

    def forward(self, x):
        x = self.first(x)
        for b in self.blocks:
            x = torch.tanh(b(x))
        return x


@pytest.fixture
def mgr(monkeypatch):
    m = ModelManager()
    torch.manual_seed(0)
    models = {c: nn.Linear(256, 256).to(torch.bfloat16) for c in Component}
    models[Component.TRANSFORMER] = TinyDiT().to(torch.bfloat16)

    def ensure_cpu(c):
        if m._models[c] is None:
            mod = models[c]
            m._models[c] = LoadedModel(module=mod, info=_fake_info(c.value), fingerprint=c.value,
                                       cpu_tensors={n: t for n, t in list(mod.named_parameters())})
            m._models[c].cpu_tensors = {n: p.data for n, p in mod.named_parameters()}
            m.status[c].residency = "cpu"
        return m._models[c]

    monkeypatch.setattr(m, "_ensure_cpu", ensure_cpu)
    yield m
    for c in Component:
        m.release_cpu(c)  # unregisters page-locked buffers before their memory can be reused


def test_component_is_on_gpu_only_inside_use(mgr):
    with mgr.use(Component.VAE) as lm:
        assert mgr.gpu_owner is Component.VAE
        assert all(p.is_cuda for p in lm.module.parameters())
        y = lm.module(torch.ones(1, 256, device="cuda", dtype=torch.bfloat16))
        assert y.is_cuda
    assert mgr.gpu_owner is None
    assert cuda_tensors_of(lm.module) == []
    assert mgr.status[Component.VAE].residency == "cpu"


def test_only_one_component_at_a_time(mgr):
    with mgr.use(Component.TEXT_ENCODER):
        with pytest.raises(ModelError):
            with mgr.use(Component.TRANSFORMER):
                pass
    # after release the next stage may proceed
    with mgr.use(Component.TRANSFORMER) as lm:
        assert all(p.is_cuda for p in lm.module.parameters())


def test_cleanup_runs_when_stage_fails(mgr):
    with pytest.raises(RuntimeError, match="boom"):
        with mgr.use(Component.TRANSFORMER) as lm:
            raise RuntimeError("boom")
    assert mgr.gpu_owner is None
    assert cuda_tensors_of(lm.module) == []
    assert mgr.post_failure_cleanup() is None
    with mgr.use(Component.VAE):  # GPU usable again
        pass


def test_release_from_ram_when_policy_says_so(mgr):
    mgr.keep_in_ram[Component.TEXT_ENCODER] = False
    with mgr.use(Component.TEXT_ENCODER) as lm:
        pass
    assert mgr._models[Component.TEXT_ENCODER] is None
    assert mgr.status[Component.TEXT_ENCODER].residency == "not_loaded"
    # even though `lm` (and the module object) is still referenced here, its weights were freed
    assert all(p.device.type == "meta" for p in lm.module.parameters())


def test_block_streaming_matches_fully_resident(mgr, monkeypatch):
    x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    with mgr.use(Component.TRANSFORMER) as lm:
        assert mgr.status[Component.TRANSFORMER].streaming["streamed_blocks"] == []
        ref = lm.module(x).clone()

    real_plan = mgr._plan_transformer

    def force_stream(lm, w, h, **kw):
        p = real_plan(lm, w, h, **kw)
        p.update(streamed_blocks=[1, 2, 3], streamed_set={1, 2, 3}, resident_set={0}, resident_blocks=1)
        return p

    monkeypatch.setattr(mgr, "_plan_transformer", force_stream)
    with mgr.use(Component.TRANSFORMER) as lm:
        assert not lm.module.blocks[2].weight.is_cuda  # streamed block not resident between forwards
        out1 = lm.module(x).clone()
        out2 = lm.module(x).clone()  # second pass exercises wrap-around prefetch
        assert lm.module.blocks[0].weight.is_cuda
    assert torch.equal(ref, out1) and torch.equal(ref, out2)
    assert mgr.status[Component.TRANSFORMER].streaming["transfers"] >= 6
    assert cuda_tensors_of(lm.module) == []


def test_weights_preserved_across_offload(mgr):
    with mgr.use(Component.VAE) as lm:
        before = lm.module.weight.detach().float().cpu().clone()
    after = lm.module.weight.detach().float()
    assert torch.equal(before, after)
