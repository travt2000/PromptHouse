"""LoRA inspection (key formats), runtime application math, stacking, and model-manager integration."""

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from app import config
from app.lora import LoraLibrary, LoraPatch, inspect_lora, krea2_targets
from app.quant import FP8Linear

D = 6144  # Krea 2 width


def _krea_lora(path, keys: dict[str, tuple], rank=2, extra=None, dtype=torch.bfloat16, meta=None):
    """keys: lora key prefix -> (in, out) ; writes A/B with the given suffix style encoded in the prefix tuple."""
    sd = {}
    for k, (style, fin, fout) in keys.items():
        a, b = (".lora_A.weight", ".lora_B.weight") if style == "peft" else (".lora_down.weight", ".lora_up.weight")
        sd[k + a] = torch.randn(rank, fin, dtype=dtype)
        sd[k + b] = torch.randn(fout, rank, dtype=dtype)
    sd.update(extra or {})
    save_file(sd, str(path), metadata=meta)
    return str(path)


# ---------------------------------------------------------------------------------------------- inspection


@pytest.mark.parametrize("prefix_q,prefix_mlp,style,expect", [
    ("diffusion_model.blocks.0.attn.wq", "diffusion_model.blocks.5.mlp.down", "peft", "krea2 original"),
    ("base_model.model.blocks.0.attn.wq", "base_model.model.blocks.5.mlp.down", "peft", "krea2 original"),
    ("transformer.transformer_blocks.0.attn.to_q", "transformer.transformer_blocks.5.ff.down", "peft", "diffusers"),
    ("lora_unet_blocks_0_attn_wq", "lora_unet_blocks_5_mlp_down", "kohya", "kohya"),
    ("lora_transformer_transformer_blocks_0_attn_to_q", "lora_transformer_transformer_blocks_5_ff_down", "kohya", "kohya"),
])
def test_key_formats_resolve_to_the_same_modules(tmp_path, prefix_q, prefix_mlp, style, expect):
    mlp_in = krea2_targets()["blocks.5.mlp.down"][0]
    p = _krea_lora(tmp_path / "l.safetensors", {prefix_q: (style, D, D), prefix_mlp: (style, mlp_in, D)})
    info = inspect_lora(p)
    assert info.compatible, info.problems
    assert set(info.plan) == {"blocks.0.attn.wq", "blocks.5.mlp.down"}
    assert info.key_style == expect and info.rank == "2" and info.blocks_covered == 2
    assert info.gpu_bytes == 2 * 2 * (D + D) + 2 * 2 * (mlp_in + D)


def test_alpha_text_encoder_and_metadata(tmp_path):
    p = _krea_lora(tmp_path / "a.safetensors", {"lora_unet_blocks_1_attn_wk": ("kohya", D, krea2_targets()["blocks.1.attn.wk"][1])},
                   extra={"lora_unet_blocks_1_attn_wk.alpha": torch.tensor(1.0), "lora_te1_text_model_x.lora_down.weight": torch.zeros(2, 8)},
                   meta={"modelspec.title": "Test", "modelspec.trigger_phrase": "tok"})
    info = inspect_lora(p)
    assert info.compatible and info.has_alpha and info.ignored_text_encoder_tensors == 1
    assert any("text-encoder" in w for w in info.warnings) and info.trigger == "tok" and info.title == "Test"


@pytest.mark.parametrize("extra,needle", [
    ({"diffusion_model.blocks.0.attn.wq.hada_w1_a": torch.zeros(2, 2)}, "LoHa"),
    ({"diffusion_model.blocks.0.attn.wq.dora_scale": torch.zeros(1, D)}, "DoRA"),
    ({"diffusion_model.not_a_layer.lora_A.weight": torch.zeros(2, 4), "diffusion_model.not_a_layer.lora_B.weight": torch.zeros(4, 2)}, "do not exist"),
    ({"diffusion_model.blocks.1.attn.wq.lora_A.weight": torch.zeros(2, 100), "diffusion_model.blocks.1.attn.wq.lora_B.weight": torch.zeros(D, 2)}, "do not fit"),
    ({"diffusion_model.blocks.2.attn.wq.lora_A.weight": torch.zeros(2, D)}, "no B"),
])
def test_unsupported_or_mismatched_files_are_rejected(tmp_path, extra, needle):
    p = _krea_lora(tmp_path / "bad.safetensors", {"diffusion_model.blocks.0.attn.wq": ("peft", D, D)}, extra=extra)
    info = inspect_lora(p)
    assert not info.compatible and needle in " ".join(info.problems), info.problems


def test_missing_and_non_safetensors(tmp_path):
    assert "not found" in inspect_lora(str(tmp_path / "nope.safetensors")).problems[0]
    (tmp_path / "x.ckpt").write_bytes(b"0")
    assert not inspect_lora(str(tmp_path / "x.ckpt")).compatible


# ---------------------------------------------------------------------------------------------- runtime math (CPU)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(32, 48, bias=True)
        self.b = FP8Linear(48, 16, bias=False)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


TINY_TARGETS = {"a": (32, 48), "b": (48, 16)}


def _tiny_model():
    torch.manual_seed(0)
    m = Tiny()
    w = torch.randn(16, 48) * 0.1
    s = w.abs().max() / 400
    m.b.weight = (w / s).to(torch.float8_e4m3fn)
    m.b.weight_scale = s.reshape(())
    m.b.scale_value = float(s)
    m.a.to(torch.bfloat16)
    return m.eval().requires_grad_(False)


def _tiny_lora(path, seed, rank=4, alpha=None):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for name, (fin, fout) in TINY_TARGETS.items():
        sd[f"diffusion_model.{name}.lora_A.weight"] = torch.randn(rank, fin, generator=g, dtype=torch.float32) * 0.2
        sd[f"diffusion_model.{name}.lora_B.weight"] = torch.randn(fout, rank, generator=g, dtype=torch.float32) * 0.2
        if alpha is not None:
            sd[f"diffusion_model.{name}.alpha"] = torch.tensor(float(alpha))
    save_file(sd, str(path))
    return str(path), sd


def _reference(model, x, loras):
    """Merged-weight reference in fp32: W' = W + sum s * alpha/r * B @ A."""
    wa = model.a.weight.float().clone()
    wb = (model.b.weight.float() * model.b.scale_value).clone()
    for sd, s, alpha, r in loras:
        k = (alpha / r) if alpha is not None else 1.0
        wa += s * k * sd["diffusion_model.a.lora_B.weight"] @ sd["diffusion_model.a.lora_A.weight"]
        wb += s * k * sd["diffusion_model.b.lora_B.weight"] @ sd["diffusion_model.b.lora_A.weight"]
    h = torch.relu(x.float() @ wa.T + model.a.bias.float())
    return h @ wb.T


def test_patch_matches_merged_weights_and_stack_is_additive(tmp_path):
    model = _tiny_model()
    lib = LoraLibrary(targets=TINY_TARGETS)
    p1, sd1 = _tiny_lora(tmp_path / "one.safetensors", 1)
    p2, sd2 = _tiny_lora(tmp_path / "two.safetensors", 2, rank=8, alpha=4)
    x = torch.randn(5, 32, dtype=torch.bfloat16)
    base = model(x).clone()
    stack = [{"path": p1, "strength": 0.8, "enabled": True}, {"path": p2, "strength": -0.5, "enabled": True}]
    adapters = lib.prepare(stack, [])
    patch = LoraPatch(model, adapters, torch.device("cpu"))
    try:
        y = model(x).float()
    finally:
        patch.close()
    ref = _reference(model, x, [(sd1, 0.8, None, 4), (sd2, -0.5, 4, 8)])
    assert patch.patched_modules == 2
    assert torch.allclose(y, ref, atol=3e-2, rtol=3e-2), (y - ref).abs().max()
    assert not torch.allclose(y, base.float(), atol=1e-2)  # the LoRA actually changed the output
    # reversed order gives the same result (additive updates) up to rounding
    patch = LoraPatch(model, list(reversed(adapters)), torch.device("cpu"))
    y_rev = model(x).float()
    patch.close()
    assert torch.allclose(y, y_rev, atol=1e-2)
    # after close the base model is exactly restored and no hooks remain
    assert torch.equal(model(x), base)
    assert not model.a._forward_hooks and not model.b._forward_hooks


def test_disabled_and_zero_strength_entries_are_not_applied(tmp_path):
    model = _tiny_model()
    lib = LoraLibrary(targets=TINY_TARGETS)
    p1, _ = _tiny_lora(tmp_path / "one.safetensors", 1)
    p2, _ = _tiny_lora(tmp_path / "two.safetensors", 2)
    x = torch.randn(3, 32, dtype=torch.bfloat16)
    base = model(x).clone()
    adapters = lib.prepare([{"path": p1, "strength": 0.0, "enabled": True}, {"path": p2, "strength": 1.0, "enabled": False}], [])
    assert [l.info.filename for l, _ in adapters] == ["one.safetensors"]  # disabled entry is not even loaded
    patch = LoraPatch(model, adapters, torch.device("cpu"))
    assert patch.patched_modules == 0 and torch.equal(model(x), base)
    patch.close()


def test_library_caches_and_evicts(tmp_path):
    lib = LoraLibrary(targets=TINY_TARGETS)
    p1, _ = _tiny_lora(tmp_path / "one.safetensors", 1)
    p2, _ = _tiny_lora(tmp_path / "two.safetensors", 2)
    a = lib.prepare([{"path": p1, "strength": 1.0, "enabled": True}], [p1])[0][0]
    b = lib.prepare([{"path": p1, "strength": 0.3, "enabled": True}], [p1])[0][0]
    assert a is b  # strength change: no reload
    lib.prepare([{"path": p2, "strength": 1.0, "enabled": True}], [p1, p2])
    assert len(lib.snapshot()["loaded"]) == 2
    lib.prepare([{"path": p2, "strength": 1.0, "enabled": True}], [p2])  # p1 removed from the stack
    assert [i["filename"] for i in lib.snapshot()["loaded"]] == ["two.safetensors"]
    lib.release_all()
    assert lib.snapshot()["ram_bytes"] == 0


def test_incompatible_enabled_lora_fails_clearly(tmp_path):
    lib = LoraLibrary(targets=TINY_TARGETS)
    with pytest.raises(ValueError, match="not found"):
        lib.prepare([{"path": str(tmp_path / "gone.safetensors"), "strength": 1.0, "enabled": True}], [])
    assert lib.prepare([{"path": str(tmp_path / "gone.safetensors"), "strength": 1.0, "enabled": False}], []) == []


# ---------------------------------------------------------------------------------------------- settings


def test_stack_settings_validation():
    s = config.update_settings(config.Settings(), {"loras": [{"path": "a.safetensors", "strength": 0.75, "enabled": False}, {"path": "b.safetensors"}]})
    assert s.loras == [{"path": "a.safetensors", "strength": 0.75, "enabled": False}, {"path": "b.safetensors", "strength": 1.0, "enabled": True}]
    for bad in ([{"path": "a.safetensors"}, {"path": "A.safetensors" if __import__("os").name == "nt" else "a.safetensors"}],
                [{"path": "a.safetensors", "strength": 9}], [{"path": "a.safetensors", "strength": "1"}],
                [{"path": ""}], [{"path": "a.safetensors", "enabled": 1}], [{"path": "a", "weight": 1}],
                [{"path": f"{i}.safetensors"} for i in range(config.MAX_LORAS + 1)], "nope"):
        with pytest.raises(config.SettingsError):
            config.update_settings(config.Settings(), {"loras": bad})


def test_stack_persists(tmp_path):
    f = tmp_path / "settings.json"
    s = config.update_settings(config.Settings(), {"loras": [{"path": "x.safetensors", "strength": -1.25, "enabled": True}]})
    config.save_settings(s, f)
    assert config.load_settings(f).loras == s.loras


# ---------------------------------------------------------------------------------------------- manager integration (GPU)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_manager_applies_stack_without_touching_base_weights(tmp_path, monkeypatch):
    from tests.test_model_manager import TinyDiT, _fake_info
    from app.model_manager import Component, LoadedModel, ModelManager, cuda_tensors_of

    m = ModelManager()
    torch.manual_seed(0)
    mod = TinyDiT().to(torch.bfloat16).eval().requires_grad_(False)
    builds = []

    def ensure_cpu(c):
        if m._models[c] is None:
            builds.append(c)
            m._models[c] = LoadedModel(module=mod, info=_fake_info(c.value), fingerprint=c.value,
                                       cpu_tensors={n: p.data for n, p in mod.named_parameters()})
            m.status[c].residency = "cpu"
        return m._models[c]

    monkeypatch.setattr(m, "_ensure_cpu", ensure_cpu)
    targets = {n: (256, 256) for n, x in mod.named_modules() if isinstance(x, nn.Linear)}
    lib = LoraLibrary(targets=targets)
    g = torch.Generator().manual_seed(3)
    sd = {}
    for n in targets:
        sd[f"diffusion_model.{n}.lora_A.weight"] = torch.randn(8, 256, generator=g) * 0.1
        sd[f"diffusion_model.{n}.lora_B.weight"] = torch.randn(256, 8, generator=g) * 0.1
    p = str(tmp_path / "s.safetensors")
    save_file(sd, p)
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    snapshot = {n: t.clone() for n, t in m._ensure_cpu(Component.TRANSFORMER).cpu_tensors.items()}

    try:
        with m.use(Component.TRANSFORMER) as lm:
            base = lm.module(x).clone()
        adapters = lib.prepare([{"path": p, "strength": 1.0, "enabled": True}], [p])
        with m.use(Component.TRANSFORMER, adapters=adapters) as lm:
            with_lora = lm.module(x).clone()
        st = m.status[Component.TRANSFORMER]
        assert st.adapters["patched_modules"] == 5 and st.adapters["vram_bytes"] == 5 * 2 * 8 * 512
        assert st.streaming["lora_bytes"] == st.adapters["vram_bytes"]

        # same stack while streaming blocks -> identical output
        real_plan = m._plan_transformer

        def force_stream(lm, w, h, extra_resident_bytes=0):
            plan = real_plan(lm, w, h, extra_resident_bytes)
            plan.update(streamed_blocks=[1, 2, 3], streamed_set={1, 2, 3}, resident_set={0}, resident_blocks=1)
            return plan

        monkeypatch.setattr(m, "_plan_transformer", force_stream)
        with m.use(Component.TRANSFORMER, adapters=adapters) as lm:
            streamed = lm.module(x).clone()
        half = lib.prepare([{"path": p, "strength": 0.5, "enabled": True}], [p])
        with m.use(Component.TRANSFORMER, adapters=half) as lm:
            half_out = lm.module(x).clone()
        with m.use(Component.TRANSFORMER) as lm:  # stack removed again
            base_again = lm.module(x).clone()
        assert cuda_tensors_of(mod) == []
        for n, t in m._models[Component.TRANSFORMER].cpu_tensors.items():  # base weights never modified
            assert torch.equal(t, snapshot[n]), n
    finally:
        m.release_cpu(Component.TRANSFORMER)

    assert not torch.equal(base, with_lora)
    assert torch.equal(with_lora, streamed)
    assert not torch.equal(half_out, with_lora) and not torch.equal(half_out, base)
    assert torch.equal(base, base_again)
    assert builds == [Component.TRANSFORMER]  # the base model was built once for all stack changes
    assert all(not sub._forward_hooks for sub in mod.modules())


@cuda
def test_adapters_rejected_for_other_components():
    from app.model_manager import Component, ModelError, ModelManager

    m = ModelManager()
    with pytest.raises(ModelError):
        with m.use(Component.VAE, adapters=[(object(), 1.0)]):
            pass
