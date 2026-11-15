import json

import pytest
import torch

from app import config as C
from app.conditioning import Conditioning, PromptCache, tokenize
from app.jobs import new_random_seed


def test_settings_roundtrip(tmp_path):
    f = tmp_path / "settings.json"
    s = C.load_settings(f)  # defaults when missing
    assert s.seed == 42 and s.randomize_seed is False
    s2 = C.update_settings(s, {"seed": 123, "randomize_seed": True, "vae_path": ' "D:\\x\\vae.safetensors" '})
    C.save_settings(s2, f)
    s3 = C.load_settings(f)
    assert s3.seed == 123 and s3.randomize_seed is True and s3.vae_path == "D:\\x\\vae.safetensors"
    assert json.loads(f.read_text())["seed"] == 123
    assert s3.transformer_path == s.transformer_path  # untouched fields preserved


@pytest.mark.parametrize("patch", [{"seed": -1}, {"seed": 2**60}, {"seed": "abc"}, {"seed": True}, {"randomize_seed": "yes"}, {"bogus": 1},
                                   {"keep_in_ram": {"transformer": "no"}}, {"transformer_path": 5}])
def test_settings_validation(patch):
    with pytest.raises(C.SettingsError):
        C.update_settings(C.Settings(), patch)


def test_random_seed_range():
    seeds = {new_random_seed() for _ in range(50)}
    assert len(seeds) > 45 and all(0 <= s < 2**32 for s in seeds)


def _cond(prompt="a cat", fp="enc-A"):
    return Conditioning(prompt=prompt, embeds=torch.zeros(1, 512, 12, 2560, dtype=torch.bfloat16), mask=torch.ones(1, 512, dtype=torch.bool),
                        encoder_fingerprint=fp, encoder_filename="te.safetensors", prompt_tokens=2, truncated=False)


def test_cache_install_and_lookup():
    cache = PromptCache()
    tok = cache.invalidate()
    assert cache.install(_cond(), tok)
    assert cache.valid_for("a cat", "enc-A") is not None
    assert cache.valid_for("a cat ", "enc-A") is None  # edited prompt -> not valid
    assert cache.valid_for("a cat", "enc-B") is None  # different text encoder -> not valid
    # repeated lookups reuse the same entry (no re-encode needed)
    assert cache.valid_for("a cat", "enc-A") is cache.valid_for("a cat", "enc-A")


def test_cache_rejects_stale_encode_result():
    cache = PromptCache()
    tok = cache.invalidate()
    cache.invalidate()  # e.g. text encoder changed while encoding
    assert not cache.install(_cond(), tok)
    assert cache.get() is None


def test_reencode_replaces_and_invalidate_clears():
    cache = PromptCache()
    cache.install(_cond("one"), cache.invalidate())
    tok = cache.invalidate()
    assert cache.get() is None  # previous conditioning unavailable while encoding
    cache.install(_cond("two"), tok)
    assert cache.get().prompt == "two"


def test_cache_refuses_cuda_tensors():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    c = _cond()
    c.embeds = c.embeds.cuda()
    with pytest.raises(ValueError):
        PromptCache().install(c, 0)


def test_tokenize_matches_official_layout():
    from transformers import AutoTokenizer

    from app.config import Settings

    d = Settings().text_encoder_config_dir
    try:
        tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
    except Exception:
        pytest.skip("Qwen3-VL tokenizer assets not present")
    ids, mask, n, trunc = tokenize(tok, "a red fox")
    assert ids.shape == (1, 512 + 34) and mask.shape == ids.shape and mask.dtype == torch.bool
    assert not trunc and n == int(mask.sum()) - 34 - 5
    assert mask[0, -5:].all() and not mask[0, 34 + n:-5].any()  # padding sits between prompt and suffix
    _, _, _, trunc = tokenize(tok, "word " * 2000)
    assert trunc
