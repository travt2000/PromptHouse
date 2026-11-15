"""Stage 1: Krea 2 prompt encoding with Qwen3-VL and a single-entry CPU prompt cache.

Encoding reproduces the official Krea 2 `Qwen3VLConditioner.forward` (krea-2/encoder.py), which the
Diffusers `Krea2Pipeline.get_text_hidden_states` mirrors:
  * system-prompt template prefix + prompt, right-padded to 512 + 34 - 5 tokens, then the 5-token
    assistant suffix appended (so padding sits between prompt and suffix, masked out);
  * Qwen3VLModel forward with the boolean attention mask (mask-aware M-RoPE positions);
  * hidden states of layers (2, 5, ..., 35) stacked on dim 2, first 34 (system prefix) tokens dropped.
Result: embeds [1, 512, 12, 2560] BF16 and mask [1, 512] bool, stored on CPU.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

import torch

from .model_manager import Component, ModelManager

PROMPT_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_TOKENS = 34
SUFFIX_TOKENS = 5
MAX_LENGTH = 512
SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
NUM_LAYERS = len(SELECT_LAYERS)
TEXT_DIM = 2560


@dataclass
class Conditioning:
    prompt: str
    embeds: torch.Tensor  # [1, 512, 12, 2560] bf16, CPU
    mask: torch.Tensor  # [1, 512] bool, CPU
    encoder_fingerprint: str
    encoder_filename: str
    prompt_tokens: int
    truncated: bool
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)

    def describe(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "encoder": self.encoder_filename,
            "shape": list(self.embeds.shape),
            "dtype": str(self.embeds.dtype).replace("torch.", ""),
            "device": self.embeds.device.type,
            "prompt_tokens": self.prompt_tokens,
            "truncated": self.truncated,
            "bytes": self.embeds.numel() * self.embeds.element_size() + self.mask.numel(),
            "created": self.created,
        }


class PromptCache:
    """Single-entry cache. `invalidate()` makes conditioning unavailable before a new encode starts."""

    def __init__(self):
        self._lock = threading.Lock()
        self._entry: Conditioning | None = None
        self._generation = 0  # bumps on every invalidate; a finishing encode only installs if unchanged

    def invalidate(self) -> int:
        with self._lock:
            self._entry = None
            self._generation += 1
            return self._generation

    def install(self, cond: Conditioning, token: int) -> bool:
        if cond.embeds.device.type != "cpu" or cond.mask.device.type != "cpu":
            raise ValueError("Prompt cache only stores CPU tensors")
        with self._lock:
            if token != self._generation:
                return False  # invalidated meanwhile (settings changed / newer encode)
            self._entry = cond
            return True

    def get(self) -> Conditioning | None:
        with self._lock:
            return self._entry

    def valid_for(self, prompt: str, encoder_fingerprint: str) -> Conditioning | None:
        e = self.get()
        if e is None or e.prompt != prompt or e.encoder_fingerprint != encoder_fingerprint:
            return None
        return e


def tokenize(tokenizer, prompt: str) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
    text = [PROMPT_PREFIX + prompt]
    suffix = tokenizer([PROMPT_SUFFIX], return_tensors="pt")
    if suffix["input_ids"].shape[1] != SUFFIX_TOKENS:
        raise RuntimeError(f"Tokenizer mismatch: suffix is {suffix['input_ids'].shape[1]} tokens, expected {SUFFIX_TOKENS}")
    prefix_len = len(tokenizer(PROMPT_PREFIX)["input_ids"])
    if prefix_len != PREFIX_TOKENS:
        raise RuntimeError(f"Tokenizer mismatch: system prefix is {prefix_len} tokens, expected {PREFIX_TOKENS}")
    budget = MAX_LENGTH + PREFIX_TOKENS - SUFFIX_TOKENS
    full_len = len(tokenizer(text[0])["input_ids"])
    inputs = tokenizer(
        text, truncation=True, return_length=False, return_overflowing_tokens=False,
        padding="max_length", max_length=budget, return_tensors="pt",
    )
    input_ids = torch.cat([inputs["input_ids"], suffix["input_ids"]], dim=1)
    mask = torch.cat([inputs["attention_mask"].bool(), suffix["attention_mask"].bool()], dim=1)
    return input_ids, mask, min(full_len, budget) - PREFIX_TOKENS, full_len > budget


def encode_prompt(manager: ModelManager, prompt: str) -> Conditioning:
    """Load Qwen3-VL to GPU, encode, move results to CPU, unload. Never leaves CUDA tensors behind."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty")
    fingerprint = manager.text_encoder_fingerprint()
    with manager.use(Component.TEXT_ENCODER) as lm:
        tokenizer = lm.extra["tokenizer"]
        input_ids, mask, n_tokens, truncated = tokenize(tokenizer, prompt)
        manager.stage_cb("Encoding prompt")
        dev = manager.device
        with torch.inference_mode():
            ids_d, mask_d = input_ids.to(dev), mask.to(dev)
            out = lm.module(input_ids=ids_d, attention_mask=mask_d, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            expected = lm.module.config.text_config.num_hidden_layers + 1
            if len(hs) != expected:
                raise RuntimeError(f"Expected {expected} hidden states (embeddings + layers), got {len(hs)}")
            hiddens = torch.stack([hs[i] for i in SELECT_LAYERS], dim=2)[:, PREFIX_TOKENS:]
            embeds = hiddens.to("cpu", dtype=torch.bfloat16).contiguous()
            cmask = mask_d[:, PREFIX_TOKENS:].to("cpu").contiguous()
            del out, hs, hiddens, ids_d, mask_d
        if embeds.shape != (1, MAX_LENGTH, NUM_LAYERS, TEXT_DIM) or cmask.shape != (1, MAX_LENGTH):
            raise RuntimeError(f"Unexpected conditioning shape {tuple(embeds.shape)} / {tuple(cmask.shape)}")
        if not torch.isfinite(embeds).all():
            raise RuntimeError("Text encoder produced non-finite values")
        filename = lm.info.filename
    return Conditioning(
        prompt=prompt, embeds=embeds, mask=cmask, encoder_fingerprint=fingerprint,
        encoder_filename=filename, prompt_tokens=n_tokens, truncated=truncated,
    )
