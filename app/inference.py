"""Stage 2: Krea 2 Turbo flow-matching Euler sampling from cached conditioning.

`prepare`, `timesteps` and the Euler loop follow the official functional sampler (krea-2/sampling.py,
Apache-2.0). Turbo settings are fixed: 8 steps, CFG disabled (guidance 0.0, no unconditional pass),
constant timestep-shift mu = 1.15 (the README's recommended `--mu 1.15` for oss_turbo; Diffusers'
Krea2Pipeline also pins mu=1.15 for the distilled checkpoint).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from einops import rearrange, repeat

from .conditioning import NUM_LAYERS, TEXT_DIM, Conditioning
from .model_manager import Component, ModelManager

STEPS = 8
GUIDANCE = 0.0
MU = 1.15
PATCH = 2
LATENT_CHANNELS = 16
VAE_COMPRESSION = 8
SAMPLER = "flow-matching Euler (official Krea 2 functional sampler)"


@dataclass(frozen=True)
class SamplingSettings:
    steps: int = STEPS
    guidance: float = GUIDANCE
    mu: float = MU


def prepare(img, txtlen, patch, txtmask):
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((txtmask, imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return img, pos, mask


def timesteps(seq_len, steps, x1=None, x2=None, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


def initial_noise(seed: int, width: int, height: int, device: torch.device) -> torch.Tensor:
    """Seeded latent noise from a local generator on the inference device (global RNG untouched)."""
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(1, LATENT_CHANNELS, height // VAE_COMPRESSION, width // VAE_COMPRESSION, device=device, dtype=torch.bfloat16, generator=gen)


def check_conditioning_compat(cond: Conditioning, manager: ModelManager) -> None:
    info = manager.inspect(Component.TRANSFORMER)
    exp_layers = info.details.get("text_layers_expected")
    exp_dim = info.details.get("text_dim_expected")
    if cond.embeds.shape[2] != exp_layers or cond.embeds.shape[3] != exp_dim:
        raise ValueError(
            f"Cached conditioning ({cond.embeds.shape[2]} layers x {cond.embeds.shape[3]}) is incompatible with the "
            f"selected transformer (expects {exp_layers} x {exp_dim}). Re-encode the prompt."
        )
    if (exp_layers, exp_dim) != (NUM_LAYERS, TEXT_DIM):
        raise ValueError("Transformer expects a different text conditioning layout")


def sample_latents(
    manager: ModelManager,
    cond: Conditioning,
    width: int,
    height: int,
    seed: int,
    progress: Callable[[int, int], None] | None = None,
    settings: SamplingSettings = SamplingSettings(),
    adapters: list | None = None,
) -> torch.Tensor:
    """Load transformer (resident + streamed blocks), run Euler sampling, return CPU latents [1,16,H/8,W/8]."""
    if width % (VAE_COMPRESSION * PATCH) or height % (VAE_COMPRESSION * PATCH):
        raise ValueError(f"{width}x{height} must be multiples of {VAE_COMPRESSION * PATCH}")
    if settings.guidance != 0.0:
        raise ValueError("CFG is disabled for Krea 2 Turbo in this MVP")
    check_conditioning_compat(cond, manager)
    dev = manager.device
    with manager.use(Component.TRANSFORMER, width=width, height=height, adapters=adapters) as lm:
        model = lm.module
        manager.stage_cb("Sampling")
        with torch.inference_mode():
            txt = cond.embeds.to(dev, non_blocking=True)
            txtmask = cond.mask.to(dev, non_blocking=True)
            noise = initial_noise(seed, width, height, dev)
            x, pos, mask = prepare(noise, txt.shape[1], PATCH, txtmask)
            del noise
            ts = timesteps(x.shape[1], settings.steps, mu=settings.mu)
            n = len(ts) - 1
            for i, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
                t = torch.full((1,), tcurr, dtype=x.dtype, device=dev)
                v = model(img=x, context=txt, t=t, pos=pos, mask=mask)
                x = x + (tprev - tcurr) * v
                del v
                torch.cuda.synchronize(dev)
                if progress:
                    progress(i + 1, n)
            latents = rearrange(
                x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=PATCH, pw=PATCH,
                h=height // (VAE_COMPRESSION * PATCH), w=width // (VAE_COMPRESSION * PATCH),
            )
            latents = latents.to("cpu").contiguous()
            del x, pos, mask, txt, txtmask
        if not torch.isfinite(latents).all():
            raise RuntimeError("Sampling produced non-finite latents (NaN/Inf); result discarded")
    return latents
