"""Stage 3: Qwen-Image VAE decode of Krea 2 latents (only after the transformer has left the GPU).

Matches the official `QwenAutoencoder.decode` + sampler post-processing (krea-2/autoencoder.py,
sampling.py): latents * latents_std + latents_mean, decode as a single-frame video, clamp to [-1, 1],
map to [0, 255] and truncate to uint8.

Memory: a full-frame BF16 decode was measured at ~4.4 GiB peak for 1024x1024 and ~13.8 GiB for 2048x2048
(RTX 5070, Windows). Larger than the free VRAM, the Windows driver silently spills to shared system memory
(~100x slower) instead of failing. The decode mode is therefore chosen up front from measured free VRAM:
full-frame when it fits, otherwise Diffusers' tiled decode (1024 px tiles, 128 px blended overlap). The
choice is reported in the result and image metadata - it is never a silent retry.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from .model_manager import Component, ModelManager

GiB = 1024**3
TILE = 1024
TILE_STRIDE = 896  # 128 px overlap, blended by Diffusers' tiled_decode


def estimate_full_decode_bytes(width: int, height: int) -> int:
    mp = width * height / (1024 * 1024)
    return int((1.25 + 3.15 * mp) * 1.15 * GiB)  # linear fit of the two measurements above + 15%


def decode_latents(manager: ModelManager, latents: torch.Tensor) -> tuple[Image.Image, dict]:
    if latents.device.type != "cpu":
        raise ValueError("decode_latents expects CPU latents (the transformer stage must have released the GPU)")
    if manager.gpu_owner is not None:
        raise RuntimeError(f"Cannot decode: {manager.gpu_owner.value} is still on the GPU")
    dev = manager.device
    h, w = latents.shape[2] * 8, latents.shape[3] * 8
    with manager.use(Component.VAE) as lm:
        vae = lm.module
        free, _ = torch.cuda.mem_get_info(dev)
        need = estimate_full_decode_bytes(w, h)
        tiled = need > free - GiB
        if tiled:
            vae.enable_tiling(tile_sample_min_height=TILE, tile_sample_min_width=TILE,
                              tile_sample_stride_height=TILE_STRIDE, tile_sample_stride_width=TILE_STRIDE)
        else:
            vae.disable_tiling()
        info = {
            "mode": f"tiled ({TILE}px tiles, {TILE - TILE_STRIDE}px blended overlap)" if tiled else "full frame",
            "tiled": tiled,
            "estimated_full_decode_bytes": need,
            "free_vram_bytes": int(free),
        }
        manager.stage_cb("Decoding" + (" (tiled)" if tiled else ""))
        with torch.inference_mode():
            mean = torch.tensor(vae.config.latents_mean, dtype=torch.bfloat16, device=dev).view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std, dtype=torch.bfloat16, device=dev).view(1, -1, 1, 1, 1)
            x = rearrange(latents.to(dev, dtype=torch.bfloat16), "b c h w -> b c 1 h w")
            x = x * std + mean
            img = vae.decode(x).sample
            img = rearrange(img, "b c 1 h w -> b c h w")
            img = img.clamp(-1, 1) * 0.5 + 0.5
            arr = rearrange(img * 255.0, "b c h w -> b h w c").to("cpu").byte().numpy()
            del x, img, mean, std
        vae.disable_tiling()
    info["peak_allocated_bytes"] = manager.status[Component.VAE].last_peak_allocated_bytes
    return Image.fromarray(np.ascontiguousarray(arr[0])), info
