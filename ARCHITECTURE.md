# Architecture

## Environment verified (Phase 1)

| | |
|---|---|
| OS | Windows 11 Pro, WDDM driver 610.88 |
| GPU | NVIDIA GeForce RTX 5070, 12 GB (Blackwell, sm_120), BF16 supported |
| System RAM | 32 GB (≈14 GB typically used by other applications on this machine) |
| Model drive | D: is a SATA **HDD** (~200 MB/s): reading the 13 GB transformer takes ~60–100 s |
| Python / torch | 3.12.7 / torch 2.11.0+cu128 (Blackwell requires CUDA ≥ 12.8) |
| Libraries | transformers 4.57.1 (official Krea pin), diffusers 0.39.0 — see `requirements.txt` |

### Checkpoints found in `models/krea2turbo/` (headers inspected, not assumed)

| File | Layout | Storage |
|---|---|---|
| `krea2_turbo_fp8_scaled.safetensors` (13.1 GB) | **Original Krea key names** (`blocks.N.attn.wq`, `txtfusion.*`, …), 28 blocks, width 6144, `txtfusion.projector` [1,12] | 256 linear weights FP8 E4M3 + per-tensor F32 `weight_scale`; rest BF16. Header `_quantization_metadata` marks 96 layers `full_precision_matrix_mult`. No activation scales. |
| `qwen3vl_4b_fp8_scaled.safetensors` (4.9 GB) | `model.layers.N.*` (36 layers, hidden 2560) + `model.visual.*` vision tower | 252 linear weights FP8 E4M3 + per-tensor F32 scale + `comfy_quant` uint8 JSON markers; rest BF16. **No tokenizer/config inside.** |
| `qwen_image_vae.safetensors` (0.25 GB) | Original Wan-2.1/Qwen-Image VAE key names | BF16 |

The Qwen3-VL-4B-Instruct `config.json` + tokenizer files (≈11 MB, no weights) were placed once, by hand,
in `models/krea2turbo/qwen3vl_4b_config/`. The application never downloads anything.

## Reference implementation

The official `krea-ai/krea-2` repository (Apache-2.0) was read in full and is the reference; the Diffusers
`Krea2Pipeline` was used to cross-check it. They agree on every point used here:

* **Prompt encoding**: system-prompt template prefix (34 tokens) + prompt, right-padded to `512+34-5`,
  then the 5-token assistant suffix; Qwen3-VL forward with the boolean mask (mask-aware M-RoPE positions);
  hidden states `(2,5,…,35)` (index 0 = embeddings) stacked → `[1, 512, 12, 2560]`, prefix dropped.
* **Sampling**: `prepare` (patchify 2×2, text positions 0, image (h,w) positions), timesteps
  `linspace(1,0,steps+1)` shifted by `exp(mu)/(exp(mu)+(1/t-1))`, plain Euler `x += (t_prev-t)·v`.
  Turbo: 8 steps, CFG disabled (no unconditional pass), constant `mu=1.15`.
* **Decode**: `latents·std + mean` (Qwen-Image VAE constants), single-frame decode, clamp, `·255`, uint8.

`app/krea2_mmdit.py` is the official model definition, vendored with only `torch.compile` removed (no
Triton on Windows) and the attention kernel restricted to cuDNN/memory-efficient SDPA (never the math
kernel). Its parameter names match the checkpoint, so weights load with `strict=True` and no remapping.

## Component loading (single files, no Diffusers repositories)

| Component | Construction | Weights |
|---|---|---|
| Transformer | `SingleStreamDiT(KREA2_CONFIG)` on the meta device | FP8 layers → `FP8Linear`; `load_state_dict(strict=True, assign=True)` |
| Text encoder | `transformers.Qwen3VLModel._from_config(config)` under `accelerate.init_empty_weights` | keys remapped `model.*` → `language_model.*`; FP8 layers → `FP8Linear`; the vision tower is never loaded (text-only) |
| VAE | `diffusers.AutoencoderKLQwenImage()` (default config = Qwen-Image VAE) | original keys converted with Diffusers' own `convert_wan_vae_to_diffusers`; `strict=True` |

`app/checkpoints.py` reads only the safetensors header to detect component type, storage format
(bf16 / fp16 / fp32 / fp8_scaled), validates shapes against the Krea 2 / Qwen3-VL-4B / Qwen-Image VAE
configurations and validates the quantization metadata. Unsupported layouts (activation-scaled FP8,
block-wise or non-FP8 quantization formats, unscaled FP8, Diffusers-format transformer) are rejected with
a message — nothing is silently substituted.

### FP8 execution (weight-only)

`FP8Linear` keeps the FP8 tensor as stored (1 byte/param in RAM and VRAM). At matmul time it upcasts that
one layer's weight to BF16 (FP8 values are exact in BF16) and applies the per-tensor scale to the output
in fp32 opmath: `y = (x @ W_fp8ᵀ)·s + b`. This is mathematically the dequantized layer; activations are
never quantized, so every layer effectively runs "full precision matrix mult". FP8 tensor-core matmuls
are not used. Verified on the real text encoder: FP8 checkpoint uses **4.11 GiB** RAM/VRAM vs **7.49 GiB**
for a BF16 checkpoint, and the conditioning matches (per-vector cosine ≥ 0.9995).

## GPU residency (`app/model_manager.py`)

`ModelManager.use(component)` is the only way a component reaches the GPU. It refuses if any other
component (or any tensor of one) is on the GPU, uploads, yields, and in `finally`:
removes every uploaded tensor (restores CPU masters), runs `gc` + `synchronize` + `empty_cache`,
**verifies no parameter/buffer is on CUDA**, and only then marks the component unloaded.
Failures inside the stage still run that cleanup; the job runner runs a second cleanup after the
exception (and its traceback frames holding activations) has been dropped, then probes CUDA; an
unrecoverable context is reported as "restart required".

Stage order per generation (each acquires and fully releases the GPU):

```
Encode:   [Qwen3-VL → GPU] encode → embeds+mask to CPU → [unload]        → CPU prompt cache
Generate: [transformer → GPU] 8 Euler steps → latents to CPU → [unload]
          [VAE → GPU] decode → pixels to CPU → [unload] → save PNG + JSON
```

### Transformer block streaming

13.1 GB of FP8 weights do not fit a 12 GB card next to activations. Per generation the manager measures
device-wide free VRAM (`cudaMemGetInfo`), subtracts an activation estimate for the requested resolution
and a safety margin (1 GiB + 8 % of the card, calibrated from measurement), keeps all non-block weights
and as many of the 28 blocks as fit resident, and streams the rest:

* Each streamed block is packed once into **one exact-size host buffer** page-locked in place with
  `cudaHostRegister` (`pin_memory()` rounds each tensor to a power of two, ~40 % extra RAM here).
* Two fixed GPU slots form a ring buffer: block *k+1* is copied (one memcpy, ~23 GB/s) on a side stream
  into the free slot while block *k* computes from the other; parameters are views into the slot.
  CUDA events order "copy done" → compute and "compute done" → next copy into that slot.
* Streaming produces bit-identical results to full residency (unit-tested).

Measured on the RTX 5070: 1024² → 12/28 blocks resident, device VRAM peak 9.3 GiB, 8 steps in ~20 s;
2048² → 3/28 resident, peak 9.2 GiB.

### VAE decode

Full-frame decode peaks at ~4.4 GiB (1024²) and ~13.8 GiB (2048²). Under WDDM an over-allocation silently
spills to shared system memory (~100× slower) instead of raising OOM. The decode mode is therefore chosen
**before** decoding from measured free VRAM: full frame when it fits, otherwise Diffusers' tiled decode
(1024 px tiles, 128 px blended overlap). The mode is recorded in the result and the image metadata. The
manager also flags any stage whose peak allocation exceeded physical VRAM.

## System RAM policy

Per-component "keep in RAM" (Settings). Defaults: transformer **yes** (re-reading 13 GB from the HDD costs
~1–2 min per image), VAE yes, text encoder **no** (encoding is infrequent; RAM is tight on this machine).
Before loading, available RAM is checked; under pressure the manager releases other components' RAM
caches (never one on the GPU), largest first, records why, and otherwise refuses with a clear message.
Release really returns memory to the OS:

* module tensors are replaced by empty meta tensors (lingering references cannot pin gigabytes);
* page-locked buffers are unregistered before being freed;
* PyTorch's Windows CPU allocator (mimalloc) is configured to decommit freed memory
  (`app/__init__.py`), and release runs on the GPU-worker thread that allocated the weights
  (mimalloc defers cross-thread frees). Verified: server RSS returns to ~0.5 GiB after releasing all models.

## Prompt cache and jobs

Single-entry CPU cache (`app/conditioning.py`): `[1,512,12,2560]` BF16 + `[1,512]` bool. An encode request
invalidates it immediately; a result is installed only if no newer invalidation happened (stale-result
protection). It is valid only for the exact prompt text and the exact text-encoder fingerprint
(path, size, mtime, tokenizer files). Changing the text encoder invalidates it; changing the transformer
does not (the transformer's expected conditioning shape, read from its header, is checked at generate).

`app/jobs.py`: one `ThreadPoolExecutor(max_workers=1)`; a second GPU job while one runs gets HTTP 409.
Model-path changes are rejected while a job runs, otherwise executed on the worker thread.
FastAPI endpoints stay responsive during inference (`/api/system` measured at 2 ms mean, 26 ms max).

## LoRA adapters (`app/lora.py`)

* **Inspection** (header only, cached by path/size/mtime): keys are normalised to Krea 2 module names from
  three layouts — original (`diffusion_model.` / `model.diffusion_model.` / `base_model.model.` prefixes),
  Diffusers `Krea2Transformer2DModel` names (inverse of Diffusers' own Krea 2 LoRA converter), kohya flattened
  names — then validated against the transformer's 264 linear layers (built once on the meta device):
  A `[r, in]`, B `[out, r]`, scalar alpha. Unsupported adapter types and unknown modules make the file
  incompatible; text-encoder and non-LoRA tensors are reported as not applied.
* **RAM cache** (`LoraLibrary`): loaded files are kept as BF16 A/B plus `alpha/r`, keyed by fingerprint,
  independent of the base checkpoint. Entries no longer in the stack are evicted on the GPU-worker thread.
* **Application** (`LoraPatch`, attached inside `ModelManager.use(TRANSFORMER, adapters=...)`): for each
  targeted layer, all enabled non-zero adapters are concatenated into one `A [Σr, in]` / `B [out, Σr]` on the
  GPU (strength·alpha/r folded into B, in fp32 before the BF16 cast) and a forward hook adds `B(Ax)` to the
  layer output. This works identically for `nn.Linear` and `FP8Linear`, and for resident or streamed blocks
  (streaming swaps the base parameters; the adapter tensors are separate). Adapter bytes are added to the
  resident part of the VRAM plan (`streaming.lora_bytes`). On exit the hooks are removed and the adapter
  tensors dropped before the usual "nothing left on CUDA" verification.
* Because base weights are never modified, a stack change needs no reload and no "unmerge"; strength 0 and
  disabled entries are skipped entirely (bit-identical to no LoRA).
* The generate request captures the stack at submit time; the PUT `/api/loras` endpoint replaces the whole
  ordered stack and is allowed while a job runs.
