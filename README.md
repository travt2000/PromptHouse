# Krea 2 Turbo — local FastAPI image generator

A local, single-user web app that runs **Krea 2 Turbo** text-to-image inference in its own Python process
with explicit, application-controlled GPU residency:

1. **Encode Prompt** — Qwen3-VL text encoder → GPU, encode, conditioning cached in system RAM, encoder unloaded.
2. **Generate Image** — transformer → GPU, 8-step Euler sampling from the cached conditioning, unloaded;
   then VAE → GPU, decode, unloaded; PNG + JSON metadata saved to `outputs/`.

Only one model component is ever on the GPU. No ComfyUI, no external inference servers, no downloads.
Design details and measurements: [ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements

* Windows or Linux, NVIDIA GPU with BF16 support (tested: RTX 5070 12 GB, driver 610.88)
* Python 3.12, ~16 GB free system RAM for the FP8 transformer (+ ~4–8 GB while encoding)
* Model files on an SSD are strongly recommended (13 GB transformer; an HDD takes 1–2 min to read it)

## Installation

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
# optional, for tests and the browser check:  -r requirements-dev.txt
```

## Model assets

All paths are configurable in **Settings** (gear icon) and stored in `settings.json`. Defaults:

| Setting | Default path | Accepted |
|---|---|---|
| Transformer | `models/krea2turbo/krea2_turbo_fp8_scaled.safetensors` | single-file Krea 2 Turbo, original Krea key names, BF16/FP16/FP32 or scaled FP8 (per-tensor `weight_scale`) |
| Text encoder | `models/krea2turbo/qwen3vl_4b_fp8_scaled.safetensors` | single-file Qwen3-VL-4B, BF16 or scaled FP8 |
| Text encoder config dir | `models/krea2turbo/qwen3vl_4b_config/` | Qwen3-VL-4B-Instruct `config.json` + tokenizer files |
| VAE | `models/krea2turbo/qwen_image_vae.safetensors` | Qwen-Image VAE (original or Diffusers key names) |

A single `.safetensors` text encoder does not contain its tokenizer/config. Put these small files (≈11 MB,
no weights) in the config directory — they are the files of
`https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct`:
`config.json, tokenizer.json, tokenizer_config.json, vocab.json, merges.txt` (`generation_config.json`,
`chat_template.json` optional). Example:

```powershell
$d="models\krea2turbo\qwen3vl_4b_config"; mkdir $d -Force
foreach ($f in 'config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt') {
  curl.exe -sSfL -o "$d\$f" "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/main/$f" }
```

The app validates every path (component type, shapes, storage format, quantization metadata) from the
file header and refuses incompatible files; it never loads a different model or downloads anything.

## Running

```powershell
.venv\Scripts\python.exe run.py            # http://127.0.0.1:7860  (localhost only, one worker)
```

Workflow: type a prompt → **Encode Prompt** (state turns *Encoded*) → choose resolution →
**Generate Image**. Generate again (other seed/resolution) without re-encoding. Editing the prompt or
changing the text encoder marks the prompt *Not Encoded*. Seed (default 42) and *Randomize seed* are in
Settings; the seed actually used is shown and saved in the metadata.

### LoRA stack

The **LoRA** panel holds an ordered stack (up to 16): **+ Add LoRA** (file browser, starts in `models/loras/`),
per-entry enable checkbox, strength slider/number (−5…5; slider ±2), ↑/↓ or drag the ⋮⋮ handle to reorder,
× to remove. The stack is saved in `settings.json` and applies to the next **Generate** (a running job keeps
the stack it was started with). No re-encode is needed: LoRAs only affect the transformer.

* **Adapters are never merged into the base weights.** Each targeted layer gets a runtime hook adding
  `strength · alpha/rank · B(Ax)`; the BF16/FP8 base checkpoint is untouched, so adding, removing, re-weighting,
  disabling or reordering LoRAs never reloads it (with *Keep in RAM → Transformer* on, the default).
* LoRA files are cached in system RAM separately (shown under *System*); files removed from the stack are freed.
* Accepted: single `.safetensors`, standard LoRA (`lora_A/lora_B` or `lora_down/lora_up`, optional `alpha`),
  Krea-original (`diffusion_model.blocks.N.attn.wq`, AI-Toolkit `base_model.model.`), Diffusers
  (`transformer.transformer_blocks.N.attn.to_q`) or kohya (`lora_unet_blocks_N_attn_wq`) key names.
  Rejected with a message: LoHa/LoKr/DoRA/OFT, modules that don't exist in Krea 2, shape mismatches.
  Text-encoder parts of a LoRA are reported and **not applied**.
* Order does not change the result: LoRA updates are additive (it only matters for rounding).
* Used LoRAs (file, strength, rank) are recorded in the image metadata.

**Windows tip:** in NVIDIA Control Panel → Manage 3D settings → Program settings → `python.exe`, set
*CUDA – Sysmem Fallback Policy* to **Prefer No Sysmem Fallback**. Otherwise the driver silently spills
over-allocations into (very slow) shared system memory instead of raising an out-of-memory error. The app
plans memory to avoid this and flags it if it happens anyway.

## Tests and checks

```powershell
.venv\Scripts\python.exe -m pytest -q tests                 # 191 tests; GPU lifecycle tests use tiny modules
.venv\Scripts\python.exe scripts\smoke_test.py --repeat 2   # real encode -> sample -> decode, same modules as the app
.venv\Scripts\python.exe scripts\ui_e2e.py                  # drives the running UI in Edge (needs run.py)
.venv\Scripts\python.exe scripts\lora_check.py [--full]    # LoRA math on real FP8 layers; --full = real sampling
.venv\Scripts\python.exe scripts\compare_text_encoders.py A.safetensors B.safetensors
.venv\Scripts\python.exe scripts\make_bf16_reference.py IN_fp8.safetensors OUT_bf16.safetensors
```

Unit tests cover dimension calculation, seeds, settings persistence, prompt-cache invalidation and stale
results, cached-conditioning reuse, rejection of simultaneous GPU jobs, cleanup after exceptions/OOM,
stage separation/order, FP8 layer math, block streaming equivalence, RAM release, LoRA key formats/rejections,
LoRA math vs. merged weights, stacking, disable/zero strength, streaming + LoRA, no base reload on stack changes. They do **not** prove
Krea 2 generation — the smoke test and UI test do that with real weights.

## Verification status (this machine)

| Item | Status |
|---|---|
| FP8-scaled transformer + FP8-scaled text encoder → real images | ✅ verified (1024², 1360×768, 2048²) |
| Conditioning reused across generations, text encoder not re-run | ✅ verified (smoke test assertion, UI test) |
| One component on the GPU at a time; nothing left on CUDA after each stage | ✅ verified after every stage |
| BF16 text encoder (single file) | ✅ verified with a BF16 file **derived from the FP8 checkpoint** (`scripts/make_bf16_reference.py`), switched in the running app, image generated; conditioning matches FP8 (cos ≥ 0.9995) |
| Switching transformer path keeps conditioning; switching text encoder invalidates it | ✅ verified through the API |
| **BF16 transformer checkpoint** (`krea2_turbo_bf16.safetensors`) | ❌ **not verified — incomplete acceptance item.** The loader path is implemented (same code as FP8 minus `FP8Linear`), but no BF16 transformer file is present, and its ~24.5 GB of weights would need ~26 GB of free RAM for the RAM-resident streaming design; this machine typically has ~18 GB free. |
| LoRA: key formats, math on **real FP8 checkpoint layers** (blocks 0 and 27) on the GPU | ✅ `scripts/lora_check.py`: matches merged-weight reference within BF16 rounding (rel. 2.9e-3); 3 layouts identical; strength 0 bit-identical; base weights unchanged |
| LoRA panel (add/refuse incompatible/duplicate, strength, disable, reorder by arrows and drag, persist, remove) | ✅ Edge via Playwright |
| A *trained* Krea 2 LoRA in a full real generation through the UI | ✅ tested manually by the user — loaded a LoRA and generated; worked as expected |
| Automated full-pipeline LoRA check (`lora_check.py --full`) | not run — needs ~16 GB free RAM. Streaming + LoRA equivalence and "no base reload" are unit-tested with a stand-in model. |
| Real CUDA OOM | handling tested with simulated OOM only; memory planning avoided OOM in all real runs |

## Known limitations

* The transformer always streams part of its blocks on a 12 GB card (planned per resolution); speed is
  ~20 s per 1024² image after the first load, ~1.5–2 min at 2048².
* FP8 checkpoints run as weight-only FP8 (BF16 compute); FP8 tensor-core matmul is not used.
* Prompts longer than 512 tokens are truncated (official behavior); the UI shows when this happens.
* At large sizes the VAE decodes in 1024 px tiles with blended overlaps (reported, not silent).
* Fixed Turbo settings (8 steps, no CFG, mu 1.15); no negative prompts, img2img, batching or history.
* LoRA: transformer only (text-encoder LoRA weights are ignored and reported); standard LoRA only (no LoHa/LoKr/DoRA).
  Adapter tensors are fully GPU-resident during sampling and counted in the VRAM plan, so a large stack means
  fewer resident transformer blocks (slightly slower), never an over-allocation.
* Pinned the official Krea dependency `transformers==4.57.1`; other versions are untested.
