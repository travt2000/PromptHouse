"use strict";

const $ = (id) => document.getElementById(id);
const GiB = 1024 ** 3;
const state = {
  res: null,              // resolution table from the server
  status: null,           // last /api/status
  settings: null,
  lastEncodeJob: null,    // {prompt, status, error}
  genJobId: null,
  pollingJob: false,
  loras: [],              // ordered stack [{path, strength, enabled, info}]
  loraMeta: { default_dir: "", strength_limit: 5, max_loras: 16 },
  loraLocked: null,       // the active character's LoRA (not part of the editable stack)
  workspaceSig: undefined, // JSON of the last seen workspace, to detect changes
  wsEpoch: 0,             // bumped by every workspace change made here; older status snapshots are dropped
  wsToken: 0,             // latest onWorkspace refresh (older async refreshes give up)
  characters: [],         // landing-page list
  editingId: null,        // character editor: id being edited, null = new
};

function store(key, value) { try { localStorage.setItem(key, value); } catch (_) {} }
function recall(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  let body = null;
  try { body = await r.json(); } catch (_) {}
  if (!r.ok) throw new Error((body && body.detail) || `${r.status} ${r.statusText}`);
  return body;
}
const fmtGiB = (b) => (b / GiB).toFixed(1);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------------------------ resolution
function currentResolution() {
  if (!state.res) return null;
  const key = `${$("aspect").value}|${Number($("megapixels").value)}|${Number($("multiple").value)}`;
  return state.res.entries[key] || null;
}
function renderResolution() {
  const r = currentResolution();
  if (!r) return;
  $("dims").textContent = `${r.width} × ${r.height}`;
  $("actual-mp").textContent = `${r.actual_megapixels.toFixed(3)} MP`;
  $("res-warnings").textContent = r.warnings.join(" ");
  store("res", JSON.stringify({ a: $("aspect").value, m: $("megapixels").value, k: $("multiple").value }));
}
async function initResolution() {
  state.res = await api("/api/resolutions");
  const fill = (el, items, fmt) => { el.innerHTML = items.map((v) => `<option value="${v}">${fmt(v)}</option>`).join(""); };
  fill($("aspect"), state.res.aspects, (v) => v);
  fill($("megapixels"), state.res.megapixels, (v) => `${Number(v).toFixed(1)} MP`);
  fill($("multiple"), state.res.multiples, (v) => `${v}`);
  const d = state.res.defaults;
  let saved = null;
  try { saved = JSON.parse(recall("res") || "null"); } catch (_) {}
  $("aspect").value = saved?.a ?? d.aspect;
  $("megapixels").value = saved?.m ?? String(d.megapixels);
  $("multiple").value = saved?.k ?? String(d.multiple);
  if (!currentResolution()) { $("aspect").value = d.aspect; $("megapixels").value = String(d.megapixels); $("multiple").value = String(d.multiple); }
  // option values for megapixels are e.g. "1" for 1.0 — normalise to the table's key format
  for (const id of ["aspect", "megapixels", "multiple"]) $(id).addEventListener("change", () => { renderResolution(); updateControls(); });
  renderResolution();
}

// ------------------------------------------------------------------------------------ encode state
function encodeState() {
  const s = state.status;
  const prompt = effectivePrompt();
  const job = s?.active_job;
  if (job && job.kind === "encode") return { label: "Encoding", cls: "busy" };
  const c = s?.conditioning;
  if (c && c.prompt === prompt && s.conditioning_valid_for_current_encoder) return { label: "Encoded", cls: "ok" };
  const le = state.lastEncodeJob;
  if (le && le.status === "error" && le.prompt === prompt) return { label: "Error", cls: "err" };
  return { label: "Not Encoded", cls: "" };
}
function updateControls() {
  const s = state.status;
  const busy = !!s?.busy;
  const enc = encodeState();
  $("encode-state").textContent = enc.label;
  $("encode-state").className = `pill ${enc.cls}`;
  const promptOk = $("prompt").value.trim().length > 0;
  const unavailable = !!s?.startup_error && !s?.models;
  $("encode-btn").disabled = busy || !promptOk || unavailable;
  $("generate-btn").disabled = busy || enc.label !== "Encoded" || !currentResolution() || unavailable;

  const c = s?.conditioning;
  let detail = "";
  if (enc.label === "Encoded") detail = `${c.prompt_tokens} tokens · cached on ${c.device.toUpperCase()} · ${c.encoder}${c.truncated ? " · TRUNCATED to 512 tokens" : ""}`;
  else if (enc.label === "Error") detail = state.lastEncodeJob.error;
  else if (c && c.prompt !== effectivePrompt()) detail = "Prompt changed since the last encode.";
  else if (c && !s.conditioning_valid_for_current_encoder) detail = "Text encoder changed; encode again.";
  $("encode-detail").textContent = detail;
  $("encode-detail").className = `hint ${enc.label === "Error" ? "err" : ""}`;

  const eff = effectivePrompt();
  const showPreview = eff !== $("prompt").value;
  $("prompt-preview").hidden = !showPreview;
  if (showPreview) $("prompt-preview").textContent = `Encodes as: ${eff}`;

  const st = state.settings;
  if (st) $("seed-mode").textContent = st.randomize_seed ? "Random seed per image" : `Fixed seed ${st.seed} (change in Settings)`;
}

// ------------------------------------------------------------------------------------ polling
async function pollStatus() {
  for (;;) {
    try {
      const epoch = state.wsEpoch;
      const st = await api("/api/status");
      if (epoch !== state.wsEpoch) { await sleep(100); continue; } // a workspace change happened meanwhile; this snapshot is stale
      state.status = st;
      const se = state.status.startup_error;
      $("startup-error").hidden = !se;
      $("startup-error").textContent = se || "";
      await onWorkspace();
      renderJobFromStatus();
      renderResidency();
      updateControls();
      renderGuidePrompts();
    } catch (e) { /* server restarting */ }
    await sleep(1000);
  }
}
async function pollSystem() {
  for (;;) {
    try {
      const m = await api("/api/system");
      if (m.gpu) {
        $("gpu-name").textContent = m.gpu.name;
        $("vram-text").textContent = `${fmtGiB(m.gpu.vram_used)} / ${fmtGiB(m.gpu.vram_total)} GiB`;
        $("vram-bar").style.width = `${(100 * m.gpu.vram_used / m.gpu.vram_total).toFixed(1)}%`;
        $("gpu-text").textContent = `${m.gpu.util_percent}%`;
        $("gpu-bar").style.width = `${m.gpu.util_percent}%`;
      } else {
        $("gpu-name").textContent = m.gpu_error || "GPU unavailable";
      }
      $("ram-text").textContent = `${fmtGiB(m.ram.used)} / ${fmtGiB(m.ram.total)} GiB`;
      $("ram-bar").style.width = `${m.ram.percent}%`;
      $("cpu-text").textContent = `${m.cpu.percent.toFixed(0)}%`;
      $("cpu-bar").style.width = `${m.cpu.percent}%`;
    } catch (_) {}
    await sleep(2000);
  }
}
function renderResidency() {
  const comps = state.status?.models?.components;
  if (!comps) { $("residency").innerHTML = ""; return; }
  const names = { text_encoder: "Text encoder", transformer: "Transformer", vae: "VAE" };
  const label = { not_loaded: "not loaded", loading_from_disk: "reading from disk", cpu: "in system RAM", loading_to_gpu: "loading to GPU", gpu: "on GPU", unloading_from_gpu: "unloading" };
  $("residency").innerHTML = Object.entries(comps).map(([k, v]) => {
    const cls = v.residency === "gpu" || v.residency === "loading_to_gpu" ? "gpu" : v.residency === "cpu" ? "cpu" : "";
    const ram = v.ram_bytes ? ` · ${fmtGiB(v.ram_bytes)} GiB RAM` : "";
    return `<span>${names[k]}</span><span class="${cls}">${label[v.residency] || v.residency}${ram}</span>`;
  }).join("") + loraResidencyRow();
}
function loraResidencyRow() {
  const lc = state.status?.lora_cache;
  if (!lc || !lc.loaded.length) return "";
  return `<span>LoRAs</span><span class="cpu">${lc.loaded.length} in system RAM · ${(lc.ram_bytes / 2 ** 20).toFixed(0)} MiB</span>`;
}

// ------------------------------------------------------------------------------------ jobs
function renderJobFromStatus() {
  const job = state.status?.active_job;
  if (job && job.kind === "generate" && !state.pollingJob) followGenerate(job.id); // page reloaded mid-job
  if (job && job.kind === "encode") { $("gen-status").textContent = "Encoding prompt"; $("gen-stage").textContent = job.stage; }
  else if (job && job.kind === "vision") { $("gen-status").textContent = "Local model (GPU busy)"; $("gen-stage").textContent = job.stage; }
  else if (!job && !state.pollingJob) {
    if (["Encoding prompt", "Local model (GPU busy)"].includes($("gen-status").textContent)) { $("gen-status").textContent = "Idle"; $("gen-stage").textContent = "-"; }
  }
}

async function encode() {
  const prompt = effectivePrompt(); // what the server will encode (trigger included)
  $("error-box").hidden = true;
  try {
    const job = await api("/api/encode", { method: "POST", body: JSON.stringify({ prompt: $("prompt").value, prepend_trigger: $("prepend").checked }) });
    state.lastEncodeJob = { prompt, status: "running" };
    state.status = { ...(state.status || {}), busy: true, active_job: job, conditioning: null };
    updateControls();
    for (;;) {
      await sleep(400);
      const j = await api(`/api/jobs/${job.id}`);
      $("gen-stage").textContent = j.stage;
      if (j.status === "complete" || j.status === "error") {
        state.lastEncodeJob = { prompt, status: j.status, error: j.error };
        if (j.status === "error") showError(j.error, j.restart_required);
        break;
      }
    }
    state.status = await api("/api/status");
    $("gen-status").textContent = "Idle";
    $("gen-stage").textContent = state.lastEncodeJob.status === "complete" ? "Ready to generate" : "Error";
    updateControls();
  } catch (e) {
    state.lastEncodeJob = { prompt, status: "error", error: e.message };
    showError(e.message);
    updateControls();
  }
}

async function generate() {
  const r = currentResolution();
  const body = { prompt: $("prompt").value, prepend_trigger: $("prepend").checked, aspect: r.aspect, megapixels: r.megapixels_target, multiple: r.multiple, width: r.width, height: r.height };
  $("error-box").hidden = true;
  try {
    const job = await api("/api/generate", { method: "POST", body: JSON.stringify(body) });
    $("gen-seed").textContent = job.request.seed;
    followGenerate(job.id);
  } catch (e) {
    showError(e.message);
  }
}

async function followGenerate(jobId) {
  state.pollingJob = true;
  $("viewer").classList.add("busy");
  $("overlay").hidden = false;
  $("progress-wrap").hidden = false;
  $("gen-status").textContent = "Generating";
  try {
    for (;;) {
      const j = await api(`/api/jobs/${jobId}`);
      $("gen-stage").textContent = j.stage;
      $("gen-seed").textContent = j.request.seed;
      $("overlay-text").textContent = j.stage;
      const p = j.progress;
      if (p) {
        const pct = j.status === "complete" ? 100 : (100 * p.step) / p.total;
        $("progress-bar").style.width = `${pct}%`;
        $("progress-text").textContent = j.stage === "Sampling" || p.step > 0 ? `step ${p.step}/${p.total}` : j.stage;
      }
      if (j.status === "complete") { showImage(j.result); $("gen-status").textContent = "Complete"; break; }
      if (j.status === "error") { showError(j.error, j.restart_required); $("gen-status").textContent = "Error"; break; }
      await sleep(500);
    }
  } catch (e) {
    showError(e.message);
    $("gen-status").textContent = "Error";
  } finally {
    state.pollingJob = false;
    $("viewer").classList.remove("busy");
    $("overlay").hidden = true;
    setTimeout(() => { if (!state.pollingJob) $("progress-wrap").hidden = true; }, 1500);
  }
}

function showImage(res) {
  const img = $("image");
  img.src = `/api/images/${res.image_id}`;
  img.hidden = false;
  $("placeholder").hidden = true;
  $("viewer").classList.remove("empty");
  const tiled = res.decode && res.decode.tiled ? ` · VAE ${res.decode.mode}` : "";
  const loras = (res.metadata.loras || []).map((l) => `${l.file.replace(/\.safetensors$/i, "")} ${l.strength}`).join(", ");
  const who = res.metadata.character ? `${res.metadata.character.name} · ` : "";
  $("image-info").textContent = `${who}${res.width} × ${res.height} · seed ${res.seed} · ${res.metadata.transformer}${loras ? ` · LoRA: ${loras}` : ""} · ${res.seconds}s${tiled}`;
  if (res.warnings && res.warnings.length) showError(res.warnings.join("\n"));
  const a = $("download");
  a.href = `/api/images/${res.image_id}?download=true`;
  a.hidden = false;
}
function showError(msg, restart) {
  const box = $("error-box");
  box.textContent = (restart ? "RESTART REQUIRED: " : "") + msg;
  box.hidden = false;
}

// ------------------------------------------------------------------------------------ settings
const MODEL_FIELDS = [
  ["transformer_path", "Krea 2 Turbo diffusion transformer (.safetensors)", "file"],
  ["text_encoder_path", "Qwen3-VL-4B text encoder (.safetensors)", "file"],
  ["text_encoder_config_dir", "Qwen3-VL config + tokenizer directory", "dir"],
  ["vae_path", "Qwen-Image VAE (.safetensors)", "file"],
];

function infoHtml(v, comp) {
  if (!v) return "";
  if (v.problems && v.problems.length) return `<div class="hint err">${v.problems.map(esc).join("<br>")}</div>`;
  if (v.format === undefined) {
    return `<div class="hint ok">Valid${v.details?.model_type ? ` (${esc(v.details.model_type)}, ${v.details.layers} layers)` : ""}</div>`;
  }
  const rows = [
    ["Checkpoint", v.filename],
    ["Detected format", v.format],
    ["Stored precision", v.stored_precision],
    ["Execution precision", v.execution_precision],
    ["Dequantized at load", v.dequantized_at_load ? "yes" : "no"],
    ["Handling", v.dequant_note],
    ["Weights to load", `${fmtGiB(v.weight_bytes_to_load)} GiB (file ${fmtGiB(v.size_bytes)} GiB)`],
  ];
  if (comp) {
    if (comp.last_vram_weight_bytes) rows.push(["Measured VRAM (weights, last load)", `${fmtGiB(comp.last_vram_weight_bytes)} GiB`]);
    if (comp.last_peak_allocated_bytes) rows.push(["Last peak allocated", `${fmtGiB(comp.last_peak_allocated_bytes)} GiB`]);
    if (comp.streaming) {
      const s = comp.streaming;
      rows.push(["Last VRAM plan", `${s.resident_blocks}/28 blocks resident, ${s.streamed_blocks.length} streamed; est. ${fmtGiB(s.estimated_weight_vram_bytes)} GiB weights + ${fmtGiB(s.activation_estimate_bytes)} GiB activations`]);
    }
    if (comp.adapters) {
      const a = comp.adapters;
      rows.push(["Last LoRA stack", a.applied.length
        ? `${a.applied.map((x) => `${x.file} (${x.strength})`).join(", ")}; ${a.patched_modules} layers patched, ${(a.vram_bytes / 2 ** 20).toFixed(0)} MiB VRAM`
        : "none"]);
    }
  }
  return `<div class="model-info">${rows.map(([k, val]) => `<span class="k">${esc(k)}</span><span>${esc(String(val))}</span>`).join("")}</div>`;
}
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function openSettings(section) {
  const data = await api("/api/settings");
  state.settings = data.settings;
  const comps = state.status?.models?.components || {};
  const compOf = { transformer_path: comps.transformer, text_encoder_path: comps.text_encoder, vae_path: comps.vae };
  $("model-fields").innerHTML = MODEL_FIELDS.map(([key, label, type]) => `
    <div class="model-field" data-key="${key}">
      <label>${esc(label)}</label>
      <div class="path-row">
        <input type="text" value="${esc(data.settings[key])}" data-path="${key}">
        <button class="btn small" data-validate="${key}">Validate</button>
        <button class="btn small" data-browse="${key}" data-type="${type}">Browse…</button>
      </div>
      <div class="info" id="info-${key}">${infoHtml(data.validation[key], compOf[key])}</div>
    </div>`).join("");
  $("seed").value = data.settings.seed;
  $("seed").max = data.max_seed;
  $("randomize").checked = data.settings.randomize_seed;
  document.querySelectorAll("[data-keep]").forEach((el) => { el.checked = !!data.settings.keep_in_ram[el.dataset.keep]; });
  renderVisionSettings(data.vision_problems);
  $("settings-msg").textContent = "";
  $("settings").hidden = false;
  if (section === "vision") $("vision-heading").scrollIntoView({ block: "start" });
}

async function validateField(key) {
  const path = document.querySelector(`[data-path="${key}"]`).value;
  const v = await api("/api/validate-path", { method: "POST", body: JSON.stringify({ kind: key, path }) });
  $(`info-${key}`).innerHTML = infoHtml(v);
}

async function saveSettings() {
  const patch = {};
  document.querySelectorAll("#model-fields [data-path]").forEach((el) => { patch[el.dataset.path] = el.value; });
  const seed = $("seed").value.trim();
  if (!/^\d+$/.test(seed)) { $("settings-msg").textContent = "Seed must be a non-negative integer."; $("settings-msg").className = "hint err"; return; }
  patch.seed = Number(seed);
  patch.randomize_seed = $("randomize").checked;
  patch.keep_in_ram = {};
  document.querySelectorAll("[data-keep]").forEach((el) => { patch.keep_in_ram[el.dataset.keep] = el.checked; });
  patch.vision = collectVision();
  try {
    const r = await api("/api/settings", { method: "PUT", body: JSON.stringify(patch) });
    state.settings = r.settings;
    for (const [key] of MODEL_FIELDS) $(`info-${key}`).innerHTML = infoHtml(r.validation[key]);
    const inv = r.invalidated.length ? ` Reloaded on next use: ${r.invalidated.join(", ")}.` : "";
    const enc = r.invalidated.includes("text_encoder") ? " Prompt conditioning invalidated — encode again." : "";
    renderVisionProblems(r.vision_problems);
    $("settings-msg").textContent = `Saved.${inv}${enc}`;
    $("settings-msg").className = "hint ok";
    if (!$("style").hidden) refreshStyleSummary();
    state.status = await api("/api/status");
    updateControls();
  } catch (e) {
    $("settings-msg").textContent = e.message;
    $("settings-msg").className = "hint err";
  }
}

// server-side file browser
let browseTarget = null;
// onPick: optional callback for the chosen path (navigation keeps the current target's callback)
// exts: file extensions to list (default .safetensors), e.g. ".gguf" or ".exe"
async function browse(key, type, path, onPick, exts) {
  if (!browseTarget || browseTarget.key !== key || onPick !== undefined || path === undefined) browseTarget = { key, type, onPick: onPick || null, exts: exts || ".safetensors" };
  let start = path;
  if (start === undefined) {
    const cur = document.querySelector(`[data-path="${key}"]`).value;
    start = cur ? (type === "dir" ? cur : cur.replace(/[\\/][^\\/]*$/, "")) : "";
  }
  let data;
  const ext = encodeURIComponent(browseTarget.exts);
  try { data = await api(`/api/browse?path=${encodeURIComponent(start)}&ext=${ext}`); }
  catch (_) { data = await api(`/api/browse?path=&ext=${ext}`); }
  $("browser-path").textContent = data.path || "Drives";
  const items = [];
  if (data.parent !== null && data.path) items.push(`<li class="dir" data-dir="${esc(data.parent)}">..</li>`);
  for (const e of data.entries) {
    if (e.type === "dir") items.push(`<li class="dir" data-dir="${esc(e.path)}">${esc(e.name)}</li>`);
    else if (type === "file") items.push(`<li data-file="${esc(e.path)}"><span>${esc(e.name)}</span><span class="hint">${fmtGiB(e.size)} GiB</span></li>`);
  }
  $("browser-list").innerHTML = items.join("") || `<li class="hint">${type === "dir" ? "No subfolders here" : `No ${esc(browseTarget.exts.replaceAll(",", " / "))} files here`}</li>`;
  $("browser-pick-dir").hidden = type !== "dir" || !data.path;
  $("browser-pick-dir").dataset.path = data.path;
  $("browser").hidden = false;
}
function pickPath(p) {
  $("browser").hidden = true;
  if (browseTarget.onPick) { browseTarget.onPick(p); return; }
  document.querySelector(`[data-path="${browseTarget.key}"]`).value = p;
  validateField(browseTarget.key);
}

// ------------------------------------------------------------------------------------ LoRA stack
// The stack is edited locally and sent whole to PUT /api/loras (add/remove/reorder/enable/strength).
const baseName = (p) => p.split(/[\\/]/).pop();
const stem = (p) => baseName(p).replace(/\.safetensors$/i, "");

function loraMetaHtml(e) {
  const v = e.info || {};
  if (v.problems && v.problems.length) return `<div class="lora-meta err">${esc(v.problems.join(" "))}</div>`;
  const parts = [];
  if (v.rank) parts.push(`rank ${esc(v.rank)}`);
  if (v.modules) parts.push(`${v.modules} layers${v.blocks_covered ? ` · ${v.blocks_covered}/28 blocks` : ""}`);
  if (v.key_style) parts.push(esc(v.key_style));
  if (v.gpu_bytes) parts.push(`${(v.gpu_bytes / 2 ** 20).toFixed(0)} MiB`);
  let html = parts.join(" · ");
  if (v.trigger) html += `<br>trigger: <span class="mono">${esc(v.trigger)}</span>`;
  if (v.warnings && v.warnings.length) html += `<br><span class="warn">${esc(v.warnings.join(" "))}</span>`;
  return `<div class="lora-meta">${html}</div>`;
}

function renderLoraCount() {
  const L = state.loraLocked;
  const n = state.loras.length + (L ? 1 : 0);
  const active = state.loras.filter((e) => e.enabled && e.strength !== 0 && !e.shadowed).length + (L && L.strength !== 0 ? 1 : 0);
  $("lora-count").textContent = n ? `${active} of ${n} active` : "";
}

function renderLockedLora() {
  const L = state.loraLocked;
  if (!L) { $("lora-locked").innerHTML = ""; return; }
  const bad = L.info && !L.info.compatible;
  $("lora-locked").innerHTML = `<div class="lora-item locked${bad ? " bad" : ""}" id="locked-lora">
      <div class="lora-top">
        <span class="lora-lock" title="Locked by the active character">🔒</span>
        <span class="lora-name" title="${esc(L.path)}">${esc(stem(L.path))}</span>
        <span class="lora-fixed" title="Strength (change it in the character)">${esc(L.strength)}</span>
      </div>
      <div class="lora-meta">Character LoRA for ${esc(L.character)}; strength is set in the character.</div>
      ${loraMetaHtml(L)}
    </div>`;
}

function renderLoras() {
  const lim = state.loraMeta.strength_limit || 5;
  const n = state.loras.length;
  renderLockedLora();
  $("lora-list").innerHTML = state.loras.map((e, i) => {
    const bad = e.info && !e.info.compatible;
    const sliderMax = Math.max(2, Math.ceil(Math.abs(e.strength)));
    const note = e.shadowed ? `<div class="lora-meta warn">Not used: the character's LoRA is the same file.</div>` : "";
    return `<li class="lora-item${e.enabled && !e.shadowed ? "" : " off"}${bad ? " bad" : ""}" data-i="${i}">
      <div class="lora-top">
        <span class="lora-handle" title="Drag to reorder">⋮⋮</span>
        <input type="checkbox" data-act="toggle" ${e.enabled ? "checked" : ""} title="${e.enabled ? "Disable" : "Enable"}" aria-label="Enable ${esc(stem(e.path))}">
        <span class="lora-name" title="${esc(e.path)}">${esc(stem(e.path))}</span>
        <button class="icon-btn sm" data-act="up" title="Move up" aria-label="Move up" ${i === 0 ? "disabled" : ""}>↑</button>
        <button class="icon-btn sm" data-act="down" title="Move down" aria-label="Move down" ${i === n - 1 ? "disabled" : ""}>↓</button>
        <button class="icon-btn sm" data-act="remove" title="Remove" aria-label="Remove">×</button>
      </div>
      <div class="lora-strength">
        <input type="range" data-act="slider" min="${-sliderMax}" max="${sliderMax}" step="0.05" value="${e.strength}" aria-label="Strength">
        <input type="number" data-act="num" min="${-lim}" max="${lim}" step="0.05" value="${e.strength}" aria-label="Strength value">
      </div>
      ${note}${loraMetaHtml(e)}
    </li>`;
  }).join("");
  renderLoraCount();
}

function setLoraMsg(text, cls = "") { $("lora-msg").textContent = text; $("lora-msg").className = `hint ${cls}`; }

let loraSaveTimer = null;
let loraSaveSeq = 0;
function saveLoras(delay = 0) {
  clearTimeout(loraSaveTimer);
  loraSaveTimer = setTimeout(async () => {
    const seq = ++loraSaveSeq;
    const stack = state.loras.map(({ path, strength, enabled }) => ({ path, strength, enabled }));
    try {
      const r = await api("/api/loras", { method: "PUT", body: JSON.stringify({ stack }) });
      if (seq !== loraSaveSeq) return; // a newer edit superseded this one
      applyLoraView(r, true);
      setLoraMsg(state.status?.busy ? "Saved — applies to the next image." : "");
    } catch (e) {
      setLoraMsg(e.message, "err");
      try { applyLoraView(await api("/api/loras")); } catch (_) {}
    }
  }, delay);
}

function applyLoraView(v, fromSave) {
  state.loraMeta = { default_dir: v.default_dir, strength_limit: v.strength_limit, max_loras: v.max_loras };
  state.loraLocked = v.locked || null;
  const ae = document.activeElement;
  const editing = ae && $("lora-list").contains(ae) && (ae.type === "range" || ae.type === "number");
  if (fromSave && editing) {
    // don't rebuild the control the user is dragging/typing in; just refresh validation info
    v.stack.forEach((e, i) => { if (state.loras[i] && state.loras[i].path === e.path) { state.loras[i].info = e.info; state.loras[i].shadowed = e.shadowed; } });
    renderLockedLora();
    return;
  }
  state.loras = v.stack;
  renderLoras();
}

async function addLora(path) {
  if (state.loraLocked && state.loraLocked.path.toLowerCase() === path.toLowerCase()) { setLoraMsg(`${baseName(path)} is the character's LoRA and is already applied.`, "warn"); return; }
  if (state.loras.some((e) => e.path.toLowerCase() === path.toLowerCase())) { setLoraMsg(`${baseName(path)} is already in the stack.`, "warn"); return; }
  if (state.loras.length >= (state.loraMeta.max_loras || 16)) { setLoraMsg("The LoRA stack is full.", "err"); return; }
  let info;
  try { info = await api("/api/validate-path", { method: "POST", body: JSON.stringify({ kind: "lora", path }) }); }
  catch (e) { setLoraMsg(e.message, "err"); return; }
  if (!info.compatible) { setLoraMsg(`${baseName(path)}: ${info.problems.join(" ")}`, "err"); return; }
  store("loraDir", path.replace(/[\\/][^\\/]*$/, ""));
  state.loras.push({ path, strength: 1, enabled: true, info });
  setLoraMsg("");
  renderLoras();
  saveLoras();
}

function wireLoras() {
  $("add-lora").addEventListener("click", () => browse("lora", "file", recall("loraDir") || state.loraMeta.default_dir || "", addLora));
  const list = $("lora-list");
  const idxOf = (el) => Number(el.closest("li.lora-item").dataset.i);
  list.addEventListener("click", (ev) => {
    const act = ev.target.dataset?.act;
    if (!["up", "down", "remove"].includes(act)) return;
    const i = idxOf(ev.target);
    if (act === "remove") state.loras.splice(i, 1);
    else {
      const j = act === "up" ? i - 1 : i + 1;
      if (j < 0 || j >= state.loras.length) return;
      [state.loras[i], state.loras[j]] = [state.loras[j], state.loras[i]];
    }
    renderLoras();
    saveLoras();
  });
  list.addEventListener("change", (ev) => {
    const act = ev.target.dataset?.act;
    if (act === "toggle") {
      state.loras[idxOf(ev.target)].enabled = ev.target.checked;
      renderLoras();
      saveLoras();
    } else if (act === "num") {
      const lim = state.loraMeta.strength_limit || 5;
      let v = Number(ev.target.value);
      if (ev.target.value.trim() === "" || !Number.isFinite(v)) v = 1;
      v = Math.max(-lim, Math.min(lim, Math.round(v * 100) / 100));
      state.loras[idxOf(ev.target)].strength = v;
      ev.target.blur();
      renderLoras();
      saveLoras();
    }
  });
  list.addEventListener("input", (ev) => {
    if (ev.target.dataset?.act !== "slider") return;
    const v = Math.round(Number(ev.target.value) * 100) / 100;
    state.loras[idxOf(ev.target)].strength = v;
    ev.target.closest("li.lora-item").querySelector('[data-act="num"]').value = v;
    renderLoraCount();
    saveLoras(300);
  });
  // drag to reorder, started only from the handle so the slider stays usable
  let dragFrom = null;
  list.addEventListener("mousedown", (ev) => { if (ev.target.classList.contains("lora-handle")) ev.target.closest("li").draggable = true; });
  list.addEventListener("dragstart", (ev) => {
    const li = ev.target.closest("li.lora-item");
    dragFrom = Number(li.dataset.i);
    li.classList.add("dragging");
    ev.dataTransfer.effectAllowed = "move";
    ev.dataTransfer.setData("text/plain", String(dragFrom));
  });
  list.addEventListener("dragover", (ev) => {
    const li = ev.target.closest("li.lora-item");
    if (dragFrom === null || !li) return;
    ev.preventDefault();
    list.querySelectorAll(".drag-over").forEach((x) => x.classList.remove("drag-over"));
    if (Number(li.dataset.i) !== dragFrom) li.classList.add("drag-over");
  });
  list.addEventListener("drop", (ev) => {
    const li = ev.target.closest("li.lora-item");
    if (dragFrom === null || !li) return;
    ev.preventDefault();
    const to = Number(li.dataset.i);
    if (to !== dragFrom) {
      const [moved] = state.loras.splice(dragFrom, 1);
      state.loras.splice(to, 0, moved);
      saveLoras();
    }
    dragFrom = null;
    renderLoras();
  });
  list.addEventListener("dragend", () => { if (dragFrom !== null) { dragFrom = null; renderLoras(); } });
}

// ------------------------------------------------------------------------------------ characters
// The server holds the workspace: null = landing page, {character: {...} | null} = generator (null = no character).
const activeCharacter = () => state.status?.workspace?.character || null;

// Mirror of app/characters.py:compose_prompt — the UI compares its result with the cached conditioning.
function composePrompt(prompt, trigger, prepend) {
  trigger = (trigger || "").trim();
  if (!prepend || !trigger) return prompt;
  const body = prompt.replace(/^\s+/, "");
  const re = new RegExp("^" + trigger.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "(?![0-9A-Za-z_])", "i");
  if (re.test(body)) return prompt;
  return `${trigger}, ${body}`;
}
function effectivePrompt() {
  const p = $("prompt").value;
  const c = activeCharacter();
  if (!c || !p.trim()) return p;
  return composePrompt(p, c.trigger, $("prepend").checked);
}

async function onWorkspace() {
  const ws = state.status?.workspace ?? null;
  const sig = JSON.stringify(ws);
  if (sig === state.workspaceSig) return;
  const prevId = state.workspaceSig ? JSON.parse(state.workspaceSig)?.character?.id : undefined;
  state.workspaceSig = sig;
  const token = ++state.wsToken;
  $("landing").hidden = !!ws;
  $("app").hidden = !ws;
  if (!ws) { await loadLanding(); return; }
  const c = ws.character;
  if (c && c.id !== prevId) {
    const saved = recall(`prepend:${c.id}`);
    $("prepend").checked = saved === null ? c.prepend_trigger : saved === "1";
  }
  renderCharBar();
  try {
    const v = await api("/api/loras");
    if (token !== state.wsToken) return; // superseded by a newer workspace change
    applyLoraView(v);
  } catch (_) {}
  updateControls();
}

// Apply a workspace change made from this page and refresh immediately (not waiting for the poll).
async function refreshWorkspace() {
  state.wsEpoch++;
  state.status = await api("/api/status");
  state.workspaceSig = undefined;
  await onWorkspace();
}

function renderCharBar() {
  const c = activeCharacter();
  document.querySelector(".char-bar").classList.toggle("none", !c);
  $("char-name").textContent = c ? c.name : "No character";
  $("edit-active-char").hidden = !c;
  $("docs-active-char").hidden = !c;
  $("style-active-char").hidden = !c;
  const bits = [];
  if (c && c.lora_path) bits.push(`LoRA ${stem(c.lora_path)} @ ${c.lora_strength}`);
  if (c && !c.lora_path) bits.push("No LoRA set");
  if (c && c.trigger) bits.push(`trigger “${c.trigger}”`);
  $("char-detail").textContent = c ? bits.join(" · ") : "Free generation — no character LoRA or trigger.";
  $("prepend-wrap").hidden = !(c && c.trigger);
  $("trigger-chip").textContent = c?.trigger || "";
  document.title = c ? `${c.name} · PromptHouse` : "PromptHouse";
  renderGuidePrompts();
}

function setLandingMsg(text, cls = "") { $("landing-msg").textContent = text; $("landing-msg").className = `hint ${cls}`; }

async function loadLanding(selectId) {
  let data;
  try { data = await api("/api/characters"); $("landing-error").hidden = true; }
  catch (e) { $("landing-error").textContent = e.message; $("landing-error").hidden = false; data = { characters: [] }; }
  state.characters = data.characters;
  const want = String(selectId ?? $("char-select").value ?? recall("lastCharacter") ?? "");
  const sel = $("char-select");
  sel.innerHTML = state.characters.length
    ? state.characters.map((c) => `<option value="${c.id}">${esc(c.name)}</option>`).join("")
    : `<option value="">No characters yet — create one</option>`;
  const pick = state.characters.find((c) => String(c.id) === want) || state.characters.find((c) => String(c.id) === recall("lastCharacter"));
  if (pick) sel.value = String(pick.id);
  renderCharSummary();
}

const selectedCharacter = () => state.characters.find((c) => String(c.id) === $("char-select").value) || null;

function renderCharSummary() {
  const c = selectedCharacter();
  const none = !c;
  for (const id of ["char-load", "char-edit", "char-docs", "char-style", "char-delete"]) $(id).disabled = none;
  if (!c) { $("char-summary").innerHTML = ""; return; }
  const rows = [];
  if (c.lora_path) {
    const bad = c.lora_info && !c.lora_info.compatible;
    rows.push(`<div><span class="k">LoRA</span> <span class="mono" title="${esc(c.lora_path)}">${esc(baseName(c.lora_path))}</span> @ ${esc(c.lora_strength)}</div>`);
    if (bad) rows.push(`<div class="hint err">${esc(c.lora_info.problems.join(" "))}</div>`);
  } else rows.push(`<div class="hint">No LoRA set</div>`);
  if (c.trigger) rows.push(`<div><span class="k">Trigger</span> <span class="mono">${esc(c.trigger)}</span>${c.prepend_trigger ? "" : ' <span class="hint">(not prepended by default)</span>'}</div>`);
  const t = c.character_type_info;
  if (t) rows.push(`<div><span class="k">Type</span> ${esc(t.label)}${t.missing ? ' <span class="hint warn">(not in config/character_types.json)</span>' : ""}</div>`);
  if (c.description) rows.push(`<div class="desc">${esc(c.description)}</div>`);
  $("char-summary").innerHTML = rows.join("");
  $("char-load").disabled = !!(c.lora_path && c.lora_info && !c.lora_info.compatible);
}

async function enterWorkspace(characterId) {
  setLandingMsg("");
  state.wsEpoch++;
  try {
    await api("/api/workspace", { method: "POST", body: JSON.stringify({ character_id: characterId }) });
    if (characterId !== null) store("lastCharacter", String(characterId));
    await refreshWorkspace();
  } catch (e) { setLandingMsg(e.message, "err"); }
}

async function leaveWorkspace() {
  state.wsEpoch++;
  try { await api("/api/workspace", { method: "DELETE" }); } catch (_) {}
  try { await refreshWorkspace(); } catch (e) { showError(e.message); }
}

async function deleteCharacter() {
  const c = selectedCharacter();
  if (!c) return;
  if (!confirm(`Delete the character "${c.name}"?\n\nThis removes the character record and its style libraries (image analyses and guide history). The LoRA file, the character's document folder (including written guides) and all image files are kept on disk.`)) return;
  try {
    const r = await api(`/api/characters/${c.id}`, { method: "DELETE" });
    setLandingMsg(`Deleted ${c.name}.${r.docs_kept ? ` Documents kept in ${r.docs_kept}` : ""}`, "ok");
    await loadLanding("");
  } catch (e) { setLandingMsg(e.message, "err"); }
}

// editor
function setEditorMsg(text, cls = "") { $("ce-msg").textContent = text; $("ce-msg").className = `hint ${cls}`; }

async function openEditor(c) {
  state.editingId = c ? c.id : null;
  await loadTypes();
  fillTypeSelect($("ce-type"), c?.character_type || "");
  $("ce-type-desc").textContent = typeDescription(c?.character_type || "");
  $("ce-title").textContent = c ? `Edit ${c.name}` : "New character";
  $("ce-name").value = c?.name || "";
  $("ce-lora").value = c?.lora_path || "";
  $("ce-strength").value = c ? c.lora_strength : 1;
  $("ce-trigger").value = c?.trigger || "";
  $("ce-prepend").checked = c ? c.prepend_trigger : true;
  $("ce-description").value = c?.description || "";
  setEditorMsg("");
  $("ce-lora-info").innerHTML = "";
  $("char-editor").hidden = false;
  if (c?.lora_path) checkEditorLora(false);
  $("ce-name").focus();
}

async function checkEditorLora(autofill) {
  const path = $("ce-lora").value.trim().replace(/^"|"$/g, "");
  if (!path) { $("ce-lora-info").innerHTML = ""; return true; }
  let v;
  try { v = await api("/api/validate-path", { method: "POST", body: JSON.stringify({ kind: "lora", path }) }); }
  catch (e) { $("ce-lora-info").innerHTML = `<span class="err">${esc(e.message)}</span>`; return false; }
  $("ce-lora-info").innerHTML = loraMetaHtml({ info: v });
  if (v.compatible && autofill) {
    if (!$("ce-trigger").value.trim() && v.trigger) $("ce-trigger").value = v.trigger;
    if (!$("ce-name").value.trim()) $("ce-name").value = stem(path);
  }
  return v.compatible;
}

async function saveEditor() {
  const strength = Number($("ce-strength").value);
  if ($("ce-strength").value.trim() === "" || !Number.isFinite(strength)) { setEditorMsg("LoRA strength must be a number.", "err"); return; }
  const body = {
    name: $("ce-name").value,
    lora_path: $("ce-lora").value.trim().replace(/^"|"$/g, ""),
    lora_strength: strength,
    trigger: $("ce-trigger").value,
    prepend_trigger: $("ce-prepend").checked,
    description: $("ce-description").value,
    character_type: $("ce-type").value,
  };
  if (!body.name.trim()) { setEditorMsg("Name is required.", "err"); $("ce-name").focus(); return; }
  try {
    const id = state.editingId;
    const c = await api(id === null ? "/api/characters" : `/api/characters/${id}`, { method: id === null ? "POST" : "PUT", body: JSON.stringify(body) });
    $("char-editor").hidden = true;
    if (!$("landing").hidden) { await loadLanding(c.id); setLandingMsg(`Saved ${c.name}.`, "ok"); }
    if (activeCharacter()?.id === c.id) await refreshWorkspace(); // character bar + locked LoRA
  } catch (e) { setEditorMsg(e.message, "err"); }
}

function wireCharacters() {
  $("char-select").addEventListener("change", () => { renderCharSummary(); setLandingMsg(""); });
  $("char-load").addEventListener("click", () => { const c = selectedCharacter(); if (c) enterWorkspace(c.id); });
  $("free-mode").addEventListener("click", () => enterWorkspace(null));
  $("char-new").addEventListener("click", () => openEditor(null));
  $("char-edit").addEventListener("click", () => { const c = selectedCharacter(); if (c) openEditor(c); });
  $("char-delete").addEventListener("click", deleteCharacter);
  $("switch-char").addEventListener("click", leaveWorkspace);
  $("edit-active-char").addEventListener("click", () => { const c = activeCharacter(); if (c) openEditor(c); });
  $("prepend").addEventListener("change", () => {
    const c = activeCharacter();
    if (c) store(`prepend:${c.id}`, $("prepend").checked ? "1" : "0");
    updateControls();
  });
  $("ce-close").addEventListener("click", () => { $("char-editor").hidden = true; });
  $("ce-cancel").addEventListener("click", () => { $("char-editor").hidden = true; });
  $("ce-save").addEventListener("click", saveEditor);
  $("ce-lora").addEventListener("change", () => checkEditorLora(true));
  $("ce-lora-clear").addEventListener("click", () => { $("ce-lora").value = ""; $("ce-lora-info").innerHTML = ""; });
  $("ce-lora-browse").addEventListener("click", () => {
    const cur = $("ce-lora").value.trim();
    const start = cur ? cur.replace(/[\\/][^\\/]*$/, "") : (recall("loraDir") || state.loraMeta.default_dir || "");
    browse("char_lora", "file", start, (p) => { $("ce-lora").value = p; store("loraDir", p.replace(/[\\/][^\\/]*$/, "")); checkEditorLora(true); });
  });
}

// ------------------------------------------------------------------------------------ character documents
// Markdown files in the character's folder. Saves send the file's mtime; the server refuses (409) if the
// file changed on disk meanwhile (e.g. edited in another editor), so nothing is silently overwritten.
const docs = { cid: null, files: [], cur: null }; // cur: {path, mtime, original}

const docsDirty = () => !!docs.cur && $("docs-text").value !== docs.cur.original;
function setDocsMsg(html, cls = "") { $("docs-msg").innerHTML = html; $("docs-msg").className = `hint ${cls}`; }
function confirmDiscard() { return !docsDirty() || confirm(`Discard unsaved changes to ${docs.cur.path}?`); }

async function openDocs(c) {
  docs.cid = c.id;
  docs.cur = null;
  $("docs-title").textContent = `${c.name} — documents`;
  $("docs-results").innerHTML = "";
  $("docs-q").value = "";
  showDoc(null);
  setDocsMsg("");
  $("docs").hidden = false;
  await refreshDocs();
}

function closeDocs() {
  if (!confirmDiscard()) return;
  $("docs").hidden = true;
  docs.cur = null;
}

async function refreshDocs() {
  try {
    const r = await api(`/api/characters/${docs.cid}/docs`);
    docs.files = r.files;
    $("docs-folder").textContent = r.folder;
    $("docs-folder").title = r.folder;
    renderDocList();
  } catch (e) { setDocsMsg(esc(e.message), "err"); }
}

function renderDocList() {
  const cur = docs.cur?.path;
  $("docs-list").innerHTML = docs.files.length ? docs.files.map((f) => `
    <li data-path="${esc(f.path)}" class="${f.path === cur ? "active" : ""}" title="${esc(f.title)}">
      <span class="doc-path">${esc(f.path)}</span>
      <span class="doc-meta">${esc(f.type)}${f.tags.length ? ` · ${esc(f.tags.join(", "))}` : ""} · ${(f.size / 1024).toFixed(1)} KB</span>
    </li>`).join("") : `<li class="hint">No Markdown files yet.</li>`;
}

function showDoc(d) {
  docs.cur = d ? { path: d.path, mtime: d.mtime, original: d.content } : null;
  $("docs-text").value = d ? d.content : "";
  $("docs-text").disabled = !d;
  $("docs-path").textContent = d ? d.path : "";
  $("docs-delete").disabled = !d;
  updateDocsDirty();
  renderDocList();
}

function updateDocsDirty() {
  const dirty = docsDirty();
  $("docs-dirty").textContent = dirty ? "● unsaved" : "";
  $("docs-save").disabled = !dirty;
}

async function loadDoc(path) {
  try {
    showDoc(await api(`/api/characters/${docs.cid}/docs/file?path=${encodeURIComponent(path)}`));
    setDocsMsg("");
  } catch (e) { setDocsMsg(esc(e.message), "err"); await refreshDocs(); }
}

async function saveDoc() {
  if (!docs.cur) return;
  const content = $("docs-text").value;
  try {
    const r = await api(`/api/characters/${docs.cid}/docs/file`, { method: "PUT", body: JSON.stringify({ path: docs.cur.path, content, expected_mtime: docs.cur.mtime }) });
    docs.cur = { path: r.path, mtime: r.mtime, original: content };
    updateDocsDirty();
    setDocsMsg(`Saved ${esc(r.path)}.`, "ok");
    await refreshDocs();
  } catch (e) {
    const reload = e.message.includes("changed on disk") || e.message.includes("deleted on disk");
    setDocsMsg(esc(e.message) + (reload ? ` <button class="btn small" id="docs-reload">Reload from disk (discard my edits)</button>` : ""), "err");
  }
}

async function newDoc() {
  if (!confirmDiscard()) return;
  let name = prompt("New document path inside the character's folder (e.g. notes/wardrobe.md):", "notes/");
  if (name === null) return;
  name = name.trim();
  if (!name || name.endsWith("/")) { setDocsMsg("Enter a file name.", "err"); return; }
  if (!/\.md$/i.test(name)) name += ".md";
  const title = baseName(name).replace(/\.md$/i, "").replace(/[-_]+/g, " ");
  const content = `# ${title.charAt(0).toUpperCase()}${title.slice(1)}\n\n`;
  try {
    const r = await api(`/api/characters/${docs.cid}/docs/file`, { method: "PUT", body: JSON.stringify({ path: name, content, expected_mtime: null }) });
    await refreshDocs();
    await loadDoc(r.path);
    $("docs-text").focus();
  } catch (e) { setDocsMsg(esc(e.message), "err"); }
}

async function deleteDoc() {
  if (!docs.cur || !confirm(`Delete ${docs.cur.path}? The file is removed from disk.`)) return;
  try {
    await api(`/api/characters/${docs.cid}/docs/file?path=${encodeURIComponent(docs.cur.path)}`, { method: "DELETE" });
    setDocsMsg(`Deleted ${esc(docs.cur.path)}.`, "ok");
    showDoc(null);
    await refreshDocs();
  } catch (e) { setDocsMsg(esc(e.message), "err"); }
}

async function searchDocs() {
  const q = $("docs-q").value.trim();
  if (!q) { $("docs-results").innerHTML = ""; return; }
  try {
    const r = await api(`/api/characters/${docs.cid}/context?k=5&q=${encodeURIComponent(q)}`);
    const warn = r.warnings.length ? `<div class="hint warn">${r.warnings.map((w) => `${esc(w.path)}: ${esc(w.warning)}`).join("<br>")}</div>` : "";
    $("docs-results").innerHTML = warn + (r.results.length ? r.results.map((h) => `
      <div class="docs-hit">
        <div class="src">${esc(h.path)}${h.headings.length ? ` › ${esc(h.headings.join(" › "))}` : ""} · lines ${h.start_line}–${h.end_line} · score ${h.score}</div>
        <pre>${esc(h.text)}</pre>
      </div>`).join("") : `<div class="hint">No matches in ${r.chunks_total} sections.</div>`);
  } catch (e) { $("docs-results").innerHTML = `<div class="hint err">${esc(e.message)}</div>`; }
}

function wireDocs() {
  $("char-docs").addEventListener("click", () => { const c = selectedCharacter(); if (c) openDocs(c); });
  $("docs-active-char").addEventListener("click", () => { const c = activeCharacter(); if (c) openDocs(c); });
  $("docs-close").addEventListener("click", closeDocs);
  $("docs-refresh").addEventListener("click", refreshDocs);
  $("docs-new").addEventListener("click", newDoc);
  $("docs-save").addEventListener("click", saveDoc);
  $("docs-delete").addEventListener("click", deleteDoc);
  $("docs-open").addEventListener("click", async () => {
    try { await api(`/api/characters/${docs.cid}/docs/open`, { method: "POST" }); } catch (e) { setDocsMsg(esc(e.message), "err"); }
  });
  $("docs-list").addEventListener("click", (ev) => {
    const li = ev.target.closest("li[data-path]");
    if (!li || li.dataset.path === docs.cur?.path || !confirmDiscard()) return;
    loadDoc(li.dataset.path);
  });
  $("docs-text").addEventListener("input", updateDocsDirty);
  $("docs-text").addEventListener("keydown", (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "s") { ev.preventDefault(); if (docsDirty()) saveDoc(); }
  });
  $("docs-msg").addEventListener("click", (ev) => { if (ev.target.id === "docs-reload" && docs.cur) loadDoc(docs.cur.path); });
  $("docs-search").addEventListener("click", searchDocs);
  $("docs-q").addEventListener("keydown", (ev) => { if (ev.key === "Enter") searchDocs(); });
  window.addEventListener("beforeunload", (ev) => { if (!$("docs").hidden && docsDirty()) { ev.preventDefault(); ev.returnValue = ""; } });
}

// ------------------------------------------------------------------------------------ vision model settings
// The `vision` section of settings.json (app/style/vsettings.py), edited in the Settings dialog.
const VISION_PATHS = [
  ["llama_server_path", "llama-server executable", ".exe"],
  ["model_path", "Vision model (.gguf)", ".gguf"],
  ["mmproj_path", "Vision projector — mmproj (.gguf, must match the model)", ".gguf"],
];
const VISION_MAIN = [["n_ctx", "Context size", "int"], ["n_gpu_layers", "GPU layers", "int"], ["extra_args", "Extra llama-server arguments", "text", "e.g. --n-cpu-moe 20"]];
const VISION_ADVANCED = [
  ["external_url", "Endpoint URL (instead of llama-server)", "text", "http://127.0.0.1:1234"], ["external_model", "Endpoint model name", "text"],
  ["startup_timeout_s", "Startup timeout (s)", "int"], ["request_timeout_s", "Request timeout (s)", "int"], ["port", "Port (0 = automatic)", "int"],
  ["image_max_edge", "Image max edge (px)", "int"], ["jpeg_quality", "JPEG quality", "int"], ["temperature", "Temperature", "float"],
  ["max_tokens", "Max output tokens", "int"], ["structured_mode", "Structured output", "select"],
  ["chat_template_kwargs", "Chat template arguments (JSON)", "text"], ["guide_max_tokens", "Guide max tokens", "int"],
  ["guide_temperature", "Guide temperature", "float"], ["prompt_max_tokens", "Prompt-writing max tokens", "int"],
  ["prompt_temperature", "Prompt-writing temperature", "float"],
];

function visionInput([key, label, type, placeholder]) {
  const v = state.settings?.vision?.[key] ?? "";
  if (type === "select") {
    const opts = ["json_schema", "json_object", "none"].map((o) => `<option value="${o}" ${o === v ? "selected" : ""}>${o === "none" ? "none (prompt only)" : o}</option>`).join("");
    return `<label>${esc(label)}<select data-vision="${key}" data-vtype="text">${opts}</select></label>`;
  }
  const num = type === "int" || type === "float";
  return `<label>${esc(label)}<input type="${num ? "number" : "text"}" ${type === "float" ? 'step="0.05"' : ""} data-vision="${key}" data-vtype="${type}" value="${esc(v)}" placeholder="${esc(placeholder || "")}"></label>`;
}

function renderVisionSettings(problems) {
  $("vision-fields").innerHTML = VISION_PATHS.map(([key, label, ext]) => `
    <div class="vision-field">
      <label>${esc(label)}</label>
      <div class="path-row"><input type="text" data-vision="${key}" data-vtype="text" value="${esc(state.settings?.vision?.[key] ?? "")}">
        <button class="btn small" data-vbrowse="${key}" data-ext="${ext}">Browse…</button></div>
    </div>`).join("") + `<div class="grid3">${VISION_MAIN.map(visionInput).join("")}</div>`;
  $("vision-advanced").innerHTML = VISION_ADVANCED.map(visionInput).join("");
  document.querySelector('[data-vision="release_krea_ram"]').checked = !!state.settings?.vision?.release_krea_ram;
  renderVisionProblems(problems);
}

function renderVisionProblems(problems) {
  $("vision-problems").textContent = problems && problems.length ? problems.join(" ") : "";
}

function collectVision() {
  const out = {};
  document.querySelectorAll("[data-vision]").forEach((el) => {
    const key = el.dataset.vision;
    if (el.type === "checkbox") out[key] = el.checked;
    else if (el.dataset.vtype === "int" || el.dataset.vtype === "float") out[key] = el.value.trim() === "" ? el.value : Number(el.value);
    else out[key] = el.value;
  });
  return out;
}

// ------------------------------------------------------------------------------------ character types
// config/character_types.json, re-read by the server on every request.
const ctypes = { list: [], error: null };

async function loadTypes() {
  try { const r = await api("/api/style/character-types"); ctypes.list = r.types; ctypes.error = r.error; }
  catch (e) { ctypes.list = []; ctypes.error = e.message; }
}

function fillTypeSelect(sel, current) {
  sel.innerHTML = `<option value="">— not set —</option>` + ctypes.list.map((t) => `<option value="${esc(t.id)}">${esc(t.label)}</option>`).join("");
  if (current && !ctypes.list.some((t) => t.id === current)) {
    // the stored id was removed or renamed in the JSON: keep showing it rather than silently clearing it
    sel.insertAdjacentHTML("beforeend", `<option value="${esc(current)}">${esc(current)} (no longer in the config)</option>`);
  }
  sel.value = current || "";
}
const typeInfo = (id) => ctypes.list.find((t) => t.id === id) || null;
function typeDescription(id) {
  if (ctypes.error) return `Character types file: ${ctypes.error}`;
  return typeInfo(id)?.description || "";
}

// ------------------------------------------------------------------------------------ style analysis
// Per-character image libraries read by a local vision model, and the guides written from them
// (app/style). Jobs run on the GPU worker, so generation waits while one runs.
const STYLE_KINDS = {
  character: {
    intro: "Images of this character. Each one is assessed for hair, colouring, build and what flatters them; together they become the Self-Style Guide.",
    placeholder: "Folder of images of the character, e.g. D:\\Characters\\Bailey\\refs",
    guide: "Self-Style Guide",
  },
  inspiration: {
    intro: "Images of content this character should make. Each one is read for format, outfit, pose, setting, light and mood — the person in it is never described. Together they become the Content Guide.",
    placeholder: "Folder of inspiration images",
    guide: "Content Guide",
  },
};
const STYLE_PAGE = 60;
const sty = { cid: null, name: "", kind: "character", filter: "", offset: 0, token: 0, selected: new Set(), lastPicked: null,
              data: null, status: null, polling: false, analyzed: 0, guide: null, localError: false };

// tiny DOM builder (text is never parsed as HTML)
function h(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== "") node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return node;
}
const pretty = (v) => String(v).replaceAll("_", " ");
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const styleUrl = (tail = "") => `/api/characters/${sty.cid}/style${tail}`;
const folderKey = () => `styleFolder:${sty.cid}:${sty.kind}`;

async function openStyle(c) {
  sty.cid = c.id;
  sty.name = c.name;
  sty.kind = recall("styleKind") === "inspiration" ? "inspiration" : "character";
  sty.guide = null;
  $("style-title").textContent = `${c.name} — Style`;
  $("style-log-text").hidden = true;
  $("style").hidden = false;
  await loadTypes();
  await refreshStyleSummary(true);
  showStyleKind(sty.kind);
  pollStyle();
}

function closeStyle() {
  $("style").hidden = true;
  if (!$("landing").hidden) loadLanding(sty.cid);
}

async function refreshStyleSummary(first) {
  try {
    const d = await api(styleUrl());
    sty.data = d;
    if (first) $("style-recursive").checked = d.recursive;
    const probs = d.problems || [];
    $("style-problems").hidden = !probs.length;
    $("style-problems").innerHTML = probs.length ? `${probs.map(esc).join("<br>")} <button class="btn small" id="style-fix">Open settings</button>` : "";
    $("style-model").textContent = probs.length ? "" : `Model: ${d.model_label}`;
    fillTypeSelect($("style-type"), d.character.character_type);
    $("style-type-desc").textContent = typeDescription(d.character.character_type);
    showStyleCounts(d.counts[sty.kind]);
    renderStyleStatus(d.status);
  } catch (e) { $("style-run-msg").textContent = e.message; }
}

function showStyleKind(kind) {
  sty.kind = kind;
  store("styleKind", kind);
  const k = STYLE_KINDS[kind];
  document.querySelectorAll("#style .tab").forEach((t) => t.classList.toggle("active", t.dataset.kind === kind));
  $("style-intro").textContent = k.intro;
  $("style-guide").textContent = `Write ${k.guide}`;
  $("style-folder").placeholder = k.placeholder;
  $("style-folder").value = recall(folderKey()) || "";
  $("style-scan-result").textContent = "";
  showStyleError("");
  clearStyleSelection();
  if (sty.data) showStyleCounts(sty.data.counts[kind]);
  loadStyleGallery(true);
  loadStyleGuides();
  renderStyleStatus(sty.status);
}

function showStyleCounts(c) {
  if (!c) return;
  if (sty.data) sty.data.counts[sty.kind] = c;
  sty.analyzed = c.ok;
  $("style-counts").textContent = `${plural(c.images, "image")} · ${c.ok} analyzed · ${c.pending} pending` + (c.failed ? ` · ${c.failed} failed` : "");
  updateGuideButton();
}

// ---- gallery
function badgeHtml(item) {
  const b = item.analysis_status === "ok" ? `<span class="s-badge ok">analyzed</span>` : item.analysis_status === "failed" ? `<span class="s-badge failed">failed</span>` : "";
  const dup = item.near_dup_of ? `<span class="s-badge dup" title="Looks like image #${item.near_dup_of}">similar</span>` : "";
  return b + dup;
}

async function loadStyleGallery(reset) {
  if (reset) { sty.offset = 0; $("style-gallery").innerHTML = ""; }
  const token = ++sty.token; // a newer load (tab switch, filter) wins
  const q = new URLSearchParams({ kind: sty.kind, limit: STYLE_PAGE, offset: sty.offset });
  if (sty.filter) q.set("status", sty.filter);
  let data;
  try { data = await api(`${styleUrl("/images")}?${q}`); } catch (e) { $("style-run-msg").textContent = e.message; return; }
  if (token !== sty.token) return;
  $("style-gallery").insertAdjacentHTML("beforeend", data.items.map((item) => `
    <div class="s-card${sty.selected.has(item.id) ? " selected" : ""}" data-id="${item.id}">
      <button class="open" title="${esc(item.path)}"><img src="/api/style/images/${item.id}/thumb" loading="lazy" alt=""></button>
      <label class="pick" title="Select (shift-click to select a range)"><input type="checkbox" ${sty.selected.has(item.id) ? "checked" : ""} aria-label="Select image ${item.id}"></label>
      <span class="badges">${badgeHtml(item)}</span>
    </div>`).join(""));
  sty.offset += data.items.length;
  $("style-more").hidden = sty.offset >= data.total;
  if (!data.total) $("style-gallery").innerHTML = `<div class="hint">${sty.filter ? "No images with this status." : "No images yet: scan a folder above."}</div>`;
  showStyleCounts(data.counts);
  showStyleSelection();
}

// refresh the badges of the cards already shown (during a batch), without losing the scroll position
async function refreshStyleBadges() {
  const shown = shownStyleIds();
  if (!shown.length || sty.filter) { loadStyleGallery(true); return; }
  const q = new URLSearchParams({ kind: sty.kind, limit: Math.min(500, Math.max(sty.offset, STYLE_PAGE)), offset: 0 });
  try {
    const data = await api(`${styleUrl("/images")}?${q}`);
    const byId = new Map(data.items.map((i) => [i.id, i]));
    for (const card of document.querySelectorAll("#style-gallery .s-card")) {
      const item = byId.get(Number(card.dataset.id));
      if (item) card.querySelector(".badges").innerHTML = badgeHtml(item);
    }
    showStyleCounts(data.counts);
  } catch (_) {}
}

// ---- selection & removal
const shownStyleIds = () => [...document.querySelectorAll("#style-gallery .s-card")].map((c) => Number(c.dataset.id));

function pickStyle(id, on, range) {
  let ids = [id];
  if (range && sty.lastPicked !== null) { // shift-click: everything between the last pick and this one
    const shown = shownStyleIds(), a = shown.indexOf(sty.lastPicked), b = shown.indexOf(id);
    if (a >= 0 && b >= 0) ids = shown.slice(Math.min(a, b), Math.max(a, b) + 1);
  }
  for (const i of ids) on ? sty.selected.add(i) : sty.selected.delete(i);
  sty.lastPicked = id;
  showStyleSelection();
}

function showStyleSelection() {
  for (const card of document.querySelectorAll("#style-gallery .s-card")) {
    const on = sty.selected.has(Number(card.dataset.id));
    card.classList.toggle("selected", on);
    card.querySelector("input").checked = on;
  }
  const n = sty.selected.size;
  const busyHere = !!sty.status?.active && sty.status.character_id === sty.cid;
  $("style-sel-count").textContent = n ? `${n} selected` : "";
  $("style-sel-clear").hidden = !n;
  $("style-sel-remove").disabled = !n || busyHere;
  $("style-sel-remove").title = busyHere ? "Wait for the running style job to finish." : "";
}

function clearStyleSelection() { sty.selected.clear(); sty.lastPicked = null; showStyleSelection(); }

async function removeStyleSelection() {
  const n = sty.selected.size;
  if (!n || !confirm(`Remove ${plural(n, "image")} from this library?\n\nTheir analyses are deleted too. The files on disk are not touched, and a rescan of the folder will skip them unless you tick "Re-add removed".`)) return;
  try {
    const r = await api(styleUrl("/remove"), { method: "POST", body: JSON.stringify({ image_ids: [...sty.selected] }) });
    $("style-scan-result").textContent = `Removed ${plural(r.removed, "image")}.`;
    clearStyleSelection();
    loadStyleGallery(true);
  } catch (e) { alert(e.message); }
}

// ---- scan
async function scanStyleFolder() {
  const out = $("style-scan-result");
  const folder = $("style-folder").value.trim();
  if (!folder) { out.textContent = "Enter or browse to a folder first."; return; }
  store(folderKey(), folder);
  out.textContent = "Scanning (hashing every image; a large folder on a slow disk takes a while)…";
  $("style-scan").disabled = true;
  try {
    const r = await api(styleUrl("/scan"), { method: "POST", body: JSON.stringify({ kind: sty.kind, folder, recursive: $("style-recursive").checked, include_removed: $("style-readd").checked }) });
    $("style-readd").checked = false; // one-shot, so a later rescan doesn't silently undo removals
    out.textContent = `Found ${r.found}: ${r.added} new, ${r.duplicates} already in this library` +
      (r.near_duplicates ? `, ${r.near_duplicates} look similar to an existing image` : "") +
      (r.restored ? `, ${r.restored} previously removed re-added` : "") +
      (r.skipped_removed ? `, ${r.skipped_removed} previously removed skipped` : "") +
      (r.unreadable.length ? `, ${r.unreadable.length} unreadable` : "") + ".";
    loadStyleGallery(true);
  } catch (e) { out.textContent = e.message; }
  $("style-scan").disabled = false;
}

// ---- jobs
function showStyleError(msg) {
  sty.localError = !!msg;
  $("style-run-error").textContent = msg || "";
  $("style-run-error").hidden = !msg;
}

async function startStyleRun(reanalyze) {
  showStyleError("");
  try {
    const r = await api(styleUrl("/analyze"), { method: "POST", body: JSON.stringify({ kind: sty.kind, reanalyze }) });
    if (r.reused) $("style-scan-result").textContent = `${plural(r.reused, "image")} reused an analysis of the same file from another character.`;
    if (!r.queued) { refreshStyleBadges(); return; }
  } catch (e) { showStyleError(e.message); return; }
  pollStyle();
}

async function writeStyleGuide() {
  showStyleError("");
  try { await api(styleUrl("/guides"), { method: "POST", body: JSON.stringify({ kind: sty.kind }) }); }
  catch (e) { showStyleError(e.message); return; }
  pollStyle();
}

function gpuBusyElsewhere() {
  const j = state.status?.active_job;
  return !!(j && j.kind !== "vision");
}

function updateGuideButton() {
  const active = !!sty.status?.active;
  $("style-guide").disabled = active || !sty.analyzed || gpuBusyElsewhere();
  $("style-guide-hint").textContent = !sty.analyzed
    ? "Analyze some images first; the guide is written from their analyses."
    : `Written from the ${plural(sty.analyzed, "image")} analyzed with the current model, shaped by the character type, into the character's documents (guides/).`;
}

function renderStyleStatus(s) {
  if (!s) return;
  sty.status = s;
  const active = !!s.active;
  const mine = s.character_id === sty.cid && s.kind === sty.kind;
  const busy = gpuBusyElsewhere();
  $("style-analyze").disabled = $("style-reanalyze").disabled = active || busy;
  $("style-cancel").disabled = !active;
  updateGuideButton();
  showStyleSelection();
  const what = s.job === "guide" ? `the ${STYLE_KINDS[s.kind]?.guide || "guide"}` : `${STYLE_KINDS[s.kind] ? (s.kind === "character" ? "character images" : "inspiration images") : "images"}`;
  let msg = "";
  const doing = s.job === "prompts" ? "writing image prompts" : `${s.job === "guide" ? "writing " : "analyzing "}${what}`;
  if (active && !mine) msg = `A local-model job is running (${doing}${s.character_id !== sty.cid ? " for another character" : ""}): ${s.message}. Only one runs at a time.`;
  else if (mine && s.state !== "idle") msg = s.message + (s.job === "analyze" && s.total ? ` (${s.done}/${s.total}${s.failed ? `, ${s.failed} failed` : ""}${s.reused ? `; ${s.reused} reused` : ""})` : "");
  else if (busy) msg = "Image generation is using the GPU; style jobs can start when it finishes.";
  $("style-run-msg").textContent = msg;
  $("style-run-msg").className = `hint ${mine && s.state === "error" ? "err" : ""}`;
  const pct = mine && s.total ? Math.round((100 * s.done) / s.total) : 0;
  $("style-bar").style.width = `${active || s.state === "done" ? pct : 0}%`;
  $("style-progress-text").textContent = mine && active && s.job === "analyze" && s.total ? `${s.done}/${s.total}` : "";
  const showErr = mine && !active && !!s.error;
  if (showErr) { $("style-run-error").textContent = s.error; $("style-run-error").hidden = false; }
  else if (!sty.localError) $("style-run-error").hidden = true;
}

async function pollStyle() {
  if (sty.polling) return;
  sty.polling = true;
  let sawActive = false, lastDone = -1;
  try {
    while (!$("style").hidden) {
      const s = await api("/api/style/status");
      renderStyleStatus(s);
      const here = s.character_id === sty.cid;
      sawActive ||= s.active;
      if (s.active && here && s.kind === sty.kind && s.job === "analyze" && s.done !== lastDone) { lastDone = s.done; refreshStyleBadges(); }
      if (!s.active) {
        if (sawActive && here) {
          await refreshStyleSummary();
          if (s.kind === sty.kind) { refreshStyleBadges(); loadStyleGuides(); }
          if (s.job === "guide" && s.state === "done" && s.guide_id) openStyleGuide(s.guide_id);
        }
        break;
      }
      await sleep(1500);
    }
  } catch (e) { $("style-run-msg").textContent = e.message; }
  sty.polling = false;
}

// ---- guides
async function loadStyleGuides() {
  const forKind = sty.kind, forCid = sty.cid;
  let items = [];
  try { items = (await api(`${styleUrl("/guides")}?kind=${forKind}`)).items; } catch (_) {}
  if (forKind !== sty.kind || forCid !== sty.cid) return;
  $("style-guides").innerHTML = items.map((g, i) => `
    <li>
      <span class="when">${esc(new Date(g.created_at * 1000).toLocaleString())}</span>
      <span class="hint">${plural(g.image_count, "image")} · ${esc(g.model_label)}${g.character_type ? ` · ${esc(typeInfo(g.character_type)?.label || g.character_type)}` : ""}</span>
      ${g.truncated ? `<span class="hint warn" title="The model hit its output limit">cut short</span>` : ""}
      ${i === 0 && g.doc_path ? `<span class="current" title="${esc(g.doc_path)}">current document</span>` : ""}
      <button class="btn small" data-guide="${g.id}">View</button>
      ${i === 0 && g.doc_path ? `<button class="btn small" data-guide-doc="${esc(g.doc_path)}">Open in Docs</button>` : ""}
    </li>`).join("");
}

// Minimal Markdown -> HTML for the guide viewer. Text is HTML-escaped before any tag is added.
const mdInline = (s) => esc(s)
  .replace(/`([^`]+)`/g, "<code>$1</code>")
  .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
  .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\w)/g, "$1<em>$2</em>");

function renderMarkdown(md) {
  const lines = md.replace(/\r/g, "").split("\n");
  const out = [];
  let para = [];
  const flush = () => { if (para.length) out.push(`<p>${para.map(mdInline).join("<br>")}</p>`); para = []; };
  const LIST = /^\s*([-*+]|\d+[.)])\s+(.*)$/;
  for (let i = 0; i < lines.length;) {
    const line = lines[i];
    let m;
    if (!line.trim()) { flush(); i++; }
    else if ((m = line.match(/^(#{1,4})\s+(.*)$/))) { flush(); out.push(`<h${m[1].length}>${mdInline(m[2])}</h${m[1].length}>`); i++; }
    else if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); out.push("<hr>"); i++; }
    else if (line.startsWith(">")) {
      flush();
      const quote = [];
      while (i < lines.length && lines[i].startsWith(">")) quote.push(lines[i++].replace(/^>\s?/, ""));
      out.push(`<blockquote>${quote.filter((q) => q.trim()).map(mdInline).join("<br>")}</blockquote>`);
    } else if (line.trim().startsWith("|")) {
      flush();
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(lines[i++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => mdInline(c.trim()));
      const body = rows.filter((r, n) => !(n === 1 && /^[\s|:-]+$/.test(r)));
      const [head, ...rest] = body;
      out.push(`<table><thead><tr>${cells(head).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
        rest.map((r) => `<tr>${cells(r).map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") + "</tbody></table>");
    } else if ((m = line.match(LIST))) {
      flush();
      const ordered = /\d/.test(m[1]);
      const items = [];
      while (i < lines.length && (m = lines[i].match(LIST))) { items.push(`<li>${mdInline(m[2])}</li>`); i++; }
      out.push(ordered ? `<ol>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`);
    } else { para.push(line); i++; }
  }
  flush();
  return out.join("\n");
}

async function openStyleGuide(id) {
  let g;
  try { g = await api(`/api/style/guides/${id}`); } catch (e) { alert(e.message); return; }
  sty.guide = g;
  $("sg-title").textContent = g.kind === "character" ? "Self-Style Guide" : "Content Guide";
  $("sg-body").innerHTML = renderMarkdown(g.markdown);
  $("sg-source").textContent = g.markdown;
  $("sg-body").hidden = false;
  $("sg-source").hidden = true;
  $("sg-raw").textContent = "Show raw";
  $("sg-copy").textContent = "Copy Markdown";
  const latest = (await api(`${styleUrl("/guides")}?kind=${g.kind}`).catch(() => ({ items: [] }))).items[0];
  const current = latest && latest.id === g.id && g.doc_path;
  $("sg-docs").hidden = !current;
  $("sg-meta").textContent = `${new Date(g.created_at * 1000).toLocaleString()} · ${plural(g.image_count, "image")}` +
    (current ? ` · saved as ${g.doc_path}` : " · an earlier version (stored in the database; the file was archived under guides/archive/ when a newer guide was written)");
  $("style-guide-view").hidden = false;
}

async function openGuideInDocs(path) {
  $("style-guide-view").hidden = true;
  await openDocs({ id: sty.cid, name: sty.name });
  await loadDoc(path);
}

// ---- image detail
const SHORT = 28;
function describeItem(item) {
  if (item.description !== undefined) return `${item.color ? item.color + " " : ""}${item.description}${item.category ? ` (${pretty(item.category)})` : ""}`;
  return Object.values(item).map(pretty).join(", ");
}

function renderRecord(rec) {
  const root = h("div");
  for (const [section, part] of Object.entries(rec)) {
    if (!part || typeof part !== "object") continue;
    const chips = h("div", { class: "chips" });
    const extras = [];
    for (const [k, v] of Object.entries(part)) {
      if (typeof v === "string") {
        if (!v) continue;
        if (v.length <= SHORT) chips.append(h("span", { class: "chip" }, h("b", {}, pretty(k)), pretty(v)));
        else extras.push(h("p", {}, h("b", {}, pretty(k) + ": "), v));
      } else if (Array.isArray(v) && v.length) {
        if (typeof v[0] === "object") extras.push(h("p", {}, h("b", {}, pretty(k) + ": "), v.map(describeItem).join("; ")));
        else if (v.every((x) => String(x).length <= SHORT)) extras.push(h("div", { class: "chips" }, v.map((x) => h("span", { class: "chip" }, h("b", {}, pretty(k)), pretty(x)))));
        else extras.push(h("div", {}, h("b", {}, pretty(k) + ":"), h("ul", {}, v.map((x) => h("li", {}, x)))));
      }
    }
    root.append(h("div", { class: "sect" }, h("h3", {}, pretty(section)), chips, ...extras));
  }
  return root;
}

async function openStyleDetail(id) {
  let d;
  try { d = await api(`/api/style/images/${id}`); } catch (e) { alert(e.message); return; }
  $("style-detail-img").src = `/api/style/images/${id}/file`;
  const info = $("style-detail-info");
  info.replaceChildren(h("p", { class: "hint mono" }, d.image.path, ` · ${d.image.width}×${d.image.height}`));
  const current = d.analyses.find((a) => a.model_label === d.current_model && a.schema_version === d.current_schema_version);
  const shown = current || d.analyses[0];
  if (!shown) info.append(h("p", {}, "Not analyzed yet."));
  else {
    if (!current) info.append(h("p", { class: "hint" }, "Not analyzed with the current model and schema yet; showing the most recent older result."));
    info.append(h("p", { class: "hint" }, `${shown.model_label} · schema v${shown.schema_version}` + (shown.seconds ? ` · ${shown.seconds.toFixed(1)}s` : "") +
      (shown.reused_from ? " · reused from the same file in another character's library" : "")));
    if (shown.status === "ok") {
      info.append(renderRecord(shown.record));
      info.append(h("details", {}, h("summary", { class: "hint" }, "Raw JSON"), h("pre", { class: "pre-box" }, JSON.stringify(shown.record, null, 2))));
    } else {
      info.append(h("pre", { class: "pre-box err" }, shown.error || "Failed."));
      if (shown.raw_text) info.append(h("details", {}, h("summary", { class: "hint" }, "Raw model output"), h("pre", { class: "pre-box" }, shown.raw_text)));
    }
  }
  $("style-detail").hidden = false;
}

async function saveStyleType() {
  const id = $("style-type").value;
  $("style-type-desc").textContent = typeDescription(id);
  try {
    await api(`/api/characters/${sty.cid}`, { method: "PUT", body: JSON.stringify({ character_type: id }) });
    $("style-type-msg").textContent = "Saved.";
    $("style-type-msg").className = "hint ok";
  } catch (e) { $("style-type-msg").textContent = e.message; $("style-type-msg").className = "hint err"; }
}

function wireStyle() {
  $("char-style").addEventListener("click", () => { const c = selectedCharacter(); if (c) openStyle(c); });
  $("style-active-char").addEventListener("click", () => { const c = activeCharacter(); if (c) openStyle(c); });
  $("style-close").addEventListener("click", closeStyle);
  document.querySelectorAll("#style .tab").forEach((t) => t.addEventListener("click", () => { if (t.dataset.kind !== sty.kind) showStyleKind(t.dataset.kind); }));
  $("style-settings").addEventListener("click", () => openSettings("vision"));
  $("style-problems").addEventListener("click", (ev) => { if (ev.target.id === "style-fix") openSettings("vision"); });
  $("style-type").addEventListener("change", saveStyleType);
  $("style-folder").addEventListener("change", () => store(folderKey(), $("style-folder").value.trim()));
  $("style-browse").addEventListener("click", () => {
    const cur = $("style-folder").value.trim();
    browse("style_folder", "dir", cur, (p) => { $("style-folder").value = p; store(folderKey(), p); });
  });
  $("style-scan").addEventListener("click", scanStyleFolder);
  $("style-analyze").addEventListener("click", () => startStyleRun(false));
  $("style-reanalyze").addEventListener("click", () => {
    if (confirm(`Re-run the analysis for every image in this library with the current model?`)) startStyleRun(true);
  });
  $("style-cancel").addEventListener("click", () => api("/api/style/cancel", { method: "POST" }).catch(() => {}));
  $("style-guide").addEventListener("click", writeStyleGuide);
  $("style-log").addEventListener("click", async () => {
    const pre = $("style-log-text");
    if (!pre.hidden) { pre.hidden = true; return; }
    try { pre.textContent = (await api("/api/style/server-log")).log || "(no log yet: the vision model has not been started)"; } catch (e) { pre.textContent = e.message; }
    pre.hidden = false;
  });
  document.querySelectorAll("#style .filter").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#style .filter").forEach((x) => x.classList.toggle("active", x === b));
    sty.filter = b.dataset.status;
    clearStyleSelection();
    loadStyleGallery(true);
  }));
  $("style-more").addEventListener("click", () => loadStyleGallery(false));
  $("style-gallery").addEventListener("click", (ev) => {
    const card = ev.target.closest(".s-card");
    if (!card) return;
    const id = Number(card.dataset.id);
    if (ev.target.matches("input[type=checkbox]")) pickStyle(id, ev.target.checked, ev.shiftKey);
    else if (ev.target.closest(".open")) openStyleDetail(id);
  });
  $("style-sel-all").addEventListener("click", () => { shownStyleIds().forEach((i) => sty.selected.add(i)); showStyleSelection(); });
  $("style-sel-clear").addEventListener("click", clearStyleSelection);
  $("style-sel-remove").addEventListener("click", removeStyleSelection);
  $("style-guides").addEventListener("click", (ev) => {
    const t = ev.target;
    if (t.dataset.guide) openStyleGuide(Number(t.dataset.guide));
    if (t.dataset.guideDoc) openGuideInDocs(t.dataset.guideDoc);
  });
  $("style-detail-close").addEventListener("click", () => { $("style-detail").hidden = true; });
  $("sg-close").addEventListener("click", () => { $("style-guide-view").hidden = true; });
  $("sg-raw").addEventListener("click", () => {
    const raw = $("sg-source").hidden;
    $("sg-source").hidden = !raw;
    $("sg-body").hidden = raw;
    $("sg-raw").textContent = raw ? "Show formatted" : "Show raw";
  });
  $("sg-copy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(sty.guide?.markdown || ""); $("sg-copy").textContent = "Copied"; }
    catch (_) { $("sg-copy").textContent = "Copy failed"; }
  });
  $("sg-docs").addEventListener("click", () => { if (sty.guide?.doc_path) openGuideInDocs(sty.guide.doc_path); });
  $("ce-type").addEventListener("change", () => { $("ce-type-desc").textContent = typeDescription($("ce-type").value); });
  $("vision-fields").addEventListener("click", (ev) => {
    const t = ev.target;
    if (!t.dataset.vbrowse) return;
    const input = document.querySelector(`[data-vision="${t.dataset.vbrowse}"]`);
    const cur = input.value.trim();
    browse(`vision_${t.dataset.vbrowse}`, "file", cur ? cur.replace(/[\\/][^\\/]*$/, "") : "", (p) => { input.value = p; }, t.dataset.ext);
  });
}

// ------------------------------------------------------------------------------------ prompts from guides
// The local model (Settings → Style analysis) writes prompts from the active character's guides/ documents.
// The job runs on the GPU worker; its result arrives in /api/status → style.prompts (watched by pollStatus).
const gp = { cid: null, prompts: [], used: -1, waiting: false };

function setGpMsg(text, cls = "") { $("gp-msg").textContent = text; $("gp-msg").className = `hint ${cls}`; }

function drawGuidePrompts() {
  $("gp-list").innerHTML = gp.prompts.map((p, i) =>
    `<li class="${i === gp.used ? "used" : ""}"><span>${esc(p)}</span><button class="btn small" data-gp="${i}">Use</button></li>`).join("");
}

function renderGuidePrompts() {
  const c = activeCharacter();
  $("gp").hidden = !c;
  if (!c) return;
  if (gp.cid !== c.id) { // switched character: show that character's last prompts
    gp.cid = c.id;
    gp.waiting = false;
    gp.used = -1;
    try { gp.prompts = JSON.parse(recall(`gp:${c.id}`) || "[]"); } catch (_) { gp.prompts = []; }
    drawGuidePrompts();
    setGpMsg("");
  }
  const s = state.status?.style;
  const mine = !!s && s.job === "prompts" && s.character_id === c.id;
  const active = !!s?.active;
  $("gp-write").disabled = active || !!state.status?.busy;
  $("gp-cancel").hidden = !(mine && active);
  if (mine && active) { gp.waiting = true; setGpMsg(s.message); return; }
  if (!gp.waiting || !mine) return;
  gp.waiting = false; // the job we started (or saw running) has finished
  if (s.state === "done" && s.prompts?.length) {
    gp.prompts = s.prompts;
    gp.used = -1;
    store(`gp:${c.id}`, JSON.stringify(gp.prompts));
    drawGuidePrompts();
    setGpMsg(`${s.message} Pick one to use it.`, "ok");
  } else if (s.state === "error") setGpMsg(`${s.message} ${s.error || ""}`.trim(), "err");
  else if (s.state === "cancelled") setGpMsg("Cancelled.");
}

async function writeGuidePrompts() {
  const c = activeCharacter();
  if (!c) return;
  setGpMsg("Starting…");
  $("gp-write").disabled = true;
  try {
    const r = await api(`/api/characters/${c.id}/prompts`, { method: "POST",
      body: JSON.stringify({ idea: $("gp-idea").value, count: Number($("gp-count").value) }) });
    gp.waiting = true;
    setGpMsg(`Queued (from ${r.guides.map((g) => g.replace(/^guides\//, "")).join(" + ")})…`);
  } catch (e) {
    setGpMsg(e.message, "err");
    $("gp-write").disabled = false;
  }
}

function useGuidePrompt(i) {
  const p = gp.prompts[i];
  if (p === undefined) return;
  $("prompt").value = p;
  store("prompt", p);
  gp.used = i;
  drawGuidePrompts();
  updateControls();
}

function wireGuidePrompts() {
  $("gp-write").addEventListener("click", writeGuidePrompts);
  $("gp-cancel").addEventListener("click", () => api("/api/style/cancel", { method: "POST" }).catch(() => {}));
  $("gp-list").addEventListener("click", (ev) => { const b = ev.target.closest("[data-gp]"); if (b) useGuidePrompt(Number(b.dataset.gp)); });
  $("gp-idea").value = recall("gpIdea") || "";
  $("gp-idea").addEventListener("input", () => store("gpIdea", $("gp-idea").value));
  $("gp-idea").addEventListener("keydown", (ev) => { if (ev.key === "Enter" && !$("gp-write").disabled) writeGuidePrompts(); });
  const n = recall("gpCount");
  if (n) $("gp-count").value = n;
  $("gp-count").addEventListener("change", () => store("gpCount", $("gp-count").value));
}

// ------------------------------------------------------------------------------------ wiring
function wire() {
  const saved = recall("prompt");
  if (saved) $("prompt").value = saved;
  $("prompt").addEventListener("input", () => { store("prompt", $("prompt").value); updateControls(); });
  $("encode-btn").addEventListener("click", encode);
  $("generate-btn").addEventListener("click", generate);
  $("open-settings").addEventListener("click", openSettings);
  $("close-settings").addEventListener("click", () => { $("settings").hidden = true; });
  $("save-settings").addEventListener("click", saveSettings);
  $("model-fields").addEventListener("click", (ev) => {
    const t = ev.target;
    if (t.dataset.validate) validateField(t.dataset.validate);
    if (t.dataset.browse) browse(t.dataset.browse, t.dataset.type);
  });
  $("close-browser").addEventListener("click", () => { $("browser").hidden = true; });
  $("browser-list").addEventListener("click", (ev) => {
    const li = ev.target.closest("li");
    if (!li) return;
    if (li.dataset.dir !== undefined) browse(browseTarget.key, browseTarget.type, li.dataset.dir);
    else if (li.dataset.file) pickPath(li.dataset.file);
  });
  $("browser-pick-dir").addEventListener("click", (ev) => pickPath(ev.target.dataset.path));
  wireLoras();
  wireCharacters();
  wireDocs();
  wireStyle();
  wireGuidePrompts();
  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape") return;
    // close the topmost dialog only
    if (!$("browser").hidden) { $("browser").hidden = true; return; }
    if (!$("char-editor").hidden) { $("char-editor").hidden = true; return; }
    if (!$("settings").hidden) { $("settings").hidden = true; return; }
    if (!$("docs").hidden) { closeDocs(); return; }
    if (!$("style-guide-view").hidden) { $("style-guide-view").hidden = true; return; }
    if (!$("style-detail").hidden) { $("style-detail").hidden = true; return; }
    if (!$("style").hidden) closeStyle();
  });
}

(async function main() {
  wire();
  try {
    await initResolution();
    state.settings = (await api("/api/settings")).settings;
    applyLoraView(await api("/api/loras"));
    state.status = await api("/api/status");
    await onWorkspace(); // landing page unless a character (or free mode) is already loaded
  } catch (e) {
    $("app").hidden = false;
    showError(`Failed to reach the server: ${e.message}`);
  }
  updateControls();
  pollStatus();
  pollSystem();
})();
