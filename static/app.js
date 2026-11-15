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
  const prompt = $("prompt").value;
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
  else if (c && c.prompt !== $("prompt").value) detail = "Prompt changed since the last encode.";
  else if (c && !s.conditioning_valid_for_current_encoder) detail = "Text encoder changed; encode again.";
  $("encode-detail").textContent = detail;
  $("encode-detail").className = `hint ${enc.label === "Error" ? "err" : ""}`;

  const st = state.settings;
  if (st) $("seed-mode").textContent = st.randomize_seed ? "Random seed per image" : `Fixed seed ${st.seed} (change in Settings)`;
}

// ------------------------------------------------------------------------------------ polling
async function pollStatus() {
  for (;;) {
    try {
      state.status = await api("/api/status");
      const se = state.status.startup_error;
      $("startup-error").hidden = !se;
      $("startup-error").textContent = se || "";
      renderJobFromStatus();
      renderResidency();
      updateControls();
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
  else if (!job && !state.pollingJob) {
    if ($("gen-status").textContent === "Encoding prompt") $("gen-status").textContent = "Idle";
  }
}

async function encode() {
  const prompt = $("prompt").value;
  $("error-box").hidden = true;
  try {
    const job = await api("/api/encode", { method: "POST", body: JSON.stringify({ prompt }) });
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
  const body = { prompt: $("prompt").value, aspect: r.aspect, megapixels: r.megapixels_target, multiple: r.multiple, width: r.width, height: r.height };
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
  $("image-info").textContent = `${res.width} × ${res.height} · seed ${res.seed} · ${res.metadata.transformer}${loras ? ` · LoRA: ${loras}` : ""} · ${res.seconds}s${tiled}`;
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

async function openSettings() {
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
  $("settings-msg").textContent = "";
  $("settings").hidden = false;
}

async function validateField(key) {
  const path = document.querySelector(`[data-path="${key}"]`).value;
  const v = await api("/api/validate-path", { method: "POST", body: JSON.stringify({ kind: key, path }) });
  $(`info-${key}`).innerHTML = infoHtml(v);
}

async function saveSettings() {
  const patch = {};
  document.querySelectorAll("[data-path]").forEach((el) => { patch[el.dataset.path] = el.value; });
  const seed = $("seed").value.trim();
  if (!/^\d+$/.test(seed)) { $("settings-msg").textContent = "Seed must be a non-negative integer."; $("settings-msg").className = "hint err"; return; }
  patch.seed = Number(seed);
  patch.randomize_seed = $("randomize").checked;
  patch.keep_in_ram = {};
  document.querySelectorAll("[data-keep]").forEach((el) => { patch.keep_in_ram[el.dataset.keep] = el.checked; });
  try {
    const r = await api("/api/settings", { method: "PUT", body: JSON.stringify(patch) });
    state.settings = r.settings;
    for (const [key] of MODEL_FIELDS) $(`info-${key}`).innerHTML = infoHtml(r.validation[key]);
    const inv = r.invalidated.length ? ` Reloaded on next use: ${r.invalidated.join(", ")}.` : "";
    const enc = r.invalidated.includes("text_encoder") ? " Prompt conditioning invalidated — encode again." : "";
    $("settings-msg").textContent = `Saved.${inv}${enc}`;
    $("settings-msg").className = "hint ok";
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
async function browse(key, type, path, onPick) {
  if (!browseTarget || browseTarget.key !== key || onPick !== undefined || path === undefined) browseTarget = { key, type, onPick: onPick || null };
  let start = path;
  if (start === undefined) {
    const cur = document.querySelector(`[data-path="${key}"]`).value;
    start = cur ? (type === "dir" ? cur : cur.replace(/[\\/][^\\/]*$/, "")) : "";
  }
  let data;
  try { data = await api(`/api/browse?path=${encodeURIComponent(start)}`); }
  catch (_) { data = await api("/api/browse?path="); }
  $("browser-path").textContent = data.path || "Drives";
  const items = [];
  if (data.parent !== null && data.path) items.push(`<li class="dir" data-dir="${esc(data.parent)}">..</li>`);
  for (const e of data.entries) {
    if (e.type === "dir") items.push(`<li class="dir" data-dir="${esc(e.path)}">${esc(e.name)}</li>`);
    else if (type === "file") items.push(`<li data-file="${esc(e.path)}"><span>${esc(e.name)}</span><span class="hint">${fmtGiB(e.size)} GiB</span></li>`);
  }
  $("browser-list").innerHTML = items.join("") || `<li class="hint">No .safetensors files here</li>`;
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
  const n = state.loras.length;
  const active = state.loras.filter((e) => e.enabled && e.strength !== 0).length;
  $("lora-count").textContent = n ? `${active} of ${n} active` : "";
}

function renderLoras() {
  const lim = state.loraMeta.strength_limit || 5;
  const n = state.loras.length;
  $("lora-list").innerHTML = state.loras.map((e, i) => {
    const bad = e.info && !e.info.compatible;
    const sliderMax = Math.max(2, Math.ceil(Math.abs(e.strength)));
    return `<li class="lora-item${e.enabled ? "" : " off"}${bad ? " bad" : ""}" data-i="${i}">
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
      ${loraMetaHtml(e)}
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
  const ae = document.activeElement;
  const editing = ae && $("lora-list").contains(ae) && (ae.type === "range" || ae.type === "number");
  if (fromSave && editing) {
    // don't rebuild the control the user is dragging/typing in; just refresh validation info
    v.stack.forEach((e, i) => { if (state.loras[i] && state.loras[i].path === e.path) state.loras[i].info = e.info; });
    return;
  }
  state.loras = v.stack;
  renderLoras();
}

async function addLora(path) {
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
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") { $("browser").hidden = true; $("settings").hidden = true; }
  });
}

(async function main() {
  wire();
  try {
    await initResolution();
    state.settings = (await api("/api/settings")).settings;
    applyLoraView(await api("/api/loras"));
  } catch (e) { showError(`Failed to reach the server: ${e.message}`); }
  updateControls();
  pollStatus();
  pollSystem();
})();
