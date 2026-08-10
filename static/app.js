"use strict";

// ---------------------------------------------------------------- elements
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const off = document.createElement("canvas");     // offscreen, 1px per cell
const octx = off.getContext("2d");

const $ = (id) => document.getElementById(id);
const elGen = $("stat-gen"), elPop = $("stat-pop"), elSize = $("stat-size"),
      elCursor = $("stat-cursor"), elConn = $("conn"),
      btnPlay = $("btn-play"), slider = $("fps"), fpsLabel = $("fps-label"),
  timeline = $("timeline"), timelineLabel = $("timeline-label"),
  btnRewind = $("btn-rewind"), eventsBox = $("events");

// ------------------------------------------------------------------- state
const st = { width: 0, height: 0, generation: 0, population: 0,
             running: false, fps: 10, edge: "wrap", timelineMin: 0,
             timelineLatestSnapshot: 0, timelineLatestGeneration: 0,
             cells: "" };
const view = { scale: 6, ox: 0, oy: 0, fitted: false };
let img = null, buf32 = null;
let ws = null, wsOk = false;
let mode = "draw"; // draw | erase | pan
let timelineDragging = false;
let rewindSendTimer = null;
const preview = { active: false, generation: 0, population: 0, cells: null };
let previewRequestSeq = 0;
let latestPreviewRequested = 0;

const pack = (r, g, b) => ((0xff << 24) | (b << 16) | (g << 8) | r) >>> 0;
const DEAD = pack(0x10, 0x16, 0x1e);
const ALIVE_NOW = pack(0x7c, 0xe3, 0x8b);
const ALIVE_REWIND = pack(0x66, 0xb3, 0xff);

// -------------------------------------------------------------- websocket
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { wsOk = true; elConn.classList.add("ok"); };
  ws.onclose = () => {
    wsOk = false; elConn.classList.remove("ok");
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type === "state") applyState(m);
    else if (m.type === "preview_state") applyPreview(m);
    else if (m.type === "event") addEvent(m);
    else if (m.type === "backlog") m.events.forEach(addEvent);
  };
}
const send = (obj) => { if (wsOk) ws.send(JSON.stringify(obj)); };

function updateRewindButtonState() {
  const gen = Number(timeline.value);
  btnRewind.disabled = !Number.isInteger(gen) || gen === st.generation;
}

// ------------------------------------------------------------ state apply
function applyState(m) {
  if (m.width !== st.width || m.height !== st.height) {
    off.width = m.width; off.height = m.height;
    img = octx.createImageData(m.width, m.height);
    buf32 = new Uint32Array(img.data.buffer);
    view.fitted = false;
  }
  Object.assign(st, { width: m.width, height: m.height, edge: m.edge,
                      generation: m.generation, population: m.population,
                      running: m.running, fps: m.fps,
                      timelineMin: m.timeline_min_generation ?? 0,
                      timelineLatestSnapshot: m.timeline_latest_snapshot ?? m.generation,
                      cells: m.cells,
                      timelineLatestGeneration: m.timeline_latest_generation ?? m.generation });

  if (preview.active && (st.running || st.generation !== preview.generation)) {
    clearPreview();
  }

  if (!preview.active) {
    renderCells(st.cells, ALIVE_NOW);
    elGen.textContent = st.generation;
    elPop.textContent = st.population;
  }
  elSize.textContent = `${st.width}×${st.height} ${st.edge}`;
  btnPlay.innerHTML = st.running ? "&#10074;&#10074; Pause" : "&#9654; Play";
  btnPlay.classList.toggle("active", st.running);
  if (document.activeElement !== slider) {
    slider.value = st.fps;
    fpsLabel.textContent = Math.round(st.fps);
  }
  timeline.min = String(st.timelineMin);
  timeline.max = String(Math.max(st.generation, st.timelineLatestGeneration));
  if (!timelineDragging && !preview.active) timeline.value = String(st.generation);
  timelineLabel.textContent = timeline.value;
  updateRewindButtonState();
  if (!view.fitted) fit();
  draw();
}

function renderCells(cellsB64, aliveColor) {
  const raw = atob(cellsB64);
  const n = st.width * st.height;
  for (let i = 0; i < n; i++) {
    buf32[i] = (raw.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1 ? aliveColor : DEAD;
  }
  octx.putImageData(img, 0, 0);
}

function applyPreview(m) {
  if (typeof m.request_id === "number" && m.request_id < latestPreviewRequested) return;

  preview.active = true;
  preview.generation = m.generation;
  preview.population = m.population;
  preview.cells = m.cells;

  timeline.value = String(preview.generation);
  timelineLabel.textContent = timeline.value;
  updateRewindButtonState();

  renderCells(preview.cells, ALIVE_REWIND);
  elGen.textContent = preview.generation;
  elPop.textContent = preview.population;
  draw();
}

function clearPreview() {
  if (!preview.active) return;
  preview.active = false;
  preview.cells = null;
  timeline.value = String(st.generation);
  timelineLabel.textContent = timeline.value;
  updateRewindButtonState();
  elGen.textContent = st.generation;
  elPop.textContent = st.population;
  if (st.cells) renderCells(st.cells, ALIVE_NOW);
  draw();
}

function requestPreviewFromSlider(immediate = false) {
  const gen = Number(timeline.value);
  if (!Number.isInteger(gen)) return;
  if (gen === st.generation) {
    if (rewindSendTimer) clearTimeout(rewindSendTimer);
    clearPreview();
    return;
  }
  if (rewindSendTimer) clearTimeout(rewindSendTimer);
  const sendPreview = () => {
    previewRequestSeq += 1;
    latestPreviewRequested = previewRequestSeq;
    send({ action: "preview_generation", generation: gen, request_id: previewRequestSeq });
  };
  if (immediate) {
    sendPreview();
    return;
  }
  rewindSendTimer = setTimeout(sendPreview, 40);
}

// -------------------------------------------------------------- rendering
function draw() {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0b0e13";
  ctx.fillRect(0, 0, cw, ch);
  if (!img) return;

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, st.width, st.height,
                view.ox, view.oy, st.width * view.scale, st.height * view.scale);

  ctx.strokeStyle = "rgba(140,160,180,0.35)";
  ctx.strokeRect(view.ox - 0.5, view.oy - 0.5,
                 st.width * view.scale + 1, st.height * view.scale + 1);

  if (view.scale >= 6) drawGrid(cw, ch);
}

function drawGrid(cw, ch) {
  const s = view.scale;
  const x0 = Math.max(0, Math.floor((0 - view.ox) / s));
  const x1 = Math.min(st.width, Math.ceil((cw - view.ox) / s));
  const y0 = Math.max(0, Math.floor((0 - view.oy) / s));
  const y1 = Math.min(st.height, Math.ceil((ch - view.oy) / s));
  ctx.strokeStyle = "rgba(120,140,160,0.10)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = x0; x <= x1; x++) {
    const px = Math.round(view.ox + x * s) + 0.5;
    ctx.moveTo(px, view.oy + y0 * s);
    ctx.lineTo(px, view.oy + y1 * s);
  }
  for (let y = y0; y <= y1; y++) {
    const py = Math.round(view.oy + y * s) + 0.5;
    ctx.moveTo(view.ox + x0 * s, py);
    ctx.lineTo(view.ox + x1 * s, py);
  }
  ctx.stroke();
}

function fit() {
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!st.width || !cw) return;
  let s = Math.min((cw - 24) / st.width, (ch - 24) / st.height);
  view.scale = s >= 1 ? Math.max(1, Math.floor(s)) : Math.max(0.2, s);
  view.ox = (cw - st.width * view.scale) / 2;
  view.oy = (ch - st.height * view.scale) / 2;
  view.fitted = true;
}

// ------------------------------------------------------------ interaction
function cellAt(e) {
  const r = canvas.getBoundingClientRect();
  const x = Math.floor((e.clientX - r.left - view.ox) / view.scale);
  const y = Math.floor((e.clientY - r.top - view.oy) / view.scale);
  return (x >= 0 && x < st.width && y >= 0 && y < st.height) ? { x, y } : null;
}

function lineCells(a, b) {
  const out = [];
  let x0 = a.x, y0 = a.y;
  const dx = Math.abs(b.x - x0), dy = -Math.abs(b.y - y0);
  const sx = x0 < b.x ? 1 : -1, sy = y0 < b.y ? 1 : -1;
  let err = dx + dy;
  for (;;) {
    out.push([x0, y0]);
    if (x0 === b.x && y0 === b.y) break;
    const e2 = 2 * err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
  }
  return out;
}

let drag = null; // {kind:"pan"|"paint", value, last, sx, sy, sox, soy}

canvas.addEventListener("pointerdown", (e) => {
  if (preview.active) clearPreview();
  canvas.setPointerCapture(e.pointerId);
  if (e.button === 1 || mode === "pan") {
    drag = { kind: "pan", sx: e.clientX, sy: e.clientY, sox: view.ox, soy: view.oy };
    canvas.style.cursor = "grabbing";
    return;
  }
  const value = (e.button === 2 || mode === "erase") ? 0 : 1;
  const c = cellAt(e);
  if (!c) { drag = null; return; }
  drag = { kind: "paint", value, last: c };
  send({ action: "paint", cells: [[c.x, c.y]], value });
});

canvas.addEventListener("pointermove", (e) => {
  const c = cellAt(e);
  elCursor.textContent = c ? `(${c.x}, ${c.y})` : "";
  if (!drag) return;
  if (drag.kind === "pan") {
    view.ox = drag.sox + (e.clientX - drag.sx);
    view.oy = drag.soy + (e.clientY - drag.sy);
    draw();
    return;
  }
  if (c && (c.x !== drag.last.x || c.y !== drag.last.y)) {
    send({ action: "paint", cells: lineCells(drag.last, c), value: drag.value });
    drag.last = c;
  }
});

const endDrag = () => {
  drag = null;
  canvas.style.cursor = mode === "pan" ? "grab" : "crosshair";
};
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const wx = (mx - view.ox) / view.scale, wy = (my - view.oy) / view.scale;
  view.scale = Math.min(48, Math.max(0.2, view.scale * Math.exp(-e.deltaY * 0.0015)));
  view.ox = mx - wx * view.scale;
  view.oy = my - wy * view.scale;
  view.fitted = true; // manual view: stop auto-refit
  draw();
}, { passive: false });

// ---------------------------------------------------------------- controls
btnPlay.onclick = () => {
  if (!st.running) clearPreview();
  send(st.running ? { action: "pause" } : { action: "play", fps: +slider.value });
};
$("btn-step").onclick = () => {
  clearPreview();
  send({ action: "step" });
};
$("btn-clear").onclick = () => {
  clearPreview();
  send({ action: "clear" });
};
$("btn-fit").onclick = () => { fit(); draw(); };
slider.oninput = () => {
  fpsLabel.textContent = slider.value;
  send({ action: "fps", fps: +slider.value });
};

timeline.oninput = () => {
  timelineDragging = true;
  timelineLabel.textContent = timeline.value;
  updateRewindButtonState();
  requestPreviewFromSlider(false);
};

timeline.onchange = () => {
  timelineDragging = false;
  requestPreviewFromSlider(true);
  updateRewindButtonState();
};

btnRewind.onclick = () => {
  const gen = Number(timeline.value);
  if (!Number.isInteger(gen) || gen === st.generation) return;
  clearPreview();
  send({ action: "rewind", generation: gen });
};

for (const m of ["draw", "erase", "pan"]) {
  $(`mode-${m}`).onclick = () => {
    mode = m;
    for (const n of ["draw", "erase", "pan"])
      $(`mode-${n}`).classList.toggle("active", n === m);
    canvas.style.cursor = m === "pan" ? "grab" : "crosshair";
  };
}

$("mcp-cmd").onclick = () => {
  navigator.clipboard?.writeText($("mcp-cmd").textContent.trim());
  $("mcp-cmd").style.color = "#7ce38b";
  setTimeout(() => ($("mcp-cmd").style.color = ""), 500);
};

window.addEventListener("resize", () => { if (!view.fitted) fit(); draw(); });

// ------------------------------------------------------------ event feed
function addEvent(m) {
  const empty = eventsBox.querySelector(".evt-empty");
  if (empty) empty.remove();
  const nearBottom =
    eventsBox.scrollTop + eventsBox.clientHeight >= eventsBox.scrollHeight - 40;
  const div = document.createElement("div");
  div.className = `evt ${m.source || ""}`;
  const t = document.createElement("time");
  t.textContent = new Date(m.ts * 1000).toLocaleTimeString();
  div.appendChild(t);
  div.appendChild(document.createTextNode(`${m.source}: ${m.text}`));
  eventsBox.appendChild(div);
  while (eventsBox.childElementCount > 300) eventsBox.firstElementChild.remove();
  if (nearBottom) eventsBox.scrollTop = eventsBox.scrollHeight;
}

connect();
draw();
