"use strict";

// The server connection, and reflecting what it sends into the UI.
//
// Deliberately one-way: this module never imports from controls.js, so the
// dependency runs controls -> socket and there is no cycle.

import { colors, el, preview, st, ui, view } from "./store.js";
import { draw, fit, renderCells, resizeBuffers } from "./renderer.js";

let ws = null, wsOk = false;

export function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { wsOk = true; el.conn.classList.add("ok"); };
  ws.onclose = () => {
    wsOk = false; el.conn.classList.remove("ok");
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

export const send = (obj) => { if (wsOk) ws.send(JSON.stringify(obj)); };

export function formatFps(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "0";
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

export function updateRewindButtonState() {
  const gen = Number(el.timeline.value);
  el.rewind.disabled = !Number.isInteger(gen) || gen === st.generation;
}

// ------------------------------------------------------------ state apply

function applyState(m) {
  if (m.width !== st.width || m.height !== st.height) {
    resizeBuffers(m.width, m.height);
    view.fitted = false;
  }
  Object.assign(st, { width: m.width, height: m.height, edge: m.edge,
                      generation: m.generation, population: m.population,
                      running: m.running, fps: m.fps,
                      timelineMin: m.timeline_min_generation ?? 0,
                      timelineLatestSnapshot: m.timeline_latest_snapshot ?? m.generation,
                      cells: m.cells,
                      timelineLatestGeneration: m.timeline_latest_generation ?? m.generation });

  if (ui.pendingRewindGeneration !== null && st.generation === ui.pendingRewindGeneration) {
    ui.pendingRewindGeneration = null;
  }

  if (preview.active && (st.running || st.generation !== preview.generation)) {
    clearPreview();
  }

  if (!preview.active) {
    renderCells(st.cells, colors.ALIVE_NOW);
    el.gen.textContent = st.generation;
    el.pop.textContent = st.population;
  }
  el.size.textContent = `${st.width}×${st.height} ${st.edge}`;
  el.play.innerHTML = st.running ? "&#10074;&#10074; Pause" : "&#9654; Play";
  el.play.classList.toggle("active", st.running);
  if (document.activeElement !== el.fps) {
    el.fps.value = st.fps;
    el.fpsLabel.textContent = formatFps(st.fps);
  }
  el.timeline.min = String(st.timelineMin);
  el.timeline.max = String(Math.max(st.generation, st.timelineLatestGeneration));
  if (!ui.timelineDragging && !preview.active) el.timeline.value = String(st.generation);
  el.timelineLabel.textContent = el.timeline.value;
  updateRewindButtonState();
  if (!view.fitted) fit();
  draw();
}

function applyPreview(m) {
  if (ui.pendingRewindGeneration !== null) return;
  if (typeof m.request_id === "number" && m.request_id < ui.latestPreviewRequested) return;

  preview.active = true;
  preview.generation = m.generation;
  preview.population = m.population;
  preview.cells = m.cells;

  el.timeline.value = String(preview.generation);
  el.timelineLabel.textContent = el.timeline.value;
  updateRewindButtonState();

  renderCells(preview.cells, colors.ALIVE_REWIND);
  el.gen.textContent = preview.generation;
  el.pop.textContent = preview.population;
  draw();
}

export function clearPreview(keepSelection = false) {
  if (!preview.active) return;
  preview.active = false;
  preview.cells = null;
  if (!keepSelection) {
    el.timeline.value = String(st.generation);
    el.timelineLabel.textContent = el.timeline.value;
  }
  updateRewindButtonState();
  el.gen.textContent = st.generation;
  el.pop.textContent = st.population;
  if (st.cells) renderCells(st.cells, colors.ALIVE_NOW);
  draw();
}

export function requestPreviewFromSlider(immediate = false) {
  const gen = Number(el.timeline.value);
  if (!Number.isInteger(gen)) return;
  if (gen === st.generation) {
    if (ui.rewindSendTimer) clearTimeout(ui.rewindSendTimer);
    clearPreview();
    return;
  }
  if (ui.rewindSendTimer) clearTimeout(ui.rewindSendTimer);
  const sendPreview = () => {
    ui.previewRequestSeq += 1;
    ui.latestPreviewRequested = ui.previewRequestSeq;
    send({ action: "preview_generation", generation: gen, request_id: ui.previewRequestSeq });
  };
  if (immediate) {
    sendPreview();
    return;
  }
  ui.rewindSendTimer = setTimeout(sendPreview, 40);
}

// ------------------------------------------------------------ event feed

function addEvent(m) {
  const empty = el.events.querySelector(".evt-empty");
  if (empty) empty.remove();
  const nearBottom =
    el.events.scrollTop + el.events.clientHeight >= el.events.scrollHeight - 40;
  const div = document.createElement("div");
  div.className = `evt ${m.source || ""}`;
  const t = document.createElement("time");
  t.textContent = new Date(m.ts * 1000).toLocaleTimeString();
  div.appendChild(t);
  div.appendChild(document.createTextNode(`${m.source}: ${m.text}`));
  el.events.appendChild(div);
  while (el.events.childElementCount > 300) el.events.firstElementChild.remove();
  if (nearBottom) el.events.scrollTop = el.events.scrollHeight;
}
