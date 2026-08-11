"use strict";

// All user input: painting and panning on the canvas, the toolbar buttons,
// the speed and timeline sliders, and the sidebar resizer.

import { $, canvas, el, preview, st, ui, view } from "./store.js";
import { draw, fit } from "./renderer.js";
import {
  clearPreview, formatFps, requestPreviewFromSlider, send, updateRewindButtonState,
} from "./socket.js";

let mode = "draw";        // draw | erase | pan
let drag = null;          // {kind:"pan"|"paint", value, last, sx, sy, sox, soy}

// ------------------------------------------------------------ canvas input

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
  el.cursor.textContent = c ? `(${c.x}, ${c.y})` : "";
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

el.play.onclick = () => {
  if (!st.running) clearPreview();
  send(st.running ? { action: "pause" } : { action: "play", fps: +el.fps.value });
};
$("btn-step").onclick = () => {
  clearPreview();
  send({ action: "step" });
};
$("btn-clear").onclick = () => {
  clearPreview();
  send({ action: "clear" });
};
el.create.onclick = () => {
  clearPreview();
  const width = Number(el.createWidth.value);
  const height = Number(el.createHeight.value);
  const randomFill = Number(el.createRandom.value);
  const edge = el.createEdge.value === "dead" ? "dead" : "wrap";
  if (!Number.isInteger(width) || !Number.isInteger(height)) return;
  if (!Number.isFinite(randomFill)) return;
  send({ action: "create", width, height, edge, random_fill: randomFill });
};
el.advance.onclick = () => {
  clearPreview();
  const generations = Number(el.advanceGenerations.value);
  if (!Number.isInteger(generations) || generations < 1 || generations > 100) return;
  send({ action: "advance", generations });
};
$("btn-fit").onclick = () => { fit(); draw(); };
el.fps.oninput = () => {
  el.fpsLabel.textContent = formatFps(el.fps.value);
  send({ action: "fps", fps: +el.fps.value });
};

el.timeline.oninput = () => {
  ui.timelineDragging = true;
  el.timelineLabel.textContent = el.timeline.value;
  updateRewindButtonState();
  requestPreviewFromSlider(false);
};

el.timeline.onchange = () => {
  ui.timelineDragging = false;
  requestPreviewFromSlider(true);
  updateRewindButtonState();
};

el.rewind.onclick = () => {
  const gen = Number(el.timeline.value);
  if (!Number.isInteger(gen) || gen === st.generation) return;
  ui.timelineDragging = false;
  el.timeline.value = String(gen);
  el.timelineLabel.textContent = el.timeline.value;
  ui.previewRequestSeq += 1;
  ui.latestPreviewRequested = ui.previewRequestSeq;
  ui.pendingRewindGeneration = gen;
  clearPreview(true);
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

if (el.leftResizer && el.leftPanel) {
  let resizeDrag = null;
  el.leftResizer.addEventListener("pointerdown", (e) => {
    resizeDrag = { startX: e.clientX, startWidth: el.leftPanel.getBoundingClientRect().width };
    el.leftResizer.setPointerCapture(e.pointerId);
  });
  el.leftResizer.addEventListener("pointermove", (e) => {
    if (!resizeDrag) return;
    const next = Math.max(220, Math.min(520, Math.round(resizeDrag.startWidth + (e.clientX - resizeDrag.startX))));
    document.documentElement.style.setProperty("--left-panel-width", `${next}px`);
  });
  const stopResize = () => { resizeDrag = null; };
  el.leftResizer.addEventListener("pointerup", stopResize);
  el.leftResizer.addEventListener("pointercancel", stopResize);
}

window.addEventListener("resize", () => { if (!view.fitted) fit(); draw(); });
