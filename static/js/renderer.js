"use strict";

// Canvas painting: the offscreen 1px-per-cell buffer, and scaling it onto the
// visible canvas.

import { buf, canvas, colors, ctx, st, view } from "./store.js";

export function resizeBuffers(width, height) {
  buf.canvas.width = width;
  buf.canvas.height = height;
  buf.img = buf.ctx.createImageData(width, height);
  buf.buf32 = new Uint32Array(buf.img.data.buffer);
}

export function renderCells(cellsB64, aliveColor) {
  const raw = atob(cellsB64);
  const n = st.width * st.height;
  for (let i = 0; i < n; i++) {
    buf.buf32[i] = (raw.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1 ? aliveColor : colors.DEAD;
  }
  buf.ctx.putImageData(buf.img, 0, 0);
}

export function draw() {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0b0e13";
  ctx.fillRect(0, 0, cw, ch);
  if (!buf.img) return;

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(buf.canvas, 0, 0, st.width, st.height,
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

export function fit() {
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!st.width || !cw) return;
  const s = Math.min((cw - 24) / st.width, (ch - 24) / st.height);
  view.scale = s >= 1 ? Math.max(1, Math.floor(s)) : Math.max(0.2, s);
  view.ox = (cw - st.width * view.scale) / 2;
  view.oy = (ch - st.height * view.scale) / 2;
  view.fitted = true;
}
