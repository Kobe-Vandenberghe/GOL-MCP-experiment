"use strict";

// Shared mutable state and DOM handles.
//
// Every exported binding is a const object mutated in place, never reassigned:
// an ES module's imported bindings are read-only in the importing module, so
// anything reassigned (the pixel buffers, the drag flags) has to live inside
// an object rather than as a bare `let`.

export const $ = (id) => document.getElementById(id);

export const canvas = $("view");
export const ctx = canvas.getContext("2d");

export const el = {
  gen: $("stat-gen"),
  pop: $("stat-pop"),
  size: $("stat-size"),
  cursor: $("stat-cursor"),
  conn: $("conn"),
  play: $("btn-play"),
  fps: $("fps"),
  fpsLabel: $("fps-label"),
  timeline: $("timeline"),
  timelineLabel: $("timeline-label"),
  rewind: $("btn-rewind"),
  events: $("events"),
  createWidth: $("create-width"),
  createHeight: $("create-height"),
  createEdge: $("create-edge"),
  createRandom: $("create-random"),
  create: $("btn-create"),
  advanceGenerations: $("advance-generations"),
  advance: $("btn-advance"),
  leftPanel: document.querySelector(".left-panel"),
  leftResizer: $("left-resizer"),
};

export const st = {
  width: 0, height: 0, generation: 0, population: 0,
  running: false, fps: 10, edge: "wrap", timelineMin: 0,
  timelineLatestSnapshot: 0, timelineLatestGeneration: 0,
  cells: "",
};

export const view = { scale: 6, ox: 0, oy: 0, fitted: false };

export const preview = { active: false, generation: 0, population: 0, cells: null };

// Transient interaction state shared between the socket and the controls.
export const ui = {
  timelineDragging: false,
  rewindSendTimer: null,
  previewRequestSeq: 0,
  latestPreviewRequested: 0,
  pendingRewindGeneration: null,
};

// Offscreen 1px-per-cell buffer the visible canvas is scaled up from.
const off = document.createElement("canvas");
export const buf = {
  canvas: off,
  ctx: off.getContext("2d"),
  img: null,
  buf32: null,
};

const pack = (r, g, b) => ((0xff << 24) | (b << 16) | (g << 8) | r) >>> 0;
export const colors = {
  DEAD: pack(0x10, 0x16, 0x1e),
  ALIVE_NOW: pack(0x7c, 0xe3, 0x8b),
  ALIVE_REWIND: pack(0x66, 0xb3, 0xff),
};
