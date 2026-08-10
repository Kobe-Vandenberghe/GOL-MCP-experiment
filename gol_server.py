"""Shared Game of Life server.

One process serves three surfaces:
  - Browser UI      http://localhost:8000       (static viewer + controls)
  - Live state feed ws://localhost:8000/ws      (broadcasts board + events)
  - MCP endpoint    http://localhost:8000/mcp   (Streamable HTTP, for agents)

The server owns the authoritative board. Every mutation — from an MCP tool
call or the browser — broadcasts fresh state to all connected viewers.
"""
from __future__ import annotations

import asyncio
import base64
import bisect
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mcp.server import MCPServer

from gol_world import (
    MAX_DIM,
    MIN_DIM,
    SimulationError,
    World,
    next_grid_of,
    parse_rle,
    simulate,
    validate_sim_params,
)

STATIC_DIR = Path(__file__).parent / "static"
MAX_ADVANCE = 100          # per user spec: cap generations per advance() call
MAX_OBSERVE_CELLS = 20000  # cap ASCII output size for observe_world
MAX_FPS = 1000.0
MIN_FPS = 1.0
HISTORY_SNAPSHOT_INTERVAL = 10
HISTORY_MAX_SNAPSHOTS = 2000


def _pack_grid(grid: np.ndarray) -> bytes:
    return np.packbits(grid, axis=None).tobytes()


def _unpack_grid(packed: bytes, width: int, height: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=width * height)
    return bits.reshape((height, width)).astype(np.uint8, copy=False)


class State:
    def __init__(self):
        self.world = World(160, 100, edge="wrap")
        self.lock = asyncio.Lock()
        self.sockets: set[WebSocket] = set()
        self.events: deque[dict] = deque(maxlen=200)
        self.running = False
        self.fps = 10.0
        self.run_task: asyncio.Task | None = None
        self.snapshot_interval = HISTORY_SNAPSHOT_INTERVAL
        self.history_generations: list[int] = []
        self.history_snapshots: dict[int, bytes] = {}
        self.max_reachable_generation = 0
        self._reset_history_locked()

    def _reset_history_locked(self) -> None:
        self.history_generations.clear()
        self.history_snapshots.clear()
        self.max_reachable_generation = self.world.generation
        self._record_snapshot_locked(force=True)

    def _record_snapshot_locked(self, force: bool = False) -> bool:
        gen = self.world.generation
        if not force and (gen % self.snapshot_interval != 0):
            return False

        if gen not in self.history_snapshots:
            bisect.insort(self.history_generations, gen)
        self.history_snapshots[gen] = _pack_grid(self.world.grid)
        if gen > self.max_reachable_generation:
            self.max_reachable_generation = gen

        while len(self.history_generations) > HISTORY_MAX_SNAPSHOTS:
            drop_gen = self.history_generations.pop(0)
            if drop_gen == gen and self.history_generations:
                drop_gen = self.history_generations.pop(0)
            self.history_snapshots.pop(drop_gen, None)
        return True

    def _truncate_future_locked(self) -> int:
        idx = bisect.bisect_right(self.history_generations, self.world.generation)
        to_drop = self.history_generations[idx:]
        if not to_drop:
            return 0
        del self.history_generations[idx:]
        for gen in to_drop:
            self.history_snapshots.pop(gen, None)
        self.max_reachable_generation = self.world.generation
        return len(to_drop)

    def _timeline_bounds_locked(self) -> tuple[int, int]:
        if not self.history_generations:
            return self.world.generation, self.world.generation
        return self.history_generations[0], self.history_generations[-1]

    def _reconstruct_generation_locked(self, target_generation: int) -> np.ndarray:
        idx = bisect.bisect_right(self.history_generations, target_generation) - 1
        if idx < 0:
            raise ValueError("target generation is older than retained history")

        start_gen = self.history_generations[idx]
        packed = self.history_snapshots[start_gen]
        g = _unpack_grid(packed, self.world.width, self.world.height)
        steps = target_generation - start_gen
        for _ in range(steps):
            g = next_grid_of(g, self.world.edge)
        return g


S = State()

# ------------------------------------------------------------------ broadcast


def state_message() -> dict:
    w = S.world
    min_gen, latest_snapshot = S._timeline_bounds_locked()
    return {
        "type": "state",
        "width": w.width,
        "height": w.height,
        "edge": w.edge,
        "generation": w.generation,
        "population": w.population(),
        "running": S.running,
        "fps": S.fps,
        "timeline_min_generation": min_gen,
        "timeline_latest_snapshot": latest_snapshot,
        "timeline_latest_generation": S.max_reachable_generation,
        "timeline_snapshot_interval": S.snapshot_interval,
        "timeline_snapshot_count": len(S.history_generations),
        "cells": base64.b64encode(np.packbits(w.grid, axis=None).tobytes()).decode("ascii"),
    }


async def _send_all(payload: dict) -> None:
    if not S.sockets:
        return
    text = json.dumps(payload)
    dead = []
    for ws in list(S.sockets):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        S.sockets.discard(ws)


async def broadcast_state() -> None:
    await _send_all(state_message())


async def log_event(source: str, text: str) -> None:
    evt = {"type": "event", "source": source, "text": text, "ts": time.time()}
    S.events.append(evt)
    await _send_all(evt)


# -------------------------------------------------------------------- autorun


async def _run_loop() -> None:
    try:
        while S.running:
            t0 = time.monotonic()
            async with S.lock:
                S._truncate_future_locked()
                S.world.step(1)
                S._record_snapshot_locked()
            await broadcast_state()
            delay = 1.0 / S.fps - (time.monotonic() - t0)
            await asyncio.sleep(delay if delay > 0 else 0.001)
    finally:
        S.running = False
        await broadcast_state()


async def set_running(running: bool, fps: float | None = None) -> None:
    if fps is not None:
        S.fps = min(MAX_FPS, max(MIN_FPS, float(fps)))
    if running and not S.running:
        S.running = True
        S.run_task = asyncio.create_task(_run_loop())
    elif not running:
        S.running = False  # loop exits on its own and broadcasts
    await broadcast_state()


async def rewind_world(target_generation: int, actor: str) -> str:
    target = int(target_generation)
    await set_running(False)
    async with S.lock:
        current = S.world.generation
        if target < 0:
            raise ValueError("generation must be >= 0")
        if target > S.max_reachable_generation:
            raise ValueError(
                f"generation {target} is outside retained history "
                f"(max reachable is {S.max_reachable_generation})"
            )

        grid = S._reconstruct_generation_locked(target)
        S.world.grid = grid
        S.world.generation = target
        # Committing a rewind establishes a new present on a linear timeline.
        # Any generations ahead of the target are discarded immediately.
        dropped = S._truncate_future_locked()
        S._record_snapshot_locked(force=True)
        pop = S.world.population()

    await broadcast_state()
    await log_event(actor, f"rewind_to_generation({target}) -> pop {pop}, dropped {dropped} future snapshots")
    return (
        f"Moved world from generation {current} to {target}. Population is now {pop}. "
        f"Dropped {dropped} future snapshots from the timeline."
    )


# ------------------------------------------------------------------ MCP tools

mcp = MCPServer(
    "game-of-life",
    instructions=(
        "Controls a shared Conway's Game of Life world that the user is watching "
        "live in their browser — every change you make appears there instantly. "
        "Coordinates are (x, y) with (0,0) at the top-left; x grows rightward, "
        "y grows downward. Use observe_world to see the board as ASCII art."
    ),
)


def _status_text() -> str:
    w = S.world
    box = w.bounding_box()
    box_txt = f"x=[{box[0]}..{box[2]}] y=[{box[1]}..{box[3]}]" if box else "none (board is empty)"
    return (
        f"World {w.width}x{w.height}, edge={w.edge} "
        f"({'toroidal wrap-around' if w.edge == 'wrap' else 'cells beyond the border count as dead'})\n"
        f"Generation {w.generation}, population {w.population()}\n"
        f"Live-cell bounding box: {box_txt}\n"
        f"Dynamics: {w.dynamics()}\n"
        f"Autorun: {'ON at ' + str(S.fps) + ' fps' if S.running else 'off'}\n"
        f"Connected browser viewers: {len(S.sockets)}"
    )


@mcp.tool()
async def get_world_status() -> str:
    """Get the world's dimensions, edge behavior, generation, population,
    live-cell bounding box, dynamics (empty/static/period-2/evolving), and
    autorun state. Cheap - call this to orient yourself before other tools."""
    async with S.lock:
        text = _status_text()
    await log_event("agent", "get_world_status")
    return text


@mcp.tool()
async def create_world(width: int, height: int, edge: str = "wrap", random_fill: float = 0.0) -> str:
    """Replace the world with a new board (this discards the current one).

    width/height: 8..1024 cells. The default world is 160x100.
    edge: "wrap" = toroidal (left edge touches right, top touches bottom);
          "dead" = cells beyond the border count as permanently dead.
    random_fill: probability 0..1 that each cell starts alive (0 = empty board).
    Coordinates afterwards: (x, y), 0-indexed from the top-left."""
    if not 0.0 <= random_fill <= 1.0:
        raise ValueError("random_fill must be between 0 and 1")
    async with S.lock:
        S.world = World(width, height, edge=edge, random_fill=random_fill)
        S._reset_history_locked()
        pop = S.world.population()
    await broadcast_state()
    fill_txt = f", random_fill={random_fill}" if random_fill else ""
    await log_event("agent", f"create_world {width}x{height} edge={edge}{fill_txt} -> pop {pop}")
    return f"Created {width}x{height} world (edge={edge}), generation 0, population {pop}."


@mcp.tool()
async def clear_world() -> str:
    """Kill every cell and reset the generation counter to 0.
    Dimensions and edge behavior are kept."""
    async with S.lock:
        S.world.clear()
        S._reset_history_locked()
        w, h = S.world.width, S.world.height
    await broadcast_state()
    await log_event("agent", "clear_world")
    return f"Cleared: {w}x{h} board is empty, generation reset to 0."


@mcp.tool()
async def observe_world(x: int = 0, y: int = 0, width: int = 0, height: int = 0) -> str:
    """See a region of the board as ASCII art: '#' = alive, '.' = dead.
    Includes an x-axis ruler (tens digit row + units digit row) and absolute
    y labels on each row so you can read off exact coordinates.

    (x, y) is the top-left corner of the window; width/height its size.
    Defaults (all 0) show the full board. Output is capped at 20000 cells -
    for big boards, pass a window (get_world_status's bounding box tells you
    where the live cells are)."""
    async with S.lock:
        w = S.world
        vw = width if width > 0 else w.width
        vh = height if height > 0 else w.height
        if vw * vh > MAX_OBSERVE_CELLS:
            raise ValueError(
                f"requested window is {vw}x{vh} = {vw * vh} cells; max is {MAX_OBSERVE_CELLS}. "
                "Pass a smaller width/height window."
            )
        view = w.ascii_view(x, y, vw, vh)
    await log_event("agent", f"observe_world x={x} y={y} {vw}x{vh}")
    return view


@mcp.tool()
async def set_cells(alive: list[list[int]] | None = None, dead: list[list[int]] | None = None) -> str:
    """Set individual cells. alive/dead are lists of [x, y] pairs, e.g.
    alive=[[10, 5], [11, 5], [12, 5]]. x = column from the left, y = row from
    the top, both 0-indexed. On wrap worlds out-of-range coordinates wrap
    around; on dead-edge worlds they are skipped and reported."""
    alive = alive or []
    dead = dead or []
    if not alive and not dead:
        raise ValueError("pass at least one of alive/dead as a list of [x, y] pairs")
    async with S.lock:
        S._truncate_future_locked()
        n_alive, skip_a = S.world.set_cells(alive, 1)
        n_dead, skip_d = S.world.set_cells(dead, 0)
        S._record_snapshot_locked(force=True)
        pop = S.world.population()
    await broadcast_state()
    await log_event("agent", f"set_cells +{n_alive} alive, +{n_dead} dead -> pop {pop}")
    msg = f"Set {n_alive} cells alive and {n_dead} dead. Population is now {pop}."
    skipped = skip_a + skip_d
    if skipped:
        msg += f" Skipped {len(skipped)} out-of-bounds cells: {skipped[:10]}"
    return msg


@mcp.tool()
async def place_pattern(rle: str, x: int, y: int, clear_rect: bool = False) -> str:
    """Stamp a pattern in standard Game of Life RLE format with its top-left
    corner at (x, y). In RLE, 'b' = dead, 'o' = alive, '$' = next row, '!' =
    end, digits repeat the next symbol; an "x = w, y = h" header line and '#'
    comments are allowed and ignored.

    Example - a glider: "bob$2bo$3o!"
    Example - a blinker: "3o!"

    By default only the pattern's LIVE cells are stamped (existing cells
    elsewhere survive). Set clear_rect=true to wipe the pattern's bounding
    rectangle first, so the pattern lands on a clean area."""
    cells, pw, ph = parse_rle(rle)
    async with S.lock:
        S._truncate_future_locked()
        w = S.world
        if clear_rect:
            rect = [(x + dx, y + dy) for dy in range(ph) for dx in range(pw)]
            w.set_cells(rect, 0)
        placed, skipped = w.set_cells([(x + dx, y + dy) for dx, dy in cells], 1)
        S._record_snapshot_locked(force=True)
        pop = w.population()
    await broadcast_state()
    await log_event("agent", f"place_pattern {pw}x{ph} ({placed} cells) at ({x},{y})")
    msg = (
        f"Placed {pw}x{ph} pattern with {placed} live cells at ({x},{y})"
        f"..({x + pw - 1},{y + ph - 1}). Population is now {pop}."
    )
    if skipped:
        msg += f" {len(skipped)} cells fell outside the dead-edge board and were skipped."
    return msg


@mcp.tool()
async def advance(generations: int = 1) -> str:
    """Advance the simulation 1..100 generations (capped at 100 per call).
    Multi-generation advances are animated in the user's browser at ~100
    generations/second. Reports the population change and the board's
    dynamics afterwards (empty / static / period-2 / evolving)."""
    n = int(generations)
    if not 1 <= n <= MAX_ADVANCE:
        raise ValueError(f"generations must be between 1 and {MAX_ADVANCE}")
    async with S.lock:
        S._truncate_future_locked()
        pop_before = S.world.population()
    for i in range(n):
        async with S.lock:
            S.world.step(1)
            S._record_snapshot_locked(force=(i == n - 1))
        await broadcast_state()
        if n > 1:
            await asyncio.sleep(0.01)  # let viewers see the animation
    async with S.lock:
        gen = S.world.generation
        pop = S.world.population()
        dyn = S.world.dynamics()
    await log_event("agent", f"advance({n}) -> gen {gen}, pop {pop} ({dyn})")
    return (
        f"Advanced {n} generation{'s' if n != 1 else ''} -> generation {gen}. "
        f"Population {pop_before} -> {pop} ({pop - pop_before:+d}). Dynamics: {dyn}."
    )


@mcp.tool()
async def set_autorun(enabled: bool, fps: float = 10) -> str:
    """Start or stop continuous simulation on the server (the user watches it
    run live). fps is generations per second, clamped to 1..1000. All other
    tools keep working while it runs - observe_world to peek, set_cells to
    interfere. Remember to stop it when the show is over."""
    await set_running(enabled, fps)
    await log_event("agent", f"set_autorun {'ON at ' + str(S.fps) + ' fps' if enabled else 'OFF'}")
    if enabled:
        return f"Autorun ON at {S.fps} generations/second. The world is now evolving continuously."
    async with S.lock:
        return f"Autorun OFF at generation {S.world.generation}, population {S.world.population()}."


def _sim_summary_text(summary: dict) -> str:
    bits = [f"gen {summary['start_generation']} -> {summary['end_generation']}", f"pop {summary['final_population']}"]
    if summary["extinct"]:
        bits.append(f"extinct at gen {summary['extinct_at_generation']}")
    if summary["stable"]:
        bits.append(f"stable at gen {summary['stable_at_generation']}")
    if summary["repeating_period"]:
        bits.append(f"period {summary['repeating_period']}")
    return ", ".join(bits)


def _timeline_status_text() -> str:
    current = S.world.generation
    min_gen, latest_snapshot = S._timeline_bounds_locked()
    max_gen = S.max_reachable_generation
    return (
        f"Timeline status\n"
        f"Current generation: {current}\n"
        f"Earliest retained snapshot: {min_gen}\n"
        f"Latest retained snapshot: {latest_snapshot}\n"
        f"Latest reachable generation: {max_gen}\n"
        f"Snapshot interval target: every {S.snapshot_interval} generations\n"
        f"Stored snapshots: {len(S.history_generations)} (cap {HISTORY_MAX_SNAPSHOTS})\n"
        f"Rewind range right now: [{min_gen}..{max_gen}]"
    )


@mcp.tool()
async def get_timeline_status() -> str:
    """Inspect rewind history: current generation, retained snapshot range,
    snapshot interval, and how many snapshots are stored."""
    async with S.lock:
        text = _timeline_status_text()
    await log_event("agent", "get_timeline_status")
    return text


@mcp.tool()
async def rewind_to_generation(generation: int) -> str:
    """Rewind the shared world to an earlier generation on the current
    timeline. If you continue simulating after rewinding, future history is
    discarded (linear timeline, no branching yet)."""
    return await rewind_world(generation, actor="agent")


@mcp.tool()
async def advance_generations(count: int, sample_every: int = 1) -> str:
    """Advance and PERMANENTLY commit the world by `count` generations
    (1..20000, lower for large boards - see below) - far more efficient than
    repeated advance() calls for large counts, since it doesn't send the
    full board back to you every step.

    Returns compact JSON, never the full board:
      {"samples": [...], "summary": {...}}
    "samples" has one entry every `sample_every` generations (always
    including the final generation, even if that's not on the sample_every
    grid), each with: generation, population, births, deaths (both since the
    previous sample), bbox (live-cell bounding box, or null if empty),
    state_hash (identical only for a bit-for-bit identical board) and
    shape_hash (translation-independent - stays the same while a pattern
    like a glider moves, changes only if its actual shape changes).
    "summary" covers the whole run: start/end generation, min/max
    population, final population/bbox, and whether the board went extinct,
    settled into a static (unchanging) state, or fell into a short exact
    repeat (repeating_period, detected within a 64-generation lookback -
    null if none found; a moving pattern's position generally won't repeat
    even if its shape does, so this mainly catches in-place oscillators).

    Two caps, both returning a structured {"error": ...} object instead of
    raising if exceeded (check for an "error" key before reading "samples"):
    at most 500 samples per call ({"error": "too_many_samples", ...,
    "min_sample_every": N} tells you the smallest sample_every that fits);
    and count is capped by board size, not just a flat number - a big board
    allows fewer generations per call than a small one, so a single call
    stays fast either way ({"error": "count_too_large_for_board", ...,
    "max_count_for_board": N} tells you the limit for the current board;
    call it repeatedly for more). The browser keeps animating live: it's
    updated once per sample point (and thus every generation if you pick
    sample_every=1)."""
    async with S.lock:
        S._truncate_future_locked()
        start_grid = S.world.grid.copy()
        edge = S.world.edge
        start_gen = S.world.generation
        cell_count = S.world.width * S.world.height

    try:
        validate_sim_params(count, sample_every, cell_count)
    except SimulationError as e:
        return json.dumps(e.to_dict())

    samples: list[dict] = []
    it = simulate(start_grid, edge, count, sample_every, start_gen)
    summary = None
    while True:
        try:
            sample, g = next(it)
        except StopIteration as stop:
            summary = stop.value["summary"]
            break
        samples.append(sample)
        async with S.lock:
            S.world.grid = g
            S.world.generation = sample["generation"]
            S._record_snapshot_locked(force=(sample["generation"] == summary["end_generation"] if summary else False))
        await broadcast_state()
        await asyncio.sleep(0)

    async with S.lock:
        S._record_snapshot_locked(force=True)

    await log_event(
        "agent",
        f"advance_generations(count={count}, sample_every={sample_every}) -> {_sim_summary_text(summary)}",
    )
    return json.dumps({"samples": samples, "summary": summary})


@mcp.tool()
async def preview_generations(count: int, sample_every: int = 1) -> str:
    """Run the SAME simulation as advance_generations from the current
    state, WITHOUT modifying the real world or the browser - use this to
    test what will happen (e.g. "does this pattern die out?") before
    committing with advance_generations. Same parameters, same sample-count
    and board-size-scaled generation caps, same {"samples": [...],
    "summary": {...}} shape (generation, population, births, deaths, bbox,
    state_hash, shape_hash per sample; extinct/stable/repeating_period
    detection in the summary), and the same structured {"error": ...}
    objects for invalid input - guaranteed to match advance_generations
    exactly for the same (count, sample_every) since both share the same
    simulation code. Nothing is committed: call advance_generations
    afterwards if you like what you see."""
    async with S.lock:
        start_grid = S.world.grid.copy()
        edge = S.world.edge
        start_gen = S.world.generation
        cell_count = S.world.width * S.world.height

    try:
        validate_sim_params(count, sample_every, cell_count)
    except SimulationError as e:
        return json.dumps(e.to_dict())

    samples: list[dict] = []
    it = simulate(start_grid, edge, count, sample_every, start_gen)
    summary = None
    n = 0
    while True:
        try:
            sample, _g = next(it)
        except StopIteration as stop:
            summary = stop.value["summary"]
            break
        samples.append(sample)
        n += 1
        if n % 25 == 0:
            await asyncio.sleep(0)  # cooperative yield only - world is untouched

    await log_event(
        "agent",
        f"preview_generations(count={count}, sample_every={sample_every}) -> "
        f"(not committed) {_sim_summary_text(summary)}",
    )
    return json.dumps({"samples": samples, "summary": summary})


# --------------------------------------------------------------- FastAPI app

# stateless + JSON responses: each MCP request is self-contained (no session
# handshake to break on server restart), plain JSON replies instead of SSE.
mcp_app = mcp.streamable_http_app(stateless_http=True, json_response=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        yield
    S.running = False
    if S.run_task:
        S.run_task.cancel()


app = FastAPI(title="Game of Life MCP", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


async def _handle_ui(msg: dict) -> None:
    action = msg.get("action")
    if action == "paint":
        cells = msg.get("cells") or []
        value = 1 if msg.get("value", 1) else 0
        async with S.lock:
            S._truncate_future_locked()
            S.world.set_cells(cells, value)
            S._record_snapshot_locked(force=True)
        await broadcast_state()
    elif action == "play":
        await set_running(True, msg.get("fps"))
        await log_event("user", f"pressed play ({S.fps} fps)")
    elif action == "pause":
        await set_running(False)
        await log_event("user", "pressed pause")
    elif action == "step":
        async with S.lock:
            S._truncate_future_locked()
            S.world.step(1)
            S._record_snapshot_locked(force=True)
        await broadcast_state()
    elif action == "advance":
        try:
            n = int(msg.get("generations", 1))
            if not 1 <= n <= MAX_ADVANCE:
                raise ValueError(f"generations must be between 1 and {MAX_ADVANCE}")
            async with S.lock:
                S._truncate_future_locked()
            for i in range(n):
                async with S.lock:
                    S.world.step(1)
                    S._record_snapshot_locked(force=(i == n - 1))
                await broadcast_state()
                if n > 1:
                    await asyncio.sleep(0.01)
            await log_event("user", f"advanced {n} generations")
        except Exception as e:
            await log_event("user", f"advance failed: {e}")
    elif action == "fps":
        fps = msg.get("fps")
        if fps:
            S.fps = min(MAX_FPS, max(MIN_FPS, float(fps)))
            await broadcast_state()
    elif action == "clear":
        async with S.lock:
            S.world.clear()
            S._reset_history_locked()
        await broadcast_state()
        await log_event("user", "cleared the world")
    elif action == "create":
        try:
            width = int(msg.get("width"))
            height = int(msg.get("height"))
            edge = str(msg.get("edge", "wrap"))
            random_fill = float(msg.get("random_fill", 0.0))
            if edge not in ("wrap", "dead"):
                raise ValueError('edge must be "wrap" or "dead"')
            if not 0.0 <= random_fill <= 1.0:
                raise ValueError("random_fill must be between 0 and 1")

            async with S.lock:
                S.world = World(width, height, edge=edge, random_fill=random_fill)
                S._reset_history_locked()
                pop = S.world.population()

            await broadcast_state()
            fill_txt = f", random_fill={random_fill}" if random_fill else ""
            await log_event("user", f"create_world {width}x{height} edge={edge}{fill_txt} -> pop {pop}")
        except Exception as e:
            await log_event("user", f"create failed: {e}")
    elif action == "rewind":
        try:
            target = int(msg.get("generation"))
            await rewind_world(target, actor="user")
        except Exception as e:
            await log_event("user", f"rewind failed: {e}")


async def _handle_ui_for_socket(ws: WebSocket, msg: dict) -> None:
    action = msg.get("action")
    if action != "preview_generation":
        await _handle_ui(msg)
        return

    try:
        target = int(msg.get("generation"))
    except Exception:
        return
    req_id = msg.get("request_id")

    async with S.lock:
        if target < 0 or target > S.max_reachable_generation:
            return
        g = S._reconstruct_generation_locked(target)
        pop = int(g.sum())
        payload = {
            "type": "preview_state",
            "generation": target,
            "population": pop,
            "cells": base64.b64encode(np.packbits(g, axis=None).tobytes()).decode("ascii"),
        }
        if isinstance(req_id, int):
            payload["request_id"] = req_id

    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        pass


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    S.sockets.add(ws)
    try:
        await ws.send_text(json.dumps(state_message()))
        await ws.send_text(json.dumps({"type": "backlog", "events": list(S.events)}))
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except json.JSONDecodeError:
                continue
            await _handle_ui_for_socket(ws, msg)
    except WebSocketDisconnect:
        pass
    finally:
        S.sockets.discard(ws)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Mounted last as a catch-all: routes above win; /mcp falls through to the
# MCP streamable HTTP app (its own internal path is /mcp).
app.mount("/", mcp_app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
