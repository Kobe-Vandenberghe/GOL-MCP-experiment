"""Shared Game of Life server.

One process serves three surfaces:
  - Browser UI      http://localhost:8000       (static viewer + controls)
  - Live state feed ws://localhost:8000/ws      (broadcasts board + events)
  - MCP endpoint    http://localhost:8000/mcp   (Streamable HTTP, for agents)

WorldService owns the authoritative board; this module is the transport layer
on top of it. MCP tools and browser commands both call the same service
methods, so there is exactly one implementation of each operation — what
differs here is only how the result is phrased and who it is reported to.
Every mutation broadcasts fresh state to all connected viewers.
"""
from __future__ import annotations

import base64
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mcp.server import MCPServer

from world_service import (
    AdvanceResult,
    CreateResult,
    PlacementResult,
    RewindResult,
    SetCellsResult,
    SimulationRun,
    StateSnapshot,
    TimelineStatus,
    WorldService,
    WorldStatus,
)

STATIC_DIR = Path(__file__).parent / "static"

service = WorldService()
SOCKETS: set[WebSocket] = set()
EVENTS: deque[dict] = deque(maxlen=200)


# ------------------------------------------------------------------ broadcast


def state_message() -> dict:
    s: StateSnapshot = service.state_snapshot()
    return {
        "type": "state",
        "width": s.width,
        "height": s.height,
        "edge": s.edge,
        "generation": s.generation,
        "population": s.population,
        "running": s.running,
        "fps": s.fps,
        "timeline_min_generation": s.timeline_min_generation,
        "timeline_latest_snapshot": s.timeline_latest_snapshot,
        "timeline_latest_generation": s.timeline_latest_generation,
        "timeline_snapshot_interval": s.timeline_snapshot_interval,
        "timeline_snapshot_count": s.timeline_snapshot_count,
        "cells": base64.b64encode(s.packed_cells).decode("ascii"),
    }


async def _send_all(payload: dict) -> None:
    if not SOCKETS:
        return
    text = json.dumps(payload)
    dead = []
    for ws in list(SOCKETS):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        SOCKETS.discard(ws)


async def broadcast_state() -> None:
    await _send_all(state_message())


async def log_event(source: str, text: str) -> None:
    evt = {"type": "event", "source": source, "text": text, "ts": time.time()}
    EVENTS.append(evt)
    await _send_all(evt)


service.set_change_listener(broadcast_state)


# ------------------------------------------------------------- prose helpers
# The service returns dataclasses; MCP tools return text. These turn one into
# the other so the phrasing lives in one place per operation.


def _status_text(st: WorldStatus) -> str:
    box_txt = (
        f"x=[{st.bbox[0]}..{st.bbox[2]}] y=[{st.bbox[1]}..{st.bbox[3]}]"
        if st.bbox else "none (board is empty)"
    )
    edge_txt = (
        "toroidal wrap-around" if st.edge == "wrap"
        else "cells beyond the border count as dead"
    )
    return (
        f"World {st.width}x{st.height}, edge={st.edge} ({edge_txt})\n"
        f"Generation {st.generation}, population {st.population}\n"
        f"Live-cell bounding box: {box_txt}\n"
        f"Dynamics: {st.dynamics}\n"
        f"Autorun: {'ON at ' + str(st.fps) + ' fps' if st.running else 'off'}\n"
        f"Connected browser viewers: {len(SOCKETS)}"
    )


def _timeline_status_text(t: TimelineStatus) -> str:
    return (
        f"Timeline status\n"
        f"Current generation: {t.current_generation}\n"
        f"Earliest retained snapshot: {t.earliest_snapshot}\n"
        f"Latest retained snapshot: {t.latest_snapshot}\n"
        f"Latest reachable generation: {t.max_reachable_generation}\n"
        f"Snapshot interval target: every {t.snapshot_interval} generations\n"
        f"Stored snapshots: {t.snapshot_count} (cap {t.snapshot_cap})\n"
        f"Rewind range right now: [{t.earliest_snapshot}..{t.max_reachable_generation}]"
    )


def _create_text(r: CreateResult) -> str:
    return (
        f"Created {r.width}x{r.height} world (edge={r.edge}), "
        f"generation 0, population {r.population}."
    )


def _create_log(r: CreateResult) -> str:
    fill_txt = f", random_fill={r.random_fill}" if r.random_fill else ""
    return f"create_world {r.width}x{r.height} edge={r.edge}{fill_txt} -> pop {r.population}"


def _set_cells_text(r: SetCellsResult) -> str:
    msg = (
        f"Set {r.set_alive} cells alive and {r.set_dead} dead. "
        f"Population is now {r.population}."
    )
    if r.skipped:
        msg += f" Skipped {len(r.skipped)} out-of-bounds cells: {r.skipped[:10]}"
    return msg


def _placement_text(r: PlacementResult) -> str:
    msg = (
        f"Placed {r.pattern_width}x{r.pattern_height} pattern with {r.placed_cells} "
        f"live cells at ({r.x},{r.y})..({r.x + r.pattern_width - 1},"
        f"{r.y + r.pattern_height - 1}). Population is now {r.population}."
    )
    if r.skipped:
        msg += (
            f" {len(r.skipped)} cells fell outside the dead-edge board and were skipped."
        )
    return msg


def _advance_text(r: AdvanceResult) -> str:
    return (
        f"Advanced {r.generations} generation{'s' if r.generations != 1 else ''} "
        f"-> generation {r.generation}. Population {r.population_before} -> "
        f"{r.population} ({r.population_delta:+d}). Dynamics: {r.dynamics}."
    )


def _rewind_text(r: RewindResult) -> str:
    return (
        f"Moved world from generation {r.from_generation} to {r.to_generation}. "
        f"Population is now {r.population}. "
        f"Dropped {r.dropped_snapshots} future snapshots from the timeline."
    )


def _sim_summary_text(summary: dict) -> str:
    bits = [
        f"gen {summary['start_generation']} -> {summary['end_generation']}",
        f"pop {summary['final_population']}",
    ]
    if summary["extinct"]:
        bits.append(f"extinct at gen {summary['extinct_at_generation']}")
    if summary["stable"]:
        bits.append(f"stable at gen {summary['stable_at_generation']}")
    if summary["repeating_period"]:
        bits.append(f"period {summary['repeating_period']}")
    return ", ".join(bits)


def _sim_run_json(run: SimulationRun) -> str:
    if run.failed:
        return json.dumps(run.error)
    return json.dumps({"samples": run.samples, "summary": run.summary})


async def rewind_world(target_generation: int, actor: str) -> str:
    """Shared by the MCP tool and the browser's rewind button."""
    result = await service.rewind(target_generation)
    await log_event(
        actor,
        f"rewind_to_generation({result.to_generation}) -> pop {result.population}, "
        f"dropped {result.dropped_snapshots} future snapshots",
    )
    return _rewind_text(result)


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


@mcp.tool()
async def get_world_status() -> str:
    """Get the world's dimensions, edge behavior, generation, population,
    live-cell bounding box, dynamics (empty/static/period-2/evolving), and
    autorun state. Cheap - call this to orient yourself before other tools."""
    text = _status_text(await service.status())
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
    result = await service.create_world(width, height, edge=edge, random_fill=random_fill)
    await log_event("agent", _create_log(result))
    return _create_text(result)


@mcp.tool()
async def clear_world() -> str:
    """Kill every cell and reset the generation counter to 0.
    Dimensions and edge behavior are kept."""
    result = await service.clear()
    await log_event("agent", "clear_world")
    return f"Cleared: {result.width}x{result.height} board is empty, generation reset to 0."


@mcp.tool()
async def observe_world(x: int = 0, y: int = 0, width: int = 0, height: int = 0) -> str:
    """See a region of the board as ASCII art: '#' = alive, '.' = dead.
    Includes an x-axis ruler (tens digit row + units digit row) and absolute
    y labels on each row so you can read off exact coordinates.

    (x, y) is the top-left corner of the window; width/height its size.
    Defaults (all 0) show the full board. Output is capped at 20000 cells -
    for big boards, pass a window (get_world_status's bounding box tells you
    where the live cells are)."""
    result = await service.observe(x, y, width, height)
    await log_event(
        "agent", f"observe_world x={result.x} y={result.y} {result.width}x{result.height}"
    )
    return result.text


@mcp.tool()
async def set_cells(alive: list[list[int]] | None = None, dead: list[list[int]] | None = None) -> str:
    """Set individual cells. alive/dead are lists of [x, y] pairs, e.g.
    alive=[[10, 5], [11, 5], [12, 5]]. x = column from the left, y = row from
    the top, both 0-indexed. On wrap worlds out-of-range coordinates wrap
    around; on dead-edge worlds they are skipped and reported."""
    result = await service.set_cells(alive, dead)
    await log_event(
        "agent",
        f"set_cells +{result.set_alive} alive, +{result.set_dead} dead -> pop {result.population}",
    )
    return _set_cells_text(result)


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
    result = await service.place_pattern(rle, x, y, clear_rect=clear_rect)
    await log_event(
        "agent",
        f"place_pattern {result.pattern_width}x{result.pattern_height} "
        f"({result.placed_cells} cells) at ({result.x},{result.y})",
    )
    return _placement_text(result)


@mcp.tool()
async def advance(generations: int = 1) -> str:
    """Advance the simulation 1..100 generations (capped at 100 per call).
    Multi-generation advances are animated in the user's browser at ~100
    generations/second. Reports the population change and the board's
    dynamics afterwards (empty / static / period-2 / evolving)."""
    result = await service.advance(generations)
    await log_event(
        "agent",
        f"advance({result.generations}) -> gen {result.generation}, "
        f"pop {result.population} ({result.dynamics})",
    )
    return _advance_text(result)


@mcp.tool()
async def set_autorun(enabled: bool, fps: float = 10) -> str:
    """Start or stop continuous simulation on the server (the user watches it
    run live). fps is generations per second, clamped to 1..1000. All other
    tools keep working while it runs - observe_world to peek, set_cells to
    interfere. Remember to stop it when the show is over."""
    result = await service.set_autorun(enabled, fps)
    await log_event(
        "agent", f"set_autorun {'ON at ' + str(result.fps) + ' fps' if enabled else 'OFF'}"
    )
    if enabled:
        return (
            f"Autorun ON at {result.fps} generations/second. "
            "The world is now evolving continuously."
        )
    return f"Autorun OFF at generation {result.generation}, population {result.population}."


@mcp.tool()
async def get_timeline_status() -> str:
    """Inspect rewind history: current generation, retained snapshot range,
    snapshot interval, and how many snapshots are stored."""
    text = _timeline_status_text(await service.timeline_status())
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
    run = await service.advance_generations(count, sample_every)
    if not run.failed:
        await log_event(
            "agent",
            f"advance_generations(count={count}, sample_every={sample_every}) -> "
            f"{_sim_summary_text(run.summary)}",
        )
    return _sim_run_json(run)


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
    run = await service.preview_generations(count, sample_every)
    if not run.failed:
        await log_event(
            "agent",
            f"preview_generations(count={count}, sample_every={sample_every}) -> "
            f"(not committed) {_sim_summary_text(run.summary)}",
        )
    return _sim_run_json(run)


# --------------------------------------------------------------- FastAPI app

# stateless + JSON responses: each MCP request is self-contained (no session
# handshake to break on server restart), plain JSON replies instead of SSE.
mcp_app = mcp.streamable_http_app(stateless_http=True, json_response=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        yield
    await service.shutdown()


app = FastAPI(title="Game of Life MCP", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ------------------------------------------------------- browser commands
# Each branch maps a UI action onto the same service method the matching MCP
# tool uses; only the phrasing of the activity-log line differs.


async def _handle_ui(msg: dict) -> None:
    action = msg.get("action")
    if action == "paint":
        cells = msg.get("cells") or []
        if not cells:
            return          # an empty stroke was a no-op before the service
        value = 1 if msg.get("value", 1) else 0
        if value:
            await service.set_cells(alive=cells)
        else:
            await service.set_cells(dead=cells)
    elif action == "play":
        result = await service.set_autorun(True, msg.get("fps"))
        await log_event("user", f"pressed play ({result.fps} fps)")
    elif action == "pause":
        await service.set_autorun(False)
        await log_event("user", "pressed pause")
    elif action == "step":
        await service.advance(1)
    elif action == "advance":
        try:
            result = await service.advance(int(msg.get("generations", 1)))
            await log_event("user", f"advanced {result.generations} generations")
        except Exception as e:
            await log_event("user", f"advance failed: {e}")
    elif action == "fps":
        fps = msg.get("fps")
        if fps:
            await service.set_autorun(service.running, fps)
    elif action == "clear":
        await service.clear()
        await log_event("user", "cleared the world")
    elif action == "create":
        try:
            result = await service.create_world(
                int(msg.get("width")),
                int(msg.get("height")),
                edge=str(msg.get("edge", "wrap")),
                random_fill=float(msg.get("random_fill", 0.0)),
            )
            await log_event("user", _create_log(result))
        except Exception as e:
            await log_event("user", f"create failed: {e}")
    elif action == "rewind":
        try:
            await rewind_world(int(msg.get("generation")), actor="user")
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

    preview = await service.preview_generation(target)
    if preview is None:
        return

    payload = {
        "type": "preview_state",
        "generation": preview.generation,
        "population": preview.population,
        "cells": base64.b64encode(preview.packed_cells).decode("ascii"),
    }
    req_id = msg.get("request_id")
    if isinstance(req_id, int):
        payload["request_id"] = req_id

    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        pass


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    SOCKETS.add(ws)
    try:
        await ws.send_text(json.dumps(state_message()))
        await ws.send_text(json.dumps({"type": "backlog", "events": list(EVENTS)}))
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except json.JSONDecodeError:
                continue
            await _handle_ui_for_socket(ws, msg)
    except WebSocketDisconnect:
        pass
    finally:
        SOCKETS.discard(ws)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Mounted last as a catch-all: routes above win; /mcp falls through to the
# MCP streamable HTTP app (its own internal path is /mcp).
app.mount("/", mcp_app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
