# Game of Life · MCP

A shared Conway's Game of Life world that an AI agent controls through **MCP
tools** while you watch it evolve live in your browser. One Python process
serves everything:

| Surface | URL |
|---|---|
| Browser viewer | `http://localhost:8000` |
| Live state WebSocket | `ws://localhost:8000/ws` |
| MCP endpoint (Streamable HTTP) | `http://localhost:8000/mcp` |

The server owns the authoritative board (NumPy). Every MCP tool call mutates
it and immediately broadcasts the new state to all connected browser tabs.

```
Codex / Claude Code ──MCP──▶ ┌───────────────────┐
                             │  Python server    │──▶ shared GoL world
Browser UI ◀──WebSocket────  │ (FastAPI+FastMCP) │
                             └───────────────────┘
```

## Quickstart

```powershell
.\run.ps1
```

(First run creates the venv and installs dependencies. If PowerShell blocks the
script, run it as `powershell -ExecutionPolicy Bypass -File run.ps1`.)

…or manually:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn gol_server:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>. The server binds to localhost only and has no
auth — it's a local toy, don't expose it.

## Connect an agent

**Claude Code** (run in any terminal; the server must be running when the
agent starts):

```bash
claude mcp add --transport http gol http://localhost:8000/mcp
```

**Codex CLI** — add to `~/.codex/config.toml` (exact key names vary by Codex
version; check its MCP docs):

```toml
[mcp_servers.gol]
url = "http://localhost:8000/mcp"
```

## MCP tools

| Tool | What it does |
|---|---|
| `get_world_status()` | Dimensions, generation, population, live-cell bounding box, dynamics, autorun state |
| `observe_world(x, y, width, height)` | ASCII view of a region (`#` alive, `.` dead) with coordinate rulers; defaults to the full board, capped at 20 000 cells |
| `create_world(width, height, edge, random_fill)` | New board, 8–1024 per side; `edge="wrap"` (toroidal, default) or `"dead"`; optional random soup |
| `clear_world()` | Kill everything, reset generation to 0 |
| `set_cells(alive, dead)` | Set individual cells from lists of `[x, y]` pairs |
| `place_pattern(rle, x, y, clear_rect)` | Stamp a standard RLE pattern (e.g. glider `bob$2bo$3o!`) at a position |
| `advance(generations)` | Step the simulation, 1–100 per call, animated in the browser |
| `advance_generations(count, sample_every)` | Efficiently commit up to thousands of generations in one call, returning compact per-sample stats (not the full board) plus a run summary |
| `preview_generations(count, sample_every)` | Same as `advance_generations` but doesn't touch the world — test what would happen before committing |
| `set_autorun(enabled, fps)` | Continuous simulation on the server, 1–1000 gen/s |

Coordinates everywhere: `(x, y)`, 0-indexed, `(0,0)` top-left, x → right,
y ↓ down.

### `advance_generations` / `preview_generations`

Both return `{"samples": [...], "summary": {...}}` as a JSON string — never
the full board, so responses stay small even over thousands of generations.
`sample_every` controls how often a sample is recorded (always including the
final generation); each sample has `generation`, `population`, `births` and
`deaths` (since the previous sample), `bbox`, a `state_hash` (matches only a
bit-for-bit identical board) and a `shape_hash` (translation-independent —
stays the same while a pattern like a glider moves). The `summary` reports
start/end generation, min/max population, and whether the run went extinct,
settled into a static state, or fell into a short repeating cycle.

Two caps apply, both surfaced as a `{"error": ...}` object in the response
(not a thrown error) when exceeded:

- **At most 500 samples per call.** Raise `sample_every` if you hit this —
  the error tells you the minimum value that fits.
- **`count` is capped by board size, not a flat number.** The real cost of a
  run is `width × height × count`, so a 1024×1024 board allows far fewer
  generations per call than the default 160×100 board — the error reports
  the exact ceiling for the current board. Call the tool repeatedly for
  more.

`advance_generations` commits each sample to the live world as it goes (so
the browser animates progress at the `sample_every` cadence); `preview_generations`
never touches the world or the browser. Both walk the identical simulation
code, so a preview and a subsequent advance with the same arguments always
agree exactly.

## Browser viewer

- **Play / Pause / Step / speed** — drives the same server-side simulation the
  agent uses.
- **Draw / Erase / Pan** — left-drag paints (right-drag always erases), wheel
  zooms, middle-drag pans, **Fit** re-centers.
- **Activity panel** — live feed of every MCP tool call, so you can watch the
  agent think.

## Challenge ideas

- "Place a Gosper glider gun and report the population after 200 generations."
- "I've drawn a mess in the middle — stabilize the board into only still lifes."
- "Make two gliders collide head-on and tell me what survives."
- "`create_world(300, 200, random_fill=0.2)`, run it until it settles, then
  find and report the coordinates of every oscillator."
- "Write my initials using still lifes."

## Tests

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest
```

`tests/test_world.py`, `test_rle.py` and `test_simulate.py` cover the pure
simulation layer. `test_history.py` covers snapshot retention, generation
reconstruction and rewind. `test_service.py` covers `WorldService`'s structured
results and asserts that each operation behaves identically whether it arrives
from MCP or from the browser. `test_concurrency.py` covers what happens when
the browser writes to the world while a long `advance_generations` is
committing to it.

A handful of tests are `xfail(strict=True)` — they document known bugs that
are not fixed yet, and will start failing loudly (as unexpected passes) once
they are:

- `create_world` during `advance_generations` leaves `world.grid` at the old
  shape while `width`/`height` report the new one, so the broadcast frame is
  sized from one and packed from the other.
- `_handle_ui` validates `create`/`rewind` but not `paint`/`fps`, so a
  malformed WebSocket message escapes to `ws_endpoint` (which only catches
  `WebSocketDisconnect`) and drops the browser connection.

## Files

- `gol_world.py` — pure simulation: NumPy stepping, RLE parser, ASCII renderer
- `world_service.py` — authoritative world: state, locking, rewind history,
  autorun, and every mutating operation. Returns dataclasses, knows nothing
  about HTTP or WebSockets.
- `gol_server.py` — transport: FastAPI app, WebSocket hub, MCP tool
  definitions, and the prose formatting that turns service results into text
  for the agent.
- `static/` — the browser viewer (vanilla JS + canvas)
- `tests/` — pytest suite (see above)

MCP tools and browser commands both call the same `WorldService` methods, so
each operation has exactly one implementation; only the phrasing of the reply
and the activity-log line differ.
