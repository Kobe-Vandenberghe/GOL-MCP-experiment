"""MCP tool definitions.

Thin contracts over WorldService: each tool calls one service method, logs the
result and phrases it for the agent. The docstrings are the agent's only
contract, so behaviour changes belong in them as much as in the code.
"""
from __future__ import annotations

from mcp.server import MCPServer

import formatting
import hub

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
    text = formatting.status_text(await hub.service.status(), len(hub.SOCKETS))
    await hub.log_event("agent", "get_world_status")
    return text


@mcp.tool()
async def create_world(width: int, height: int, edge: str = "wrap", random_fill: float = 0.0) -> str:
    """Replace the world with a new board (this discards the current one).

    width/height: 8..1024 cells. The default world is 160x100.
    edge: "wrap" = toroidal (left edge touches right, top touches bottom);
          "dead" = cells beyond the border count as permanently dead.
    random_fill: probability 0..1 that each cell starts alive (0 = empty board).
    Coordinates afterwards: (x, y), 0-indexed from the top-left."""
    result = await hub.service.create_world(width, height, edge=edge, random_fill=random_fill)
    await hub.log_event("agent", formatting.create_log(result))
    return formatting.create_text(result)


@mcp.tool()
async def clear_world() -> str:
    """Kill every cell and reset the generation counter to 0.
    Dimensions and edge behavior are kept."""
    result = await hub.service.clear()
    await hub.log_event("agent", "clear_world")
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
    result = await hub.service.observe(x, y, width, height)
    await hub.log_event(
        "agent", f"observe_world x={result.x} y={result.y} {result.width}x{result.height}"
    )
    return result.text


@mcp.tool()
async def set_cells(alive: list[list[int]] | None = None, dead: list[list[int]] | None = None) -> str:
    """Set individual cells. alive/dead are lists of [x, y] pairs, e.g.
    alive=[[10, 5], [11, 5], [12, 5]]. x = column from the left, y = row from
    the top, both 0-indexed. On wrap worlds out-of-range coordinates wrap
    around; on dead-edge worlds they are skipped and reported."""
    result = await hub.service.set_cells(alive, dead)
    await hub.log_event(
        "agent",
        f"set_cells +{result.set_alive} alive, +{result.set_dead} dead -> pop {result.population}",
    )
    return formatting.set_cells_text(result)


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
    result = await hub.service.place_pattern(rle, x, y, clear_rect=clear_rect)
    await hub.log_event(
        "agent",
        f"place_pattern {result.pattern_width}x{result.pattern_height} "
        f"({result.placed_cells} cells) at ({result.x},{result.y})",
    )
    return formatting.placement_text(result)


@mcp.tool()
async def advance(generations: int = 1) -> str:
    """Advance the simulation 1..100 generations (capped at 100 per call).
    Multi-generation advances are animated in the user's browser at ~100
    generations/second. Reports the population change and the board's
    dynamics afterwards (empty / static / period-2 / evolving)."""
    result = await hub.service.advance(generations)
    await hub.log_event(
        "agent",
        f"advance({result.generations}) -> gen {result.generation}, "
        f"pop {result.population} ({result.dynamics})",
    )
    return formatting.advance_text(result)


@mcp.tool()
async def set_autorun(enabled: bool, fps: float = 10) -> str:
    """Start or stop continuous simulation on the server (the user watches it
    run live). fps is generations per second, clamped to 1..1000. All other
    tools keep working while it runs - observe_world to peek, set_cells to
    interfere. Remember to stop it when the show is over."""
    result = await hub.service.set_autorun(enabled, fps)
    await hub.log_event(
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
    text = formatting.timeline_status_text(await hub.service.timeline_status())
    await hub.log_event("agent", "get_timeline_status")
    return text


@mcp.tool()
async def rewind_to_generation(generation: int) -> str:
    """Rewind the shared world to an earlier generation on the current
    timeline. If you continue simulating after rewinding, future history is
    discarded (linear timeline, no branching yet)."""
    return await hub.rewind_world(generation, actor="agent")


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
    sample_every=1).

    Concurrent edits: the run is computed from a snapshot of the board taken
    when the call starts, so if anything else writes to the world while it
    runs (the user drawing in the browser, pressing Clear or Play, or another
    tool call), this stops early rather than overwriting that change with
    generations derived from a board that no longer exists. You then get:
      {"samples": [...], "summary": null,
       "interrupted": {"at_generation": N, "reason": "concurrent_edit", ...}}
    The generations in "samples" ARE committed - only the remainder was
    skipped. "summary" is null because the run never finished, so check for
    an "interrupted" key before reading "summary". Call again to continue
    from wherever the world ended up."""
    run = await hub.service.advance_generations(count, sample_every)
    if run.interrupted:
        await hub.log_event("agent", formatting.sim_interrupted_log("advance_generations", run))
    elif not run.failed:
        await hub.log_event(
            "agent",
            f"advance_generations(count={count}, sample_every={sample_every}) -> "
            f"{formatting.sim_summary_text(run.summary)}",
        )
    return formatting.sim_run_json(run)


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
    exactly for the same (count, sample_every), provided nothing else
    modifies the world in between, since both walk the same simulation code
    from the same starting board. Nothing is committed and this can never be
    interrupted (it works on its own snapshot): call advance_generations
    afterwards if you like what you see."""
    run = await hub.service.preview_generations(count, sample_every)
    if not run.failed:
        await hub.log_event(
            "agent",
            f"preview_generations(count={count}, sample_every={sample_every}) -> "
            f"(not committed) {formatting.sim_summary_text(run.summary)}",
        )
    return formatting.sim_run_json(run)
