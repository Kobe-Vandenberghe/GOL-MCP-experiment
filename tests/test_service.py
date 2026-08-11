"""WorldService: structured results, and the MCP/browser equivalence that the
extraction exists to guarantee.

The point of moving operations behind the service is that `create_world` (say)
has exactly one implementation, whichever surface asked for it. The
equivalence tests below fail if the two paths ever drift apart again.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from gol_world import World
from world_service import MAX_ADVANCE, WorldService, unpack_grid

from conftest import seeded_grid


def fingerprint(svc: WorldService) -> tuple:
    """Everything a caller could observe about the world's state."""
    s = svc.state_snapshot()
    return (s.width, s.height, s.edge, s.generation, s.population,
            s.timeline_min_generation, s.timeline_latest_generation,
            s.timeline_snapshot_count, s.packed_cells)


def seeded_service(width=32, height=32, density=0.3, seed=8) -> WorldService:
    w = World(width, height, edge="wrap")
    if density:
        w.grid = seeded_grid(width, height, density, seed)
    svc = WorldService(w)
    return svc


# ------------------------------------------------- standalone construction

def test_service_is_usable_without_any_transport():
    """The whole reason for the extraction: a world you can construct and
    drive in a test, with no server, socket or global state."""
    svc = WorldService(World(16, 16))
    assert svc.world.width == 16
    assert svc.history_generations == [0]


async def test_service_operations_work_with_no_change_listener():
    svc = WorldService(World(16, 16))
    await svc.set_cells(alive=[[1, 1], [2, 1], [3, 1]])
    result = await svc.advance(2)
    assert result.generation == 2
    assert result.population == 3


async def test_change_listener_fires_once_per_committed_generation():
    calls = {"n": 0}

    async def on_change():
        calls["n"] += 1

    svc = WorldService(World(16, 16), on_change=on_change)
    await svc.advance(5)
    assert calls["n"] == 5


async def test_preview_generations_never_notifies_or_commits():
    calls = {"n": 0}

    async def on_change():
        calls["n"] += 1

    svc = seeded_service()
    svc.set_change_listener(on_change)
    before = svc.world.grid.copy()
    run = await svc.preview_generations(count=20, sample_every=5)

    assert calls["n"] == 0
    assert run.committed is False
    assert svc.world.generation == 0
    assert np.array_equal(svc.world.grid, before)


# ----------------------------------------------------------- the revision

async def test_revision_starts_at_zero():
    assert WorldService(World(16, 16)).revision == 0


@pytest.mark.parametrize("name,op", [
    ("create_world", lambda s: s.create_world(24, 24)),
    ("clear", lambda s: s.clear()),
    ("set_cells", lambda s: s.set_cells(alive=[[1, 1]])),
    ("place_pattern", lambda s: s.place_pattern("3o!", 2, 2)),
    ("advance", lambda s: s.advance(1)),
])
async def test_every_mutation_bumps_the_revision(name, op):
    svc = seeded_service()
    before = svc.revision
    await op(svc)
    assert svc.revision > before, f"{name} did not bump the revision"


async def test_advance_bumps_once_per_generation():
    svc = seeded_service()
    before = svc.revision
    await svc.advance(5)
    assert svc.revision == before + 5


async def test_rewind_bumps_the_revision():
    svc = seeded_service()
    await svc.advance(10)
    before = svc.revision
    await svc.rewind(3)
    assert svc.revision > before


@pytest.mark.parametrize("name,op", [
    ("status", lambda s: s.status()),
    ("timeline_status", lambda s: s.timeline_status()),
    ("observe", lambda s: s.observe()),
    ("preview_generation", lambda s: s.preview_generation(0)),
    ("preview_generations", lambda s: s.preview_generations(count=10, sample_every=5)),
])
async def test_read_only_operations_do_not_bump_the_revision(name, op):
    svc = seeded_service()
    await svc.advance(5)
    before = svc.revision
    await op(svc)
    assert svc.revision == before, f"{name} bumped the revision but changes nothing"


async def test_status_reports_the_revision():
    svc = seeded_service()
    await svc.advance(3)
    st = await svc.status()
    assert st.revision == svc.revision


async def test_advance_generations_bumps_once_per_committed_sample():
    svc = seeded_service()
    before = svc.revision
    run = await svc.advance_generations(count=20, sample_every=5)
    assert not run.interrupted
    assert svc.revision == before + len(run.samples)


# ------------------------------------------------------ structured results

async def test_create_result_reports_the_new_board():
    svc = WorldService()
    r = await svc.create_world(40, 24, edge="dead", random_fill=0.0)
    assert (r.width, r.height, r.edge, r.population) == (40, 24, "dead", 0)
    assert r.random_fill == 0.0


async def test_create_world_rejects_a_bad_random_fill():
    svc = WorldService()
    with pytest.raises(ValueError, match="random_fill"):
        await svc.create_world(32, 32, random_fill=1.5)


async def test_create_world_rejects_a_bad_edge():
    svc = WorldService()
    with pytest.raises(ValueError, match="wrap"):
        await svc.create_world(32, 32, edge="bogus")


async def test_set_cells_result_counts_alive_dead_and_skipped():
    svc = WorldService(World(16, 16, edge="dead"))
    r = await svc.set_cells(alive=[[1, 1], [2, 2], [99, 99]], dead=[[1, 1]])
    assert r.set_alive == 2
    assert r.set_dead == 1
    assert r.skipped == [(99, 99)]
    assert r.population == 1


async def test_set_cells_requires_at_least_one_list():
    svc = WorldService()
    with pytest.raises(ValueError, match="at least one"):
        await svc.set_cells()


async def test_placement_result_describes_the_stamp():
    svc = WorldService(World(32, 32))
    r = await svc.place_pattern("bob$2bo$3o!", x=5, y=5)
    assert (r.pattern_width, r.pattern_height) == (3, 3)
    assert (r.x, r.y) == (5, 5)
    assert r.placed_cells == 5
    assert r.population == 5
    assert r.bbox == (5, 5, 7, 7)
    assert r.generation == 0


async def test_place_pattern_with_clear_rect_wipes_the_rectangle_first():
    svc = WorldService(World(32, 32))
    await svc.set_cells(alive=[[5, 5], [6, 6], [7, 7]])
    r = await svc.place_pattern("3o!", x=5, y=5, clear_rect=True)
    # the 3x1 rect at (5,5) is cleared, so (5,5) is replaced but (6,6)/(7,7) live
    assert r.population == 5


async def test_place_pattern_rejects_bad_rle_without_touching_the_world():
    svc = seeded_service()
    before = svc.world.grid.copy()
    with pytest.raises(ValueError):
        await svc.place_pattern("not-an-rle-%%%", x=0, y=0)
    assert np.array_equal(svc.world.grid, before)


async def test_advance_result_reports_the_population_delta():
    svc = WorldService(World(16, 16))
    await svc.set_cells(alive=[[5, 5], [6, 5], [7, 5]])   # blinker
    r = await svc.advance(1)
    assert r.generations == 1
    assert r.generation == 1
    assert r.population_before == 3
    assert r.population == 3
    assert r.population_delta == 0
    assert r.dynamics == "period-2"


@pytest.mark.parametrize("n", [0, -1, MAX_ADVANCE + 1])
async def test_advance_rejects_out_of_range_counts(n):
    svc = WorldService()
    with pytest.raises(ValueError, match="between 1 and"):
        await svc.advance(n)


async def test_rewind_result_reports_the_move():
    svc = seeded_service()
    await svc.advance(20)
    r = await svc.rewind(5)
    assert r.from_generation == 20
    assert r.to_generation == 5
    assert r.dropped_snapshots >= 1


async def test_timeline_status_reports_the_retained_range():
    svc = seeded_service()
    await svc.advance(25)
    t = await svc.timeline_status()
    assert t.current_generation == 25
    assert t.earliest_snapshot == 0
    assert t.max_reachable_generation == 25
    assert t.snapshot_count == len(svc.history_generations)


async def test_world_status_reports_bbox_and_dynamics():
    svc = WorldService(World(32, 32))
    await svc.set_cells(alive=[[4, 4], [5, 4], [4, 5], [5, 5]])   # block
    st = await svc.status()
    assert st.bbox == (4, 4, 5, 5)
    assert st.dynamics == "static"
    assert st.population == 4
    assert st.running is False


async def test_observe_reports_the_resolved_window():
    svc = WorldService(World(40, 30))
    r = await svc.observe()                    # 0 means "whole board"
    assert (r.width, r.height) == (40, 30)
    assert "40x30" in r.text


async def test_observe_rejects_an_oversized_window():
    svc = WorldService(World(1024, 1024))
    with pytest.raises(ValueError, match="max is"):
        await svc.observe()


async def test_simulation_run_carries_a_structured_error_instead_of_raising():
    svc = WorldService(World(16, 16))
    run = await svc.advance_generations(count=999999, sample_every=1)
    assert run.failed
    assert run.error["error"] == "count_too_large"
    assert run.samples == []
    assert run.committed is False


async def test_preview_and_advance_agree_exactly():
    """The guarantee the tool docstrings make to the agent."""
    a = seeded_service(seed=31)
    b = seeded_service(seed=31)
    preview = await a.preview_generations(count=60, sample_every=10)
    advanced = await b.advance_generations(count=60, sample_every=10)
    assert preview.samples == advanced.samples
    assert preview.summary == advanced.summary


# ------------------------------------------- MCP / browser equivalence
# Each pair drives the same operation through both surfaces and compares the
# resulting world. These are what stop the two paths drifting apart again.


async def run_both(server, mcp_call, ui_msg, setup=None):
    """Run `mcp_call` on one fresh service and `ui_msg` on another, both
    starting from the same board, and return the two fingerprints."""
    results = []
    for which in ("mcp", "ui"):
        svc = seeded_service(density=0.0)
        server.service = svc
        if setup is not None:
            await setup(svc)
        if which == "mcp":
            await mcp_call()
        else:
            await server._handle_ui(ui_msg)
        results.append(fingerprint(svc))
    return results


async def test_create_world_is_identical_from_mcp_and_the_browser(server):
    a, b = await run_both(
        server,
        lambda: server.create_world(width=40, height=24, edge="dead", random_fill=0.0),
        {"action": "create", "width": 40, "height": 24, "edge": "dead", "random_fill": 0.0},
    )
    assert a == b


async def test_clear_is_identical_from_mcp_and_the_browser(server):
    async def setup(svc):
        await svc.set_cells(alive=[[1, 1], [2, 2]])
        await svc.advance(3)

    a, b = await run_both(server, lambda: server.clear_world(), {"action": "clear"}, setup)
    assert a == b


async def test_single_step_is_identical_from_mcp_and_the_browser(server):
    async def setup(svc):
        await svc.set_cells(alive=[[5, 5], [6, 5], [7, 5]])

    a, b = await run_both(
        server, lambda: server.advance(generations=1), {"action": "step"}, setup
    )
    assert a == b


async def test_multi_generation_advance_is_identical_from_mcp_and_the_browser(server):
    async def setup(svc):
        await svc.set_cells(alive=[[1, 0], [2, 1], [0, 2], [1, 2], [2, 2]])

    a, b = await run_both(
        server,
        lambda: server.advance(generations=6),
        {"action": "advance", "generations": 6},
        setup,
    )
    assert a == b


async def test_setting_cells_is_identical_from_mcp_and_the_browser(server):
    a, b = await run_both(
        server,
        lambda: server.set_cells(alive=[[3, 3], [4, 4], [5, 5]]),
        {"action": "paint", "cells": [[3, 3], [4, 4], [5, 5]], "value": 1},
    )
    assert a == b


async def test_erasing_cells_is_identical_from_mcp_and_the_browser(server):
    async def setup(svc):
        await svc.set_cells(alive=[[3, 3], [4, 4], [5, 5]])

    a, b = await run_both(
        server,
        lambda: server.set_cells(dead=[[3, 3], [4, 4]]),
        {"action": "paint", "cells": [[3, 3], [4, 4]], "value": 0},
        setup,
    )
    assert a == b


async def test_rewind_is_identical_from_mcp_and_the_browser(server):
    async def setup(svc):
        await svc.set_cells(alive=[[1, 0], [2, 1], [0, 2], [1, 2], [2, 2]])
        await svc.advance(12)

    a, b = await run_both(
        server,
        lambda: server.rewind_to_generation(generation=4),
        {"action": "rewind", "generation": 4},
        setup,
    )
    assert a == b


# --------------------------------------------------------- wire formatting

def test_state_message_frame_matches_the_declared_dimensions(server):
    server.service = seeded_service(width=48, height=32, density=0.4, seed=2)
    import base64

    msg = server.state_message()
    raw = base64.b64decode(msg["cells"])
    assert len(raw) * 8 >= msg["width"] * msg["height"]
    grid = unpack_grid(raw, msg["width"], msg["height"])
    assert int(grid.sum()) == msg["population"]


async def test_advance_generations_tool_returns_the_documented_json(server):
    server.service = seeded_service(width=24, height=24, seed=4)
    out = json.loads(await server.advance_generations(count=10, sample_every=5))
    assert set(out) == {"samples", "summary"}
    assert [s["generation"] for s in out["samples"]] == [5, 10]
    assert out["summary"]["end_generation"] == 10


async def test_simulation_cap_errors_reach_the_agent_as_json_not_an_exception(server):
    server.service = seeded_service(width=24, height=24, seed=4)
    out = json.loads(await server.advance_generations(count=10, sample_every=11))
    assert out["error"] == "sample_every_too_large"
