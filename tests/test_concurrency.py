"""Operation-level concurrency: what happens when something writes to the
world while a long-running operation is committing to it.

advance_generations commits one sample at a time, awaiting between commits, so
a browser paint / Clear / create_world can land mid-run. These tests pin what
that does today.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from gol_world import World, next_grid_of

from conftest import seeded_grid

GLIDER = [[1, 0], [2, 1], [0, 2], [1, 2], [2, 2]]


async def paint(server, cells, value=1):
    await server._handle_ui({"action": "paint", "cells": cells, "value": value})


def clean_run(width, height, density, seed, count, edge="wrap"):
    """What the board would look like with nobody interfering."""
    g = seeded_grid(width, height, density, seed)
    for _ in range(count):
        g = next_grid_of(g, edge)
    return g


# ------------------------------------------------- the aliasing regression

async def test_a_concurrent_paint_cannot_alter_the_running_simulation(
    world_factory, server, commit_hook
):
    """Regression: gol_server used to assign simulate()'s live working array
    straight onto world.grid, so an in-place write from the browser reached
    into the simulator and corrupted every generation after it.

    The run now stops at the paint rather than running to completion, so it
    yields fewer samples than an undisturbed run — but every generation it did
    commit must be bit-for-bit what the undisturbed run produced."""
    world_factory(width=48, height=48, density=0.3, seed=1234)
    baseline = json.loads(await server.advance_generations(count=120, sample_every=1))

    world_factory(width=48, height=48, density=0.3, seed=1234)
    hook = commit_hook(50, lambda: paint(server, GLIDER))
    interfered = json.loads(await server.advance_generations(count=120, sample_every=1))

    assert hook["fired"], "the paint never landed; the test proves nothing"
    committed = interfered["samples"]
    assert committed, "the run should have committed something before stopping"
    assert len(committed) < len(baseline["samples"]), "the run should have stopped early"
    assert committed == baseline["samples"][:len(committed)]


async def test_a_concurrent_clear_cannot_make_the_summary_lie(
    world_factory, server, commit_hook
):
    """Regression: pressing Clear mid-run used to zero the simulator's own
    array, so the agent got back a summary claiming the board went extinct."""
    async def do_clear():
        async with server.service.lock:
            server.service.world.clear()
            server.service._reset_history_locked()

    world_factory(width=48, height=48, density=0.35, seed=11)
    hook = commit_hook(40, do_clear)
    out = json.loads(await server.advance_generations(count=150, sample_every=1))

    assert hook["fired"]
    expected = clean_run(48, 48, 0.35, 11, 150)
    assert out["summary"]["extinct"] is False
    assert out["summary"]["final_population"] == int(expected.sum())


async def test_the_world_grid_is_never_aliased_to_the_simulator(
    world_factory, server
):
    """Sharper than the paint test: mutate world.grid in place at one commit
    and check the very next committed generation is a pure Life step from the
    board as it was *before* the mutation."""
    world_factory(width=32, height=32, density=0.3, seed=5)
    MUTATE_AT = 20
    seen = {"n": 0, "before": None, "mutated": None, "after": None}

    async def hooked():
        seen["n"] += 1
        if seen["n"] == MUTATE_AT:
            async with server.service.lock:
                seen["before"] = server.service.world.grid.copy()
                # in-place, exactly as World.set_cells writes
                server.service.world.grid[10:13, 10:13] = 1
                seen["mutated"] = server.service.world.grid.copy()
        elif seen["n"] == MUTATE_AT + 1:
            async with server.service.lock:
                seen["after"] = server.service.world.grid.copy()

    server.service.set_change_listener(hooked)
    await server.advance_generations(count=60, sample_every=1)

    assert seen["before"] is not None and seen["after"] is not None
    # Precondition: the mutation has to be observable one step later, or this
    # test would pass vacuously whatever the server does.
    assert not np.array_equal(
        next_grid_of(seen["mutated"], "wrap"), next_grid_of(seen["before"], "wrap")
    ), "chosen mutation is invisible after one step; pick another"

    assert np.array_equal(seen["after"], next_grid_of(seen["before"], "wrap"))


# --------------------------------------------------------- interrupt semantics

async def test_a_concurrent_paint_stops_the_run_and_is_kept(
    world_factory, server, commit_hook
):
    """A browser paint mid-run is no longer silently discarded: the run stops
    at that point and the paint survives."""
    world_factory(width=48, height=48, density=0.3, seed=1234)
    hook = commit_hook(50, lambda: paint(server, GLIDER))
    out = json.loads(await server.advance_generations(count=120, sample_every=1))

    assert hook["fired"]
    assert "interrupted" in out
    assert out["interrupted"]["reason"] == "concurrent_edit"
    assert out["summary"] is None

    # the painted cells are still alive — the run did not overwrite them
    async with server.service.lock:
        grid = server.service.world.grid
        assert all(grid[y, x] == 1 for x, y in GLIDER)


async def test_an_interrupted_run_keeps_what_it_committed(
    world_factory, server, commit_hook
):
    world_factory(width=48, height=48, density=0.3, seed=1234)
    commit_hook(50, lambda: paint(server, GLIDER))
    out = json.loads(await server.advance_generations(count=120, sample_every=1))

    stopped_at = out["interrupted"]["at_generation"]
    assert out["samples"], "samples committed before the interruption are kept"
    assert [s["generation"] for s in out["samples"]] == list(range(1, len(out["samples"]) + 1))
    assert out["samples"][-1]["generation"] == stopped_at
    assert stopped_at < 120, "the run should not have reached the requested end"


async def test_the_world_is_left_at_the_generation_the_run_stopped_at(
    world_factory, server, commit_hook
):
    world_factory(width=48, height=48, density=0.3, seed=1234)
    commit_hook(50, lambda: paint(server, GLIDER))
    out = json.loads(await server.advance_generations(count=120, sample_every=1))

    async with server.service.lock:
        assert server.service.world.generation == out["interrupted"]["at_generation"]


async def test_an_uninterrupted_run_reports_no_interruption(world_factory, server):
    world_factory(width=48, height=48, density=0.3, seed=1234)
    out = json.loads(await server.advance_generations(count=60, sample_every=1))
    assert "interrupted" not in out
    assert out["summary"]["end_generation"] == 60


async def test_a_run_can_be_resumed_after_being_interrupted(
    world_factory, server, commit_hook
):
    """The documented recovery path: call again to continue."""
    world_factory(width=48, height=48, density=0.3, seed=1234)
    commit_hook(30, lambda: paint(server, GLIDER))
    first = json.loads(await server.advance_generations(count=100, sample_every=1))
    stopped_at = first["interrupted"]["at_generation"]

    second = json.loads(await server.advance_generations(count=20, sample_every=1))
    assert "interrupted" not in second
    assert second["summary"]["start_generation"] == stopped_at
    assert second["summary"]["end_generation"] == stopped_at + 20


async def test_autorun_stepping_interrupts_a_committing_run(world_factory, server):
    """Autorun is a writer like any other, so it must interrupt too rather
    than fight the batch run for the generation counter."""
    world_factory(width=48, height=48, density=0.3, seed=99)
    fired = {"n": 0}

    async def hooked():
        fired["n"] += 1
        if fired["n"] == 15:
            async with server.service.lock:
                server.service.world.step(1)
                server.service._bump_locked()

    server.service.set_change_listener(hooked)
    out = json.loads(await server.advance_generations(count=100, sample_every=1))
    assert "interrupted" in out


# ------------------------------------------------------------- shape desync

async def test_create_world_mid_run_keeps_dims_and_grid_consistent(
    world_factory, server, commit_hook
):
    """Regression: create_world used to leave world.grid at the old shape while
    width/height reported the new one, because the batch run kept committing
    stale-shaped boards over the top of it."""
    world_factory(width=64, height=64, density=0.3, seed=3)
    hook = commit_hook(20, lambda: server.create_world(width=200, height=120, edge="wrap"))
    await server.advance_generations(count=400, sample_every=1)

    assert hook["fired"]
    async with server.service.lock:
        w = server.service.world
        assert w.grid.shape == (w.height, w.width)


async def test_broadcast_frame_always_matches_the_declared_dimensions(
    world_factory, server, commit_hook
):
    """Same root cause seen from the browser's side: the frame is sized from
    width/height, so a stale grid made it undecodable."""
    import base64

    world_factory(width=64, height=64, density=0.3, seed=3)
    commit_hook(20, lambda: server.create_world(width=200, height=120, edge="wrap"))
    await server.advance_generations(count=400, sample_every=1)

    msg = server.state_message()
    bits_sent = len(base64.b64decode(msg["cells"])) * 8
    assert bits_sent >= msg["width"] * msg["height"]


async def test_create_world_mid_run_wins_over_the_batch_run(
    world_factory, server, commit_hook
):
    world_factory(width=64, height=64, density=0.3, seed=3)
    commit_hook(20, lambda: server.create_world(width=200, height=120, edge="wrap"))
    await server.advance_generations(count=400, sample_every=1)

    async with server.service.lock:
        w = server.service.world
        assert (w.width, w.height) == (200, 120)
        assert w.generation == 0        # the new world, not the interrupted run's


# ------------------------------------------------------ websocket robustness

@pytest.mark.parametrize("msg", [
    {"action": "paint", "cells": [["a", "b"]], "value": 1},
    {"action": "paint", "cells": "nonsense", "value": 1},
    {"action": "fps", "fps": "fast"},
    {"action": "step", "unexpected": object()},
])
async def test_malformed_ui_messages_do_not_escape_the_socket_handler(
    server, world_factory, msg
):
    """ws_endpoint only catches WebSocketDisconnect, so anything else escaping
    the handler drops the browser's connection."""
    world_factory()
    await server._handle_ui_for_socket(None, msg)


@pytest.mark.parametrize("msg", [
    {"action": "create", "width": "wide", "height": 10},
    {"action": "rewind", "generation": "yesterday"},
    {"action": "advance", "generations": "lots"},
])
async def test_guarded_ui_actions_swallow_bad_input(server, world_factory, msg):
    """These three already validate; pinned so the behaviour isn't lost when
    validation moves behind a service."""
    world_factory()
    await server._handle_ui(msg)


async def test_an_unknown_action_is_ignored(server, world_factory):
    world_factory()
    await server._handle_ui({"action": "definitely-not-a-real-action"})
