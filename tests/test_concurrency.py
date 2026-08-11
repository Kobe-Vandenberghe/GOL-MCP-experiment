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
    into the simulator and corrupted every generation after it. Both runs must
    now produce identical samples regardless of the paint."""
    world_factory(width=48, height=48, density=0.3, seed=1234)
    baseline = json.loads(await server.advance_generations(count=120, sample_every=1))

    world_factory(width=48, height=48, density=0.3, seed=1234)
    hook = commit_hook(50, lambda: paint(server, GLIDER))
    interfered = json.loads(await server.advance_generations(count=120, sample_every=1))

    assert hook["fired"], "the paint never landed; the test proves nothing"
    assert interfered["samples"] == baseline["samples"]
    assert interfered["summary"] == baseline["summary"]


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


# ------------------------------------------------ lost updates (known gap)

async def test_a_concurrent_paint_is_silently_discarded(
    world_factory, server, commit_hook
):
    """Current behaviour, and the reason revision/interrupt semantics are the
    next step: the paint no longer corrupts the run, but it is thrown away
    without telling anyone. When advance_generations learns to stop on a
    concurrent edit, this test should be rewritten to assert that instead."""
    world_factory(width=48, height=48, density=0.3, seed=1234)
    hook = commit_hook(50, lambda: paint(server, GLIDER))
    await server.advance_generations(count=120, sample_every=1)

    assert hook["fired"]
    async with server.service.lock:
        final = server.service.world.grid.copy()
    assert np.array_equal(final, clean_run(48, 48, 0.3, 1234, 120))


# ---------------------------------------------- shape desync (known bug)

@pytest.mark.xfail(
    strict=True,
    reason="create_world during advance_generations leaves world.grid at the "
           "old shape while width/height report the new one; state_message() "
           "then sends a buffer the browser cannot decode. Needs the world "
           "dims and grid to be mutated atomically behind one service.",
)
async def test_create_world_mid_run_keeps_dims_and_grid_consistent(
    world_factory, server, commit_hook
):
    world_factory(width=64, height=64, density=0.3, seed=3)
    hook = commit_hook(20, lambda: server.create_world(width=200, height=120, edge="wrap"))
    await server.advance_generations(count=400, sample_every=1)

    assert hook["fired"]
    async with server.service.lock:
        w = server.service.world
        assert w.grid.shape == (w.height, w.width)


@pytest.mark.xfail(
    strict=True,
    reason="same root cause: the broadcast frame is sized from width/height "
           "but packed from a stale grid, so the browser reads past the buffer.",
)
async def test_broadcast_frame_always_matches_the_declared_dimensions(
    world_factory, server, commit_hook
):
    import base64

    world_factory(width=64, height=64, density=0.3, seed=3)
    commit_hook(20, lambda: server.create_world(width=200, height=120, edge="wrap"))
    await server.advance_generations(count=400, sample_every=1)

    msg = server.state_message()
    bits_sent = len(base64.b64decode(msg["cells"])) * 8
    assert bits_sent >= msg["width"] * msg["height"]


# ------------------------------------------- websocket robustness (known bug)

@pytest.mark.xfail(
    strict=True,
    reason="_handle_ui validates 'create'/'rewind' but not 'paint'/'fps', so a "
           "malformed message escapes to ws_endpoint, which only catches "
           "WebSocketDisconnect - dropping the browser connection.",
)
@pytest.mark.parametrize("msg", [
    {"action": "paint", "cells": [["a", "b"]], "value": 1},
    {"action": "paint", "cells": "nonsense", "value": 1},
    {"action": "fps", "fps": "fast"},
])
async def test_malformed_ui_messages_do_not_escape_the_handler(server, world_factory, msg):
    world_factory()
    await server._handle_ui(msg)


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
