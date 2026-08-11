"""Snapshot retention, generation reconstruction, and rewind."""
from __future__ import annotations

import numpy as np
import pytest

from gol_world import World, next_grid_of
from world_service import unpack_grid

from conftest import seeded_grid


# ------------------------------------------------------------ snapshotting

def test_a_new_world_snapshots_generation_zero(world_factory, server):
    world_factory()
    assert server.service.history_generations == [0]
    assert server.service.max_reachable_generation == 0


def test_snapshots_land_on_the_interval(world_factory, server):
    world_factory()
    server.service.snapshot_interval = 5
    for _ in range(12):
        server.service.world.step(1)
        server.service._record_snapshot_locked()
    assert server.service.history_generations == [0, 5, 10]


def test_force_records_a_snapshot_off_the_interval(world_factory, server):
    world_factory()
    server.service.snapshot_interval = 10
    server.service.world.step(3)
    assert server.service._record_snapshot_locked() is False
    assert server.service._record_snapshot_locked(force=True) is True
    assert server.service.history_generations == [0, 3]


def test_recording_the_same_generation_twice_does_not_duplicate_it(world_factory, server):
    world_factory()
    server.service.world.step(4)
    server.service._record_snapshot_locked(force=True)
    server.service._record_snapshot_locked(force=True)
    assert server.service.history_generations == [0, 4]


def test_max_reachable_generation_tracks_the_furthest_snapshot(world_factory, server):
    world_factory()
    server.service.world.step(30)
    server.service._record_snapshot_locked(force=True)
    assert server.service.max_reachable_generation == 30


def test_snapshot_retention_is_capped(world_factory, server):
    world_factory()
    server.service.snapshot_cap = 5
    server.service.snapshot_interval = 1
    for _ in range(20):
        server.service.world.step(1)
        server.service._record_snapshot_locked()
    assert len(server.service.history_generations) <= 5
    assert server.service.history_generations[-1] == 20      # newest is kept
    assert set(server.service.history_snapshots) == set(server.service.history_generations)


# --------------------------------------------------------- reconstruction

def test_reconstructs_a_generation_that_has_an_exact_snapshot(world_factory, server):
    world_factory(density=0.3, seed=42)
    server.service.snapshot_interval = 5
    for _ in range(10):
        server.service.world.step(1)
        server.service._record_snapshot_locked()
    got = server.service._reconstruct_generation_locked(5)
    want = unpack_grid(server.service.history_snapshots[5],
                       server.service.world.width, server.service.world.height)
    assert np.array_equal(got, want)


def test_reconstructs_a_generation_between_snapshots_by_replaying(world_factory, server):
    world_factory(width=32, height=32, density=0.3, seed=7)
    server.service.snapshot_interval = 10
    start = server.service.world.grid.copy()
    for _ in range(20):
        server.service.world.step(1)
        server.service._record_snapshot_locked()

    got = server.service._reconstruct_generation_locked(13)
    want = start
    for _ in range(13):
        want = next_grid_of(want, "wrap")
    assert np.array_equal(got, want)


def test_reconstruction_round_trips_every_generation(world_factory, server):
    world_factory(width=24, height=24, density=0.35, seed=99)
    server.service.snapshot_interval = 7
    expected = {0: server.service.world.grid.copy()}
    for gen in range(1, 25):
        server.service.world.step(1)
        server.service._record_snapshot_locked()
        expected[gen] = server.service.world.grid.copy()

    for gen, want in expected.items():
        assert np.array_equal(server.service._reconstruct_generation_locked(gen), want), gen


def test_reconstruction_rejects_a_generation_older_than_retained_history(world_factory, server):
    world_factory()
    server.service.history_generations.clear()
    server.service.history_snapshots.clear()
    server.service.world.step(5)
    server.service._record_snapshot_locked(force=True)
    with pytest.raises(ValueError, match="older than retained history"):
        server.service._reconstruct_generation_locked(2)


# -------------------------------------------------------------- truncation

def test_truncating_drops_snapshots_ahead_of_the_present(world_factory, server):
    world_factory()
    server.service.snapshot_interval = 5
    for _ in range(20):
        server.service.world.step(1)
        server.service._record_snapshot_locked()
    server.service.world.generation = 7
    dropped = server.service._truncate_future_locked()
    assert dropped == 3                              # drops 10, 15, 20; keeps 0, 5
    assert server.service.history_generations == [0, 5]
    assert server.service.max_reachable_generation == 7


def test_truncating_with_no_future_is_a_no_op(world_factory, server):
    world_factory()
    server.service.world.step(10)
    server.service._record_snapshot_locked(force=True)
    assert server.service._truncate_future_locked() == 0


def test_timeline_bounds_report_the_snapshot_range(world_factory, server):
    world_factory()
    server.service.snapshot_interval = 5
    for _ in range(10):
        server.service.world.step(1)
        server.service._record_snapshot_locked()
    assert server.service._timeline_bounds_locked() == (0, 10)


# ------------------------------------------------------------------ rewind

async def test_rewind_restores_the_earlier_board(world_factory, server):
    world_factory(width=32, height=32, density=0.3, seed=11)
    at_start = server.service.world.grid.copy()
    await server.advance(generations=20)

    await server.rewind_world(0, actor="test")
    assert server.service.world.generation == 0
    assert np.array_equal(server.service.world.grid, at_start)


async def test_rewind_reports_the_move_and_the_dropped_snapshots(world_factory, server):
    world_factory()
    await server.advance(generations=20)
    msg = await server.rewind_world(10, actor="test")
    assert "generation 20 to 10" in msg
    assert "Dropped" in msg


async def test_rewind_discards_the_future(world_factory, server):
    world_factory()
    await server.advance(generations=30)
    await server.rewind_world(10, actor="test")
    assert server.service.max_reachable_generation == 10
    assert all(g <= 10 for g in server.service.history_generations)


async def test_rewind_stops_autorun(world_factory, server):
    world_factory()
    server.service.running = True
    await server.rewind_world(0, actor="test")
    assert server.service.running is False


async def test_rewind_rejects_a_negative_generation(world_factory, server):
    world_factory()
    with pytest.raises(ValueError, match="must be >= 0"):
        await server.rewind_world(-1, actor="test")


async def test_rewind_rejects_a_generation_beyond_the_timeline(world_factory, server):
    world_factory()
    await server.advance(generations=5)
    with pytest.raises(ValueError, match="outside retained history"):
        await server.rewind_world(99, actor="test")


async def test_rewinding_then_advancing_produces_the_same_board_as_never_rewinding(
    world_factory, server
):
    world_factory(width=32, height=32, density=0.3, seed=77)
    await server.advance(generations=15)
    straight_through = server.service.world.grid.copy()

    await server.rewind_world(5, actor="test")
    await server.advance(generations=10)
    assert server.service.world.generation == 15
    assert np.array_equal(server.service.world.grid, straight_through)
