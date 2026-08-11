"""Shared fixtures.

gol_server holds a module-level WorldService, so tests that touch the server
install a fresh one and stub the broadcast/log coroutines that would otherwise
need a live WebSocket.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gol_server as gs  # noqa: E402
from gol_world import World  # noqa: E402
from world_service import WorldService  # noqa: E402


def seeded_grid(width: int, height: int, density: float, seed: int) -> np.ndarray:
    """Deterministic random board — World(random_fill=...) uses an unseeded
    rng, which is useless for tests that need two identical runs."""
    rng = np.random.default_rng(seed)
    return (rng.random((height, width)) < density).astype(np.uint8)


@pytest.fixture
def service():
    """A bare WorldService with no listener attached."""
    return WorldService()


@pytest.fixture
def server(monkeypatch):
    """gol_server wired to a fresh service, with broadcast/log stubbed.

    Yields the module. `server.service` is the WorldService under test.
    """
    async def noop(*args, **kwargs):
        return None

    fresh = WorldService()
    monkeypatch.setattr(gs, "service", fresh)
    monkeypatch.setattr(gs, "broadcast_state", noop)
    monkeypatch.setattr(gs, "log_event", noop)
    # The service calls its listener directly, so stubbing the module attribute
    # is not enough — point the listener at the stub too.
    fresh.set_change_listener(noop)
    yield gs
    fresh.running = False


@pytest.fixture
def world_factory(server):
    """Install a deterministic world on the server's service."""
    def make(width=48, height=48, edge="wrap", density=0.3, seed=1234):
        w = World(width, height, edge=edge)
        if density:
            w.grid = seeded_grid(width, height, density, seed)
        server.service.world = w
        server.service._reset_history_locked()
        return w
    return make


@pytest.fixture
def commit_hook(server, monkeypatch):
    """Run a callback from inside advance_generations' commit loop.

    advance_generations awaits its change listener once per committed sample,
    so replacing it lets a test inject an interfering write at an exact sample
    index instead of racing a sleep timer.
    """
    def install(at_sample, action):
        calls = {"n": 0, "fired": False}

        async def hooked():
            calls["n"] += 1
            if calls["n"] == at_sample and not calls["fired"]:
                calls["fired"] = True
                await action()

        server.service.set_change_listener(hooked)
        return calls

    return install
