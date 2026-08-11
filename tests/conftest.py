"""Shared fixtures.

The server surface is split across hub / mcp_tools / ws. The `server` fixture
presents it as one object so tests read as "call this operation on the running
server" rather than tracking which module a name currently lives in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import formatting  # noqa: E402,F401  (imported so a syntax error here fails fast)
import hub  # noqa: E402
import mcp_tools  # noqa: E402
import ws  # noqa: E402
from gol_world import World  # noqa: E402
from world_service import WorldService  # noqa: E402


def seeded_grid(width: int, height: int, density: float, seed: int) -> np.ndarray:
    """Deterministic random board — World(random_fill=...) uses an unseeded
    rng, which is useless for tests that need two identical runs."""
    rng = np.random.default_rng(seed)
    return (rng.random((height, width)) < density).astype(np.uint8)


class ServerFacade:
    """Flat view over hub / mcp_tools / ws.

    `service` is a property so that assigning to it rebinds hub.service, which
    is what every caller actually reads.
    """

    _SOURCES = (mcp_tools, ws, hub)

    @property
    def service(self) -> WorldService:
        return hub.service

    @service.setter
    def service(self, value: WorldService) -> None:
        hub.service = value

    def __getattr__(self, name):
        for module in self._SOURCES:
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(f"no server attribute {name!r} in hub/mcp_tools/ws")


@pytest.fixture
def service():
    """A bare WorldService with no listener attached."""
    return WorldService()


@pytest.fixture
def server(monkeypatch):
    """The server surface, wired to a fresh service with broadcast/log stubbed
    so nothing needs a live WebSocket."""
    async def noop(*args, **kwargs):
        return None

    fresh = WorldService()
    monkeypatch.setattr(hub, "service", fresh)
    monkeypatch.setattr(hub, "broadcast_state", noop)
    monkeypatch.setattr(hub, "log_event", noop)
    # The service calls its listener directly, so patching the module attribute
    # is not enough — point the listener at the stub too.
    fresh.set_change_listener(noop)
    yield ServerFacade()
    hub.service.running = False


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
def commit_hook(server):
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
