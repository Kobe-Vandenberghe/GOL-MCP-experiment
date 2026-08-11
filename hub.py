"""Shared transport state: the live world, the connected viewers, and the
activity log.

Everything that both surfaces need sits here, so mcp_tools and ws depend on
this rather than on each other. `service` is looked up as an attribute
(`hub.service`) rather than imported by value, so tests can swap in a fresh
WorldService and have every caller see it.
"""
from __future__ import annotations

import base64
import json
import time
from collections import deque

from fastapi import WebSocket

import formatting
from world_service import StateSnapshot, WorldService

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


# ----------------------------------------------------------- shared operations


async def rewind_world(target_generation: int, actor: str) -> str:
    """Used by both the MCP tool and the browser's rewind button; `actor` only
    decides who the activity log attributes it to."""
    result = await service.rewind(target_generation)
    await log_event(
        actor,
        f"rewind_to_generation({result.to_generation}) -> pop {result.population}, "
        f"dropped {result.dropped_snapshots} future snapshots",
    )
    return formatting.rewind_text(result)
