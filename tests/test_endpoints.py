"""The HTTP and WebSocket surface, driven through the real FastAPI app.

Everything else stubs the transport out, which means a broken route or a stale
name inside ws_endpoint would not show up. These tests run the composed app so
that the wiring in app.py is exercised too.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app as app_module
import hub
from gol_world import World
from world_service import WorldService


@pytest.fixture(scope="module")
def _app_client():
    """One client per module: the MCP session manager refuses to .run() twice
    on the same instance, and the lifespan starts it."""
    # base_url matters: the MCP transport rejects unknown Host headers, and
    # TestClient's default ("testserver") is one of them.
    with TestClient(app_module.app, base_url="http://127.0.0.1:8000") as c:
        yield c


@pytest.fixture
def client(_app_client):
    """The composed app with a fresh, small world and real broadcasting."""
    original = hub.service
    fresh = WorldService(World(32, 32))
    hub.service = fresh
    fresh.set_change_listener(hub.broadcast_state)
    hub.SOCKETS.clear()
    hub.EVENTS.clear()
    try:
        yield _app_client
    finally:
        fresh.running = False
        hub.service = original


def handshake(ws) -> tuple[dict, dict]:
    """Every connection opens with a state frame then the event backlog."""
    return ws.receive_json(), ws.receive_json()


def next_state(ws, tries: int = 40) -> dict | None:
    for _ in range(tries):
        m = ws.receive_json()
        if m["type"] == "state":
            return m
    return None


# ------------------------------------------------------------------- routes

def test_index_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<canvas" in r.text


def test_static_assets_are_served(client):
    for path in ("/static/style.css", "/static/index.html"):
        assert client.get(path).status_code == 200, path


def test_every_browser_module_is_served(client):
    """The viewer loads as ES modules, so a missing or renamed file is a blank
    page rather than a 404 the user would notice."""
    for name in ("main", "store", "renderer", "socket", "controls"):
        r = client.get(f"/static/js/{name}.js")
        assert r.status_code == 200, name


def test_the_page_loads_the_module_entry_point(client):
    body = client.get("/").text
    assert '<script type="module" src="/static/js/main.js"></script>' in body


def test_mcp_endpoint_lists_the_tools(client):
    r = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Accept": "application/json, text/event-stream"},
    )
    assert r.status_code == 200
    names = {t["name"] for t in r.json()["result"]["tools"]}
    assert "advance_generations" in names
    assert "get_world_status" in names


# --------------------------------------------------------------- websocket

def test_connection_opens_with_state_then_backlog(client):
    with client.websocket_connect("/ws") as ws:
        state, backlog = handshake(ws)
        assert state["type"] == "state"
        assert (state["width"], state["height"]) == (32, 32)
        assert state["generation"] == 0
        assert backlog["type"] == "backlog"
        assert isinstance(backlog["events"], list)


def test_painting_broadcasts_the_new_state(client):
    with client.websocket_connect("/ws") as ws:
        handshake(ws)
        ws.send_json({"action": "paint", "cells": [[5, 5], [6, 5], [7, 5]], "value": 1})
        state = next_state(ws)
        assert state is not None
        assert state["population"] == 3


def test_stepping_advances_the_generation(client):
    with client.websocket_connect("/ws") as ws:
        handshake(ws)
        ws.send_json({"action": "paint", "cells": [[5, 5], [6, 5], [7, 5]], "value": 1})
        next_state(ws)
        ws.send_json({"action": "step"})
        state = next_state(ws)
        assert state["generation"] == 1
        assert state["population"] == 3          # a blinker keeps 3 cells


def test_preview_returns_a_preview_frame(client):
    with client.websocket_connect("/ws") as ws:
        handshake(ws)
        ws.send_json({"action": "paint", "cells": [[5, 5]], "value": 1})
        next_state(ws)
        ws.send_json({"action": "step"})
        next_state(ws)
        ws.send_json({"action": "preview_generation", "generation": 0, "request_id": 7})
        for _ in range(40):
            m = ws.receive_json()
            if m["type"] == "preview_state":
                assert m["generation"] == 0
                assert m["request_id"] == 7
                break
        else:
            pytest.fail("no preview_state frame arrived")


def test_clear_empties_the_board(client):
    with client.websocket_connect("/ws") as ws:
        handshake(ws)
        ws.send_json({"action": "paint", "cells": [[1, 1], [2, 2]], "value": 1})
        next_state(ws)
        ws.send_json({"action": "clear"})
        state = next_state(ws)
        assert state["population"] == 0
        assert state["generation"] == 0


@pytest.mark.parametrize("bad", [
    {"action": "paint", "cells": [["a", "b"]], "value": 1},
    {"action": "fps", "fps": "fast"},
    {"action": "create", "width": "wide", "height": 10},
    {"action": "rewind", "generation": "yesterday"},
    {"action": "definitely-not-an-action"},
    {"not-an-action-key": 1},
])
def test_a_malformed_command_does_not_drop_the_connection(client, bad):
    """ws_endpoint only catches WebSocketDisconnect, so anything else escaping
    the handler would close the socket."""
    with client.websocket_connect("/ws") as ws:
        handshake(ws)
        ws.send_json(bad)
        # the socket must still serve a normal command afterwards
        ws.send_json({"action": "paint", "cells": [[3, 3]], "value": 1})
        state = next_state(ws)
        assert state is not None and state["population"] >= 1


def test_non_json_text_does_not_drop_the_connection(client):
    with client.websocket_connect("/ws") as ws:
        handshake(ws)
        ws.send_text("this is not json{{{")
        ws.send_json({"action": "paint", "cells": [[4, 4]], "value": 1})
        state = next_state(ws)
        assert state is not None and state["population"] >= 1


def test_the_broadcast_frame_is_decodable_by_the_browser(client):
    import base64

    with client.websocket_connect("/ws") as ws:
        state, _ = handshake(ws)
        raw = base64.b64decode(state["cells"])
        assert len(raw) * 8 >= state["width"] * state["height"]


def test_an_agent_tool_call_reaches_the_connected_browser(client):
    """The whole point of the app: an MCP mutation shows up on the socket."""
    with client.websocket_connect("/ws") as ws:
        handshake(ws)
        r = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": "place_pattern",
                             "arguments": {"rle": "3o!", "x": 4, "y": 4}}},
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert r.status_code == 200
        state = next_state(ws)
        assert state["population"] == 3
