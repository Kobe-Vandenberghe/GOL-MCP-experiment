"""Browser-facing surface: the command protocol and the WebSocket endpoint.

Each command maps onto the same WorldService method the matching MCP tool uses;
only the phrasing of the activity-log line differs. ws_endpoint is a plain
coroutine - app.py registers it on the route.
"""
from __future__ import annotations

import base64
import json

from fastapi import WebSocket, WebSocketDisconnect

import formatting
import hub


# --------------------------------------------------------- browser commands


async def handle_ui(msg: dict) -> None:
    action = msg.get("action")
    if action == "paint":
        cells = msg.get("cells") or []
        if not cells:
            return          # an empty stroke was a no-op before the service
        value = 1 if msg.get("value", 1) else 0
        if value:
            await hub.service.set_cells(alive=cells)
        else:
            await hub.service.set_cells(dead=cells)
    elif action == "play":
        result = await hub.service.set_autorun(True, msg.get("fps"))
        await hub.log_event("user", f"pressed play ({result.fps} fps)")
    elif action == "pause":
        await hub.service.set_autorun(False)
        await hub.log_event("user", "pressed pause")
    elif action == "step":
        await hub.service.advance(1)
    elif action == "advance":
        try:
            result = await hub.service.advance(int(msg.get("generations", 1)))
            await hub.log_event("user", f"advanced {result.generations} generations")
        except Exception as e:
            await hub.log_event("user", f"advance failed: {e}")
    elif action == "fps":
        fps = msg.get("fps")
        if fps:
            await hub.service.set_autorun(hub.service.running, float(fps))
    elif action == "clear":
        await hub.service.clear()
        await hub.log_event("user", "cleared the world")
    elif action == "create":
        try:
            result = await hub.service.create_world(
                int(msg.get("width")),
                int(msg.get("height")),
                edge=str(msg.get("edge", "wrap")),
                random_fill=float(msg.get("random_fill", 0.0)),
            )
            await hub.log_event("user", formatting.create_log(result))
        except Exception as e:
            await hub.log_event("user", f"create failed: {e}")
    elif action == "rewind":
        try:
            await hub.rewind_world(int(msg.get("generation")), actor="user")
        except Exception as e:
            await hub.log_event("user", f"rewind failed: {e}")


async def handle_ui_for_socket(ws: WebSocket, msg: dict) -> None:
    action = msg.get("action")
    if action != "preview_generation":
        try:
            await handle_ui(msg)
        except Exception as e:
            # A malformed command must never take the socket down with it:
            # ws_endpoint only catches WebSocketDisconnect, so anything else
            # escaping here would drop the browser's connection.
            await hub.log_event("user", f"{action or 'command'} failed: {e}")
        return

    try:
        target = int(msg.get("generation"))
    except Exception:
        return

    preview = await hub.service.preview_generation(target)
    if preview is None:
        return

    payload = {
        "type": "preview_state",
        "generation": preview.generation,
        "population": preview.population,
        "cells": base64.b64encode(preview.packed_cells).decode("ascii"),
    }
    req_id = msg.get("request_id")
    if isinstance(req_id, int):
        payload["request_id"] = req_id

    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        pass


async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    hub.SOCKETS.add(ws)
    try:
        await ws.send_text(json.dumps(hub.state_message()))
        await ws.send_text(json.dumps({"type": "backlog", "events": list(hub.EVENTS)}))
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except json.JSONDecodeError:
                continue
            await handle_ui_for_socket(ws, msg)
    except WebSocketDisconnect:
        pass
    finally:
        hub.SOCKETS.discard(ws)
