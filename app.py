"""Composition root: wires the MCP app, the WebSocket endpoint and the static
viewer onto one FastAPI application.

One process serves three surfaces:
  - Browser UI      http://localhost:8000       (static viewer + controls)
  - Live state feed ws://localhost:8000/ws      (broadcasts board + events)
  - MCP endpoint    http://localhost:8000/mcp   (Streamable HTTP, for agents)

Run with:  uvicorn app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import hub
import ws
from mcp_tools import mcp

STATIC_DIR = Path(__file__).parent / "static"

# stateless + JSON responses: each MCP request is self-contained (no session
# handshake to break on server restart), plain JSON replies instead of SSE.
mcp_app = mcp.streamable_http_app(stateless_http=True, json_response=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        yield
    await hub.service.shutdown()


app = FastAPI(title="Game of Life MCP", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.websocket("/ws")(ws.ws_endpoint)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Mounted last as a catch-all: routes above win; /mcp falls through to the
# MCP streamable HTTP app (its own internal path is /mcp).
app.mount("/", mcp_app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
