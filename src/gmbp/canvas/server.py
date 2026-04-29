import asyncio
import json
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import os

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # Fire and forget broadcasting to all connected canvas clients
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                pass

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We only expect clients to listen, but we must keep the connection alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# HTML endpoint for the canvas
@app.get("/")
async def get():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    try:
        with open(html_path, "r") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Canvas UI not found.</h1>", status_code=404)

# Global queue to bridge synchronous REPL thread with async FastAPI server
_telemetry_queue = asyncio.Queue()

async def telemetry_worker():
    """Reads from the queue and broadcasts via WebSockets."""
    while True:
        event = await _telemetry_queue.get()
        await manager.broadcast(event)
        _telemetry_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(telemetry_worker())

def _run_server():
    """Runs uvicorn on the background thread, hiding startup logs."""
    import logging
    log = logging.getLogger("uvicorn")
    log.setLevel(logging.CRITICAL)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="critical")

def start_canvas_server():
    """Spawns the telemetry server in a daemon thread."""
    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()
    return thread

def emit_event(event_type: str, payload: dict):
    """
    Called by the synchronous Session logic. Safely pushes the event
    to the asyncio loop running the WebSocket server.
    """
    event = {"event": event_type, "payload": payload}
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(_telemetry_queue.put_nowait, event)
    except RuntimeError:
        pass # If no loop is running, the server thread hasn't started yet
