import asyncio
import json
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

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
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GMBP Canvas</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        body { margin: 0; padding: 0; background-color: #000; color: #0f0; font-family: monospace; display: flex; height: 100vh; overflow: hidden; }
        #cy { width: 50vw; height: 100vh; border-right: 1px solid #0f0; }
        #panel { width: 50vw; height: 100vh; display: flex; flex-direction: column; }
        #header { padding: 10px; border-bottom: 1px solid #0f0; }
        #plot-container { flex-grow: 1; padding: 10px; overflow-y: auto; }
    </style>
</head>
<body>
<div id="cy"></div>
<div id="panel">
    <div id="header">IDLE. WAITING FOR NODE SELECTION.</div>
    <div id="plot-container"></div>
</div>
<script>
    const cy = cytoscape({
        container: document.getElementById('cy'),
        style: [
            { selector: 'node[type="variable"]', style: { 'shape': 'ellipse', 'background-color': '#000', 'border-width': 2, 'border-color': '#0f0', 'label': 'data(name)', 'color': '#0f0', 'text-valign': 'top', 'text-halign': 'center', 'text-margin-y': -5 } },
            { selector: 'node[type="factor"]', style: { 'shape': 'rectangle', 'background-color': '#0f0', 'width': 20, 'height': 20, 'label': 'data(name)', 'color': '#0f0', 'text-valign': 'top', 'text-halign': 'center', 'text-margin-y': -5 } },
            { selector: 'node[observed="true"]', style: { 'background-color': '#0f0', 'color': '#000' } },
            { selector: 'edge', style: { 'width': 1, 'line-color': '#0f0', 'curve-style': 'bezier' } }
        ],
        layout: { name: 'cose' }
    });

    let nodeBeliefs = {};

    function evaluateGMM(x, components) {
        let y = 0;
        for (const comp of components) {
            const z = (x - comp.mean) / Math.sqrt(comp.variance);
            const pdf = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI * comp.variance);
            y += comp.weight * pdf;
        }
        return y;
    }

    function renderVariablePlot(nodeId, nodeName) {
        const header = document.getElementById('header');
        const container = document.getElementById('plot-container');
        if (!nodeBeliefs[nodeId]) {
            header.innerText = `VARIABLE: ${nodeName} | STATUS: NO BELIEF COMPUTED`;
            container.innerHTML = '';
            return;
        }
        header.innerText = `VARIABLE: ${nodeName} | STATUS: MARGINAL BELIEF P(${nodeName})`;
        const components = nodeBeliefs[nodeId];
        let minMean = Math.min(...components.map(c => c.mean));
        let maxMean = Math.max(...components.map(c => c.mean));
        let maxStd = Math.max(...components.map(c => Math.sqrt(c.variance)));
        let xStart = minMean - (3 * maxStd);
        let xEnd = maxMean + (3 * maxStd);
        let xSteps = 200;
        let dx = (xEnd - xStart) / xSteps;
        let xData = [], yData = [];
        for(let i=0; i<=xSteps; i++) { let x = xStart + (i * dx); xData.push(x); yData.push(evaluateGMM(x, components)); }
        Plotly.newPlot(container, [{x: xData, y: yData, type: 'scatter', mode: 'lines', line: {color: '#0f0', width: 2}, fill: 'tozeroy', fillcolor: 'rgba(0, 255, 0, 0.2)'}], { paper_bgcolor: '#000', plot_bgcolor: '#000', xaxis: { color: '#0f0', gridcolor: '#333' }, yaxis: { color: '#0f0', gridcolor: '#333' }, margin: { t: 10, r: 10, b: 40, l: 40 } }, {displayModeBar: false});
    }

    cy.on('tap', 'node', function(evt){
        const node = evt.target;
        if (node.data('type') === 'variable') renderVariablePlot(node.id(), node.data('name'));
        else { document.getElementById('header').innerText = `FACTOR: ${node.data('name')} | PAIRWISE PLOTS PENDING`; document.getElementById('plot-container').innerHTML = ''; }
    });

    const connectWebSocket = () => {
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        ws.onopen = function() {
            console.log("WebSocket connected. Requesting initial state.");
            ws.send(JSON.stringify({command: "REQUEST_STATE"}));
        };
        ws.onmessage = function(event) {
            const msg = JSON.parse(event.data);
            if (msg.event === "GRAPH_SYNC") { cy.elements().remove(); cy.add(msg.payload.nodes); cy.add(msg.payload.edges); cy.layout({ name: 'cose', animate: true, randomize: false }).run(); } 
            else if (msg.event === "BELIEF_SYNC") { nodeBeliefs = msg.payload; const selected = cy.$(':selected'); if (selected.length > 0 && selected[0].data('type') === 'variable') renderVariablePlot(selected[0].id(), selected[0].data('name')); }
        };
        ws.onclose = function() { document.getElementById('header').innerText = "CONNECTION LOST. RESTART REPL."; };
    };

    connectWebSocket();
</script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(HTML_CONTENT)

_telemetry_queue = asyncio.Queue()

# We need a reference to the latest state to send to newly connected clients
_latest_graph_state = None
_latest_belief_state = None

async def telemetry_worker():
    global _latest_graph_state, _latest_belief_state
    while True:
        event = await _telemetry_queue.get()
        if event["event"] == "GRAPH_SYNC":
            _latest_graph_state = event
        elif event["event"] == "BELIEF_SYNC":
            _latest_belief_state = event
            
        await manager.broadcast(event)
        _telemetry_queue.task_done()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    # Immediately push the latest state to the new client
    if _latest_graph_state:
        await websocket.send_text(json.dumps(_latest_graph_state))
    if _latest_belief_state:
        await websocket.send_text(json.dumps(_latest_belief_state))
        
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(telemetry_worker())

def _run_server():
    import logging
    log = logging.getLogger("uvicorn")
    log.setLevel(logging.CRITICAL)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="critical")

def start_canvas_server():
    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()
    return thread

def emit_event(event_type: str, payload: dict):
    event = {"event": event_type, "payload": payload}
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(_telemetry_queue.put_nowait, event)
    except RuntimeError:
        pass
