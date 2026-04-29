from ..core.topology import FactorGraph
from ..inference.bp import BeliefPropagation
from .fluent import FluentFactor, Observer, FluentVariable
from ..canvas.server import emit_event, start_canvas_server
import time

class Session:
    def __init__(self):
        self.graph = FactorGraph()
        self.engine = BeliefPropagation(self.graph)
        self._server_thread = start_canvas_server()
        time.sleep(0.5)  # Extended wait to ensure server is bound and ready to queue

    def _sync_graph(self):
        nodes = []
        edges = []
        
        for v_id, var in self.graph.variables.items():
            nodes.append({"data": {"id": v_id, "name": var.name, "type": "variable", "dim": var.dim, "observed": var.observed_value is not None}})
            
        for f_id, factor in self.graph.factors.items():
            nodes.append({"data": {"id": f_id, "name": factor.name, "type": "factor"}})
            for v_id in factor.scope:
                edges.append({"data": {"source": f_id, "target": v_id}})
                
        emit_event("GRAPH_SYNC", {"nodes": nodes, "edges": edges})

    def add_variable(self, *names):
        fluents = []
        for name in names:
            self.graph.add_variable(name)
            print(f"[Session] Variable added: {name}")
            fluents.append(FluentVariable(self, name))
        self._sync_graph()
        if len(fluents) == 1:
            return fluents[0]
        return fluents

    def add_factor(self, name, scope):
        self.graph.add_factor(name, scope)
        print(f"[Session] Factor added: {name} with scope {scope}")
        self._sync_graph()
        return FluentFactor(self, name)
        
    def factor(self, name):
        return FluentFactor(self, name)
        
    def variable(self, name):
        return FluentVariable(self, name)

    def observe(self):
        return Observer(self)

    def step(self):
        self.engine.step()
        print("[Session] Belief Propagation step completed.")
        self._emit_beliefs()
        
    def compute_beliefs(self):
        self.engine.compute_beliefs()
        print("[Session] Marginal beliefs computed.")
        self._emit_beliefs()

    def _emit_beliefs(self):
        beliefs = {}
        for v_id, var in self.graph.variables.items():
            if var.belief is not None:
                comps = []
                for w, comp in zip(var.belief.weights, var.belief.components):
                    comps.append({
                        "weight": float(w),
                        "mean": float(comp.mean[0]),
                        "variance": float(comp.cov[0, 0])
                    })
                beliefs[v_id] = comps
        emit_event("BELIEF_SYNC", beliefs)

    def clear(self):
        self.graph = FactorGraph()
        self.engine = BeliefPropagation(self.graph)
        print("[Session] Graph cleared.")
        self._sync_graph()
