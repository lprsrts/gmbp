from ..core.topology import FactorGraph
from ..inference.bp import BeliefPropagation
from .fluent import FluentFactor, Observer

class Session:
    """The central state manager for the active graph and canvas telemetry."""
    def __init__(self):
        self.graph = FactorGraph()
        self.engine = BeliefPropagation(self.graph)

    def add_variable(self, *names):
        """Adds one or more variables to the graph."""
        for name in names:
            self.graph.add_variable(name)
            # TODO: Emit telemetry event to canvas here
            print(f"[Session] Variable added: {name}")

    def add_factor(self, name, scope):
        """Adds a factor connecting the given scope and returns a FluentFactor."""
        self.graph.add_factor(name, scope)
        # TODO: Emit telemetry event to canvas here
        print(f"[Session] Factor added: {name} with scope {scope}")
        return FluentFactor(self, name)
        
    def factor(self, name):
        """Retrieves an existing factor for fluent chaining."""
        if name not in self.graph.factors:
            raise ValueError(f"Factor '{name}' not found in graph.")
        return FluentFactor(self, name)

    def observe(self):
        """Starts an observation chain."""
        return Observer(self)

    def step(self):
        """Executes one iteration of Belief Propagation."""
        self.engine.step()
        print("[Session] Belief Propagation step completed.")
        
    def compute_beliefs(self):
        """Computes marginal beliefs and triggers visual update."""
        self.engine.compute_beliefs()
        print("[Session] Marginal beliefs computed.")
        # TODO: Broadcast distribution updates to canvas

    def clear(self):
        """Resets the session state."""
        self.graph = FactorGraph()
        self.engine = BeliefPropagation(self.graph)
        print("[Session] Graph cleared.")
