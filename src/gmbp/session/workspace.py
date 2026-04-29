from ..core.topology import FactorGraph
from ..inference.bp import BeliefPropagation
from .fluent import FluentFactor, Observer, FluentVariable

class Session:
    def __init__(self):
        self.graph = FactorGraph()
        self.engine = BeliefPropagation(self.graph)

    def add_variable(self, *names):
        for name in names:
            self.graph.add_variable(name)
            print(f"[Session] Variable added: {name}")

    def add_factor(self, name, scope):
        self.graph.add_factor(name, scope)
        print(f"[Session] Factor added: {name} with scope {scope}")
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
        
    def compute_beliefs(self):
        self.engine.compute_beliefs()
        print("[Session] Marginal beliefs computed.")

    def clear(self):
        self.graph = FactorGraph()
        self.engine = BeliefPropagation(self.graph)
        print("[Session] Graph cleared.")
