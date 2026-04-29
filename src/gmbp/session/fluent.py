from ..inference.em import fit_gmm_to_data

class FluentFactor:
    def __init__(self, session, factor_name):
        self.session = session
        self.factor_id = session.graph.resolve_id(factor_name)
        self.node = session.graph.factors[self.factor_id]

    def fit_gm(self, data, N=1):
        self.node.potential = fit_gmm_to_data(data, n_components=N)
        print(f"[Factor:{self.node.name}] Fitted GMM with {N} components.")
        return self

    def rename(self, new_name):
        old_name = self.node.name
        self.session.graph.rename_node(old_name, new_name)
        print(f"[Factor:{old_name}] Renamed to {new_name}.")
        return self

class FluentVariable:
    def __init__(self, session, variable_name):
        self.session = session
        self.variable_id = session.graph.resolve_id(variable_name)
        self.node = session.graph.variables[self.variable_id]

    def at(self, value):
        self.node.observed_value = value
        print(f"[Variable:{self.node.name}] Observed at {value}.")
        return self
        
    def rename(self, new_name):
        old_name = self.node.name
        self.session.graph.rename_node(old_name, new_name)
        print(f"[Variable:{old_name}] Renamed to {new_name}.")
        return self

class Observer:
    def __init__(self, session):
        self.session = session

    def variable(self, name):
        return FluentVariable(self.session, name)

def variable_fluent(session, name):
    return FluentVariable(session, name)
