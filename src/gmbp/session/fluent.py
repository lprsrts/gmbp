from ..inference.em import fit_gmm_to_data

class FluentFactor:
    """Builder object for chaining operations on a FactorNode."""
    def __init__(self, session, factor_name):
        self.session = session
        self.factor_name = factor_name
        self.node = session.graph.factors[factor_name]

    def fit_gm(self, data, N=1):
        """Fits a Gaussian Mixture to the factor's potential using empirical data."""
        self.node.potential = fit_gmm_to_data(data, n_components=N)
        print(f"[Factor:{self.factor_name}] Fitted GMM with {N} components.")
        # TODO: Broadcast visual update to canvas
        return self

class FluentVariable:
    """Builder object for chaining operations on a VariableNode."""
    def __init__(self, session, variable_name):
        self.session = session
        self.variable_name = variable_name
        self.node = session.graph.variables[variable_name]

    def at(self, value):
        """Observes the variable at a specific deterministic value."""
        self.node.observed_value = value
        print(f"[Variable:{self.variable_name}] Observed at {value}.")
        # TODO: Broadcast visual update to canvas
        return self

class Observer:
    """Intermediate object allowing `observe().variable("X").at(0.4)` syntax."""
    def __init__(self, session):
        self.session = session

    def variable(self, name):
        if name not in self.session.graph.variables:
            raise ValueError(f"Variable '{name}' not found in graph.")
        return FluentVariable(self.session, name)
