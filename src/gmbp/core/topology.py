import networkx as nx

class VariableNode:
    """Represents a random variable in the factor graph."""
    def __init__(self, name, dim=1):
        self.name = name
        self.dim = dim
        self.belief = None          # GaussianMixture representing the marginal distribution
        self.observed_value = None  # Float or array if the variable is observed

    def __repr__(self):
        obs_str = f", observed={self.observed_value}" if self.observed_value is not None else ""
        return f"VariableNode(name={self.name}, dim={self.dim}{obs_str})"


class FactorNode:
    """Represents a factor (potential function) in the factor graph."""
    def __init__(self, name, scope):
        self.name = name
        self.scope = tuple(scope)   # Ordered tuple of variable names this factor connects to
        self.potential = None       # GaussianMixture representing the joint distribution of the scope

    def __repr__(self):
        return f"FactorNode(name={self.name}, scope={self.scope})"


class FactorGraph:
    """The bipartite graph managing Variables, Factors, and their topological edges."""
    def __init__(self):
        self._g = nx.Graph()
        self.variables = {}
        self.factors = {}

    def add_variable(self, name, dim=1):
        if name in self._g:
            raise ValueError(f"Node '{name}' already exists in the graph.")
            
        node = VariableNode(name, dim)
        self.variables[name] = node
        self._g.add_node(name, bipartite=0, obj=node, type='variable')
        return node

    def add_factor(self, name, scope):
        if name in self._g:
            raise ValueError(f"Node '{name}' already exists in the graph.")
            
        # Enforce topology: all variables in the factor's scope must exist
        for var_name in scope:
            if var_name not in self.variables:
                raise ValueError(f"Variable '{var_name}' must be added before factor '{name}'.")
                
        node = FactorNode(name, scope)
        self.factors[name] = node
        self._g.add_node(name, bipartite=1, obj=node, type='factor')
        
        # Draw edges between the factor and its scoped variables
        for var_name in scope:
            self._g.add_edge(name, var_name)
            
        return node

    def get_neighbors(self, node_name):
        """Returns the names of connected nodes (Factor -> Variables, Variable -> Factors)."""
        return list(self._g.neighbors(node_name))

    def __repr__(self):
        return f"FactorGraph(variables={len(self.variables)}, factors={len(self.factors)})"
