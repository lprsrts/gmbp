import networkx as nx
import uuid

def generate_id():
    return uuid.uuid4().hex[:8]

class VariableNode:
    def __init__(self, name, dim=1):
        self.id = generate_id()
        self.name = name
        self.dim = dim
        self.belief = None
        self.observed_value = None

    def __repr__(self):
        obs = f", obs={self.observed_value}" if self.observed_value is not None else ""
        return f"VariableNode(id={self.id}, name={self.name}, dim={self.dim}{obs})"

class FactorNode:
    def __init__(self, name, scope_ids):
        self.id = generate_id()
        self.name = name
        self.scope = tuple(scope_ids)
        self.potential = None

    def __repr__(self):
        return f"FactorNode(id={self.id}, name={self.name})"

class FactorGraph:
    def __init__(self):
        self._g = nx.Graph()
        self.variables = {}    # ID -> VariableNode
        self.factors = {}      # ID -> FactorNode
        self.name_to_id = {}   # Name -> ID

    def resolve_id(self, name):
        if name not in self.name_to_id:
            raise ValueError(f"Node '{name}' not found.")
        return self.name_to_id[name]

    def add_variable(self, name, dim=1):
        if name in self.name_to_id:
            raise ValueError(f"Name '{name}' already exists.")
        node = VariableNode(name, dim)
        self.variables[node.id] = node
        self.name_to_id[name] = node.id
        self._g.add_node(node.id, bipartite=0, obj=node, type='variable')
        return node

    def add_factor(self, name, scope_names):
        if name in self.name_to_id:
            raise ValueError(f"Name '{name}' already exists.")
        scope_ids = [self.resolve_id(n) for n in scope_names]
        node = FactorNode(name, scope_ids)
        self.factors[node.id] = node
        self.name_to_id[name] = node.id
        self._g.add_node(node.id, bipartite=1, obj=node, type='factor')
        for v_id in scope_ids:
            self._g.add_edge(node.id, v_id)
        return node

    def rename_node(self, old_name, new_name):
        if new_name in self.name_to_id:
            raise ValueError(f"Name '{new_name}' already in use.")
        node_id = self.resolve_id(old_name)
        del self.name_to_id[old_name]
        self.name_to_id[new_name] = node_id
        
        if node_id in self.variables:
            self.variables[node_id].name = new_name
        elif node_id in self.factors:
            self.factors[node_id].name = new_name
