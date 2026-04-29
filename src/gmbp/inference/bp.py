from ..core.topology import FactorGraph, VariableNode, FactorNode
from ..core.messages import MessageManager
from ..core.distributions import GaussianMixture, Gaussian
import numpy as np

class BeliefPropagation:
    """Synchronous Gaussian Mixture Belief Propagation Engine."""
    
    def __init__(self, graph: FactorGraph):
        self.graph = graph
        self.manager = MessageManager()
        
        for f_name, factor in self.graph.factors.items():
            for v_name in factor.scope:
                self.manager.initialize_edge(f_name, v_name)

    def _multiply_messages(self, payloads):
        if not payloads:
            return None
        result = payloads[0]
        for p in payloads[1:]:
            result = result * p
            result.prune(threshold=1e-4) 
        return result

    def _upcast_message(self, gm, source_var_name, factor_scope):
        """Upcasts a variable message to the factor's full dimensionality."""
        dim = len(factor_scope)
        target_idx = factor_scope.index(source_var_name)
        
        new_components = []
        for comp in gm.components:
            # Emulate uniform distribution over other variables with huge variance
            new_cov = np.eye(dim) * 1e8
            new_cov[target_idx, target_idx] = comp.cov[0, 0]
            
            new_mean = np.zeros(dim)
            new_mean[target_idx] = comp.mean[0]
            
            new_components.append(Gaussian(new_mean, new_cov))
            
        return GaussianMixture(gm.weights.copy(), new_components)

    def _variable_to_factor(self, var_name, factor_name):
        incoming = self.manager.get_incoming_payloads(var_name, exclude_source=factor_name)
        var_node = self.graph.variables[var_name]
        
        if var_node.observed_value is not None:
            obs_cov = np.eye(var_node.dim) * 1e-6
            obs_g = Gaussian(np.atleast_1d(var_node.observed_value), obs_cov)
            obs_gm = GaussianMixture([1.0], [obs_g])
            incoming.append(obs_gm)
            
        out_msg = self._multiply_messages(incoming)
        self.manager.update(var_name, factor_name, out_msg)

    def _factor_to_variable(self, factor_name, var_name):
        incoming = self.manager.get_incoming_payloads(factor_name, exclude_source=var_name)
        factor_node = self.graph.factors[factor_name]
        
        if factor_node.potential is None:
            return 
            
        # We must know which variables the incoming messages came from.
        # So instead of a flat list, we retrieve them by neighbor.
        upcasted_incoming = []
        for neighbor in factor_node.scope:
            if neighbor != var_name:
                msg = self.manager.get_payload(neighbor, factor_name)
                if msg is not None:
                    upcasted_incoming.append(self._upcast_message(msg, neighbor, factor_node.scope))
        
        payloads = upcasted_incoming + [factor_node.potential]
        joint_distribution = self._multiply_messages(payloads)
        
        if joint_distribution is None:
            return
            
        scope_indices = {name: idx for idx, name in enumerate(factor_node.scope)}
        target_idx = scope_indices[var_name]
        
        out_msg = joint_distribution.marginalize([target_idx])
        out_msg.prune(threshold=1e-4)
        
        self.manager.update(factor_name, var_name, out_msg)

    def step(self):
        for f_name, factor in self.graph.factors.items():
            for v_name in factor.scope:
                self._variable_to_factor(v_name, f_name)
                
        for f_name, factor in self.graph.factors.items():
            for v_name in factor.scope:
                self._factor_to_variable(f_name, v_name)

    def compute_beliefs(self):
        for v_name, var_node in self.graph.variables.items():
            incoming = self.manager.get_incoming_payloads(v_name)
            belief = self._multiply_messages(incoming)
            if belief is not None:
                belief.prune(threshold=1e-4)
            var_node.belief = belief
