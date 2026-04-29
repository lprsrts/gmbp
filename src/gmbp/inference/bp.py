from ..core.topology import FactorGraph, VariableNode, FactorNode
from ..core.messages import MessageManager
from ..core.distributions import GaussianMixture, Gaussian
import numpy as np

class BeliefPropagation:
    """Synchronous Gaussian Mixture Belief Propagation Engine."""
    
    def __init__(self, graph: FactorGraph):
        self.graph = graph
        self.manager = MessageManager()
        
        # Initialize all edge messages
        for f_name, factor in self.graph.factors.items():
            for v_name in factor.scope:
                self.manager.initialize_edge(f_name, v_name)

    def _multiply_messages(self, payloads):
        """Helper to multiply a list of GaussianMixture payloads."""
        if not payloads:
            return None
        
        result = payloads[0]
        for p in payloads[1:]:
            result = result * p
            # Prune during multiplication to prevent exponential explosion
            result.prune(threshold=1e-4) 
        return result

    def _variable_to_factor(self, var_name, factor_name):
        """Calculates message from Variable to Factor."""
        incoming = self.manager.get_incoming_payloads(var_name, exclude_source=factor_name)
        
        var_node = self.graph.variables[var_name]
        
        # If the variable is observed, it sends a deterministic/highly-peaked message
        if var_node.observed_value is not None:
            # Create a narrow Gaussian around the observed value
            obs_cov = np.eye(var_node.dim) * 1e-6
            obs_g = Gaussian(np.atleast_1d(var_node.observed_value), obs_cov)
            obs_gm = GaussianMixture([1.0], [obs_g])
            incoming.append(obs_gm)
            
        out_msg = self._multiply_messages(incoming)
        self.manager.update(var_name, factor_name, out_msg)

    def _factor_to_variable(self, factor_name, var_name):
        """Calculates message from Factor to Variable."""
        incoming = self.manager.get_incoming_payloads(factor_name, exclude_source=var_name)
        factor_node = self.graph.factors[factor_name]
        
        if factor_node.potential is None:
            return # Cannot send message if potential is unfitted
            
        # 1. Multiply incoming messages with the factor's potential
        payloads = incoming + [factor_node.potential]
        joint_distribution = self._multiply_messages(payloads)
        
        if joint_distribution is None:
            return
            
        # 2. Marginalize out all variables EXCEPT the target variable
        # We need to map the target variable name to its index in the factor's scope
        scope_indices = {name: idx for idx, name in enumerate(factor_node.scope)}
        target_idx = scope_indices[var_name]
        
        out_msg = joint_distribution.marginalize([target_idx])
        out_msg.prune(threshold=1e-4)
        
        self.manager.update(factor_name, var_name, out_msg)

    def step(self):
        """Performs one full synchronous iteration of message passing."""
        # Step 1: Variables -> Factors
        for f_name, factor in self.graph.factors.items():
            for v_name in factor.scope:
                self._variable_to_factor(v_name, f_name)
                
        # Step 2: Factors -> Variables
        for f_name, factor in self.graph.factors.items():
            for v_name in factor.scope:
                self._factor_to_variable(f_name, v_name)

    def compute_beliefs(self):
        """Computes the final marginal belief for each variable."""
        for v_name, var_node in self.graph.variables.items():
            incoming = self.manager.get_incoming_payloads(v_name)
            belief = self._multiply_messages(incoming)
            if belief is not None:
                belief.prune(threshold=1e-4)
            var_node.belief = belief
