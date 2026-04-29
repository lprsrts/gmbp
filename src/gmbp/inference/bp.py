from ..core.topology import FactorGraph
from ..core.messages import MessageManager
from ..core.distributions import GaussianMixture, Gaussian
import numpy as np

class BeliefPropagation:
    def __init__(self, graph: FactorGraph):
        self.graph = graph
        self.manager = MessageManager()
        for f_id, factor in self.graph.factors.items():
            for v_id in factor.scope:
                self.manager.initialize_edge(f_id, v_id)

    def _multiply_messages(self, payloads):
        if not payloads:
            return None
        result = payloads[0]
        for p in payloads[1:]:
            result = result * p
            result.prune(threshold=1e-4) 
        return result

    def _upcast_message(self, gm, source_var_id, factor_scope):
        dim = len(factor_scope)
        target_idx = factor_scope.index(source_var_id)
        new_components = []
        for comp in gm.components:
            new_cov = np.eye(dim) * 1e8
            new_cov[target_idx, target_idx] = comp.cov[0, 0]
            new_mean = np.zeros(dim)
            new_mean[target_idx] = comp.mean[0]
            new_components.append(Gaussian(new_mean, new_cov))
        return GaussianMixture(gm.weights.copy(), new_components)

    def _variable_to_factor(self, var_id, factor_id):
        incoming = self.manager.get_incoming_payloads(var_id, exclude_source=factor_id)
        var_node = self.graph.variables[var_id]
        if var_node.observed_value is not None:
            obs_cov = np.eye(var_node.dim) * 1e-6
            obs_g = Gaussian(np.atleast_1d(var_node.observed_value), obs_cov)
            obs_gm = GaussianMixture([1.0], [obs_g])
            incoming.append(obs_gm)
        out_msg = self._multiply_messages(incoming)
        self.manager.update(var_id, factor_id, out_msg)

    def _factor_to_variable(self, factor_id, var_id):
        incoming = self.manager.get_incoming_payloads(factor_id, exclude_source=var_id)
        factor_node = self.graph.factors[factor_id]
        if factor_node.potential is None:
            return 
        upcasted_incoming = []
        for neighbor_id in factor_node.scope:
            if neighbor_id != var_id:
                msg = self.manager.get_payload(neighbor_id, factor_id)
                if msg is not None:
                    upcasted_incoming.append(self._upcast_message(msg, neighbor_id, factor_node.scope))
        payloads = upcasted_incoming + [factor_node.potential]
        joint_distribution = self._multiply_messages(payloads)
        if joint_distribution is None:
            return
        target_idx = factor_node.scope.index(var_id)
        out_msg = joint_distribution.marginalize([target_idx])
        out_msg.prune(threshold=1e-4)
        self.manager.update(factor_id, var_id, out_msg)

    def step(self):
        for f_id, factor in self.graph.factors.items():
            for v_id in factor.scope:
                self._variable_to_factor(v_id, f_id)
        for f_id, factor in self.graph.factors.items():
            for v_id in factor.scope:
                self._factor_to_variable(f_id, v_id)

    def compute_beliefs(self):
        for v_id, var_node in self.graph.variables.items():
            incoming = self.manager.get_incoming_payloads(v_id)
            belief = self._multiply_messages(incoming)
            if belief is not None:
                belief.prune(threshold=1e-4)
            var_node.belief = belief
