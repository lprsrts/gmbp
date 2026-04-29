from .distributions import GaussianMixture

class Message:
    """Represents a probabilistic message passed along an edge in the factor graph."""
    def __init__(self, source, target, payload: GaussianMixture = None):
        self.source = source
        self.target = target
        self.payload = payload  # The GaussianMixture being transmitted

    def __repr__(self):
        state = "empty" if self.payload is None else f"GM({len(self.payload.components)} comps)"
        return f"Message({self.source} -> {self.target}, {state})"

class MessageManager:
    """Manages the state of all messages currently propagating through the bipartite graph."""
    def __init__(self):
        # Maps (source_name, target_name) -> Message
        self._messages = {}

    def initialize_edge(self, node_a, node_b):
        """Initializes empty message containers in both directions for a given edge."""
        self._messages[(node_a, node_b)] = Message(node_a, node_b)
        self._messages[(node_b, node_a)] = Message(node_b, node_a)

    def update(self, source, target, payload: GaussianMixture):
        """Updates the payload of a specific directional message."""
        if (source, target) not in self._messages:
            self._messages[(source, target)] = Message(source, target)
        self._messages[(source, target)].payload = payload

    def get_payload(self, source, target):
        """Retrieves the current GaussianMixture payload for a specific direction."""
        msg = self._messages.get((source, target))
        return msg.payload if msg else None

    def get_incoming_payloads(self, target, exclude_source=None):
        """
        Retrieves all incoming messages to a target node.
        Crucial for Belief Propagation: Variable nodes multiply all incoming messages, 
        and when calculating an outgoing message to a specific neighbor, that neighbor's 
        incoming message is excluded.
        """
        payloads = []
        for (src, tgt), msg in self._messages.items():
            if tgt == target and src != exclude_source:
                if msg.payload is not None:
                    payloads.append(msg.payload)
        return payloads

    def __repr__(self):
        return f"MessageManager({len(self._messages)} directional edges)"
