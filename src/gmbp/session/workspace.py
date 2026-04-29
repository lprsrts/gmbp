class Session:
    """The central state manager for the active graph and canvas telemetry."""
    def __init__(self):
        self.variables = set()
        self.factors = {}

    def add_variable(self, *names):
        for name in names:
            self.variables.add(name)
            # TODO: Emit telemetry event to canvas here
            print(f"[Session] Variable added: {name}")

    def add_factor(self, name, scope):
        self.factors[name] = scope
        # TODO: Emit telemetry event to canvas here
        print(f"[Session] Factor added: {name} with scope {scope}")
        # Return a fluent factor object eventually
        return name
