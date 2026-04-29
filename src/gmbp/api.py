from .session.workspace import Session

# The invisible global session
_active_session = Session()

def add_variable(*names):
    """Adds variables to the active session graph."""
    return _active_session.add_variable(*names)

def add_factor(name, scope):
    """Adds a factor to the active session graph connecting the given scope."""
    return _active_session.add_factor(name, scope)
