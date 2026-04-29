from .session.workspace import Session, load_data

_active_session = Session()

def add_variable(*names):
    return _active_session.add_variable(*names)

def add_factor(name, scope):
    return _active_session.add_factor(name, scope)
    
def factor(name):
    return _active_session.factor(name)
    
def variable(name):
    return _active_session.variable(name)

def observe():
    return _active_session.observe()
    
def step():
    return _active_session.step()

def compute_beliefs():
    return _active_session.compute_beliefs()
    
def clear():
    return _active_session.clear()
