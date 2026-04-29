from gmbp import add_variable, add_factor, clear
from gmbp.api import _active_session

clear()
add_variable("X").rename("Temperature").at(25.5)
print(f"Nodes: {list(_active_session.graph.name_to_id.keys())}")

vars = add_variable("A", "B")
vars[0].rename("Alpha")
vars[1].rename("Beta")
print(f"Nodes: {list(_active_session.graph.name_to_id.keys())}")

add_factor("f_x", ("Temperature",)).rename("f_temp")
print(f"Nodes: {list(_active_session.graph.name_to_id.keys())}")
