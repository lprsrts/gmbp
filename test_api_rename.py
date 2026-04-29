import numpy as np
from gmbp import add_variable, add_factor, factor, variable, observe, step, compute_beliefs, clear
from gmbp.api import _active_session

clear()
add_variable("X", "Y")
add_factor("f_xy", ("X", "Y"))

# Rename test
variable("X").rename("Temperature")
variable("Y").rename("Pressure")
factor("f_xy").rename("f_temp_press")

# Data fitting on renamed factor
np.random.seed(42)
data = np.random.multivariate_normal([0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], 500)
factor("f_temp_press").fit_gm(data, N=1)

# Observe renamed variable
observe().variable("Temperature").at(0.4)

# Inference
step()
compute_beliefs()

# Print belief of renamed variable
y_id = _active_session.graph.resolve_id("Pressure")
y_node = _active_session.graph.variables[y_id]
print(f"Belief of Pressure: {y_node.belief}")
