import numpy as np
from gmbp import add_variable, add_factor, factor, observe, step, compute_beliefs, clear
from gmbp.api import _active_session

# 1. Clear session
clear()

# 2. Scaffold the graph
add_variable("X", "Y")
add_factor("f_xy", ("X", "Y"))

# 3. Generate synthetic correlated data
np.random.seed(42)
data_cluster_1 = np.random.multivariate_normal([0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], 500)
data_cluster_2 = np.random.multivariate_normal([5.0, 5.0], [[1.0, -0.8], [-0.8, 1.0]], 500)
synthetic_data = np.vstack([data_cluster_1, data_cluster_2])

# 4. Fluent chaining
f0 = factor("f_xy")
f0.fit_gm(synthetic_data, N=2)

observe().variable("X").at(0.4)

# 5. Inference
step()
compute_beliefs()

# 6. Verification
print("\n--- INFERENCE RESULTS ---")
y_node = _active_session.graph.variables["Y"]
if y_node.belief:
    print(f"Variable Y Belief -> {y_node.belief}")
    for idx, (w, comp) in enumerate(zip(y_node.belief.weights, y_node.belief.components)):
        print(f"  Component {idx}: weight={w:.4f}, mean={comp.mean[0]:.4f}, var={comp.cov[0][0]:.4f}")
else:
    print("Belief of Y is None.")
