# gmbp

Gaussian Mixture Belief Propagation (GMBP) — A high-performance inference engine and dynamic canvas for continuous probabilistic networks.

## Design Philosophy

**Say Less, Tell More.**
This library abstracts rigorous bipartite graph topology and sum-product message passing behind a fluent, REPL-friendly API. It decouple the mathematical control plane from the visual telemetry, allowing non-programmers to define networks naturally while viewing real-time topological shifts and probability density functions in the browser.

## Architecture

- **`gmbp.core`**: Canonical form operations for Gaussian / Gaussian Mixture Models (multiplication, exact analytic marginalization, Schur complement conditioning). Bipartite topology data structures.
- **`gmbp.inference`**: Synchronous Belief Propagation engine. Prevents exponential mixture explosion via dynamic pruning. Backed by `scikit-learn` for EM data fitting.
- **`gmbp.session`**: Facade pattern managing the implicit global state and resolving string names to immutable UUID hashes.
- **`gmbp.canvas`**: Asynchronous `FastAPI` / `WebSocket` daemon bridging the REPL to a brutalist, physics-based `Cytoscape` / `Plotly` local web UI.

## Usage

Start your REPL, then open `http://127.0.0.1:8080` in your browser.

```python
import numpy as np
from gmbp import *

# 1. Scaffold the network dynamically
add_variable("X", "Y")
add_factor("f_xy", ("X", "Y"))

# 2. Fluent modifications
variable("X").rename("Temperature")
variable("Y").rename("Pressure")

# 3. Fit empirical data to factors using Expectation-Maximization
data = np.random.multivariate_normal([0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], 500)
factor("f_xy").fit_gm(data, N=2)

# 4. Inject deterministic observations
observe().variable("Temperature").at(25.5)

# 5. Execute inference
step()
compute_beliefs()
```

When you execute `step()` and `compute_beliefs()`, the canvas will automatically re-render the mathematical $P(X)$ density functions.
