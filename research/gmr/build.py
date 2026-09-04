"""The concept note's graph: 3 classes x 3 states, 4 sensors, 9 factors."""
import numpy as np
from .gm import GM
from .graph import HybridGraph, ClassSensorFactor, SensorPairFactor


def class_conditional(rng, base, spread, modes=3, var=0.35):
    mu = base + spread * np.linspace(-1.0, 1.0, modes) + rng.normal(0, 0.15, modes)
    w = rng.dirichlet(np.ones(modes) * 2.0)
    return GM(w, mu.reshape(-1, 1), np.full((modes, 1, 1), var))


def build_graph(seed=0, modes=3, states=3, obs_strength="medium",
                obs_at=2.0, sensor_var=0.35):
    rng = np.random.default_rng(seed)
    g = HybridGraph()
    for c in ("C1", "C2", "C3"):
        g.add_class(c, states)
    for s in ("S1", "S2", "S3", "S4"):
        g.add_sensor(s)

    def bank(tag):
        return [class_conditional(rng, base=rng.uniform(-4, 4), spread=2.2,
                                  modes=modes, var=sensor_var)
                for _ in range(states)]

    # S1 and S2 are shared by all three classes -> three neighbours each
    for c in ("C1", "C2", "C3"):
        for s in ("S1", "S2"):
            g.add_factor(ClassSensorFactor(f"f_{c}_{s}", c, s, bank(f"{c}{s}")))
    g.add_factor(ClassSensorFactor("f_C1_S3", "C1", "S3", bank("C1S3")))
    g.add_factor(ClassSensorFactor("f_C2_S4", "C2", "S4", bank("C2S4")))
    g.add_factor(SensorPairFactor("f_S3_S4", "S3", "S4",
                                  GM([1.0], [[0.5]], [[[0.6]]])))

    if obs_strength != "none":
        var = {"strong": 0.05, "medium": 0.4, "weak": 3.0}[obs_strength]
        g.observe("S1", GM([1.0], [[obs_at]], [[[var]]]))
    return g


def build_chain(seed=0, K=6, modes=3, trans_spread=1.4, trans_var=0.25,
                obs_at=None, obs_var=0.2, obs_node=None, both_ends=None):
    """A continuous chain x_0 - x_1 - ... - x_K with mixture transitions.

    Unlike the note's graph, nothing here regenerates a dropped hypothesis:
    the message along the chain is the only carrier of history. This is the
    structure the irreversibility argument in the note actually assumes.
    """
    from .graph import HybridGraph, SensorPairFactor
    rng = np.random.default_rng(seed)
    g = HybridGraph()
    for k in range(K + 1):
        g.add_sensor(f"x{k}")
    for k in range(K):
        mu = trans_spread * np.linspace(-1, 1, modes) + rng.normal(0, 0.1, modes)
        w = rng.dirichlet(np.ones(modes) * 2.0)
        disp = GM(w, mu.reshape(-1, 1), np.full((modes, 1, 1), trans_var))
        g.add_factor(SensorPairFactor(f"t{k}", f"x{k}", f"x{k+1}", disp))
    if obs_at is not None:
        node = obs_node if obs_node is not None else "x0"
        g.observe(node, GM([1.0], [[obs_at]], [[[obs_var]]]))
    if both_ends is not None:
        # observing only one end leaves every message travelling the other
        # way carrying no information, so the relevance for half the edges
        # is flat by construction rather than by any property of the method
        g.observe(f"x{K}", GM([1.0], [[both_ends]], [[[obs_var]]]))
    return g
