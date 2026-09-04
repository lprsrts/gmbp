"""S11. Is the downstream information doing the work?

The note's own testable-hypothesis section rests on the claim that a
downstream-aware criterion can make a different decision from any local
rule *and that the difference is an improvement*. A weighted ISD with any
non-uniform weight is already a different criterion from Runnalls. So the
control that matters is a relevance field that is real in shape but wrong
in placement.

  true lambda      the method as specified
  shuffled lambda  each edge is given another edge's relevance
  fixed lambda     one arbitrary Gaussian everywhere, never updated
  uniform lambda   weighted ISD degenerates to plain ISD

If shuffled and fixed do as well as true, the relevance propagation is
decoration.
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gmr.build import build_graph
from gmr.gm import GM
from gmr.graph import GMBP
from gmr.relevance import RelevanceField, UNIFORM
from gmr.reduce import reduce_weighted_isd
from gmr.adaptive import run_reference, run_local
from gmr.metrics import tv


FIXED = GM([1.0], [[0.0]], [[[4.0]]])


def run_variant(g, T, budget, mode, seed=0, alpha=0.5):
    bp = GMBP(g, budget=budget, reducer=reduce_weighted_isd)
    field = RelevanceField(bp, alpha=alpha, lam_budget=6)
    rng = np.random.default_rng(seed)
    perm = {}

    def lookup(key):
        if mode == "true":
            return field.lookup(key)
        if mode == "fixed":
            return FIXED
        if mode == "uniform":
            return UNIFORM
        if mode == "fv_only":
            # relevance only on messages arriving at a variable node
            return field.lookup(key) if key[0] in field.g.factors else UNIFORM
        if mode == "vf_only":
            # relevance only on messages leaving a variable node
            return field.lookup(key) if key[0] not in field.g.factors else UNIFORM
        if mode == "shuffled":
            keys = [k for k, v in field.lam.items()]
            if not keys:
                return UNIFORM
            if key not in perm:
                perm[key] = keys[int(rng.integers(len(keys)))]
            return field.lam.get(perm[key], UNIFORM)
        raise ValueError(mode)

    snaps = []
    for t in range(T):
        bp.sweep(lam_lookup=lookup)
        field.update(bp.msg)
        snaps.append({v: bp.belief(v) for v in g.sensors})
    return snaps


def run(seeds=(0, 1, 2, 3, 4, 5), regimes=("strong", "medium", "none"),
        budgets=(4, 6), T=6):
    rows = []
    for sd, reg, bud in itertools.product(seeds, regimes, budgets):
        g = build_graph(seed=sd, obs_strength=reg)
        _, ref = run_reference(g, T)
        row = {"seed": sd, "regime": reg, "budget": bud}
        g2 = build_graph(seed=sd, obs_strength=reg)
        _, sn = run_local(g2, T, bud, "runnalls")
        row["runnalls"] = float(np.mean([np.mean([tv(ref[t][v], sn[t][v])
                                                  for v in g2.sensors])
                                         for t in range(T)]))
        for mode in ("true", "shuffled", "fixed", "uniform", "fv_only", "vf_only"):
            g2 = build_graph(seed=sd, obs_strength=reg)
            sn = run_variant(g2, T, bud, mode, seed=sd)
            row[mode] = float(np.mean([np.mean([tv(ref[t][v], sn[t][v])
                                                for v in g2.sensors])
                                       for t in range(T)]))
        rows.append(row)
    return rows


if __name__ == "__main__":
    rows = run()
    print(f"S11 Controls on the relevance field, {len(rows)} configurations")
    print("    mean TV gap to the unreduced reference, compared sweep by sweep\n")
    base = np.array([r["runnalls"] for r in rows])
    print(f"    {'relevance':<30}{'mean gap':>11}{'median ratio to Runnalls':>27}"
          f"{'beats Runnalls':>17}{'beats true lambda':>20}")
    truev = np.array([r["true"] for r in rows])
    for m, label in (("runnalls", "Runnalls (local)"), ("uniform", "uniform (plain ISD)"),
                     ("fixed", "fixed Gaussian"), ("shuffled", "shuffled across edges"),
                     ("vf_only", "true, variable->factor only"),
                     ("fv_only", "true, factor->variable only"),
                     ("true", "true downstream, all edges")):
        v = np.array([r[m] for r in rows])
        print(f"    {label:<30}{v.mean():>11.5f}{np.median(v/np.maximum(base,1e-300)):>26.3f}"
              f"{np.mean(v < base):>16.0%}{np.mean(v < truev):>19.0%}")
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "..",
                                      "results", "s11_controls.json"), "w"), indent=1)
