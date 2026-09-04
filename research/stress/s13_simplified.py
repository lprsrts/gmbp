"""S13. What is left when the apparatus is removed?

S4 found the backward pass contributes nothing and damping hurts; S6, S8 and
S9 found the reserve harmful. Those were measured one at a time. This
removes all of them at once and asks whether the remainder still works.

The remainder is a single local rule with no propagation, no damping, no
reserve and no state of its own:

    when reducing the message a -> b, weight the ISD by
    lambda(x) = ( prod_{c != a} m_{c->b}(x) )^2,
    rebuilt from the current messages every sweep.

Compared against Runnalls and against Algorithm 1 as written, on the note's
graph and on the late-evidence graph.
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gmr.build import build_graph
from gmr.adaptive import run_reference, run_local, run_adaptive
from gmr.metrics import tv, class_tv
from gmr import fast1d as F
from s04_principal_hypothesis import build_late_graph
from gmr.gm import GM

VARIANTS = {
    "algorithm 1 as written": dict(alpha=0.5, tau=0.02, use_reserve=True,
                                   backward=True),
    "minus the reserve": dict(alpha=0.5, tau=None, use_reserve=False,
                              backward=True),
    "minus reserve and damping": dict(alpha=1.0, tau=None, use_reserve=False,
                                      backward=True),
    "single-step rule only": dict(alpha=1.0, tau=None, use_reserve=False,
                                  backward=False),
}


def gap(ref, sn, g):
    return float(np.mean([np.mean([tv(ref[t][v], sn[t][v]) for v in g.sensors])
                          for t in range(len(sn))]))


def grid(builder, seeds, regimes, budgets, T, hook=None):
    rows = []
    for sd, reg, bud in itertools.product(seeds, regimes, budgets):
        g = builder(sd, reg)
        _, ref = run_reference(g, T, hook=hook)
        row = {"seed": sd, "regime": reg, "budget": bud}
        for m in ("runnalls", "isd"):
            g2 = builder(sd, reg)
            _, sn = run_local(g2, T, bud, m, hook=hook)
            row[m] = gap(ref, sn, g2)
        for label, kw in VARIANTS.items():
            g2 = builder(sd, reg)
            F.reset_counters()
            _, sn, _, _, _ = run_adaptive(g2, T, bud, hook=hook, **kw)
            row[label] = gap(ref, sn, g2)
            row[label + "|cost"] = F.COUNTERS["pair_costs"]
        rows.append(row)
    return rows


def report(rows, title):
    print(f"\n    {title}, {len(rows)} configurations")
    base = np.array([r["runnalls"] for r in rows])
    cost0 = None
    print(f"      {'variant':<28}{'mean gap':>10}{'median ratio':>14}"
          f"{'beats Runnalls':>16}{'relative cost':>15}")
    g0 = None
    for m in ["runnalls", "isd"] + list(VARIANTS):
        v = np.array([r[m] for r in rows])
        c = (np.mean([r[m + "|cost"] for r in rows]) if m in VARIANTS
             else np.nan)
        if m == "runnalls":
            cost0 = 1.0
        rel = "" if np.isnan(c) else f"{c / ROOT_COST:>14.1f}x"
        print(f"      {m:<28}{v.mean():>10.5f}"
              f"{np.median(v / np.maximum(base, 1e-300)):>14.3f}"
              f"{np.mean(v < base):>15.0%}{rel:>15}")


if __name__ == "__main__":
    g = build_graph(seed=0)
    F.reset_counters(); run_local(g, 6, 6, "runnalls")
    ROOT_COST = F.COUNTERS["pair_costs"]

    A = grid(lambda sd, reg: build_graph(seed=sd, obs_strength=reg),
             seeds=range(5), regimes=("strong", "medium", "none"),
             budgets=(4, 6, 8), T=6)
    print("S13 Removing the apparatus")
    report(A, "the note's graph")

    def hook(t, gg):
        if t == 3:
            gg.observe("S1", GM([1.0], [[6.0]], [[[0.15]]]))
        elif t < 3:
            gg.obs.pop("S1", None)

    B = grid(lambda sd, reg: build_late_graph(sd), seeds=range(8),
             regimes=("none",), budgets=(6,), T=8, hook=hook)
    report(B, "the late-evidence graph")
    json.dump({"a": A, "b": B, "root_cost": ROOT_COST},
              open(os.path.join(os.path.dirname(__file__), "..", "results",
                                "s13_simplified.json"), "w"), indent=1)
