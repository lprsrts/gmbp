"""S4. The principal hypothesis.

    D_final(b_ref, b_adaptive) < D_final(b_ref, b_local)

at equal active-component budget.  Every run is compared against an
unreduced reference *at the same sweep*, which is the comparison the note
insists on.

Part A sweeps the note's own graph over observation regimes, budgets and
seeds.  Part B builds the case the note says should favour it most: a weak
distant hypothesis that only later information supports.
"""
import sys, os, json, itertools, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gmr.build import build_graph
from gmr.gm import GM
from gmr.adaptive import run_reference, run_local, run_adaptive
from gmr.metrics import tv, class_tv


def gap(ref_snaps, snaps, g):
    """Mean TV against the reference, per sweep, over sensors and classes."""
    sens, cls = [], []
    for t in range(len(snaps)):
        sens.append(np.mean([tv(ref_snaps[t][v], snaps[t][v]) for v in g.sensors]))
        cls.append(np.mean([class_tv(ref_snaps[t][c], snaps[t][c]) for c in g.classes]))
    return float(np.mean(sens)), float(np.mean(cls)), float(sens[-1]), float(cls[-1])


def part_a(seeds=range(6), regimes=("strong", "medium", "weak", "none"),
           budgets=(4, 6, 8), T=6):
    rows = []
    for sd, reg, bud in itertools.product(seeds, regimes, budgets):
        g = build_graph(seed=sd, obs_strength=reg)
        _, ref = run_reference(g, T)
        res = {}
        for meth in ("prune", "runnalls", "isd"):
            g2 = build_graph(seed=sd, obs_strength=reg)
            _, sn = run_local(g2, T, bud, meth)
            res[meth] = gap(ref, sn, g)
        for tag, kw in (("adaptive", dict(alpha=0.5, tau=None, use_reserve=False)),
                        ("adaptive nobwd", dict(alpha=0.5, tau=None,
                                                use_reserve=False, backward=False)),
                        ("adaptive a=1.0", dict(alpha=1.0, tau=None, use_reserve=False))):
            g2 = build_graph(seed=sd, obs_strength=reg)
            _, sn, _, _, _ = run_adaptive(g2, T, bud, **kw)
            res[tag] = gap(ref, sn, g)
        rows.append({"seed": sd, "regime": reg, "budget": bud,
                     **{f"{k}_{m}": v for k, vals in res.items()
                        for m, v in zip(("sens_mean", "cls_mean", "sens_final",
                                         "cls_final"), vals)}})
    return rows


def build_late_graph(seed, obs_at=None, decoy=6.0):
    """A sensor carrying a dominant mode and a weak distant one; the distant
    one is what the rest of the graph will later support."""
    g = build_graph(seed=seed, obs_strength="none")
    for name, f in g.factors.items():
        if name.startswith("f_C1_S1"):
            new = []
            for gmc in f.per_state:
                w = np.append(gmc.w * 0.94, 0.06)
                m = np.vstack([gmc.mu, [[decoy]]])
                s = np.concatenate([gmc.S, [[[0.4]]]], axis=0)
                new.append(GM(w, m, s))
            f.per_state = new
    return g


def part_b(seeds=range(8), T=8, budget=6, arrive=3, decoy=6.0):
    rows = []
    for sd in seeds:
        def hook(t, g):
            if t == arrive:
                g.observe("S1", GM([1.0], [[decoy]], [[[0.15]]]))
            elif t < arrive:
                g.obs.pop("S1", None)
        g = build_late_graph(sd, decoy=decoy)
        _, ref = run_reference(g, T, hook=hook)
        res = {}
        for meth in ("prune", "runnalls", "isd"):
            g2 = build_late_graph(sd, decoy=decoy)
            _, sn = run_local(g2, T, budget, meth, hook=hook)
            res[meth] = gap(ref, sn, g)
        for tag, kw in (("adaptive", dict(alpha=0.5, tau=None, use_reserve=False)),
                        ("adaptive nobwd", dict(alpha=0.5, tau=None,
                                                use_reserve=False, backward=False)),
                        ("adaptive a=1.0", dict(alpha=1.0, tau=None, use_reserve=False))):
            g2 = build_late_graph(sd, decoy=decoy)
            _, sn, _, _, _ = run_adaptive(g2, T, budget, hook=hook, **kw)
            res[tag] = gap(ref, sn, g)
        rows.append({"seed": sd, **{f"{k}_{m}": v for k, vals in res.items()
                                    for m, v in zip(("sens_mean", "cls_mean",
                                                     "sens_final", "cls_final"),
                                                    vals)}})
    return rows


METHODS = ("prune", "runnalls", "isd", "adaptive", "adaptive nobwd", "adaptive a=1.0")


def summarise(rows, key, title):
    print(f"\n    {title}  ({key})")
    base = np.array([r[f"runnalls_{key}"] for r in rows])
    for m in METHODS:
        v = np.array([r[f"{m}_{key}"] for r in rows])
        win = np.mean(v < base)
        med = np.median(v / np.maximum(base, 1e-300))
        print(f"      {m:<18} mean {v.mean():.5f}   median ratio to Runnalls {med:6.3f}"
              f"   beats Runnalls in {win:.0%} of configurations")


if __name__ == "__main__":
    t0 = time.time()
    A = part_a()
    print(f"S4a  The note's graph: {len(A)} configurations "
          f"(6 seeds x 4 regimes x 3 budgets), 6 sweeps, "
          f"reference compared sweep by sweep")
    for key, title in (("sens_mean", "sensor beliefs, mean over sweeps"),
                       ("sens_final", "sensor beliefs, final sweep"),
                       ("cls_mean", "class beliefs, mean over sweeps")):
        summarise(A, key, title)
    B = part_b()
    print(f"\nS4b  Weak distant hypothesis, evidence for it arriving at sweep 3 "
          f"({len(B)} seeds, 8 sweeps, budget 6)")
    for key, title in (("sens_mean", "sensor beliefs, mean over sweeps"),
                       ("sens_final", "sensor beliefs, final sweep")):
        summarise(B, key, title)
    print(f"\n    elapsed {time.time()-t0:.0f}s")
    json.dump({"part_a": A, "part_b": B},
              open(os.path.join(os.path.dirname(__file__), "..", "results",
                                "s04_principal.json"), "w"), indent=1)
