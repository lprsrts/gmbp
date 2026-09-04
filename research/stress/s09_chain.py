"""S9. The structure the reserve argument actually assumes.

S8a showed a deleted hypothesis reappearing on the next sweep, because in
the note's graph a factor->sensor message is rebuilt from the potential and
the current class message. In a chain there is no such regeneration: the
message is the only carrier of history, so a deletion is permanent.

This repeats S8a and S8b on a chain, and re-runs the principal hypothesis
there, to find whether the method's premises hold where they are supposed
to.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gmr.build import build_chain
from gmr.gm import GM
from gmr.graph import GMBP, unit
from gmr.adaptive import run_reference, run_local, run_adaptive

REF_BUDGET = 150   # a chain grows 3x per sweep, so an exact reference is not
                   # affordable past a few sweeps; validated in main()


def reference(g, T):
    return run_local(g, T, REF_BUDGET, "runnalls")
from gmr.metrics import tv
from gmr.relevance import RelevanceField
from gmr.reduce import reduce_weighted_isd
from s08_irreversibility import drop_component, run_variant, pick_target


def part_a(seeds=(0, 1, 2, 3, 4), T=8, t0=3, K=6):
    rows = []
    for sd in seeds:
        g = build_chain(seed=sd, K=K, obs_at=1.0, both_ends=2.0)
        _, ref = reference(g, T)
        g2 = build_chain(seed=sd, K=K, obs_at=1.0, both_ends=2.0)
        bp = GMBP(g2, budget=None, reducer=None)
        traj, target = [], None
        for t in range(T):
            bp.sweep()
            if t == t0:
                best = pick_target(bp, g2.sensors)
                if best is None:
                    break
                imm, key, k, wk = best
                target = {"edge": f"{key[0]}->{key[1]}", "weight": wk,
                          "immediate_tv": float(imm)}
                bp.msg[key] = drop_component(bp.msg[key], k)
            traj.append(np.mean([tv(ref[t][v], bp.belief(v)) for v in g2.sensors]))
        rows.append({"seed": sd, "target": target, "tv": traj[t0:],
                     "recovery": float(traj[-1] / max(traj[t0], 1e-300))})
    return rows


def part_b(seeds=(0, 1, 2, 3), T=8, budget=6, K=6):
    rows = []
    settings = [("no reserve", None, False),
                ("as written, tau=0.1", 1e-1, False),
                ("repaired,   tau=0.1", 1e-1, True),
                ("as written, tau=1", 1.0, False),
                ("repaired,   tau=1", 1.0, True),
                ("relative,   k=1", 1.0, "relative"),
                ("relative,   k=3", 3.0, "relative"),
                ("relative,   k=10", 10.0, "relative")]
    for label, tau, repair in settings:
        gaps, reacts = [], []
        for sd in seeds:
            g = build_chain(seed=sd, K=K, obs_at=1.0, both_ends=2.0)
            _, ref = reference(g, T)
            g2 = build_chain(seed=sd, K=K, obs_at=1.0, both_ends=2.0)
            snaps, nr = run_variant(g2, T, budget, tau, repair)
            gaps.append(np.mean([np.mean([tv(ref[t][v], snaps[t][v])
                                          for v in g2.sensors])
                                 for t in range(T)]))
            reacts.append(nr)
        rows.append({"label": label, "gap": float(np.mean(gaps)),
                     "reactivations": float(np.mean(reacts))})
    return rows


def part_c(seeds=(0, 1, 2, 3, 4, 5), T=8, budgets=(4, 6, 8), K=6):
    rows = []
    for sd in seeds:
        for bud in budgets:
            g = build_chain(seed=sd, K=K, obs_at=1.0, both_ends=2.0)
            _, ref = reference(g, T)
            res = {}
            for meth in ("prune", "runnalls", "isd"):
                g2 = build_chain(seed=sd, K=K, obs_at=1.0, both_ends=2.0)
                _, sn = run_local(g2, T, bud, meth)
                res[meth] = float(np.mean([np.mean([tv(ref[t][v], sn[t][v])
                                                    for v in g2.sensors])
                                           for t in range(T)]))
            for tag, kw in (("adaptive", dict(alpha=0.5, tau=None, use_reserve=False)),
                            ("adaptive a=1.0", dict(alpha=1.0, tau=None,
                                                    use_reserve=False))):
                g2 = build_chain(seed=sd, K=K, obs_at=1.0, both_ends=2.0)
                _, sn, _, _, _ = run_adaptive(g2, T, bud, **kw)
                res[tag] = float(np.mean([np.mean([tv(ref[t][v], sn[t][v])
                                                   for v in g2.sensors])
                                          for t in range(T)]))
            rows.append({"seed": sd, "budget": bud, **res})
    return rows


def validate_reference(seeds=(0, 1, 2), T=8, K=6):
    """Is budget 150 high enough to stand in for no reduction?"""
    out = []
    for sd in seeds:
        g = build_chain(seed=sd, K=K, obs_at=1.0, both_ends=2.0)
        _, a = run_local(g, T, 150, "runnalls")
        g = build_chain(seed=sd, K=K, obs_at=1.0, both_ends=2.0)
        _, b = run_local(g, T, 300, "runnalls")
        out.append(float(np.mean([np.mean([tv(a[t][v], b[t][v])
                                           for v in g.sensors])
                                  for t in range(T)])))
    return out


if __name__ == "__main__":
    vr = validate_reference()
    print("S9  Chain graph. Reference is Runnalls at budget 150; raising it to")
    print(f"    300 moves the reference by TV {np.mean(vr):.2e} (mean over 3 seeds),")
    print("    which bounds the error this stand-in contributes.\n")
    A = part_a()
    print("S9a Same deletion test, on a chain (no regeneration path)\n")
    print(f"    {'seed':<6}{'edge':<18}{'weight':>9}   TV to reference from the deletion onward")
    for r in A:
        print(f"    {r['seed']:<6}{(r['target'] or {}).get('edge','-'):<18}"
              f"{(r['target'] or {}).get('weight',0):>9.4f}   "
              + "  ".join(f"{v:.5f}" for v in r["tv"]))
    rec = np.array([r["recovery"] for r in A])
    print(f"\n    final TV as a fraction of the TV right after deletion: median {np.median(rec):.3f}")
    print("    (S8a on the note's graph gave 0.002 -- there the hypothesis came back)")

    B = part_b()
    print("\nS9b Does the reserve earn its cost on a chain?\n")
    base = B[0]["gap"]
    print(f"    {'variant':<24}{'mean TV gap':>13}{'x no-reserve':>14}{'reactivations':>15}")
    for r in B:
        print(f"    {r['label']:<24}{r['gap']:>13.5f}{r['gap']/base:>13.2f}x{r['reactivations']:>15.0f}")

    C = part_c()
    print("\nS9c Principal hypothesis on a chain (mean TV gap to the unreduced reference)\n")
    meths = ("prune", "runnalls", "isd", "adaptive", "adaptive a=1.0")
    basev = np.array([r["runnalls"] for r in C])
    print(f"    {'method':<18}{'mean gap':>11}{'median ratio to Runnalls':>26}{'beats Runnalls':>17}")
    for m in meths:
        v = np.array([r[m] for r in C])
        print(f"    {m:<18}{v.mean():>11.5f}{np.median(v/np.maximum(basev,1e-300)):>25.3f}"
              f"{np.mean(v < basev):>16.0%}")
    json.dump({"a": A, "b": B, "c": C},
              open(os.path.join(os.path.dirname(__file__), "..", "results",
                                "s09_chain.json"), "w"), indent=1)
