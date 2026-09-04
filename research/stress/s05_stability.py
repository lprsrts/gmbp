"""S5. Does the coupled belief-relevance iteration settle?

Reduction depends on the relevance field, the relevance field depends on
the messages, and the messages depend on the reduction.  The note's only
control on that loop is the damping factor alpha in Eq. (49).  Nothing in
the note argues the coupled map is contractive.

Measured over 20 sweeps: how much the sensor beliefs and the relevance
field still move at the end, compared with the same graph under a fixed
local criterion and under no reduction at all.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gmr.build import build_graph
from gmr.graph import GMBP
from gmr.relevance import RelevanceField
from gmr.reduce import reduce_weighted_isd, reduce_runnalls
from gmr.adaptive import run_reference, run_local
from gmr.metrics import tv


def lam_distance(a, b, n=801):
    lo = min(a.mu[:, 0].min(), b.mu[:, 0].min()) - 6
    hi = max(a.mu[:, 0].max(), b.mu[:, 0].max()) + 6
    x = np.linspace(lo, hi, n)
    pa, pb = a.pdf(x), b.pdf(x)
    pa = pa / max(pa.max(), 1e-300); pb = pb / max(pb.max(), 1e-300)
    return float(np.max(np.abs(pa - pb)))


def trace_adaptive(seed, T=20, budget=6, alpha=0.5, regime="medium"):
    g = build_graph(seed=seed, obs_strength=regime)
    bp = GMBP(g, budget=budget, reducer=reduce_weighted_isd)
    field = RelevanceField(bp, alpha=alpha, lam_budget=6)
    bel, lam_move = [], []
    prev_lam = None
    for t in range(T):
        bp.sweep(lam_lookup=field.lookup)
        field.update(bp.msg)
        bel.append({v: bp.belief(v) for v in g.sensors})
        if prev_lam is not None:
            d = [lam_distance(prev_lam[k], field.lam[k])
                 for k in field.lam if k in prev_lam]
            lam_move.append(float(np.mean(d)))
        prev_lam = {k: v for k, v in field.lam.items()}
    return bel, lam_move


def sweep_motion(bel, tail=6):
    """Mean TV between consecutive sweeps over the last `tail` sweeps."""
    out = []
    for t in range(len(bel) - tail, len(bel)):
        out.append(np.mean([tv(bel[t - 1][v], bel[t][v]) for v in bel[t]]))
    return float(np.mean(out))


def local_motion(seed, T=20, budget=6, method="runnalls", regime="medium"):
    g = build_graph(seed=seed, obs_strength=regime)
    _, sn = run_local(g, T, budget, method)
    bel = [{v: s[v] for v in g.sensors} for s in sn]
    return sweep_motion(bel)


def ref_motion(seed, T=20, regime="medium"):
    g = build_graph(seed=seed, obs_strength=regime)
    _, sn = run_reference(g, T)
    bel = [{v: s[v] for v in g.sensors} for s in sn]
    return sweep_motion(bel)


if __name__ == "__main__":
    seeds = (1, 2, 3, 4)
    alphas = (0.1, 0.25, 0.5, 0.75, 1.0)
    print("S5  Stability of the coupled belief-relevance iteration, 20 sweeps")
    print("    'motion' = mean TV between consecutive sweeps over the last 6 sweeps")
    rm = float(np.mean([ref_motion(s) for s in seeds]))
    lm = float(np.mean([local_motion(s) for s in seeds]))
    print(f"\n    unreduced reference          motion {rm:.3e}")
    print(f"    Runnalls, budget 6           motion {lm:.3e}")
    rows = [{"method": "reference", "motion": rm},
            {"method": "runnalls", "motion": lm}]
    for a in alphas:
        ms, lmv = [], []
        for s in seeds:
            bel, lam_move = trace_adaptive(s, alpha=a)
            ms.append(sweep_motion(bel))
            lmv.append(float(np.mean(lam_move[-6:])))
        print(f"    adaptive, alpha = {a:<4}         motion {np.mean(ms):.3e}"
              f"   relevance still moving {np.mean(lmv):.3e}")
        rows.append({"method": f"adaptive_a{a}", "motion": float(np.mean(ms)),
                     "lam_motion": float(np.mean(lmv))})
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "..",
                                      "results", "s05_stability.json"), "w"), indent=1)
