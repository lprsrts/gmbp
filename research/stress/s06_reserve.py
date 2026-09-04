"""S6. The reserve set and the reactivation threshold.

Eq. (57)-(58) reactivate a stored hypothesis when

    rho_j = int lambda(x) w_j N(x; mu_j, Sigma_j) dx  >  tau_react.

rho carries the scale of lambda, and lambda is a squared BP message.  BP
messages are defined only up to a positive constant, and the note's own
normalisation choices (raw / by mass / by peak) rescale rho by orders of
magnitude while leaving every reduction decision unchanged.  So tau_react
is a threshold on a quantity with no fixed scale.

Measured: the dynamic range of rho under each convention, how the
reactivation count moves with tau, whether one tau transfers between
graphs, and whether reactivation helps at all.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gmr.build import build_graph
from gmr.graph import GMBP
from gmr.relevance import RelevanceField, reactivation_scores
from gmr.reduce import reduce_weighted_isd
from gmr.adaptive import run_reference, run_adaptive
from gmr.metrics import tv, class_tv


def collect_rho(seed=1, T=6, budget=6, norm="peak", regime="medium"):
    g = build_graph(seed=seed, obs_strength=regime)
    bp = GMBP(g, budget=budget, reducer=reduce_weighted_isd)
    field = RelevanceField(bp, alpha=0.5, lam_budget=6, norm=norm)
    reserve, rhos, growth = {}, [], []
    for t in range(T):
        bp.sweep(lam_lookup=field.lookup, reserve=reserve)
        field.update(bp.msg)
        growth.append(sum(len(v) for v in reserve.values()))
        for key, items in reserve.items():
            lam = field.lookup(key)
            comps = [(w, np.atleast_1d(m), np.atleast_2d(s))
                     for (w, m, s, _c) in items]
            if comps:
                rhos.extend(reactivation_scores(comps, lam).tolist())
    return np.asarray(rhos), growth


def tau_curve(rho, taus):
    return [float(np.mean(rho > t)) for t in taus]


def reactivation_benefit(seeds=(1, 2, 3, 4), T=8, budget=6,
                         taus=(None, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    rows = []
    for tau in taus:
        gaps = []
        for sd in seeds:
            g = build_graph(seed=sd, obs_strength="medium")
            _, ref = run_reference(g, T)
            g2 = build_graph(seed=sd, obs_strength="medium")
            _, sn, _, _, res = run_adaptive(
                g2, T, budget, alpha=0.5, tau=tau,
                use_reserve=(tau is not None))
            gaps.append(np.mean([np.mean([tv(ref[t][v], sn[t][v])
                                          for v in g.sensors])
                                 for t in range(T)]))
        rows.append({"tau": tau, "gap": float(np.mean(gaps))})
    return rows


if __name__ == "__main__":
    print("S6  The reactivation threshold has no fixed scale")
    taus = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0])
    store = {}
    for norm in ("raw", "mass", "peak"):
        rho, growth = collect_rho(norm=norm)
        rho = rho[np.isfinite(rho)]
        store[norm] = {"p1": float(np.percentile(rho, 1)),
                       "p50": float(np.median(rho)),
                       "p99": float(np.percentile(rho, 99)),
                       "curve": tau_curve(rho, taus)}
        print(f"\n    lambda normalised by {norm:<5}  rho: p1 {store[norm]['p1']:.3e}"
              f"  median {store[norm]['p50']:.3e}  p99 {store[norm]['p99']:.3e}"
              f"   dynamic range {store[norm]['p99']/max(store[norm]['p1'],1e-300):.1e}")
        print("      fraction of reserve above tau: " +
              "  ".join(f"{t:.0e}:{f:.0%}" for t, f in zip(taus, store[norm]["curve"])))
    print("\n    Every reduction decision is invariant to these rescalings;")
    print("    the reactivation rule is not.")

    print("\n    Does one tau transfer between graphs? (lambda normalised by peak)")
    per_seed = {}
    for sd in (1, 2, 3, 4, 5):
        rho, growth = collect_rho(seed=sd)
        rho = rho[np.isfinite(rho)]
        per_seed[sd] = tau_curve(rho, taus)
        print(f"      seed {sd}: " + "  ".join(f"{t:.0e}:{f:.0%}"
                                               for t, f in zip(taus, per_seed[sd])))
    _, growth = collect_rho(seed=1)
    print(f"\n    reserve size over 6 sweeps (single graph, cap 200/edge): {growth}")

    print("\n    Does reactivation improve the answer at all?")
    rows = reactivation_benefit()
    for r in rows:
        lbl = "no reserve" if r["tau"] is None else f"tau = {r['tau']:.0e}"
        print(f"      {lbl:<16} mean TV gap to reference {r['gap']:.5f}")
    json.dump({"scale": store, "per_seed": {str(k): v for k, v in per_seed.items()},
               "growth": growth, "benefit": rows},
              open(os.path.join(os.path.dirname(__file__), "..", "results",
                                "s06_reserve.json"), "w"), indent=1)
