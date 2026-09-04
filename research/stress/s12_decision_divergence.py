"""S12. How often does the relevance actually change a decision?

S9c found that on a chain the adaptive method and plain unweighted ISD
agree to eight significant figures -- the same merges, every time. On the
note's own graph they do not. The difference has to come from the shape of
the relevance, so measure it directly, at every reduction the engine
performs: does argmin under lambda differ from argmin under the uniform
weight, and how flat is lambda over the message it is weighting?
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.stats import kendalltau
from gmr.build import build_graph, build_chain
from gmr.gm import GM
from gmr.graph import GMBP
from gmr.relevance import RelevanceField
from gmr.reduce import reduce_weighted_isd
from gmr import fast1d as F


def probe(g, T=6, budget=6, alpha=0.5):
    bp = GMBP(g, budget=budget, reducer=reduce_weighted_isd)
    field = RelevanceField(bp, alpha=alpha, lam_budget=6)
    events = []
    for t in range(T):
        # inspect every message that is about to be reduced, in both
        # directions -- on a chain only the factor->variable ones grow
        cand = []
        for v, fnames in g.nbrs.items():
            if v in g.classes:
                continue
            for fn in fnames:
                cand.append(((v, fn), bp.var_to_factor(v, fn)))
        for fname, f in g.factors.items():
            for v in f.scope:
                if v in g.classes:
                    continue
                cand.append(((fname, v), bp.factor_to_var(fname, v)))
        for key, m in cand:
                if not isinstance(m, GM) or m.n <= budget:
                    continue
                lam = field.lookup(key)
                lamt = (lam.w, lam.mu[:, 0], lam.S[:, 0, 0])
                w, mu, s = m.w, m.mu[:, 0], m.S[:, 0, 0]
                cw, iu, ju = F.all_pair_costs_isd(w, mu, s, lamt)
                cu, _, _ = F.all_pair_costs_isd(w, mu, s, None)
                cr, _, _ = F.all_pair_costs_runnalls(w, mu, s)
                ok = np.isfinite(cw) & np.isfinite(cu)
                if ok.sum() < 4:
                    continue
                lv = lam.pdf(mu)
                flat = float(lv.max() / max(lv.min(), 1e-300))
                events.append({
                    "t": t, "n": int(m.n),
                    "diff_vs_uniform": bool(np.argmin(cw[ok]) != np.argmin(cu[ok])),
                    "diff_vs_runnalls": bool(np.argmin(cw[ok]) != np.argmin(cr[ok])),
                    "kendall_vs_uniform": float(kendalltau(cw[ok], cu[ok]).statistic),
                    "lambda_dynamic_range": flat,
                })
        bp.sweep(lam_lookup=field.lookup)
        field.update(bp.msg)
    return events


def summarise(name, ev):
    du = np.mean([e["diff_vs_uniform"] for e in ev])
    dr = np.mean([e["diff_vs_runnalls"] for e in ev])
    kt = np.median([e["kendall_vs_uniform"] for e in ev])
    fl = np.median([e["lambda_dynamic_range"] for e in ev])
    print(f"    {name:<28}{len(ev):>8}{du:>28.0%}{dr:>26.0%}{kt:>16.3f}{fl:>18.1f}")
    return {"name": name, "events": len(ev), "diff_uniform": float(du),
            "diff_runnalls": float(dr), "kendall": float(kt),
            "lambda_range": float(fl)}


if __name__ == "__main__":
    print("S12 Does the relevance change the merge that gets chosen?\n")
    print(f"    {'graph':<28}{'events':>8}{'picks differ from plain ISD':>28}"
          f"{'differ from Runnalls':>26}{'Kendall vs ISD':>16}{'lambda range':>18}")
    out = []
    ev = []
    for sd in range(6):
        ev += probe(build_graph(seed=sd, obs_strength="medium"))
    out.append(summarise("note's graph, medium obs", ev))
    ev = []
    for sd in range(6):
        ev += probe(build_graph(seed=sd, obs_strength="none"))
    out.append(summarise("note's graph, no obs", ev))
    ev = []
    for sd in range(6):
        ev += probe(build_chain(seed=sd, K=6, obs_at=1.0, both_ends=2.0), T=8)
    out.append(summarise("chain, both ends observed", ev))
    print("\n    'lambda range' is max/min of lambda evaluated at the component means:")
    print("    1.0 would mean the weighted criterion is exactly the unweighted one.")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..",
                                     "results", "s12_divergence.json"), "w"), indent=1)
