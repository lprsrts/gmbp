"""S2. The relevance field is itself a mixture, and it blows up.

lambda = r(x)^2 squares an n-component message into n^2 components; the
damped update lambda <- (1-a)lambda + a lambda_hat concatenates, doubling
again.  So the relevance field needs its own reduction -- the exact problem
the method was introduced to solve, one level up, and the note does not
mention it.

Measured: pre-cap component counts, the cost of capping (does the capped
relevance still rank merges the way the exact relevance does), how far the
relevance drifts from the message it weights (in message sigmas, i.e. how
close it gets to the S1c underflow cliff), and the total arithmetic cost
against the cheap local criterion.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gmr.build import build_graph
from gmr.graph import GMBP, unit
from gmr.gm import GM, gm_product, gm_square
from gmr.relevance import RelevanceField, normalise_lambda, cap, UNIFORM
from gmr.reduce import reduce_weighted_isd, reduce_runnalls
from gmr import fast1d as F
from scipy.stats import kendalltau


def instrumented_run(seed=1, T=6, budget=6, alpha=0.5, lam_budget=6,
                     pre=40):
    g = build_graph(seed=seed)
    bp = GMBP(g, budget=budget, reducer=reduce_weighted_isd)
    field = RelevanceField(bp, alpha=alpha, lam_budget=lam_budget)
    raw_sizes, drifts, cap_effects = [], [], []
    for t in range(T):
        bp.sweep(lam_lookup=field.lookup)
        # what lambda would be if nothing capped it
        hat = {}
        for v in g.sensors:
            for fn in g.nbrs[v]:
                pieces = [bp.msg[(f2, v)] for f2 in g.nbrs[v] if f2 != fn]
                if v in g.obs and g.obs[v] is not None:
                    pieces.append(g.obs[v])
                r = None
                for p in pieces:
                    r = p if r is None else gm_product(r, p)
                if r is None:
                    continue
                sq = gm_square(unit(r))
                prev = field.lam.get((fn, v), UNIFORM)
                uncapped_n = sq.n + (prev.n if alpha < 1.0 else 0)
                raw_sizes.append({"t": t, "edge": f"{fn}->{v}",
                                  "r_n": r.n, "lam_sq_n": sq.n,
                                  "lam_uncapped_n": uncapped_n})
                hat[(fn, v)] = sq
                # drift of the capped lambda relative to the message
                lam_c = normalise_lambda(cap(sq, lam_budget, pre=pre), "peak")
                msg = bp.msg[(fn, v)]
                msd = float(np.sqrt(np.average(
                    msg.S[:, 0, 0] + msg.mu[:, 0] ** 2, weights=msg.w)
                    - np.average(msg.mu[:, 0], weights=msg.w) ** 2))
                mmean = float(np.average(msg.mu[:, 0], weights=msg.w))
                lpk = float(lam_c.mu[np.argmax(lam_c.w), 0])
                drifts.append({"t": t, "edge": f"{fn}->{v}",
                               "drift_sigmas": abs(lpk - mmean) / max(msd, 1e-12),
                               "msg_sd": msd})
                # ranking damage from capping lambda
                if msg.n >= 4:
                    w, m, s = msg.w, msg.mu[:, 0], msg.S[:, 0, 0]
                    lam_ex = (sq.w, sq.mu[:, 0], sq.S[:, 0, 0])
                    lam_cp = (lam_c.w, lam_c.mu[:, 0], lam_c.S[:, 0, 0])
                    ce, _, _ = F.all_pair_costs_isd(w, m, s, lam_ex)
                    cc, _, _ = F.all_pair_costs_isd(w, m, s, lam_cp)
                    ok = np.isfinite(ce) & np.isfinite(cc)
                    if ok.sum() > 3:
                        tau = kendalltau(ce[ok], cc[ok]).statistic
                        cap_effects.append({
                            "t": t, "edge": f"{fn}->{v}",
                            "kendall": float(tau),
                            "top1_same": bool(np.argmin(ce[ok]) == np.argmin(cc[ok])),
                            "lam_exact_n": sq.n, "lam_capped_n": lam_c.n})
        field.update(bp.msg)
    return raw_sizes, drifts, cap_effects


def cost_comparison(seeds=(1, 2, 3), T=6, budget=6):
    from gmr.adaptive import run_local, run_adaptive
    out = []
    for sd in seeds:
        g = build_graph(seed=sd)
        F.reset_counters()
        run_local(g, T, budget, "runnalls")
        loc = dict(F.COUNTERS)
        g = build_graph(seed=sd)
        F.reset_counters()
        run_adaptive(g, T, budget, alpha=0.5, tau=None)
        ad = dict(F.COUNTERS)
        out.append({"seed": sd, "runnalls": loc, "adaptive": ad,
                    "cost_ratio": ad["pair_costs"] / max(loc["pair_costs"], 1)})
    return out


if __name__ == "__main__":
    sizes, drifts, caps = instrumented_run()
    un = np.array([r["lam_uncapped_n"] for r in sizes])
    rn = np.array([r["r_n"] for r in sizes])
    print("S2  Relevance blow-up")
    print(f"    downstream product r has {rn.min()}-{rn.max()} components")
    print(f"    lambda = r^2 before any cap: median {int(np.median(un))}, max {int(un.max())} components")
    print(f"    the note's Algorithm 1 caps the message at 6 and says nothing about capping lambda")
    ke = np.array([c["kendall"] for c in caps])
    t1 = np.array([c["top1_same"] for c in caps])
    print(f"\n    cost of capping lambda to 6 (weight-prune to 40, then merge):")
    print(f"      Kendall tau between merge orders under exact and capped lambda: median {np.median(ke):.3f}, p10 {np.percentile(ke,10):.3f}, min {ke.min():.3f}")
    print(f"      the merge actually chosen is the same in {t1.mean():.0%} of reductions")
    dr = np.array([d["drift_sigmas"] for d in drifts])
    print(f"\n    drift of the relevance peak from the message it weights:")
    print(f"      median {np.median(dr):.2f} message-sigmas, p90 {np.percentile(dr,90):.2f}, max {dr.max():.2f}")
    print(f"      (S1c puts the underflow cliff near 20 sigmas for a narrow lambda)")
    cc = cost_comparison()
    for r in cc:
        print(f"\n    seed {r['seed']}: pair-cost evaluations  Runnalls {r['runnalls']['pair_costs']:,}"
              f"   adaptive {r['adaptive']['pair_costs']:,}   ratio {r['cost_ratio']:.1f}x")
    json.dump({"sizes": sizes, "drifts": drifts, "caps": caps, "cost": cc},
              open(os.path.join(os.path.dirname(__file__), "..", "results",
                                "s02_relevance.json"), "w"), indent=1)
