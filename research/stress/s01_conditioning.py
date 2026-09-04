"""S1. Numerical conditioning of the weighted ISD merge criterion.

The closed form is a sum of six terms that very nearly cancel: the whole
point of a good merge is that p - q is almost zero, so the criterion is a
small number assembled from large ones.  The note reports agreement with
quadrature to 5e-16 and 4e-12, but those checks are on the *value* of a
weighted overlap, not on the *difference* that ranks merges.

Measured here: the cancellation ratio sum|terms| / |result|, how it varies
with merge quality, and whether the resulting noise changes which pair the
greedy step selects.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gmr import fast1d as F

try:
    import mpmath as mp
    HAVE_MP = True
except ImportError:
    HAVE_MP = False


def mp_cost(w, m, s, lam, i, j, dps=60):
    mp.mp.dps = dps
    W = [mp.mpf(float(x)) for x in w]
    M = [mp.mpf(float(x)) for x in m]
    S = [mp.mpf(float(x)) for x in s]
    wM = W[i] + W[j]
    mM = (W[i] * M[i] + W[j] * M[j]) / wM
    sM = (W[i] * (S[i] + (M[i] - mM) ** 2) + W[j] * (S[j] + (M[j] - mM) ** 2)) / wM
    comps = [(W[i], M[i], S[i]), (W[j], M[j], S[j]), (-wM, mM, sM)]
    if lam is None:
        LK = [(mp.mpf(1), None, None)]
    else:
        LK = [(mp.mpf(float(a)), mp.mpf(float(b)), mp.mpf(float(c)))
              for a, b, c in zip(*lam)]

    def nrm(x, mu, sg):
        return mp.e ** (-(x - mu) ** 2 / (2 * sg)) / mp.sqrt(2 * mp.pi * sg)

    tot = mp.mpf(0)
    for (a, b, c) in comps:
        for (a2, b2, c2) in comps:
            ssum = c + c2
            z = nrm(b, b2, ssum)
            sst = c * c2 / ssum
            mst = (b * c2 + b2 * c) / ssum
            for (gw, gm_, gs) in LK:
                if gm_ is None:
                    tot += a * a2 * z
                else:
                    tot += gw * a * a2 * z * nrm(mst, gm_, gs + sst)
    return tot


def run(seed=0, trials=200):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(trials):
        n = int(rng.integers(6, 14))
        w = rng.dirichlet(np.ones(n))
        m = rng.normal(0, 3.0, n)
        s = rng.uniform(0.1, 1.5, n)
        K = int(rng.integers(1, 5))
        lam = (rng.random(K), rng.normal(0, 3, K), rng.uniform(0.2, 3.0, K))
        cost, iu, ju, absum = F.all_pair_costs_isd(w, m, s, lam, return_terms=True)
        ratio = absum / np.maximum(np.abs(cost), 1e-300)
        order = np.argsort(cost)
        rows.append({
            "trial": t, "n": n, "K": K,
            "cond_best": float(ratio[order[0]]),
            "cond_median": float(np.median(ratio)),
            "cond_worst_pair": float(ratio.max()),
            "cost_best": float(cost[order[0]]),
            "cost_second": float(cost[order[1]]),
            "gap_rel": float((cost[order[1]] - cost[order[0]]) /
                             max(abs(cost[order[1]]), 1e-300)),
        })
    return rows


def decision_instability(seed=1, trials=400, jitter=1e-13):
    """Does a relative perturbation at 1e-13 change which pair is merged?"""
    rng = np.random.default_rng(seed)
    flips_isd = flips_run = 0
    for _ in range(trials):
        n = int(rng.integers(8, 16))
        w = rng.dirichlet(np.ones(n)); m = rng.normal(0, 3, n)
        s = rng.uniform(0.1, 1.5, n)
        K = int(rng.integers(1, 4))
        lam = (rng.random(K), rng.normal(0, 3, K), rng.uniform(0.2, 3, K))
        c0, iu, ju = F.all_pair_costs_isd(w, m, s, lam)
        r0, _, _ = F.all_pair_costs_runnalls(w, m, s)
        pert = 1.0 + jitter * rng.normal(size=n)
        c1, _, _ = F.all_pair_costs_isd(w, m * pert, s * pert, lam)
        r1, _, _ = F.all_pair_costs_runnalls(w, m * pert, s * pert)
        flips_isd += int(np.argmin(c0) != np.argmin(c1))
        flips_run += int(np.argmin(r0) != np.argmin(r1))
    return flips_isd / trials, flips_run / trials


def exact_check(seed=2, trials=12):
    if not HAVE_MP:
        return []
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        n = int(rng.integers(6, 10))
        w = rng.dirichlet(np.ones(n)); m = rng.normal(0, 3, n)
        s = rng.uniform(0.1, 1.5, n)
        K = int(rng.integers(1, 4))
        lam = (rng.random(K), rng.normal(0, 3, K), rng.uniform(0.2, 3, K))
        cost, iu, ju, absum = F.all_pair_costs_isd(w, m, s, lam, return_terms=True)
        k = int(np.argmin(cost))
        exact = mp_cost(w, m, s, lam, int(iu[k]), int(ju[k]))
        rel = abs(mp.mpf(float(cost[k])) - exact) / abs(exact)
        # rank agreement over all pairs, double vs 60-digit
        ex_all = np.array([float(mp_cost(w, m, s, lam, int(a), int(b)))
                           for a, b in zip(iu, ju)])
        out.append({"n": n, "cond": float(absum[k] / abs(cost[k])),
                    "rel_err_best": float(rel),
                    "argmin_agrees": bool(np.argmin(ex_all) == k)})
    return out


def regime_sweep(seed=7, trials=60):
    """The dangerous regime: relevance sitting away from the mixture, which
    is exactly what a downstream-aware weight is supposed to produce."""
    rng = np.random.default_rng(seed)
    out = []
    for dist in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
        for width in (2.0, 0.5, 0.1):
            conds, flips, disagree, relerrs = [], 0, 0, []
            for _ in range(trials):
                n = int(rng.integers(8, 13))
                w = rng.dirichlet(np.ones(n))
                m = rng.normal(0, 2.0, n)
                s = rng.uniform(0.1, 1.0, n)
                lam = (np.array([1.0]),
                       np.array([m.mean() + dist]),
                       np.array([width]))
                cost, iu, ju, absum = F.all_pair_costs_isd(
                    w, m, s, lam, return_terms=True)
                if not np.all(np.isfinite(cost)):
                    continue
                k = int(np.argmin(cost))
                conds.append(absum[k] / max(abs(cost[k]), 1e-300))
                pert = 1.0 + 1e-13 * rng.normal(size=n)
                c1, _, _ = F.all_pair_costs_isd(w, m * pert, s * pert, lam)
                flips += int(np.argmin(c1) != k)
                if HAVE_MP:
                    ex = np.array([float(mp_cost(w, m, s, lam, int(a), int(b)))
                                   for a, b in zip(iu, ju)])
                    disagree += int(np.argmin(ex) != k)
                    relerrs.append(abs(cost[k] - ex[k]) / max(abs(ex[k]), 1e-300))
            out.append({"dist": dist, "width": width,
                        "cond_median": float(np.median(conds)) if conds else float("nan"),
                        "cond_max": float(np.max(conds)) if conds else float("nan"),
                        "flip_rate": flips / trials,
                        "argmin_wrong_vs_exact": disagree / trials,
                        "relerr_median": float(np.median(relerrs)) if relerrs else float("nan")})
    return out


def print_regime():
    rs = regime_sweep()
    print("\nS1b Relevance placed away from the mixture (the intended use case)")
    print("    lam offset  width   cond(median)   cond(max)   argmin wrong vs exact   rel err")
    for r in rs:
        print(f"    {r['dist']:9.1f}  {r['width']:5.2f}   {r['cond_median']:10.2e}  {r['cond_max']:10.2e}"
              f"       {r['argmin_wrong_vs_exact']:6.1%}            {r['relerr_median']:.1e}")
    json.dump(rs, open(os.path.join(os.path.dirname(__file__), "..",
                                    "results", "s01b_regimes.json"), "w"), indent=1)


if __name__ == "__main__":
    rows = run()
    cb = np.array([r["cond_best"] for r in rows])
    cm = np.array([r["cond_median"] for r in rows])
    print("S1  Cancellation in the weighted ISD merge criterion")
    print("    cond = sum|terms| / |result|;  usable digits ~ 16 - log10(cond)")
    print(f"    selected (cheapest) pair : median {np.median(cb):.3e}   p90 {np.percentile(cb,90):.3e}   max {cb.max():.3e}")
    print(f"    typical pair             : median {np.median(cm):.3e}")
    print(f"    digits left on the selected pair: median {16-np.log10(np.median(cb)):.1f}, worst {16-np.log10(cb.max()):.1f}")
    fi, fr = decision_instability()
    print(f"    argmin flips under a 1e-13 relative jitter: weighted ISD {fi:.1%}, Runnalls {fr:.1%}")
    ex = exact_check()
    if ex:
        agree = np.mean([e["argmin_agrees"] for e in ex])
        rel = np.array([e["rel_err_best"] for e in ex])
        print(f"    vs 60-digit exact: argmin agrees {agree:.0%}; rel err on the winning cost median {np.median(rel):.2e} max {rel.max():.2e}")
    print_regime()
    json.dump({"rows": rows, "flip_isd": fi, "flip_runnalls": fr, "exact": ex},
              open(os.path.join(os.path.dirname(__file__), "..", "results", "s01_conditioning.json"), "w"),
              indent=1)


