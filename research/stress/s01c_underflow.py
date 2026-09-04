"""S1c. The underflow cliff.

The whole point of a downstream-aware weight is to concentrate relevance
where the *downstream* message is large, which by construction may be far
from where the current message has mass. Push that separation and every
weighted merge cost underflows to exactly zero, at which point argmin is
whichever pair numpy happens to enumerate first.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gmr import fast1d as F


def cliff(seed=0, trials=40):
    rng = np.random.default_rng(seed)
    rows = []
    for dist in (8, 12, 16, 20, 24, 30, 40, 60, 90, 130):
        for width in (1.0, 0.25):
            zero_frac, agree, ties = [], [], []
            for _ in range(trials):
                n = 10
                w = rng.dirichlet(np.ones(n))
                m = rng.normal(0, 1.5, n)
                s = rng.uniform(0.1, 1.0, n)
                lam = (np.array([1.0]), np.array([m.mean() + dist]),
                       np.array([width]))
                cost, iu, ju = F.all_pair_costs_isd(w, m, s, lam)
                cost = np.nan_to_num(cost, nan=0.0)
                zero_frac.append(float(np.mean(cost == 0.0)))
                ties.append(int(np.sum(cost == cost.min())))
                # what an infinite-precision weighted criterion would pick:
                # rank by the same criterion computed in log space
                lg = _log_cost(w, m, s, lam)
                agree.append(int(np.argmin(cost) == np.argmin(lg)))
            rows.append({"dist": dist, "width": width,
                         "frac_costs_exactly_zero": float(np.mean(zero_frac)),
                         "mean_ties_at_min": float(np.mean(ties)),
                         "argmin_matches_logspace": float(np.mean(agree))})
    return rows


def _log_cost(w, m, s, lam):
    """Same criterion with the lambda factor pulled out in log space.

    int lam (p-q)^2 with a single Gaussian lam differs from the unweighted
    quantity by a factor that never underflows if it is kept as a log, so
    this ranks the pairs the way exact arithmetic would.
    """
    n = len(w)
    iu, ju = np.triu_indices(n, 1)
    gw, gmu, gs = lam
    out = np.empty(len(iu))
    for t, (i, j) in enumerate(zip(iu, ju)):
        wM, mM, sM = F.merged_moments(w[i], m[i], s[i], w[j], m[j], s[j])
        comps = [(w[i], m[i], s[i]), (w[j], m[j], s[j]), (-wM, mM, sM)]
        logs, sgns = [], []
        for (a, b, c) in comps:
            for (a2, b2, c2) in comps:
                ssum = c + c2
                lz = F.logN(b, b2, ssum)
                sst = c * c2 / ssum
                mst = (b * c2 + b2 * c) / ssum
                lz = lz + F.logN(mst, gmu[0], gs[0] + sst) + np.log(gw[0])
                logs.append(lz + np.log(abs(a * a2)))
                sgns.append(np.sign(a * a2))
        logs = np.array(logs); sgns = np.array(sgns)
        mx = logs.max()
        out[t] = mx + np.log(max(np.sum(sgns * np.exp(logs - mx)), 1e-300))
    return out


if __name__ == "__main__":
    rows = cliff()
    print("S1c Underflow of the relevance-weighted merge criterion")
    print("    lam offset (sd of the message ~1.5)  width   costs exactly 0   ties at min   argmin matches exact ranking")
    for r in rows:
        print(f"    {r['dist']:>10}                        {r['width']:5.2f}   "
              f"{r['frac_costs_exactly_zero']:12.1%}   {r['mean_ties_at_min']:9.1f}   "
              f"{r['argmin_matches_logspace']:22.0%}")
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "..",
                                      "results", "s01c_underflow.json"), "w"), indent=1)
