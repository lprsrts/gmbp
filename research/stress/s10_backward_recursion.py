"""S10. The backward relevance recursion, Eq. (39)-(40).

    lambda_k(x) = prod_{l>k} r_l(x)^2,   lambda_k = r_{k+1}^2 lambda_{k+1}

Each factor in that product is a squared probability density. Multiplying
K of them concentrates the result: a product of K Gaussians of variance v
has variance v/K, and the component count multiplies as well. So the
relevance the recursion produces gets narrower the further back it is
propagated, and past some depth the criterion is looking at a window
narrower than the message it is supposed to be reducing.

Measured on a chain: the width of lambda_k against depth, the fraction of
the message that lies inside it, and the component count before capping.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gmr.build import build_chain
from gmr.gm import GM, gm_square, gm_product
from gmr.graph import GMBP, unit
from gmr.relevance import cap, normalise_lambda
from gmr.reduce import reduce_runnalls
from gmr import fast1d as F


def eff_width(gm: GM):
    w = np.abs(gm.w); w = w / w.sum()
    mu = float(np.sum(w * gm.mu[:, 0]))
    var = float(np.sum(w * (gm.S[:, 0, 0] + (gm.mu[:, 0] - mu) ** 2)) - 0.0)
    return mu, np.sqrt(max(var - 0.0, 0.0))


def covered_mass(msg: GM, lam: GM, n=4001):
    """Fraction of the message's mass inside the region where lambda is
    above 1% of its peak. Outside that region the weighted criterion is
    effectively blind."""
    lo = min(msg.mu[:, 0].min(), lam.mu[:, 0].min()) - 12
    hi = max(msg.mu[:, 0].max(), lam.mu[:, 0].max()) + 12
    x = np.linspace(lo, hi, n)
    p = msg.pdf(x); p = p / np.trapezoid(p, x)
    l = lam.pdf(x)
    if l.max() <= 0:
        return 0.0
    m = l >= 0.01 * l.max()
    return float(np.trapezoid(p * m, x))


def run(seeds=(0, 1, 2, 3, 4), K=8, T=None, lam_budget=6, ref_budget=40):
    T = T or K + 2
    rows = []
    for sd in seeds:
        g = build_chain(seed=sd, K=K, obs_at=1.0)
        bp = GMBP(g, budget=ref_budget, reducer=reduce_runnalls)
        for _ in range(T):
            bp.sweep()
        # backward recursion from the far end of the chain toward x0
        lam = GM([1.0], [[0.0]], [[[1e8]]])
        uncapped = 1
        for depth, k in enumerate(range(K, 0, -1)):
            r = bp.msg[(f"t{k-1}", f"x{k}")]
            rsq = gm_square(unit(r))
            uncapped *= rsq.n
            lam = gm_product(lam, rsq) if lam.n > 1 or lam.S[0, 0, 0] < 1e7 else rsq
            pre = lam.n
            lam = normalise_lambda(cap(lam, lam_budget), "peak")
            msg = bp.msg[(f"t{k-1}", f"x{k-1}")] if k >= 1 else None
            tgt = bp.msg.get((f"x{k-1}", f"t{k-1}"))
            mu, sd_l = eff_width(lam)
            rows.append({"seed": sd, "depth": depth + 1,
                         "lam_sd": float(sd_l),
                         "lam_n_before_cap": int(pre),
                         "lam_n_uncapped": float(uncapped),
                         "covered": covered_mass(tgt, lam) if tgt is not None else float("nan")})
    return rows


if __name__ == "__main__":
    rows = run()
    print("S10 Backward relevance recursion on a chain, K = 8")
    print("    lambda_k = prod_{l>k} r_l^2, capped to 6 components at every step\n")
    print(f"    {'depth':>6}{'lambda width (sd)':>20}{'components before cap':>24}"
          f"{'uncapped count':>18}{'message mass lambda sees':>26}")
    depths = sorted(set(r["depth"] for r in rows))
    out = []
    for d in depths:
        sub = [r for r in rows if r["depth"] == d]
        sd_ = np.median([r["lam_sd"] for r in sub])
        pre = np.median([r["lam_n_before_cap"] for r in sub])
        unc = np.median([r["lam_n_uncapped"] for r in sub])
        cov = np.nanmedian([r["covered"] for r in sub])
        out.append({"depth": d, "lam_sd": float(sd_), "pre": float(pre),
                    "uncapped": float(unc), "covered": float(cov)})
        print(f"    {d:>6}{sd_:>20.4f}{pre:>24.0f}{unc:>18.3g}{cov:>25.1%}")
    print("\n    'message mass lambda sees' is the fraction of the message lying where")
    print("    lambda is above 1% of its peak. Outside it the weighted criterion is")
    print("    ranking merges by numbers that are near zero, and S1c shows what")
    print("    happens to the ranking once they underflow.")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..",
                                     "results", "s10_backward.json"), "w"), indent=1)
