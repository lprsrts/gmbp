"""S0. Verification of the implementation, before anything is stressed.

A stress test is only worth reading if the failures belong to the method
rather than the code. Four independent checks:

  1. the region-weighted ISD closed form against fine-grid quadrature
     (the note reports 5e-16 and 4e-12 for this);
  2. the vectorised 1D core against the dimension-general implementation,
     which shares no code path with it;
  3. Eq. (18), int (pr - qr)^2 = int r^2 (p-q)^2, numerically;
  4. the BP engine against an exact posterior on a tree, computed by
     enumeration and quadrature.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gmr.gm import GM
from gmr.criteria import weighted_sq_norm, numeric_weighted_sq_norm, merge_cost_isd
from gmr import fast1d as F
from gmr.graph import HybridGraph, ClassSensorFactor, GMBP
from gmr.metrics import tv


def check_closed_form(trials=200, seed=0):
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(trials):
        n = int(rng.integers(2, 6))
        r = GM(rng.normal(0, 1, n) * rng.choice([-1, 1], n),
               rng.normal(0, 2.5, (n, 1)),
               rng.uniform(0.15, 1.5, (n, 1, 1)))
        K = int(rng.integers(1, 4))
        lam = GM(rng.random(K), rng.normal(0, 2.5, (K, 1)),
                 rng.uniform(0.2, 2.5, (K, 1, 1)))
        a = weighted_sq_norm(r, lam)
        b = numeric_weighted_sq_norm(r, lam, n=200001)
        if abs(b) > 1e-12:
            errs.append(abs(a - b) / abs(b))
    return np.array(errs)


def check_fast_vs_general(trials=60, seed=1):
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(trials):
        n = int(rng.integers(4, 9))
        w = rng.dirichlet(np.ones(n)); m = rng.normal(0, 3, n)
        s = rng.uniform(0.1, 1.5, n)
        K = int(rng.integers(1, 4))
        gw, gm_, gs = rng.random(K), rng.normal(0, 3, K), rng.uniform(0.2, 3, K)
        p = GM(w, m.reshape(-1, 1), s.reshape(-1, 1, 1))
        lamGM = GM(gw, gm_.reshape(-1, 1), gs.reshape(-1, 1, 1))
        cost, iu, ju = F.all_pair_costs_isd(w, m, s, (gw, gm_, gs))
        ref = np.array([merge_cost_isd(p, int(a), int(b), lamGM)
                        for a, b in zip(iu, ju)])
        errs.append(np.max(np.abs(cost - ref) / np.abs(ref)))
    return np.array(errs)


def check_eq18(trials=200, seed=2):
    """int (p r - q r)^2 dx  ==  int r^2 (p-q)^2 dx"""
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(trials):
        n, rn = int(rng.integers(4, 9)), int(rng.integers(1, 4))
        w = rng.dirichlet(np.ones(n)); m = rng.normal(0, 2.5, n)
        s = rng.uniform(0.1, 1.2, n)
        rw = rng.dirichlet(np.ones(rn)); rm = rng.normal(0, 2.5, rn)
        rs = rng.uniform(0.15, 1.5, rn)
        i, j = rng.choice(n, 2, replace=False)
        wM, mM, sM = F.merged_moments(w[i], m[i], s[i], w[j], m[j], s[j])
        keep = np.setdiff1d(np.arange(n), [i, j])
        qw = np.append(w[keep], wM); qm = np.append(m[keep], mM)
        qs = np.append(s[keep], sM)
        lam = F.gm_square_1d(rw, rm, rs)
        lhs_w = np.concatenate([F.gm_product_1d(w, m, s, rw, rm, rs)[0],
                                -F.gm_product_1d(qw, qm, qs, rw, rm, rs)[0]])
        lhs_m = np.concatenate([F.gm_product_1d(w, m, s, rw, rm, rs)[1],
                                F.gm_product_1d(qw, qm, qs, rw, rm, rs)[1]])
        lhs_s = np.concatenate([F.gm_product_1d(w, m, s, rw, rm, rs)[2],
                                F.gm_product_1d(qw, qm, qs, rw, rm, rs)[2]])
        lhs = F.weighted_sq_norm_1d(lhs_w, lhs_m, lhs_s, None)
        rhs = F.weighted_sq_norm_1d(np.array([w[i], w[j], -wM]),
                                    np.array([m[i], m[j], mM]),
                                    np.array([s[i], s[j], sM]), lam)
        if abs(rhs) > 1e-18:
            errs.append(abs(lhs - rhs) / abs(rhs))
    return np.array(errs)


def check_bp_on_tree(seed=3):
    """C -- S1, C -- S2, S1 observed. Exact answer by enumeration."""
    rng = np.random.default_rng(seed)
    S = 3
    def bank():
        return [GM(rng.dirichlet(np.ones(2)), rng.normal(0, 3, (2, 1)),
                   rng.uniform(0.2, 1.0, (2, 1, 1))) for _ in range(S)]
    b1, b2 = bank(), bank()
    g = HybridGraph(); g.add_class("C", S)
    g.add_sensor("S1"); g.add_sensor("S2")
    g.add_factor(ClassSensorFactor("f1", "C", "S1", b1))
    g.add_factor(ClassSensorFactor("f2", "C", "S2", b2))
    obs = GM([1.0], [[0.7]], [[[0.3]]])
    g.observe("S1", obs)
    bp = GMBP(g, budget=None, reducer=None)
    for _ in range(4):
        bp.sweep()

    lik = np.array([float(np.sum([obs.w[0] * c.w[k] * np.exp(
        F.logN(obs.mu[0, 0], c.mu[k, 0], obs.S[0, 0, 0] + c.S[k, 0, 0]))
        for k in range(c.n)])) for c in b1])
    post = lik / lik.sum()
    w = np.concatenate([b2[c].w * post[c] for c in range(S)])
    m = np.concatenate([b2[c].mu[:, 0] for c in range(S)])
    s = np.concatenate([b2[c].S[:, 0, 0] for c in range(S)])
    exact_S2 = GM(w, m.reshape(-1, 1), s.reshape(-1, 1, 1))
    return (float(np.max(np.abs(np.asarray(bp.belief("C")) - post))),
            tv(exact_S2, bp.belief("S2")))


if __name__ == "__main__":
    e1 = check_closed_form()
    e2 = check_fast_vs_general()
    e3 = check_eq18()
    from gmr.graph import FLAT_VAR
    tree = []
    for fv in (1e4, 1e6, 1e8, 1e10):
        FLAT_VAR[0] = fv
        tree.append((fv,) + check_bp_on_tree())
    FLAT_VAR[0] = 1e4
    dc, dS2 = tree[0][1], tree[0][2]
    print("S0  Verification")
    print(f"    weighted ISD closed form vs quadrature : median {np.median(e1):.2e}  max {e1.max():.2e}")
    print(f"    vectorised 1D core vs general oracle    : median {np.median(e2):.2e}  max {e2.max():.2e}")
    print(f"    Eq. 18  int(pr-qr)^2 = int r^2 (p-q)^2  : median {np.median(e3):.2e}  max {e3.max():.2e}")
    print("    BP on a tree vs exact posterior, as the improper-prior stand-in widens:")
    for fv, a, b in tree:
        print(f"        flat prior variance {fv:8.0e} -> class max-abs {a:.2e}, sensor TV {b:.2e}")
    json.dump({"closed_form_median": float(np.median(e1)), "closed_form_max": float(e1.max()),
               "fast_vs_general_max": float(e2.max()), "eq18_max": float(e3.max()),
               "tree_class": dc, "tree_sensor": dS2},
              open(os.path.join(os.path.dirname(__file__), "..", "results",
                                "s00_verify.json"), "w"), indent=1)
