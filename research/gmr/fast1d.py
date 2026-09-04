"""Vectorised 1D core. The general-dimension code in gm.py / criteria.py is
kept as the correctness oracle; these routines must agree with it.

Arrays are (n,) of means m, variances s, weights w. `lam` is a triple
(gw, gm, gs) or None for the uniform weight.
"""
import numpy as np

LOG2PI = np.log(2.0 * np.pi)

COUNTERS = {"pair_costs": 0, "merges": 0}


def reset_counters():
    COUNTERS["pair_costs"] = 0
    COUNTERS["merges"] = 0


def logN(x, m, s):
    return -0.5 * ((x - m) ** 2 / s + np.log(s) + LOG2PI)


def N(x, m, s):
    return np.exp(logN(x, m, s))


def pair_product(m1, s1, m2, s2):
    ssum = s1 + s2
    z = N(m1, m2, ssum)
    sst = s1 * s2 / ssum
    mst = (m1 * s2 + m2 * s1) / ssum
    return mst, sst, z


def overlap_matrix(m, s, lam=None):
    """A[p,q] = int lambda(x) N_p N_q dx."""
    mst, sst, z = pair_product(m[:, None], s[:, None], m[None, :], s[None, :])
    if lam is None:
        return z
    gw, gm_, gs = lam
    A = np.zeros_like(z)
    for k in range(len(gw)):
        A += gw[k] * z * N(mst, gm_[k], sst + gs[k])
    return A


def triple_vec(mp, sp, mq, sq, lam):
    mst, sst, z = pair_product(mp, sp, mq, sq)
    if lam is None:
        return z
    gw, gm_, gs = lam
    out = np.zeros(np.broadcast(mst, sst).shape)
    for k in range(len(gw)):
        out = out + gw[k] * z * N(mst, gm_[k], sst + gs[k])
    return out


def merged_moments(wi, mi, si, wj, mj, sj):
    wM = wi + wj
    mM = (wi * mi + wj * mj) / wM
    sM = (wi * (si + (mi - mM) ** 2) + wj * (sj + (mj - mM) ** 2)) / wM
    return wM, mM, sM


def all_pair_costs_isd(w, m, s, lam=None, A=None, return_terms=False):
    n = len(w)
    iu, ju = np.triu_indices(n, 1)
    if A is None:
        A = overlap_matrix(m, s, lam)
    wi, mi, si = w[iu], m[iu], s[iu]
    wj, mj, sj = w[ju], m[ju], s[ju]
    wM, mM, sM = merged_moments(wi, mi, si, wj, mj, sj)
    T_ii, T_jj, T_ij = A[iu, iu], A[ju, ju], A[iu, ju]
    T_iM = triple_vec(mi, si, mM, sM, lam)
    T_jM = triple_vec(mj, sj, mM, sM, lam)
    T_MM = triple_vec(mM, sM, mM, sM, lam)
    t = (wi * wi * T_ii, wj * wj * T_jj, wM * wM * T_MM,
         2 * wi * wj * T_ij, -2 * wi * wM * T_iM, -2 * wj * wM * T_jM)
    cost = sum(t)
    COUNTERS["pair_costs"] += len(iu) * (1 if lam is None else len(lam[0]))
    if return_terms:
        return cost, iu, ju, sum(np.abs(x) for x in t)
    return cost, iu, ju


def all_pair_costs_runnalls(w, m, s, lam=None):
    n = len(w)
    iu, ju = np.triu_indices(n, 1)
    wi, mi, si = w[iu], m[iu], s[iu]
    wj, mj, sj = w[ju], m[ju], s[ju]
    wM, mM, sM = merged_moments(wi, mi, si, wj, mj, sj)
    COUNTERS["pair_costs"] += len(iu)
    return 0.5 * (wM * np.log(sM) - wi * np.log(si) - wj * np.log(sj)), iu, ju


def greedy_1d(w, m, s, budget, lam=None, criterion="isd", record=None):
    w, m, s = np.array(w, float), np.array(m, float), np.array(s, float)
    while len(w) > budget:
        if criterion == "isd":
            cost, iu, ju = all_pair_costs_isd(w, m, s, lam)
        elif criterion == "runnalls":
            cost, iu, ju = all_pair_costs_runnalls(w, m, s)
        else:
            raise ValueError(criterion)
        k = int(np.argmin(cost))
        i, j = int(iu[k]), int(ju[k])
        if record is not None:
            record.append((float(w[i]), float(m[i]), float(s[i]), float(cost[k])))
            record.append((float(w[j]), float(m[j]), float(s[j]), float(cost[k])))
        COUNTERS["merges"] += 1
        wM, mM, sM = merged_moments(w[i], m[i], s[i], w[j], m[j], s[j])
        keep = np.setdiff1d(np.arange(len(w)), [i, j])
        w = np.append(w[keep], wM)
        m = np.append(m[keep], mM)
        s = np.append(s[keep], sM)
    return w, m, s


def prune_1d(w, m, s, budget):
    w, m, s = np.asarray(w, float), np.asarray(m, float), np.asarray(s, float)
    if len(w) <= budget:
        return w.copy(), m.copy(), s.copy()
    idx = np.sort(np.argsort(-w)[:budget])
    w2 = w[idx]
    w2 = w2 * (w.sum() / w2.sum()) if w2.sum() > 0 else w2
    return w2, m[idx], s[idx]


def gm_product_1d(w1, m1, s1, w2, m2, s2):
    mst, sst, z = pair_product(m1[:, None], s1[:, None], m2[None, :], s2[None, :])
    return (w1[:, None] * w2[None, :] * z).ravel(), mst.ravel(), sst.ravel()


def gm_square_1d(w, m, s):
    return gm_product_1d(w, m, s, w, m, s)


def pdf_1d(w, m, s, x):
    x = np.asarray(x, float)
    return (w[None, :] * N(x[:, None], m[None, :], s[None, :])).sum(1)


def weighted_sq_norm_1d(w, m, s, lam=None):
    return float(np.asarray(w) @ overlap_matrix(np.asarray(m), np.asarray(s), lam) @ np.asarray(w))
