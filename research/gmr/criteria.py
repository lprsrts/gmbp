"""Reduction criteria: unweighted ISD, region/relevance-weighted ISD, Runnalls.

The weighted ISD closed form is the one the concept note relies on:

    int w(x) N(x;m1,S1) N(x;m2,S2) dx  =  z12 * N(mu*; m, Sigma* + S)

with (mu*, Sigma*, z12) the Gaussian product of the two components and
w(x) = N(x;m,S). A Gaussian-mixture weight is a sum of such terms.
"""
import numpy as np
from .gm import GM, gauss_product, log_norm_pdf_at, moment_merge, residual_of_merge


def triple_overlap(m, S, mu1, S1, mu2, S2):
    """int N(x;m,S) N(x;mu1,S1) N(x;mu2,S2) dx, returned as a log value."""
    mus, Ss, logz = gauss_product(mu1, S1, mu2, S2)
    return logz + log_norm_pdf_at(mus, m, Ss + S)


def weighted_sq_norm(r: GM, lam: GM = None, return_cond=False):
    """int lam(x) r(x)^2 dx in closed form, r may carry signed weights.

    Also returns the cancellation conditioning number
    sum|terms| / |sum terms|, which is what decides whether the criterion
    can still rank merges once the residual gets small.
    """
    n = r.n
    if r.d == 1 and not return_cond:
        from .fast1d import weighted_sq_norm_1d
        lm = None if lam is None else (lam.w, lam.mu[:, 0], lam.S[:, 0, 0])
        return weighted_sq_norm_1d(r.w, r.mu[:, 0], r.S[:, 0, 0], lm)
    terms = []
    if lam is None:
        for i in range(n):
            for j in range(n):
                lg = log_norm_pdf_at(r.mu[i], r.mu[j], r.S[i] + r.S[j])
                terms.append(r.w[i] * r.w[j] * np.exp(lg))
    else:
        for k in range(lam.n):
            for i in range(n):
                for j in range(n):
                    lg = triple_overlap(lam.mu[k], lam.S[k],
                                        r.mu[i], r.S[i], r.mu[j], r.S[j])
                    terms.append(lam.w[k] * r.w[i] * r.w[j] * np.exp(lg))
    terms = np.asarray(terms)
    val = float(np.sum(terms))
    if return_cond:
        denom = abs(val) if abs(val) > 0 else np.finfo(float).tiny
        return val, float(np.sum(np.abs(terms)) / denom)
    return val


def isd(p: GM, q: GM, lam: GM = None):
    """int lam (p-q)^2 for two mixtures, via the signed concatenation."""
    r = GM(np.concatenate([p.w, -q.w]),
           np.vstack([p.mu, q.mu]),
           np.concatenate([p.S, q.S], axis=0))
    return weighted_sq_norm(r, lam)


def merge_cost_isd(gm: GM, i, j, lam: GM = None, return_cond=False):
    """Cost of merging i,j measured as the (weighted) ISD of the change."""
    return weighted_sq_norm(residual_of_merge(gm, i, j), lam, return_cond=return_cond)


def merge_cost_runnalls(gm: GM, i, j, lam=None):
    """Runnalls' one-step upper bound on the KL increase. lam ignored."""
    wi, wj = gm.w[i], gm.w[j]
    wt, _, C = moment_merge(gm.w[[i, j]], gm.mu[[i, j]], gm.S[[i, j]])
    s, ld = np.linalg.slogdet(C)
    _, ldi = np.linalg.slogdet(gm.S[i])
    _, ldj = np.linalg.slogdet(gm.S[j])
    return 0.5 * (wt * ld - wi * ldi - wj * ldj)


def numeric_weighted_sq_norm(r: GM, lam: GM = None, lo=None, hi=None, n=400001):
    """Fine-grid reference for the 1D closed form."""
    assert r.d == 1
    if lo is None:
        sd = np.sqrt(r.S[:, 0, 0])
        lo = float(np.min(r.mu[:, 0] - 12 * sd))
        hi = float(np.max(r.mu[:, 0] + 12 * sd))
        if lam is not None:
            lsd = np.sqrt(lam.S[:, 0, 0])
            lo = min(lo, float(np.min(lam.mu[:, 0] - 12 * lsd)))
            hi = max(hi, float(np.max(lam.mu[:, 0] + 12 * lsd)))
    x = np.linspace(lo, hi, n)
    f = r.pdf(x)
    g = f * f
    if lam is not None:
        g = g * lam.pdf(x)
    return float(np.trapezoid(g, x))
