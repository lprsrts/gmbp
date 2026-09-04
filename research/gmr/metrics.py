"""Comparison metrics between a reference belief and a reduced one."""
import numpy as np
from .gm import GM


def grid_for(gms, pad=10.0, n=4001):
    lo, hi = np.inf, -np.inf
    for gm in gms:
        sd = np.sqrt(gm.S[:, 0, 0])
        lo = min(lo, float(np.min(gm.mu[:, 0] - pad * sd)))
        hi = max(hi, float(np.max(gm.mu[:, 0] + pad * sd)))
    return np.linspace(lo, hi, n)


def tv(p: GM, q: GM, x=None):
    x = grid_for([p, q]) if x is None else x
    a, b = p.pdf(x), q.pdf(x)
    a = a / np.trapezoid(a, x)
    b = b / np.trapezoid(b, x)
    return 0.5 * float(np.trapezoid(np.abs(a - b), x))


def log_ratio_error(ref: GM, red: GM, x=None, floor=1e-9):
    """|log(ref/red)| masked where the reference has essentially no mass."""
    x = grid_for([ref, red]) if x is None else x
    a, b = ref.pdf(x), red.pdf(x)
    a = a / np.trapezoid(a, x)
    b = b / np.trapezoid(b, x)
    mask = a > floor * a.max()
    e = np.zeros_like(a)
    e[mask] = np.abs(np.log(a[mask]) - np.log(np.maximum(b[mask], 1e-300)))
    return x, e, mask


def localisation(x, e, frac=0.1):
    """Share of total error carried by the worst `frac` of the axis."""
    order = np.argsort(-e)
    k = max(1, int(round(frac * len(x))))
    tot = e.sum()
    return float(e[order[:k]].sum() / tot) if tot > 0 else 0.0


def class_tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())
