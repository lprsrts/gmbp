"""Reduction policies operating to a component budget."""
import numpy as np
from .gm import GM, merge_pair
from .criteria import merge_cost_isd, merge_cost_runnalls


def weight_prune(gm: GM, budget: int) -> GM:
    if gm.n <= budget:
        return gm.copy()
    idx = np.argsort(-gm.w)[:budget]
    out = gm.subset(np.sort(idx))
    return out.scaled(gm.mass() / out.mass()) if out.mass() != 0 else out


def _fast_greedy(gm, budget, lam, criterion, record):
    from .fast1d import greedy_1d
    lm = None if lam is None else (lam.w, lam.mu[:, 0], lam.S[:, 0, 0])
    w, m, s = greedy_1d(gm.w, gm.mu[:, 0], gm.S[:, 0, 0], budget, lm,
                        criterion=criterion, record=record)
    return GM(w, m.reshape(-1, 1), s.reshape(-1, 1, 1))


def greedy_merge(gm: GM, budget: int, cost, lam: GM = None, record=None):
    """Repeatedly merge the cheapest pair until the budget is met.

    `record` (optional list) collects the residual of every merge performed,
    which is what the reserve set is built from.
    """
    cur = gm.copy()
    while cur.n > budget:
        best, bi, bj = np.inf, -1, -1
        for i in range(cur.n):
            for j in range(i + 1, cur.n):
                c = cost(cur, i, j, lam)
                if c < best:
                    best, bi, bj = c, i, j
        if bi < 0:
            break
        if record is not None:
            for k in (bi, bj):
                record.append((float(cur.w[k]), cur.mu[k].copy(),
                               cur.S[k].copy(), float(best)))
        cur = merge_pair(cur, bi, bj)
    return cur


def reduce_runnalls(gm, budget, lam=None, record=None):
    if gm.d == 1:
        return _fast_greedy(gm, budget, None, "runnalls", record)
    return greedy_merge(gm, budget, merge_cost_runnalls, None, record)


def reduce_isd(gm, budget, lam=None, record=None):
    if gm.d == 1:
        return _fast_greedy(gm, budget, None, "isd", record)
    return greedy_merge(gm, budget, merge_cost_isd, None, record)


def reduce_weighted_isd(gm, budget, lam=None, record=None):
    if gm.d == 1:
        return _fast_greedy(gm, budget, lam, "isd", record)
    return greedy_merge(gm, budget, merge_cost_isd, lam, record)
