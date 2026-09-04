"""Algorithm 1 of the concept note, plus the baselines it is measured against."""
import numpy as np
from .gm import GM
from .graph import GMBP, unit
from .reduce import reduce_runnalls, reduce_isd, reduce_weighted_isd, weight_prune
from .relevance import RelevanceField, reactivation_scores, UNIFORM


BASELINES = {
    "runnalls": reduce_runnalls,
    "isd": reduce_isd,
    "prune": lambda gm, b, lam, rec: weight_prune(gm, b),
}


def run_reference(g, T, hook=None):
    bp = GMBP(g, budget=None, reducer=None)
    snaps = []
    for t in range(T):
        if hook:
            hook(t, g)
        bp.sweep()
        snaps.append(_snapshot(bp))
    return bp, snaps


def run_local(g, T, budget, method="runnalls", hook=None):
    bp = GMBP(g, budget=budget, reducer=BASELINES[method])
    snaps = []
    for t in range(T):
        if hook:
            hook(t, g)
        bp.sweep()
        snaps.append(_snapshot(bp))
    return bp, snaps


def run_adaptive(g, T, budget, alpha=0.5, tau=None, lam_budget=6,
                 norm="peak", backward=True, use_reserve=True,
                 reserve_cap=200, trace=False, hook=None):
    """reduce -> propagate -> estimate relevance -> update -> reduce again."""
    bp = GMBP(g, budget=budget, reducer=reduce_weighted_isd)
    field = RelevanceField(bp, alpha=alpha, lam_budget=lam_budget,
                           norm=norm, backward=backward)
    reserve = {} if use_reserve else None
    snaps, diag = [], []
    for t in range(T):
        if hook:
            hook(t, g)
        bp.sweep(lam_lookup=field.lookup, reserve=reserve)
        field.update(bp.msg)
        n_react = 0
        if use_reserve and tau is not None:
            n_react = _reactivate(bp, field, reserve, tau, budget)
        for k in list(reserve or {}):
            if len(reserve[k]) > reserve_cap:
                reserve[k] = reserve[k][-reserve_cap:]
        snaps.append(_snapshot(bp))
        if trace:
            diag.append({
                "t": t,
                "reserve": sum(len(v) for v in (reserve or {}).values()),
                "reactivated": n_react,
                "lam_max": max(field.sizes().values()) if field.lam else 0,
            })
    return bp, snaps, (diag if trace else None), field, reserve


def _reactivate(bp, field, reserve, tau, budget):
    n = 0
    for key, items in list(reserve.items()):
        if not items or not isinstance(bp.msg.get(key), GM):
            continue
        lam = field.lookup(key)
        comps = [(w, np.atleast_1d(mu), np.atleast_2d(S))
                 for (w, mu, S, _c) in items]
        rho = reactivation_scores(comps, lam)
        hits = np.where(rho > tau)[0]
        if len(hits) == 0:
            continue
        cur = bp.msg[key]
        w = np.concatenate([cur.w, [items[i][0] for i in hits]])
        mu = np.vstack([cur.mu] + [np.atleast_1d(items[i][1])[None, :]
                                   for i in hits])
        S = np.concatenate([cur.S] + [np.atleast_2d(items[i][2])[None, :, :]
                                      for i in hits], axis=0)
        merged = GM(w, mu, S)
        bp.msg[key] = unit(reduce_weighted_isd(merged, budget, lam, None))
        n += len(hits)
        reserve[key] = [it for i, it in enumerate(items) if i not in set(hits)]
    return n


def _snapshot(bp):
    out = {}
    for v in bp.g.sensors:
        out[v] = bp.belief(v)
    for c in bp.g.classes:
        out[c] = np.asarray(bp.belief(c))
    return out
