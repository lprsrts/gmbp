"""S8. Is mixture reduction actually irreversible here?

The reserve set in the note exists because of one claim: a hypothesis
deleted at iteration t cannot be reconstructed later.  That is true of a
representation which only carries state forward in the message.  It is not
automatically true of a factor graph, where a factor->variable message is a
function of the factor potential and the *current* incoming message, so a
deleted mode can simply be regenerated on the next sweep.

Part A deletes a component surgically, with no reduction anywhere else, and
watches whether the belief comes back.
Part B repairs the two obvious defects in the reactivation rule -- a stale
unnormalised weight, and reinserting a hypothesis the mixture already
carries -- and asks whether the reserve then earns its cost.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gmr.build import build_graph
from gmr.gm import GM
from gmr.graph import GMBP, unit
from gmr.relevance import RelevanceField, reactivation_scores
from gmr.reduce import reduce_weighted_isd
from gmr.adaptive import run_reference
from gmr.metrics import tv


def drop_component(gm: GM, idx) -> GM:
    keep = np.setdiff1d(np.arange(gm.n), [idx])
    return unit(gm.subset(keep))


def pick_target(bp, sensors, max_w=0.2):
    """The weak hypothesis whose deletion actually moves a belief.

    Picking by distance alone lands on components carrying 1e-4 of the mass,
    whose removal is unmeasurable. This scans candidates with weight below
    max_w and takes the one with the largest immediate effect.
    """
    base = {v: bp.belief(v) for v in sensors}
    best = None
    for v in sensors:
        for fn in bp.g.nbrs[v]:
            m = bp.msg.get((fn, v))
            if not isinstance(m, GM) or m.n < 4:
                continue
            for k in range(m.n):
                if m.w[k] >= max_w:
                    continue
                saved = bp.msg[(fn, v)]
                bp.msg[(fn, v)] = drop_component(saved, k)
                d = tv(base[v], bp.belief(v))
                bp.msg[(fn, v)] = saved
                if best is None or d > best[0]:
                    best = (d, (fn, v), k, float(m.w[k]))
    return best


def part_a(seeds=(1, 2, 3, 4, 5), T=10, t0=3, regime="medium"):
    """Delete the most distant low-weight component of one message at sweep
    t0. Everything else runs unreduced."""
    rows = []
    for sd in seeds:
        g = build_graph(seed=sd, obs_strength=regime)
        _, ref = run_reference(g, T)
        g2 = build_graph(seed=sd, obs_strength=regime)
        bp = GMBP(g2, budget=None, reducer=None)
        traj = []
        target = None
        for t in range(T):
            bp.sweep()
            if t == t0:
                best = pick_target(bp, g2.sensors)
                if best is None:
                    break
                imm, key, k, wk = best
                target = {"edge": f"{key[0]}->{key[1]}", "weight": wk,
                          "immediate_tv": float(imm)}
                bp.msg[key] = drop_component(bp.msg[key], k)
            traj.append(np.mean([tv(ref[t][v], bp.belief(v))
                                 for v in g2.sensors]))
        rows.append({"seed": sd, "target": target,
                     "tv_after_delete": traj[t0:],
                     "recovery": float(traj[-1] / max(traj[t0], 1e-300))})
    return rows


def run_variant(g, T, budget, tau, repair, alpha=0.5, reserve_cap=200):
    bp = GMBP(g, budget=budget, reducer=reduce_weighted_isd)
    field = RelevanceField(bp, alpha=alpha, lam_budget=6)
    reserve, snaps, n_react = {}, [], 0
    for t in range(T):
        bp.sweep(lam_lookup=field.lookup, reserve=reserve)
        field.update(bp.msg)
        if tau is not None:
            n_react += _react(bp, field, reserve, tau, budget, repair)
        for k in list(reserve):
            if len(reserve[k]) > reserve_cap:
                reserve[k] = reserve[k][-reserve_cap:]
        snaps.append({v: bp.belief(v) for v in g.sensors})
    return snaps, n_react


def _react(bp, field, reserve, tau, budget, repair):
    """repair: False = Eq. (57)-(58) as written; True = weight rescaled and
    duplicates suppressed; "relative" = additionally replaces the absolute
    threshold with a scale-free one."""
    n = 0
    for key, items in list(reserve.items()):
        cur = bp.msg.get(key)
        if not items or not isinstance(cur, GM):
            continue
        lam = field.lookup(key)
        comps = [(w, np.atleast_1d(m), np.atleast_2d(s)) for (w, m, s, _c) in items]
        rho = reactivation_scores(comps, lam)
        if repair == "relative":
            # rho carries the scale of lambda, which is arbitrary. Compare
            # against the active components instead: bring a hypothesis back
            # only when it is more relevant than the least relevant thing
            # currently being carried.
            act = reactivation_scores(
                [(float(cur.w[i]), cur.mu[i], cur.S[i]) for i in range(cur.n)],
                lam)
            hits = list(np.where(rho > tau * float(np.min(act)))[0])
        else:
            hits = list(np.where(rho > tau)[0])
        if repair:
            # (i) drop anything the active mixture already represents
            kept = []
            for i in hits:
                mu = float(np.atleast_1d(items[i][1])[0])
                sd = np.sqrt(float(np.atleast_2d(items[i][2])[0, 0]))
                if np.min(np.abs(cur.mu[:, 0] - mu)) > 1.0 * sd:
                    kept.append(i)
            hits = kept
        if not hits:
            continue
        add_w = np.array([items[i][0] for i in hits], dtype=float)
        if repair:
            # (ii) the stored weight is from an older, differently scaled
            # mixture; bring it back in at the active mixture's scale
            add_w = add_w / max(add_w.sum(), 1e-300) * cur.mass() * 0.1
        w = np.concatenate([cur.w, add_w])
        mu = np.vstack([cur.mu] + [np.atleast_1d(items[i][1])[None, :] for i in hits])
        S = np.concatenate([cur.S] + [np.atleast_2d(items[i][2])[None, :, :]
                                      for i in hits], axis=0)
        bp.msg[key] = unit(reduce_weighted_isd(GM(w, mu, S), budget, lam, None))
        n += len(hits)
        reserve[key] = [it for i, it in enumerate(items) if i not in set(hits)]
    return n


def part_b(seeds=(1, 2, 3, 4), T=8, budget=6, regime="medium"):
    rows = []
    settings = [("no reserve", None, False)]
    for tau in (1e-2, 1e-1, 1.0):
        settings.append((f"as written, tau={tau:g}", tau, False))
        settings.append((f"repaired,   tau={tau:g}", tau, True))
    for k in (1.0, 3.0, 10.0):
        settings.append((f"relative,   k={k:g}", k, "relative"))
    for label, tau, repair in settings:
        gaps, reacts = [], []
        for sd in seeds:
            g = build_graph(seed=sd, obs_strength=regime)
            _, ref = run_reference(g, T)
            g2 = build_graph(seed=sd, obs_strength=regime)
            snaps, nr = run_variant(g2, T, budget, tau, repair)
            gaps.append(np.mean([np.mean([tv(ref[t][v], snaps[t][v])
                                          for v in g2.sensors])
                                 for t in range(T)]))
            reacts.append(nr)
        rows.append({"label": label, "gap": float(np.mean(gaps)),
                     "reactivations": float(np.mean(reacts))})
    return rows


if __name__ == "__main__":
    A = part_a()
    print("S8a Delete one distant low-weight component; no reduction anywhere else.")
    print("    Does the belief recover on later sweeps?\n")
    print(f"    {'seed':<6}{'edge':<22}{'weight':>9}   TV to reference, sweep of deletion onward")
    for r in A:
        tvs = "  ".join(f"{v:.5f}" for v in r["tv_after_delete"])
        tgt = r["target"] or {}
        print(f"    {r['seed']:<6}{tgt.get('edge','-'):<22}{tgt.get('weight',0):>9.4f}   {tvs}")
    rec = np.array([r["recovery"] for r in A])
    print(f"\n    final TV as a fraction of the TV right after deletion: "
          f"median {np.median(rec):.3f}")
    B = part_b()
    print("\nS8b Does the reserve earn its cost, as written and repaired?\n")
    print(f"    {'variant':<26}{'mean TV gap':>13}{'x no-reserve':>14}{'reactivations':>15}")
    base = [r for r in B if r["label"] == "no reserve"][0]["gap"]
    for r in B:
        print(f"    {r['label']:<26}{r['gap']:>13.5f}{r['gap']/base:>13.2f}x{r['reactivations']:>15.0f}")
    json.dump({"part_a": A, "part_b": B},
              open(os.path.join(os.path.dirname(__file__), "..", "results",
                                "s08_irreversibility.json"), "w"), indent=1)
