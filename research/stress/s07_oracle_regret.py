"""S7. Against an oracle, inside the real loop.

S3 scored the weight on a synthetic one-step downstream. This does the
decisive version: at an actual reduction event in the loopy graph, take the
candidate first merges, carry each one through to the end of inference, and
measure the damage it really did to the final beliefs.

That gives a ground-truth ranking of the merges. Every criterion is then
scored against it, including the note's relevance weight computed exactly
as the method computes it -- stale, capped, damped and all.
"""
import sys, os, json, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.stats import kendalltau
from gmr.build import build_graph
from gmr.gm import GM
from gmr.graph import GMBP, unit
from gmr.relevance import RelevanceField
from gmr.reduce import reduce_weighted_isd, reduce_runnalls
from gmr.adaptive import run_reference
from gmr.metrics import tv
from gmr import fast1d as F


def apply_merge(gm: GM, i, j) -> GM:
    w, m, s = gm.w, gm.mu[:, 0], gm.S[:, 0, 0]
    wM, mM, sM = F.merged_moments(w[i], m[i], s[i], w[j], m[j], s[j])
    keep = np.setdiff1d(np.arange(len(w)), [i, j])
    return GM(np.append(w[keep], wM), np.append(m[keep], mM).reshape(-1, 1),
              np.append(s[keep], sM).reshape(-1, 1, 1))


def one_event(seed=1, T=8, budget=6, t0=2, regime="medium"):
    """The decisive merge is the last one, not the first.

    A candidate first merge is washed out by whatever completes the
    reduction, so the message is pre-reduced to budget+1 by a fixed,
    criterion-neutral prefix (Runnalls) and the candidates are the pairs
    that take it the final step to the budget. Each candidate therefore
    produces a genuinely different reduced message.
    """
    g = build_graph(seed=seed, obs_strength=regime)
    _, ref = run_reference(g, T)

    g2 = build_graph(seed=seed, obs_strength=regime)
    bp = GMBP(g2, budget=budget, reducer=reduce_weighted_isd)
    field = RelevanceField(bp, alpha=0.5, lam_budget=6)
    for t in range(t0):
        bp.sweep(lam_lookup=field.lookup)
        field.update(bp.msg)

    key, best = None, None
    for v in g2.sensors:
        for fn in g2.nbrs[v]:
            m = bp.var_to_factor(v, fn)
            if isinstance(m, GM) and (best is None or m.n > best.n):
                key, best = (v, fn), m
    if key is None or best.n <= budget:
        return None
    pre = unit(reduce_runnalls(best, budget + 1))
    lam = field.lookup(key)
    lamt = (lam.w, lam.mu[:, 0], lam.S[:, 0, 0])

    n = pre.n
    w, mm, ss = pre.w, pre.mu[:, 0], pre.S[:, 0, 0]
    iu, ju = np.triu_indices(n, 1)
    pred = {"lambda = r^2": [], "unweighted ISD": [], "Runnalls": []}
    truth = []
    for i, j in zip(iu, ju):
        i, j = int(i), int(j)
        wM, mM, sM = F.merged_moments(w[i], mm[i], ss[i], w[j], mm[j], ss[j])
        rw = np.array([w[i], w[j], -wM]); rm = np.array([mm[i], mm[j], mM])
        rs = np.array([ss[i], ss[j], sM])
        pred["lambda = r^2"].append(F.weighted_sq_norm_1d(rw, rm, rs, lamt))
        pred["unweighted ISD"].append(F.weighted_sq_norm_1d(rw, rm, rs, None))
        pred["Runnalls"].append(0.5 * (wM * np.log(sM) - w[i] * np.log(ss[i])
                                       - w[j] * np.log(ss[j])))
        bp2 = copy.deepcopy(bp)
        f2 = copy.deepcopy(field)
        f2.bp = bp2
        bp2.msg[key] = unit(apply_merge(pre, i, j))
        per_h = [float(np.mean([tv(ref[t0 - 1][v], bp2.belief(v))
                                for v in g2.sensors]))] if t0 > 0 else [np.nan]
        for t in range(t0, T):
            bp2.sweep(lam_lookup=f2.lookup)
            f2.update(bp2.msg)
            per_h.append(float(np.mean([tv(ref[t][v], bp2.belief(v))
                                        for v in g2.sensors])))
        truth.append(per_h)
    return pred, np.array(truth)      # (candidates, horizons)


def run(seeds=(1, 2, 3, 4, 5, 6), t0s=(1, 2, 3)):
    rows = []
    for sd in seeds:
        for t0 in t0s:
            r = one_event(seed=sd, t0=t0)
            if r is None:
                continue
            pred, truth = r
            for h in range(truth.shape[1]):
                col = truth[:, h]
                spread = float(np.nanmax(col) - np.nanmin(col))
                for name, p in pred.items():
                    p = np.asarray(p)
                    ok = np.isfinite(p) & np.isfinite(col)
                    if ok.sum() < 5 or spread <= 0:
                        rows.append({"seed": sd, "t0": t0, "horizon": h,
                                     "criterion": name, "kendall": np.nan,
                                     "chosen_percentile": np.nan,
                                     "excess_frac_of_spread": np.nan,
                                     "spread": spread})
                        continue
                    tau = kendalltau(p[ok], col[ok]).statistic
                    chosen = col[ok][np.argmin(p[ok])]
                    rows.append({"seed": sd, "t0": t0, "horizon": h,
                                 "criterion": name, "kendall": float(tau),
                                 "chosen_percentile": float(np.mean(col[ok] <= chosen)),
                                 "excess_frac_of_spread": float(
                                     (chosen - col[ok].min()) / spread),
                                 "spread": spread})
    return rows


if __name__ == "__main__":
    rows = run()
    print("S7  Ranking real merges by the damage they do downstream, in the graph")
    print("    The message is pre-reduced to 7 components by Runnalls; the 21 pairs")
    print("    that take it to the budget of 6 are the candidates. Each is carried")
    print("    forward and scored against the unreduced reference at each horizon.\n")
    hs = sorted(set(r["horizon"] for r in rows))
    print(f"    {'horizon (sweeps after the merge)':<34}" +
          "".join(f"{h:>10}" for h in hs))
    sp = [np.nanmedian([r["spread"] for r in rows if r["horizon"] == h]) for h in hs]
    print(f"    {'spread of true damage':<34}" + "".join(f"{v:>10.2e}" for v in sp))
    print()
    out = {}
    for name in ("lambda = r^2", "unweighted ISD", "Runnalls"):
        line = f"    {name + ' (Kendall tau)':<34}"
        vals = []
        for h in hs:
            k = [r["kendall"] for r in rows
                 if r["horizon"] == h and r["criterion"] == name]
            v = float(np.nanmedian(k)) if np.any(np.isfinite(k)) else np.nan
            vals.append(v)
            line += f"{v:>10.3f}" if np.isfinite(v) else f"{'n/a':>10}"
        out[name] = vals
        print(line)
    print()
    for name in ("lambda = r^2", "unweighted ISD", "Runnalls"):
        line = f"    {name + ' (percentile picked)':<34}"
        for h in hs:
            k = [r["chosen_percentile"] for r in rows
                 if r["horizon"] == h and r["criterion"] == name]
            v = float(np.nanmedian(k)) if np.any(np.isfinite(k)) else np.nan
            line += f"{v:>9.0%}" + " " if np.isfinite(v) else f"{'n/a':>10}"
        print(line)
    print("\n    A spread of zero means every candidate merge left the beliefs in the")
    print("    same place, so there is nothing to rank at that horizon.")
    json.dump({"rows": rows, "tau": out},
              open(os.path.join(os.path.dirname(__file__), "..", "results",
                                "s07_oracle.json"), "w"), indent=1)
