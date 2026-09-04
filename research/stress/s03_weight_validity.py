"""S3. Is lambda = r(x)^2 the right weight?

Eq. (17)-(18) of the note derive it from

    int (p r - q r)^2 dx = int r^2 (p-q)^2 dx,

which is exact -- for an *unnormalised* product and for a downstream that is
one multiplication and nothing else. Real belief propagation normalises
messages, and most downstream paths go through at least one factor, i.e. a
marginalisation. Both break the identity.

This ranks candidate merges by the note's weight and by the local criteria,
and scores each against the damage actually done downstream under three
increasingly realistic downstream operators.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gmr import fast1d as F
from scipy.stats import kendalltau


def signed_isd(w1, m1, s1, w2, m2, s2):
    w = np.concatenate([w1, -w2]); m = np.concatenate([m1, m2])
    s = np.concatenate([s1, s2])
    return F.weighted_sq_norm_1d(w, m, s, None)


def tv_1d(w1, m1, s1, w2, m2, s2, n=4001):
    lo = min((m1 - 8 * np.sqrt(s1)).min(), (m2 - 8 * np.sqrt(s2)).min())
    hi = max((m1 + 8 * np.sqrt(s1)).max(), (m2 + 8 * np.sqrt(s2)).max())
    x = np.linspace(lo, hi, n)
    a = F.pdf_1d(w1, m1, s1, x); b = F.pdf_1d(w2, m2, s2, x)
    a = a / np.trapezoid(a, x); b = b / np.trapezoid(b, x)
    return 0.5 * float(np.trapezoid(np.abs(a - b), x))


def convolve(w, m, s, kmu, ksig):
    return w.copy(), m + kmu, s + ksig


def trial(rng, n=10, rn=3, kernel_var=0.8):
    w = rng.dirichlet(np.ones(n) * 0.8)
    m = rng.normal(0, 2.5, n)
    s = rng.uniform(0.08, 1.0, n)
    rw = rng.dirichlet(np.ones(rn))
    rm = rng.normal(0, 2.5, rn)
    rs = rng.uniform(0.1, 1.5, rn)
    lam_sq = F.gm_square_1d(rw, rm, rs)          # lambda = r^2
    lam = (lam_sq[0], lam_sq[1], lam_sq[2])

    iu, ju = np.triu_indices(n, 1)
    pred_note, pred_isd, pred_run = [], [], []
    true_unnorm, true_norm, true_marg = [], [], []
    for i, j in zip(iu, ju):
        wM, mM, sM = F.merged_moments(w[i], m[i], s[i], w[j], m[j], s[j])
        keep = np.setdiff1d(np.arange(n), [i, j])
        qw = np.append(w[keep], wM); qm = np.append(m[keep], mM)
        qs = np.append(s[keep], sM)
        # predictions
        rw_res = np.array([w[i], w[j], -wM]); rm_res = np.array([m[i], m[j], mM])
        rs_res = np.array([s[i], s[j], sM])
        pred_note.append(F.weighted_sq_norm_1d(rw_res, rm_res, rs_res, lam))
        pred_isd.append(F.weighted_sq_norm_1d(rw_res, rm_res, rs_res, None))
        pred_run.append(0.5 * (wM * np.log(sM) - w[i] * np.log(s[i])
                               - w[j] * np.log(s[j])))
        # truths
        pw, pm, ps = F.gm_product_1d(w, m, s, rw, rm, rs)
        qw2, qm2, qs2 = F.gm_product_1d(qw, qm, qs, rw, rm, rs)
        true_unnorm.append(signed_isd(pw, pm, ps, qw2, qm2, qs2))
        Zp, Zq = pw.sum(), qw2.sum()
        true_norm.append(signed_isd(pw / Zp, pm, ps, qw2 / Zq, qm2, qs2))
        cw, cm, cs = convolve(pw / Zp, pm, ps, 0.0, kernel_var)
        dw, dm, ds = convolve(qw2 / Zq, qm2, qs2, 0.0, kernel_var)
        true_marg.append(signed_isd(cw, cm, cs, dw, dm, ds))
    return (np.array(pred_note), np.array(pred_isd), np.array(pred_run),
            np.array(true_unnorm), np.array(true_norm), np.array(true_marg))


def score(pred, truth):
    ok = np.isfinite(pred) & np.isfinite(truth)
    if ok.sum() < 4:
        return np.nan, np.nan, np.nan
    tau = kendalltau(pred[ok], truth[ok]).statistic
    top1 = float(np.argmin(pred[ok]) == np.argmin(truth[ok]))
    # regret: how much worse is the merge you pick than the best one
    t = truth[ok]
    chosen = t[np.argmin(pred[ok])]
    best = max(t.min(), 1e-300)
    return tau, top1, float(chosen / best)


def staleness(trials=200, seed=11, kernel_var=0.8):
    """lambda built from a perturbed r, scored against the true downstream."""
    levels = [("exact r", 0.0), ("means +-0.25 sd", 0.25), ("means +-0.5 sd", 0.5),
              ("means +-1.0 sd", 1.0), ("means +-2.0 sd", 2.0),
              ("resampled r", None)]
    rows = []
    for label, jit in levels:
        rng = np.random.default_rng(seed)
        taus, tops, rats = [], [], []
        for _ in range(trials):
            n, rn = 10, 3
            w = rng.dirichlet(np.ones(n) * 0.8); m = rng.normal(0, 2.5, n)
            s = rng.uniform(0.08, 1.0, n)
            rw = rng.dirichlet(np.ones(rn)); rm = rng.normal(0, 2.5, rn)
            rs = rng.uniform(0.1, 1.5, rn)
            if jit is None:
                rm2 = rng.normal(0, 2.5, rn); rw2 = rng.dirichlet(np.ones(rn))
                rs2 = rng.uniform(0.1, 1.5, rn)
            else:
                rm2 = rm + jit * np.sqrt(rs) * rng.normal(size=rn)
                rw2, rs2 = rw, rs
            lam = F.gm_square_1d(rw2, rm2, rs2)
            iu, ju = np.triu_indices(n, 1)
            pred, truth = [], []
            for i, j in zip(iu, ju):
                wM, mM, sM = F.merged_moments(w[i], m[i], s[i], w[j], m[j], s[j])
                keep = np.setdiff1d(np.arange(n), [i, j])
                qw = np.append(w[keep], wM); qm = np.append(m[keep], mM)
                qs = np.append(s[keep], sM)
                pred.append(F.weighted_sq_norm_1d(
                    np.array([w[i], w[j], -wM]), np.array([m[i], m[j], mM]),
                    np.array([s[i], s[j], sM]), lam))
                pw, pm, ps = F.gm_product_1d(w, m, s, rw, rm, rs)
                qw2, qm2, qs2 = F.gm_product_1d(qw, qm, qs, rw, rm, rs)
                cw, cm, cs = convolve(pw / pw.sum(), pm, ps, 0.0, kernel_var)
                dw, dm, ds = convolve(qw2 / qw2.sum(), qm2, qs2, 0.0, kernel_var)
                truth.append(signed_isd(cw, cm, cs, dw, dm, ds))
            t, o, r = score(np.array(pred), np.array(truth))
            taus.append(t); tops.append(o); rats.append(r)
        rows.append({"label": label, "tau": float(np.nanmedian(taus)),
                     "top1": float(np.nanmean(tops)),
                     "ratio": float(np.nanmedian(rats))})
    return rows


def run(trials=300, seed=0, kernel_var=0.8):
    rng = np.random.default_rng(seed)
    names = ["lambda = r^2 (the note)", "unweighted ISD", "Runnalls"]
    truths = ["unnormalised product", "normalised product", "normalised + marginalised"]
    acc = {(a, b): [] for a in names for b in truths}
    for _ in range(trials):
        pn, pi, pr, tu, tn, tm = trial(rng, kernel_var=kernel_var)
        for a, pred in zip(names, (pn, pi, pr)):
            for b, tr in zip(truths, (tu, tn, tm)):
                acc[(a, b)].append(score(pred, tr))
    return names, truths, acc


if __name__ == "__main__":
    names, truths, acc = run()
    print("S3  Does lambda = r^2 rank merges by the damage they actually do downstream?")
    print("    300 random mixtures; Kendall tau / top-1 hit rate / cost ratio of the merge chosen vs the best one\n")
    hdr = f"    {'criterion':<26}" + "".join(f"{t:>32}" for t in truths)
    print(hdr)
    out = {}
    for a in names:
        line = f"    {a:<26}"
        for b in truths:
            arr = np.array(acc[(a, b)], dtype=float)
            tau = np.nanmedian(arr[:, 0]); top = np.nanmean(arr[:, 1])
            reg = np.nanmedian(arr[:, 2])
            out[f"{a}|{b}"] = {"tau": float(tau), "top1": float(top), "regret": float(reg)}
            line += f"{tau:>13.3f} {top:>6.0%} {reg:>11.2f}x"
        print(line)
    print("\n    (tau 1.0 and top-1 100% on the first column is the identity of Eq. 18,")
    print("     recovered numerically. The other two columns are what BP actually does.)")
    stale = staleness()
    print("\n    Relevance is always stale in a loopy graph. Perturbing r before")
    print("    squaring it, then scoring against the true downstream damage:")
    print(f"    {'r perturbation':<22}{'tau':>10}{'top-1':>9}{'chosen/best':>14}")
    for row in stale:
        print(f"    {row['label']:<22}{row['tau']:>10.3f}{row['top1']:>8.0%}{row['ratio']:>13.2f}x")
    out["staleness"] = stale
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..",
                                     "results", "s03_weight_validity.json"), "w"), indent=1)
