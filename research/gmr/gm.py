"""Gaussian mixture primitives, in log domain where it matters.

Everything here is dimension-general but the stress work runs mostly in 1D
because that is where an unreduced reference is affordable.
"""
import numpy as np


class GM:
    """A (possibly unnormalised, possibly signed) Gaussian mixture.

    weights: (N,)   means: (N, d)   covs: (N, d, d)
    Signed weights are allowed -- merge residuals p - q are represented that
    way and the ISD machinery handles them without special-casing.
    """

    __slots__ = ("w", "mu", "S")

    def __init__(self, w, mu, S):
        self.w = np.asarray(w, dtype=float).ravel()
        mu = np.asarray(mu, dtype=float)
        if mu.ndim == 1:
            mu = mu.reshape(len(self.w), -1)
        self.mu = mu
        S = np.asarray(S, dtype=float)
        if S.ndim == 1:
            S = S.reshape(len(self.w), 1, 1)
        elif S.ndim == 2 and S.shape[0] == len(self.w) and mu.shape[1] == 1:
            S = S.reshape(len(self.w), 1, 1)
        self.S = S
        assert self.mu.shape[0] == len(self.w)
        assert self.S.shape[0] == len(self.w)

    @property
    def n(self):
        return len(self.w)

    @property
    def d(self):
        return self.mu.shape[1]

    def copy(self):
        return GM(self.w.copy(), self.mu.copy(), self.S.copy())

    def mass(self):
        return float(np.sum(self.w))

    def normalised(self):
        z = np.sum(self.w)
        return GM(self.w / z, self.mu.copy(), self.S.copy())

    def scaled(self, c):
        return GM(self.w * c, self.mu.copy(), self.S.copy())

    def pdf(self, X):
        """X: (M, d) or (M,) in 1D. Returns (M,)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, self.d)
        out = np.zeros(X.shape[0])
        for k in range(self.n):
            out += self.w[k] * _norm_pdf(X, self.mu[k], self.S[k])
        return out

    def subset(self, idx):
        idx = np.asarray(idx, dtype=int)
        return GM(self.w[idx], self.mu[idx], self.S[idx])

    def concat(self, other):
        return GM(np.concatenate([self.w, other.w]),
                  np.vstack([self.mu, other.mu]),
                  np.concatenate([self.S, other.S], axis=0))

    def moments(self):
        """First two moments of the mixture, treating w as a measure."""
        z = np.sum(self.w)
        mu = (self.w[:, None] * self.mu).sum(0) / z
        d = self.mu - mu
        S = (self.w[:, None, None] * (self.S + d[:, :, None] * d[:, None, :])).sum(0) / z
        return mu, S

    def __repr__(self):
        return f"GM(n={self.n}, d={self.d}, mass={self.mass():.4g})"


def _norm_pdf(X, mu, S):
    d = len(mu)
    diff = X - mu
    L = np.linalg.cholesky(S)
    sol = np.linalg.solve(L, diff.T).T
    quad = np.einsum("ij,ij->i", sol, sol)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return np.exp(-0.5 * (quad + logdet + d * np.log(2 * np.pi)))


def log_norm_pdf_at(x, mu, S):
    """log N(x; mu, S) for a single point, robust to ill-conditioning."""
    d = len(mu)
    diff = np.asarray(x, dtype=float) - mu
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S = S + np.eye(d) * (1e-300 + 1e-12 * np.trace(S) / d)
        L = np.linalg.cholesky(S)
    sol = np.linalg.solve(L, diff)
    quad = float(sol @ sol)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (quad + logdet + d * np.log(2 * np.pi))


def gauss_product(mu1, S1, mu2, S2):
    """N(x;mu1,S1) N(x;mu2,S2) = exp(logz) * N(x;mu,S). Returns (mu, S, logz)."""
    Ssum = S1 + S2
    logz = log_norm_pdf_at(mu1, mu2, Ssum)
    P1 = np.linalg.inv(S1)
    P2 = np.linalg.inv(S2)
    S = np.linalg.inv(P1 + P2)
    S = 0.5 * (S + S.T)
    mu = S @ (P1 @ mu1 + P2 @ mu2)
    return mu, S, logz


def gm_product(p: GM, q: GM) -> GM:
    """Full n*m product. Weights carry the Gaussian overlap constants."""
    if p.d == 1 and q.d == 1:
        from .fast1d import gm_product_1d
        w, m, sv = gm_product_1d(p.w, p.mu[:, 0], p.S[:, 0, 0],
                                 q.w, q.mu[:, 0], q.S[:, 0, 0])
        return GM(w, m.reshape(-1, 1), sv.reshape(-1, 1, 1))
    ws, mus, Ss = [], [], []
    for i in range(p.n):
        for j in range(q.n):
            mu, S, logz = gauss_product(p.mu[i], p.S[i], q.mu[j], q.S[j])
            ws.append(p.w[i] * q.w[j] * np.exp(logz))
            mus.append(mu)
            Ss.append(S)
    return GM(np.array(ws), np.array(mus), np.array(Ss))


def moment_merge(w, mu, S):
    """Moment-preserving merge of a set of components into one Gaussian."""
    wt = np.sum(w)
    m = (w[:, None] * mu).sum(0) / wt
    d = mu - m
    C = (w[:, None, None] * (S + d[:, :, None] * d[:, None, :])).sum(0) / wt
    C = 0.5 * (C + C.T)
    return wt, m, C


def merge_pair(gm: GM, i, j) -> GM:
    """Return gm with components i,j replaced by their moment-preserving merge."""
    keep = [k for k in range(gm.n) if k not in (i, j)]
    wt, m, C = moment_merge(gm.w[[i, j]], gm.mu[[i, j]], gm.S[[i, j]])
    w = np.concatenate([gm.w[keep], [wt]])
    mu = np.vstack([gm.mu[keep], m[None, :]])
    S = np.concatenate([gm.S[keep], C[None, :, :]], axis=0)
    return GM(w, mu, S)


def residual_of_merge(gm: GM, i, j) -> GM:
    """The signed mixture (before - after) for merging i,j. Three components."""
    wt, m, C = moment_merge(gm.w[[i, j]], gm.mu[[i, j]], gm.S[[i, j]])
    w = np.array([gm.w[i], gm.w[j], -wt])
    mu = np.vstack([gm.mu[i], gm.mu[j], m])
    S = np.concatenate([gm.S[i][None], gm.S[j][None], C[None]], axis=0)
    return GM(w, mu, S)


def gm_square(p: GM) -> GM:
    """p(x)^2 as a Gaussian mixture. n components in, n^2 out."""
    return gm_product(p, p)
