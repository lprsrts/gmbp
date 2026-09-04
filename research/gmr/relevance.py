"""Downstream relevance propagation, per the concept note's Algorithm 1.

The note gives the relevance exactly for one case only: a message that is
subsequently multiplied by r(x), where lambda(x) = r(x)^2.  For anything
else it says the recursion is 'motivation'.  Every implementation therefore
has to invent the backward rules through factors; the choices made here are
the natural ones and are stated explicitly so that failures can be
attributed to the scheme rather than to the fill-in.
"""
import numpy as np
from .gm import GM, gm_product, gm_square, log_norm_pdf_at
from .graph import ClassSensorFactor, SensorPairFactor, unit
from .reduce import reduce_isd, reduce_runnalls


UNIFORM = GM([1.0], [[0.0]], [[[1e8]]])   # stand-in for lambda == const


def normalise_lambda(lam: GM, mode="peak", grid=None) -> GM:
    """Rescale lambda. Ranking of merges is invariant to this; the damped
    mix and the reactivation threshold are not."""
    if mode == "raw":
        return lam
    if mode == "mass":
        m = lam.mass()
        return lam.scaled(1.0 / m) if m > 0 else lam
    if mode == "peak":
        x = grid if grid is not None else _auto_grid(lam)
        pk = float(np.max(lam.pdf(x)))
        return lam.scaled(1.0 / pk) if pk > 0 else lam
    raise ValueError(mode)


def _auto_grid(lam, n=2001, pad=8.0):
    sd = np.sqrt(lam.S[:, 0, 0])
    lo = float(np.min(lam.mu[:, 0] - pad * sd))
    hi = float(np.max(lam.mu[:, 0] + pad * sd))
    return np.linspace(lo, hi, n)


def height_prune(lam: GM, budget: int) -> GM:
    """Keep the components that contribute most to lambda's *value*.

    A relevance function is not a density. What a weighted criterion reads
    off it is its pointwise height, and a very broad component carries a
    large mixture weight while contributing almost nothing to the height.
    Pruning lambda by mixture weight -- the obvious thing, and what one does
    to a message -- therefore keeps the flat background and throws away the
    informative peak. Rank by peak contribution instead.
    """
    if lam.n <= budget:
        return lam.copy()
    dets = np.array([np.linalg.slogdet(lam.S[k])[1] for k in range(lam.n)])
    height = np.log(np.maximum(np.abs(lam.w), 1e-300)) - 0.5 * dets
    idx = np.sort(np.argsort(-height)[:budget])
    return lam.subset(idx)


def cap(lam: GM, budget, method="isd", pre=40, by="height") -> GM:
    """Relevance is itself a mixture and itself blows up. Cap it.

    Squaring an n-component message gives n^2 components, so greedy merging
    straight down to the budget is O(n^4) and unaffordable; a prune to `pre`
    runs first. Which prune matters -- see height_prune. `by="weight"` is
    the naive choice, kept so the difference can be measured.
    """
    if budget is None or lam.n <= budget:
        return lam
    from .reduce import weight_prune
    if pre is not None and lam.n > pre:
        lam = height_prune(lam, pre) if by == "height" else weight_prune(lam, pre)
    red = reduce_isd if method == "isd" else reduce_runnalls
    return red(lam, budget, None, None)


class RelevanceField:
    """Holds lambda_{a->b} for every directed edge and updates it."""

    def __init__(self, bp, alpha=0.5, lam_budget=6, norm="peak",
                 backward=True, cap_by="height"):
        self.bp = bp
        self.g = bp.g
        self.alpha = alpha
        self.lam_budget = lam_budget
        self.norm = norm
        self.backward = backward
        self.cap_by = cap_by
        self.lam = {}
        for k, v in bp.msg.items():
            if isinstance(v, GM):
                self.lam[k] = UNIFORM

    def lookup(self, key):
        return self.lam.get(key, UNIFORM)

    # -- downstream estimate ------------------------------------------
    def estimate(self, msgs):
        """lambda_hat for every continuous edge from the current messages."""
        g = self.g
        out = {}
        # messages arriving at a continuous variable: exact single-step weight
        for v in g.sensors:
            for fname in g.nbrs[v]:
                pieces = [msgs[(fn, v)] for fn in g.nbrs[v] if fn != fname]
                if v in g.obs and g.obs[v] is not None:
                    pieces.append(g.obs[v])
                r = None
                for p in pieces:
                    r = p if r is None else gm_product(r, p)
                if r is None:
                    out[(fname, v)] = UNIFORM
                    continue
                lam = gm_square(unit(r))
                out[(fname, v)] = self._post(lam)
        # messages leaving a continuous variable into a factor
        for v in g.sensors:
            for fname in g.nbrs[v]:
                f = g.factors[fname]
                if not self.backward:
                    out[(v, fname)] = UNIFORM
                    continue
                if isinstance(f, SensorPairFactor):
                    other = f.b if v == f.a else f.a
                    downstream = out.get((fname, other), UNIFORM)
                    out[(v, fname)] = self._post(
                        self._pull_through_pair(f, v, downstream))
                else:
                    out[(v, fname)] = self._post(
                        self._pull_through_class(f, msgs))
        return out

    def _pull_through_pair(self, f, v, lam_down: GM) -> GM:
        """Relevance on the far side, pulled back through the displacement."""
        sgn = 1.0 if v == f.a else -1.0
        ws, mus, Ss = [], [], []
        for i in range(lam_down.n):
            for k in range(f.disp.n):
                ws.append(lam_down.w[i] * f.disp.w[k])
                mus.append(lam_down.mu[i] - sgn * f.disp.mu[k])
                Ss.append(lam_down.S[i] + f.disp.S[k])
        return GM(np.array(ws), np.vstack(mus), np.array(Ss))

    def _pull_through_class(self, f, msgs) -> GM:
        """A message into a class factor is integrated against p(x|c), so the
        class belief responds to it as

            delta b(c)  ~  r_c * int p(x|c) delta m(x) dx,

        where r_c is the product of the class node's other incoming
        messages. The squared sensitivity is therefore

            lambda(x) = sum_c r_c^2 p(x|c)^2,

        which is the discrete-side analogue of lambda = r^2 and, unlike a
        plain sum over the conditionals, actually moves as inference
        proceeds."""
        r = np.ones(self.g.classes[f.cls])
        for fn2 in self.g.nbrs[f.cls]:
            if fn2 == f.name:
                continue
            m = msgs.get((fn2, f.cls))
            if m is not None:
                r = r * np.asarray(m)
        z = r.sum()
        r = r / z if z > 0 else np.ones_like(r) / len(r)
        acc = None
        for c, gm in enumerate(f.per_state):
            sq = gm_square(unit(gm)).scaled(float(r[c]) ** 2)
            acc = sq if acc is None else acc.concat(sq)
        return acc

    def _post(self, lam: GM) -> GM:
        lam = cap(lam, self.lam_budget, by=self.cap_by)
        return normalise_lambda(lam, self.norm)

    # -- damped update -------------------------------------------------
    def update(self, msgs):
        hat = self.estimate(msgs)
        for k, lh in hat.items():
            prev = self.lam.get(k, UNIFORM)
            if self.alpha >= 1.0:
                mixed = lh
            else:
                mixed = prev.scaled(1.0 - self.alpha).concat(
                    lh.scaled(self.alpha))
            self.lam[k] = self._post(mixed)

    def sizes(self):
        return {k: v.n for k, v in self.lam.items()}


def reactivation_scores(reserve_items, lam: GM):
    """rho_j = int lambda(x) w_j N(x; mu_j, Sigma_j) dx, closed form."""
    out = []
    for comp in reserve_items:
        w, mu, S = comp
        tot = 0.0
        for k in range(lam.n):
            tot += lam.w[k] * w * np.exp(
                log_norm_pdf_at(mu, lam.mu[k], S + lam.S[k]))
        out.append(tot)
    return np.asarray(out)
