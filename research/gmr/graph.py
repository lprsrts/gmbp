"""A small hybrid loopy factor graph with Gaussian-mixture message passing.

Shape follows the concept note: discrete class variables with S states, each
class-conditional sensor density a K-component mixture, so a factor->sensor
message always carries S*K components regardless of what came in.  Sensors
with three or more neighbours therefore build products that grow
multiplicatively, which is what forces reduction.
"""
import numpy as np
from .gm import GM, gm_product, log_norm_pdf_at


class ClassSensorFactor:
    """p(sensor | class): one Gaussian mixture per class state."""

    def __init__(self, name, cls_var, sensor_var, per_state):
        self.name = name
        self.cls = cls_var
        self.sensor = sensor_var
        self.per_state = per_state       # list of GM, one per class state
        self.scope = (cls_var, sensor_var)

    def to_sensor(self, q_class):
        ws, mus, Ss = [], [], []
        for s, gm in enumerate(self.per_state):
            if q_class[s] <= 0:
                continue
            ws.append(gm.w * q_class[s])
            mus.append(gm.mu)
            Ss.append(gm.S)
        if not ws:
            ws, mus, Ss = [self.per_state[0].w * 1e-300], [self.per_state[0].mu], [self.per_state[0].S]
        return GM(np.concatenate(ws), np.vstack(mus), np.concatenate(Ss, axis=0))

    def to_class(self, m_sensor: GM):
        """q(c) proportional to int m(x) p(x|c) dx, a Gaussian overlap sum."""
        from .fast1d import N as _N
        out = np.zeros(len(self.per_state))
        mi, si, wi = m_sensor.mu[:, 0], m_sensor.S[:, 0, 0], m_sensor.w
        for s, gm in enumerate(self.per_state):
            mj, sj, wj = gm.mu[:, 0], gm.S[:, 0, 0], gm.w
            z = _N(mi[:, None], mj[None, :], si[:, None] + sj[None, :])
            out[s] = float(wi @ z @ wj)
        z = out.sum()
        return out / z if z > 0 else np.ones_like(out) / len(out)


class SensorPairFactor:
    """p(s2 | s1) = mixture over displacement: N(s2 - s1; mu_k, S_k)."""

    def __init__(self, name, a, b, disp: GM):
        self.name = name
        self.a = a
        self.b = b
        self.disp = disp
        self.scope = (a, b)

    def _convolve(self, m: GM, forward=True):
        ws, mus, Ss = [], [], []
        for i in range(m.n):
            for k in range(self.disp.n):
                sgn = 1.0 if forward else -1.0
                ws.append(m.w[i] * self.disp.w[k])
                mus.append(m.mu[i] + sgn * self.disp.mu[k])
                Ss.append(m.S[i] + self.disp.S[k])
        return GM(np.array(ws), np.vstack(mus), np.array(Ss))

    def to_other(self, src, m: GM):
        return self._convolve(m, forward=(src == self.a))


class HybridGraph:
    def __init__(self):
        self.classes = {}      # name -> n_states
        self.sensors = []      # names
        self.factors = {}      # name -> factor
        self.nbrs = {}         # var name -> [factor names]
        self.obs = {}          # sensor name -> GM likelihood (or None)

    def add_class(self, name, n_states):
        self.classes[name] = n_states
        self.nbrs.setdefault(name, [])

    def add_sensor(self, name):
        self.sensors.append(name)
        self.nbrs.setdefault(name, [])

    def add_factor(self, f):
        self.factors[f.name] = f
        for v in f.scope:
            self.nbrs.setdefault(v, []).append(f.name)

    def observe(self, sensor, gm: GM):
        self.obs[sensor] = gm


FLAT_VAR = [1e4]   # stand-in for an improper uniform prior on a sensor


def unit(gm: GM) -> GM:
    m = gm.mass()
    return gm.scaled(1.0 / m) if m > 0 else gm


class GMBP:
    """Flooding schedule. `reducer(gm, budget, lam, record) -> GM`."""

    def __init__(self, g: HybridGraph, budget=None, reducer=None,
                 normalise=True):
        self.g = g
        self.budget = budget
        self.reducer = reducer
        self.normalise = normalise
        self.msg = {}
        self.raw = {}          # pre-reduction messages, for diagnostics
        self.counts = []
        self._init_messages()

    def _init_messages(self):
        g = self.g
        for fname, f in g.factors.items():
            if isinstance(f, ClassSensorFactor):
                ns = g.classes[f.cls]
                self.msg[(fname, f.cls)] = np.ones(ns) / ns
                self.msg[(f.cls, fname)] = np.ones(ns) / ns
                flat = GM([1.0], [[0.0]], [[[FLAT_VAR[0]]]])
                self.msg[(fname, f.sensor)] = flat
                self.msg[(f.sensor, fname)] = flat
            else:
                flat = GM([1.0], [[0.0]], [[[FLAT_VAR[0]]]])
                self.msg[(fname, f.a)] = flat
                self.msg[(f.a, fname)] = flat
                self.msg[(fname, f.b)] = flat
                self.msg[(f.b, fname)] = flat

    # --- variable side -------------------------------------------------
    def var_to_factor(self, v, fname, msgs=None):
        msgs = self.msg if msgs is None else msgs
        others = [fn for fn in self.g.nbrs[v] if fn != fname]
        if v in self.g.classes:
            q = np.ones(self.g.classes[v])
            for fn in others:
                q = q * msgs[(fn, v)]
            z = q.sum()
            return q / z if z > 0 else np.ones_like(q) / len(q)
        prod = None
        pieces = [msgs[(fn, v)] for fn in others]
        if v in self.g.obs and self.g.obs[v] is not None:
            pieces.append(self.g.obs[v])
        for p in pieces:
            prod = p if prod is None else gm_product(prod, p)
        if prod is None:
            prod = GM([1.0], [[0.0]], [[[FLAT_VAR[0]]]])
        return unit(prod) if self.normalise else prod

    def belief(self, v, msgs=None):
        msgs = self.msg if msgs is None else msgs
        if v in self.g.classes:
            q = np.ones(self.g.classes[v])
            for fn in self.g.nbrs[v]:
                q = q * msgs[(fn, v)]
            z = q.sum()
            return q / z if z > 0 else np.ones_like(q) / len(q)
        prod = None
        pieces = [msgs[(fn, v)] for fn in self.g.nbrs[v]]
        if v in self.g.obs and self.g.obs[v] is not None:
            pieces.append(self.g.obs[v])
        for p in pieces:
            prod = p if prod is None else gm_product(prod, p)
        return unit(prod)

    # --- factor side ----------------------------------------------------
    def factor_to_var(self, fname, v, msgs=None):
        msgs = self.msg if msgs is None else msgs
        f = self.g.factors[fname]
        if isinstance(f, ClassSensorFactor):
            if v == f.sensor:
                return unit(f.to_sensor(msgs[(f.cls, fname)]))
            return f.to_class(msgs[(f.sensor, fname)])
        src = f.a if v == f.b else f.b
        return unit(f.to_other(src, msgs[(src, fname)]))

    # --- one flooding sweep --------------------------------------------
    def _maybe_reduce(self, key, val, lam_lookup, reserve):
        if not isinstance(val, GM):
            return val
        if self.budget is None or self.reducer is None or val.n <= self.budget:
            return val
        lam = lam_lookup(key) if lam_lookup else None
        rec = [] if reserve is not None else None
        red = self.reducer(val, self.budget, lam, rec)
        if reserve is not None and rec:
            reserve.setdefault(key, []).extend(rec)
        return unit(red) if self.normalise else red

    def sweep(self, lam_lookup=None, reserve=None):
        """Variable messages, reduced; then factor messages, reduced.

        Reduction sits inside the propagation path in both directions, so
        damage from one sweep feeds the next, as Algorithm 1 specifies.
        """
        g = self.g
        new = {}
        raw = {}
        for v, fnames in g.nbrs.items():
            for fname in fnames:
                m = self.var_to_factor(v, fname)
                raw[(v, fname)] = m.copy() if isinstance(m, GM) else m
                new[(v, fname)] = self._maybe_reduce((v, fname), m,
                                                     lam_lookup, reserve)
        for fname, f in g.factors.items():
            for v in f.scope:
                m = self.factor_to_var(fname, v, msgs=new)
                raw[(fname, v)] = m.copy() if isinstance(m, GM) else m
                new[(fname, v)] = self._maybe_reduce((fname, v), m,
                                                     lam_lookup, reserve)
        self.raw = raw
        self.msg = new
        self.counts.append({k: (v.n if isinstance(v, GM) else 0)
                            for k, v in raw.items()})

    def run(self, T, **kw):
        for _ in range(T):
            self.sweep(**kw)
        return self
