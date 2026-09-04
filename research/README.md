# Stress test: downstream-aware Gaussian mixture reduction

An implementation and adversarial evaluation of the method proposed in
*Learning where to be careful*, section "Downstream-Aware Adaptive Gaussian
Mixture Reduction". The note presents that section as a proposal with an
algorithm box and no experiments; this directory supplies both.

## Layout

    gmr/gm.py           dimension-general GM primitives (the correctness oracle)
    gmr/fast1d.py       vectorised 1D core; everything hot goes through here
    gmr/criteria.py     unweighted ISD, region-weighted ISD, Runnalls' bound
    gmr/reduce.py       greedy merging to a component budget, weight pruning
    gmr/graph.py        hybrid loopy factor graph and the GMBP engine
    gmr/build.py        the note's graph (3 classes x 3 states, 4 sensors,
                        9 factors) and a continuous chain
    gmr/relevance.py    relevance field: lambda = r^2, backward pull-through,
                        damped update, reactivation scores
    gmr/adaptive.py     Algorithm 1, plus the local baselines
    stress/             the battery, s00 (verification) through s10
    results/            machine-readable output and the printed reports

## Running

    uv venv && uv pip install numpy scipy scikit-learn mpmath
    .venv/bin/python research/run_all.py .venv/bin/python

Individual tests take from a second (s00, s03) to a few minutes
(s04, s07, s09).

## Reading it

Start with `RESULTS.md` at the repository root. `stress/s00_verify.py` is
the reason to trust the rest: it checks the closed forms against quadrature
and 60-digit arithmetic, checks the vectorised core against an independent
dimension-general implementation, and checks the BP engine against an exact
posterior on a tree.
