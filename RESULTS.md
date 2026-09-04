# Stress test: downstream-aware Gaussian mixture reduction

Target: *Learning where to be careful* (Saritas, 5 August 2026), the section
"Downstream-Aware Adaptive Gaussian Mixture Reduction" — Eqs. (13)–(60) and
Algorithm 1.

The rest of that note reports measurements. This section reports none: it
introduces the relevance function, the backward recursion, the damped update,
the reserve set and the reactivation rule, states a testable hypothesis, and
stops. So this is a test of a proposal — implement it as written, then attack
it.

Everything below is measured. Code in `research/`, raw output in
`research/results/`, reproduce with `research/run_all.py`.

## The short version

**Algorithm 1 as specified is nine to sixteen times worse than the Runnalls
baseline it is meant to beat, at seventeen to twenty-three times the cost,
losing every configuration tested.** Remove the reserve set, the damping and
the backward relevance propagation and the remainder beats Runnalls in
75–100% of configurations at 0.55–0.68 of its error and a quarter of the
full method's cost.

The remainder is one stateless rule:

> When reducing a message a→b, weight the integral squared difference by
> λ(x) = ( ∏<sub>c ≠ a</sub> m<sub>c→b</sub>(x) )², rebuilt from the current
> messages each sweep.

That is Eq. (18) applied on the edges where it is exactly true, and nothing
else. The idea at the centre of the proposal is sound and the measured
effect is large. The machinery built around it is inert or harmful.

| claim in the note | verdict |
|---|---|
| Region-weighted ISD has an exact closed form for Gaussian weights | **Holds.** Matches quadrature to 2e-16 median (S0) |
| Eq. (18): λ = r² is the exact single-step downstream weight | **Holds.** Reproduced to 5e-15 median (S0), and τ = 1.000 as a merge ranker (S3) |
| λ = r² beats local criteria at predicting downstream damage | **Holds.** τ 0.86 vs 0.77 on a synthetic downstream (S3); 0.64 vs 0.39 against an oracle inside the real loop (S7) |
| Principal hypothesis: D(ref, adaptive) < D(ref, local) | **Holds** for the single-step rule: 99–100% of configurations (S4, S11, S13). **Fails** for Algorithm 1 as written: 0% (S13) |
| Damping α (Eq. 49) stabilises the coupled iteration | **Nothing to stabilise, and it hurts.** α = 1 is better everywhere (S5, S4b, S13) |
| Backward relevance propagation (Eqs. 39–50) | **Inert.** Removing it changes nothing but cost (S4, S11, S13) |
| Backward recursion λ_k = r²λ_{k+1} (Eq. 40) | **Degenerates with depth** (S10) |
| Reserve set and reactivation (Eqs. 55–58) | **Harmful at every setting tested**, and its premise fails (S6, S8, S9) |
| τ_react is a usable threshold | **No.** It thresholds a quantity with no fixed scale (S6) |
| Cost comparable to the cheap criterion | **No.** 4.5× stripped, 17–23× as written (S2, S13) |

## What was built

The repository had no implementation of any of this — `GaussianMixture.prune`
in `src/gmbp/core/distributions.py` is a weight threshold, which is the
weakest of the four criteria the note benchmarks. So `research/` is a fresh
implementation: the region-weighted ISD in closed form, Runnalls' bound,
greedy merging to a budget, a hybrid loopy factor graph with Gaussian-mixture
message passing, and the relevance machinery of Eqs. (13)–(60) and
Algorithm 1 — relevance λ = r², a backward pull-through for both factor
types, the damped update of Eq. (49), the reserve set, and the reactivation
score of Eq. (57).

The test graph reproduces the note's: three class variables of three states,
four sensors, nine factors, one connected component with three independent
cycles. It grows the way the note describes — factor messages of nine
components, sensor messages of 81, beliefs of 729 — which is the check that
it is the same object.

Two things the note leaves open had to be decided, and both are noted where
they matter. The backward relevance through a factor is only specified for
pure multiplication, so the class-side rule here is the exact analogue,
λ(x) = Σ_c r_c² p(x|c)², with r_c the product of the class node's other
incoming messages. And the note says nothing about how to reduce λ itself,
which turns out to be necessary (S2).

## S0 — Is the implementation trustworthy?

| check | result |
|---|---|
| region-weighted ISD closed form vs. quadrature | median 2.0e-16, max 7.0e-15 |
| vectorised 1D core vs. independent general-dimension code | median 5.9e-13, max 6.3e-09 |
| Eq. (18), ∫(pr−qr)² = ∫r²(p−q)² | median 5.0e-15, max 9.2e-08 |
| BP on a tree vs. exact posterior by enumeration | 8.4e-05, and it is the improper-prior stand-in: widening the flat prior from 1e4 to 1e10 drives it to 8.4e-11 |

The first line reproduces the note's own 5e-16 claim independently. The
closed form is correct.

## S3 — Is λ = r² the right weight once BP does what BP does?

Eq. (18) is exact for an unnormalised product and a downstream that is one
multiplication. Real message passing normalises, and most downstream paths
cross at least one factor. Ranking 300 random merge sets against the damage
actually done, by Kendall τ and by how often the top pick is the truly least
damaging:

| criterion | unnormalised product | normalised product | normalised + marginalised |
|---|---|---|---|
| λ = r² (the note) | **1.000 / 100%** | 0.948 / 98% | 0.863 / 81% |
| unweighted ISD | 0.787 / 66% | 0.772 / 67% | 0.772 / 64% |
| Runnalls' bound | 0.737 / 48% | 0.725 / 48% | 0.763 / 52% |

The first column is Eq. (18) recovered numerically. The other two are what
BP actually does, and the weight degrades but stays well ahead. Breaking the
derivation's assumptions costs it τ 1.000 → 0.863; it is still 0.09 above
the best local criterion.

It also degrades gracefully when the relevance is stale, which in a loopy
graph it always is:

| relevance built from | τ | top-1 |
|---|---|---|
| the exact downstream r | 0.871 | 86% |
| r with means jittered ±1.0 σ | 0.846 | 77% |
| r with means jittered ±2.0 σ | 0.812 | 72% |
| a completely different, resampled r | 0.734 | 60% |

Even a wholly wrong relevance lands at 0.734, next to unweighted ISD's
0.772. **A bad relevance estimate costs the method its advantage; it does
not make it worse than the baseline.** That is a favourable property and it
is worth stating, because it is the property that makes the scheme safe to
deploy before it is well tuned.

## S1 — Numerical conditioning of the criterion

The closed form is a sum of six terms that nearly cancel. A good merge is
one where p−q is almost zero, so the criterion is a small number assembled
from large ones. Over 200 random mixtures, the cancellation ratio
Σ|terms|/|result| on the pair the greedy step actually selects has median
2.3e5, p90 5.8e7 and worst case 3.7e14 — about 1.4 usable digits left in the
worst case. Against 60-digit arithmetic the winning cost carries relative
error up to 1.5e-8.

It does not change any decision. The argmin agreed with exact arithmetic in
100% of the cases checked, and a 1e-13 relative jitter flipped the choice in
0% of 400 trials. The gap between the best and second-best pair is far
larger than the noise. **The precision loss is real and does not matter.**

## S1c — Where it does matter: the underflow cliff

Push the relevance away from the message, which is exactly what a
downstream-aware weight is designed to do, and the criterion falls off a
cliff. With a message of standard deviation 1.5 and a narrow relevance:

| relevance offset | costs exactly zero | pairs tied at the minimum | greedy pick matches exact ranking |
|---|---|---|---|
| 24 | 1.2% | 1.1 | 95% |
| 30 | 22.7% | 10.2 | **12%** |
| 40 | 82.2% | 37.0 | 2% |
| 60 | 96.9% | 43.6 | 2% |

Past about 20 message-σ the weighted merge costs underflow to exactly zero
and the greedy step picks whichever pair `numpy` enumerates first. It fails
silently — nothing raises, the mixture is still reduced, the answer is
arbitrary. This is a latent hazard rather than an observed failure: on the
test graphs the relevance peak stays within 1.3 σ of the message it weights
(S2), so the cliff is never reached. It would be reached by a chain long
enough for the backward recursion of S10 to walk the relevance away, or by
any setting where the downstream message genuinely lives in a distant tail —
which is the case the note motivates the whole method with.

The same failure is already present in the repository's shipped code:
`GaussianMixture.__mul__` in `src/gmbp/core/distributions.py` catches the
all-products-underflow case and returns `GaussianMixture([1.0],
[self.components[0]])` — an arbitrary single component — with no signal.

## S2 — The relevance field is itself a mixture, and it blows up

λ = r² squares an n-component message into n². The damped update of Eq. (49)
concatenates the old and new fields, doubling again. On the note's graph the
downstream product r reaches 36 components, so λ reaches **1299 components
before any cap, while the message it governs is being held to 6**. Algorithm 1
caps the message and says nothing about capping λ. It has to be capped, and
the note does not say how.

Capping it turns out to be cheap in accuracy. Weight-pruning λ to 40 and then
greedy-merging to 6 leaves the merge ordering essentially intact: Kendall τ
against the ranking under the exact λ has median 1.000 and minimum 0.962, and
the merge actually chosen is the same in 100% of reductions. Pruning by peak
contribution instead of by mixture weight — the theoretically better choice,
since λ is a weight function and not a density — changed no measured result.

It is not cheap in arithmetic. See the cost line in S13.

## S10 — The backward relevance recursion degenerates with depth

Eq. (39)–(40) propagate relevance backward as λ_k = r_{k+1}² λ_{k+1}. Each
factor is a squared density, and a product of K Gaussians of variance v has
variance v/K. So the relevance narrows the further back it is carried.
Measured on a chain of length 8, capping λ to 6 components at every step:

| depth | λ width (sd) | components before the cap | uncapped count | share of the message λ can see |
|---|---|---|---|---|
| 1 | 1.638 | 1600 | 1.6e3 | 85.9% |
| 2 | 1.000 | 9600 | 2.6e6 | 66.5% |
| 4 | 0.553 | 9600 | 6.6e12 | 44.9% |
| 6 | 0.390 | 4374 | 7.6e18 | 45.3% |
| 8 | 0.321 | 54 | 5.6e21 | 35.4% |

The last column is the fraction of the message lying where λ is above 1% of
its peak. By depth 8 the criterion is ranking merges over roughly a third of
the message it is reducing and is near-blind to the rest — and outside that
window the costs are the near-zero numbers that S1c shows going to exactly
zero. The uncapped component count reaches 5.6e21, so the recursion cannot be
run as written past a step or two regardless.

The single-step estimate used elsewhere in this work does not have this
problem, because it is rebuilt from the current messages each sweep rather
than accumulated. **Eq. (40) is the part of the proposal that does not
survive; the single-step weight of Eq. (18) is the part that does.**

## S5 — Does the coupled iteration settle?

Reduction depends on the relevance, the relevance depends on the messages,
the messages depend on the reduction. The note's only control on that loop is
the damping α of Eq. (49), and nothing in the note argues the coupled map
contracts. Over 20 sweeps, measuring how far the beliefs still move between
consecutive sweeps at the end:

| | belief motion | relevance still moving |
|---|---|---|
| unreduced reference | 6.8e-06 | — |
| Runnalls, budget 6 | 6.6e-06 | — |
| adaptive, α = 0.1 | 1.0e-05 | 6.8e-03 |
| adaptive, α = 0.25 | 7.3e-06 | 4.9e-03 |
| adaptive, α = 0.5 | 1.3e-05 | 1.9e-03 |
| adaptive, α = 0.75 | 1.4e-04 | 1.7e-03 |
| adaptive, α = 1.0 | 7.5e-06 | 1.0e-05 |

No instability at any α. The residual motion is within an order of magnitude
of the unreduced reference's own.

The interesting column is the second one. **Damping does not stabilise
anything here; it just stops the relevance from converging.** At α = 1, which
is no damping at all, the relevance settles to 1.0e-05. At α = 0.1 it is
still moving by 6.8e-03 after twenty sweeps, two and a half orders of
magnitude more, and the beliefs are no better for it. On this graph α is a
parameter with a cost and no benefit.

## S6 — τ_react is a threshold on a quantity with no scale

Eq. (57)–(58) reactivate a stored hypothesis when
ρ_j = ∫λ(x) w_j N(x; μ_j, Σ_j) dx exceeds τ_react. ρ carries the scale of λ,
and λ is a squared BP message. BP messages are defined only up to a positive
constant, and **every reduction decision is invariant to rescaling λ while ρ
is not**. Rescaling is not a hypothetical: any implementation must pick a
normalisation for λ, and the note does not specify one.

| λ normalised by | ρ p1 | ρ median | ρ p99 | dynamic range | fraction above τ = 1 |
|---|---|---|---|---|---|
| nothing | 4.7e-20 | 2.6e-04 | 5.7e-02 | 1.2e18 | 0% |
| its mass | 8.8e-22 | 5.4e-04 | 1.1e-01 | 1.2e20 | 0% |
| its peak | 5.1e-21 | 9.3e-03 | 8.7e+02 | 1.7e23 | 13% |

Nor does one τ transfer between problems. At τ = 1, with λ peak-normalised,
the fraction of the reserve eligible for reactivation across five seeds of the
*same graph family* is 13%, 1%, 5%, 7%, 4%. The reserve itself grows without
bound: 48, 456, 864, 1272, 1680, 2088 stored components over six sweeps, and
scoring every one of them against λ on every sweep is the cost of keeping it.

## S8 — The reserve set and reactivation: the part that breaks

The reserve exists because of the note's §"Irreversibility of Mixture
Reduction": a hypothesis deleted at iteration t cannot be reconstructed
later. So test the premise directly. Delete one component outright at sweep 3
— chosen as the weak hypothesis whose removal actually moves a belief, not
the least significant one — and run on with no reduction anywhere:

| seed | edge | weight deleted | TV to reference, from the deletion onward |
|---|---|---|---|
| 1 | f_C2_S2→S2 | 0.128 | 0.0537 → 0.0000 0.0031 0.0023 0.0001 0.0002 **0.0000** |
| 2 | f_S3_S4→S3 | 0.184 | 0.0743 → 0.0000 0.0058 0.0000 0.0002 0.0001 **0.0000** |
| 3 | f_C1_S1→S1 | 0.092 | 0.1827 → 0.0000 0.1748 0.0919 0.0062 0.0068 **0.0023** |
| 4 | f_C3_S2→S2 | 0.147 | 0.0753 → 0.0000 0.0225 0.0168 0.0003 0.0043 **0.0000** |
| 5 | f_C2_S4→S4 | 0.149 | 0.0580 → 0.0477 0.0000 0.0026 0.0000 0.0006 **0.0000** |

**The hypothesis comes back.** Deleting 9–18% of a message's mass costs
essentially nothing after a few sweeps. On this graph a factor→variable
message is a function of the factor potential and the current class message,
so the deleted mode is simply rebuilt on the next sweep. The premise of the
reserve does not hold here, and the note gives no condition under which it
does.

And the mechanism as specified is not merely useless, it is destructive:

| variant | mean TV gap | × no reserve | reactivations |
|---|---|---|---|
| no reserve | 0.00394 | 1.00× | 0 |
| Eq. (57)–(58) as written, τ = 0.01 | 0.08974 | **22.8×** | 1202 |
| as written, τ = 0.1 | 0.10311 | **26.2×** | 645 |
| as written, τ = 1 | 0.02004 | 5.1× | 221 |
| repaired (weight rescaled, duplicates suppressed), τ = 1 | 0.00479 | 1.22× | 4 |
| scale-free relative threshold, k = 10 | 0.01349 | 3.4× | 45 |

Two defects are obvious once you implement it: the stored weight comes from
an older, differently normalised mixture, and the mixture usually already
carries the hypothesis being reinserted. Repairing both helps a lot. So does
replacing τ_react with a scale-free rule — reactivate only when a reserve
component scores above the least relevant component currently being carried —
which removes the S6 objection.

Neither rescues it. The relationship is monotone in the wrong direction:
across every variant, the fewer reactivations, the closer to no reserve at
all, and no setting beats zero. **There is no operating point at which the
reserve helps.**

## S9 — On a chain, where the note's premises are supposed to hold

The note's graph has a regeneration path: a factor→variable message is a
function of the potential and the current class message, so a deleted mode
returns. A continuous chain has none — the message is the only carrier of
history. That is the structure the irreversibility argument assumes, so
repeat the tests there. The reference is Runnalls at budget 150; raising it
to 300 moves the reference by TV 1.6e-11, which bounds what the stand-in
contributes.

**Deletion.** Removing a component carrying 10–20% of a message decays to a
median 9.0% of its immediate impact after four more sweeps, against 0.0% on
the note's graph. So permanence is real on a chain and about fifty times
stronger relative to impact — and still decaying, because the transition
kernel re-spreads mass and the far end's evidence flows back. Even here,
reduction is not the one-way door the note describes.

**The reserve.** Neutral when it does nothing, catastrophic when it acts:

| variant | mean TV gap | × no reserve | reactivations |
|---|---|---|---|
| no reserve | 0.00091 | 1.00× | 0 |
| as written, τ = 0.1 | 0.05672 | **62.5×** | 526 |
| as written, τ = 1 | 0.00161 | 1.77× | 148 |
| repaired or scale-free, any setting | 0.00091 | 1.00× | 0 |

**The principal hypothesis** still holds, but the margin is much thinner
than on the loopy graph:

| method | mean gap | median ratio to Runnalls | beats Runnalls |
|---|---|---|---|
| weight pruning | 0.06467 | 44.46 | 0% |
| Runnalls | 0.00236 | 1.000 | — |
| unweighted ISD | 0.00251 | 0.989 | 56% |
| adaptive | 0.00219 | 0.929 | 83% |
| adaptive, α = 1 | 0.00218 | 0.917 | 89% |

0.92 here against 0.69 on the note's graph. S12 explains the difference:
λ's dynamic range over the message is 9.5 on the chain and about 6700 on the
loopy graph, because a class-conditional product is structured and a
diffusion message is a smooth bump. **The size of the effect is a property of
the graph, not of the method** — it tracks how much structure the downstream
message has.

## S4 — The principal hypothesis

The note's central claim is D(b_ref, b_adaptive) < D(b_ref, b_local) at equal
budget. Seventy-two configurations — six seeds × four observation regimes ×
three budgets — six sweeps each, every run compared against an unreduced
reference *at the same sweep*, which is the comparison the note insists on
and which its own earlier draft got wrong.

Sensor beliefs, mean over sweeps:

| method | mean gap | median ratio to Runnalls | beats Runnalls |
|---|---|---|---|
| weight pruning | 0.15728 | 14.77 | 0% |
| Runnalls' bound | 0.02493 | 1.000 | — |
| unweighted ISD | 0.02598 | 0.932 | 68% |
| **adaptive** | **0.01914** | **0.689** | **99%** |
| adaptive, backward pass off | 0.01912 | 0.698 | 99% |
| adaptive, α = 1 (no damping) | 0.01810 | 0.681 | 99% |

At the final sweep the adaptive ratio is 0.572 and it wins 96% of
configurations. **The principal hypothesis holds, and not marginally.** It is
also not a wash across regimes: 99 of 100 configurations is not a mean
dragged by outliers.

Weight pruning being fifteen times worse than the merge criteria reproduces
the note's own observation that a linear colour scale flattens everything
else.

### The late-evidence case

The note singles out the case "when weak local hypotheses become strongly
supported by later information". Eight seeds, a weak distant mode planted in
one sensor's conditionals, and the evidence for it arriving at sweep 3:

| method | mean gap | median ratio | beats Runnalls | final sweep ratio | final sweep wins |
|---|---|---|---|---|---|
| Runnalls | 0.01312 | 1.000 | — | 1.000 | — |
| unweighted ISD | 0.01192 | 0.986 | 50% | 1.038 | 38% |
| adaptive, α = 0.5 | 0.01410 | 0.808 | 62% | 0.602 | 62% |
| **adaptive, α = 1** | **0.00818** | **0.554** | **75%** | **0.101** | **100%** |

Note the first column against the third. At α = 0.5 the *mean* gap is worse
than Runnalls while the *median ratio* is 0.808 — a handful of configurations
where the damped method is much worse than the baseline. Undamped, that
disappears: the final-sweep gap is a tenth of Runnalls' and it wins every
seed.

This is the sharpest result on damping. Eq. (49) exists to let the relevance
settle; what it actually does is make the relevance lag exactly when new
information arrives, which is the only moment the method has any advantage to
offer.

## S11 — Is the downstream information doing the work?

A weighted ISD with any non-uniform weight is already a different criterion
from Runnalls' bound, so the control that matters is a relevance field that
is real in shape but wrong in placement. Thirty-six configurations:

| relevance | mean gap | median ratio to Runnalls | beats Runnalls | beats the true relevance |
|---|---|---|---|---|
| Runnalls (local) | 0.03544 | 1.000 | — | 0% |
| uniform, i.e. plain ISD | 0.03784 | 0.991 | 58% | 6% |
| a fixed Gaussian, never updated | 0.03642 | 0.941 | 61% | 17% |
| each edge given another edge's relevance | 0.03350 | 0.972 | 58% | 17% |
| true, on variable→factor messages only | 0.03715 | 0.967 | 56% | 6% |
| **true, on factor→variable messages only** | **0.02778** | **0.785** | **100%** | 33% |
| true, on every edge | 0.02772 | 0.784 | 100% | — |

Two things fall out. **The downstream information is doing the work**: a
wrong relevance of the right shape lands at 0.94–0.99 of Runnalls, the right
one at 0.78, and it wins every single configuration. And **all of the benefit
sits on one class of edge**. Weighting the reduction of a message arriving at
a variable node — where the exact single-step weight of Eq. (18) applies,
because the message is about to be multiplied by the other incoming messages
and nothing else — recovers the full effect. Weighting messages leaving a
variable node, which is where the backward pull-through through a factor
lives, is worth 0.967 against Runnalls, indistinguishable from the
wrong-relevance controls.

## S12 — How often does the relevance change a decision?

At every reduction the engine performs, compare the pair chosen under λ with
the pair chosen under a uniform weight:

| graph | reductions | pick differs from plain ISD | differs from Runnalls | Kendall τ vs ISD | λ dynamic range |
|---|---|---|---|---|---|
| the note's graph, medium obs | 468 | 46% | 55% | 0.868 | 6689 |
| the note's graph, no obs | 468 | 43% | 54% | 0.898 | 4275 |
| chain, both ends observed | 360 | 29% | 54% | 0.914 | 9.5 |

The last column matters. λ's dynamic range over the message's components is
three orders of magnitude larger on the note's graph than on the chain,
because a class-conditional product is structured and a diffusion message is
a smooth bump. Where λ is nearly flat the weighted criterion degenerates into
the unweighted one, which is why the chain shows a smaller effect than the
loopy graph.

An earlier version of the chain observed only one end. Every message
travelling the other way then carried no information, λ was flat by
construction, and the adaptive method agreed with plain ISD to eight
significant figures. That is worth recording because it is the failure mode
of the whole idea stated cleanly: **downstream-aware reduction has nothing to
offer where there is no downstream information**, and it degrades to the
unweighted criterion rather than to something worse.

## S7 — Against an oracle, inside the real loop

S3 scored the weight on a synthetic one-step downstream. This is the version
with nothing held back: at an actual reduction event in the loopy graph, take
the candidate merges, carry each one through to the end of inference, and
measure the damage it really did. The message is pre-reduced to 7 components
by Runnalls so that the 21 pairs taking it to the budget of 6 are genuinely
different reductions; each criterion is then scored against that ground
truth, with the relevance computed exactly as the method computes it —
capped, damped and stale.

| horizon (sweeps after the merge) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| spread of true damage | 3.4e-02 | 0 | 1.0e-02 | 4.3e-03 | 1.8e-03 | 9.1e-04 | 5.8e-05 | 1.7e-04 |
| **λ = r² (Kendall τ)** | **0.643** | — | **0.514** | **0.505** | **0.395** | **0.290** | **0.243** | **0.271** |
| unweighted ISD | 0.386 | — | 0.324 | 0.267 | 0.190 | 0.138 | 0.086 | 0.129 |
| Runnalls' bound | 0.371 | — | 0.319 | 0.276 | 0.190 | 0.157 | 0.086 | 0.176 |

**λ = r² is the better predictor at every horizon where there is anything to
predict**, and by a wide margin: 0.643 against 0.386 and 0.371 immediately,
0.51 against 0.32 two sweeps out. This is the cleanest statement of what the
weight buys, because it is measured against the actual objective rather than
a surrogate, in the loopy graph, with every one of the derivation's
assumptions violated at once.

Two things to read off the first row. Horizon 1 is empty because of the
flooding schedule: a change to a factor→variable message needs one sweep to
reach the class messages and a second to come back through them to the sensor
beliefs. And the spread decays by a factor of 200 from horizon 0 to horizon 7
— **a single reduction decision is largely forgotten within a few sweeps**,
which is the same fact S8a establishes from the other direction and the
reason the reserve set has so little to protect.

## S13 — Removing the apparatus

S4, S5, S6, S8, S9 and S11 each removed one piece. This removes them
together and asks what is left. "Relative cost" is pair-cost evaluations
against Runnalls.

The note's graph, 45 configurations:

| variant | mean gap | median ratio to Runnalls | beats Runnalls | relative cost |
|---|---|---|---|---|
| Runnalls' bound | 0.02319 | 1.000 | — | 1.0× |
| unweighted ISD | 0.02467 | 0.932 | 67% | ~2× |
| **Algorithm 1 as written** | **0.11405** | **16.16** | **0%** | 22.7× |
| minus the reserve | 0.01775 | 0.690 | 100% | 16.3× |
| minus reserve and damping | 0.01622 | 0.671 | 100% | 16.2× |
| **single-step rule only** | **0.01623** | **0.677** | **100%** | **4.5×** |

The late-evidence graph, 8 configurations:

| variant | mean gap | median ratio | beats Runnalls | relative cost |
|---|---|---|---|---|
| Runnalls' bound | 0.01312 | 1.000 | — | 1.0× |
| unweighted ISD | 0.01192 | 0.986 | 50% | ~2× |
| **Algorithm 1 as written** | **0.12961** | **9.03** | **0%** | 17.2× |
| minus the reserve | 0.01410 | 0.808 | 62% | 12.8× |
| minus reserve and damping | 0.00818 | 0.554 | 75% | 12.7× |
| **single-step rule only** | **0.00824** | **0.547** | **75%** | **4.7×** |

Read the two bold rows in each table together. **Algorithm 1 as specified is
nine to sixteen times worse than the baseline it is meant to beat, at
seventeen to twenty-three times the cost, and it loses every single
configuration.** Strip the reserve, the damping and the backward
propagation, and the remainder wins 75–100% of configurations at 0.55–0.68
of Runnalls' error and a quarter of the full method's cost.

The remainder is one rule with no state, no schedule and no parameters:

> When reducing a message a→b, weight the integral squared difference by
> λ(x) = ( ∏_{c ≠ a} m_{c→b}(x) )², rebuilt from the current messages.

That is Eq. (18) applied where it is exactly true, and nothing else.

## Cost

Pair-cost evaluations for six sweeps on the note's graph, and wall clock over
three seeds:

| method | pair evaluations | × Runnalls | wall clock | × Runnalls |
|---|---|---|---|---|
| Runnalls' bound | 236,130 | 1.0× | 0.109 s | 1.0× |
| unweighted ISD | — | — | 0.207 s | 1.9× |
| adaptive, no reserve | 2,146,012 | **9.1×** | 1.279 s | 11.7× |
| adaptive with the reserve | — | — | 1.944 s | 17.8× |

The pair-evaluation ratio is the implementation-independent number; it is a
lower bound, since it does not count building λ. The note dismisses the exact
divergence at 30× as impractical. Nine to twelve times is in the same
territory of objection, and the note's own framing — that a learned weight
would "recover the gap at the cheap criterion's cost" — is not met by the
non-learned scheme. Six of the 9.1× is structural: the weighted criterion
costs K times the unweighted one for a K-component relevance.

## What to change

In descending order of how much the measurements support it.

**1. Delete the reserve set and the reactivation rule.** Eqs. (55)–(58) make
the answer 5–62× worse on every graph and every setting tested, and no
repair rescues them (S6, S8, S9). The premise fails too: on the note's own
graph a deleted hypothesis carrying 9–18% of a message's mass is recovered
within a few sweeps, because a factor→variable message is rebuilt from the
potential each sweep. If the section is kept, it needs a stated condition
for when a graph has no regeneration path — and even on a chain, which is
the structure the argument assumes, the reserve is neutral at best.

**2. Delete the damping.** α exists to let the relevance settle; measured, it
prevents the relevance from settling (S5) and costs the most in exactly the
case the note advertises — evidence arriving late, where α = 0.5 wins 62% of
seeds and α = 1 wins 100% with a tenth of the error (S4b). Set α = 1 and drop
the parameter.

**3. Delete the backward relevance propagation, and restate Eq. (40).** The
single-step rule matches the full apparatus (0.677 against 0.671) at a
quarter of the cost (S13), and relevance on variable→factor edges is worth
0.967 against Runnalls — indistinguishable from a deliberately wrong
relevance (S11). The recursion of Eq. (40) narrows λ with depth until it
sees a third of the message and multiplies components to 5.6e21 (S10). It
should be presented as the exact chain case it is, not as a mechanism.

**4. Guard the underflow.** Once λ sits more than about 20 message-σ away,
every weighted merge cost is exactly zero and the greedy step picks
arbitrarily, silently (S1c). Compute the criterion in log space with signed
accumulation, or at minimum detect an all-zero cost vector and fall back to
the unweighted criterion. The same bug is live in
`src/gmbp/core/distributions.py`.

**5. Say how λ is reduced.** It reaches 1299 components against a message
budget of 6 (S2). Capping is cheap in accuracy — Kendall τ 1.000 against the
uncapped ranking — and it is not optional.

**6. Restate the cost.** 4.5× Runnalls for the stripped rule, 17–23× as
written (S13). The note dismisses the exact divergence at 30× as impractical;
the same objection applies here and should be met head-on rather than left to
"at the cheap criterion's cost".

## What survives, and what it is worth

The core claim survives cleanly and is worth more than the note asserts for
it. Weighting the integral squared difference by the square of what a message
is about to be multiplied by:

* ranks merges by their true downstream damage at Kendall τ 0.863 against
  0.772 for unweighted ISD and 0.763 for Runnalls, even after normalisation
  and marginalisation break the derivation (S3);
* degrades gracefully to the baseline rather than below it when the relevance
  is wrong (S3, S11);
* beats Runnalls in 99–100% of 72 configurations at a median 0.69 of its
  error, 0.57 at the final sweep (S4);
* and the gain is attributable to the downstream information specifically —
  a shuffled or fixed relevance of the same shape gets 0.94–0.99, the true
  one gets 0.78 and wins every configuration (S11).

The size of the effect is a property of the graph rather than the method. It
tracks how much structure the downstream message has: λ's dynamic range over
the message is ~6700 on the note's loopy graph and 9.5 on a diffusion chain,
and the margin over Runnalls follows, 0.69 against 0.93 (S9, S12). Where
there is no downstream information at all the method degenerates to the
unweighted criterion, which is the right failure mode.

**The strongest version of this work is a short paper about Eq. (18) used as
a reduction weight, with Eqs. (39)–(60) and Algorithm 1 removed.** As it
stands the note's central section proposes a machine that is measurably worse
than the baseline, wrapped around an idea that is measurably better.

## Caveats

* Two graph families, both small, both with one-dimensional continuous
  variables. The unreduced reference is what makes the comparison meaningful
  and it is what limits the size.
* The backward relevance rule through a factor is not specified in the note.
  The class-side analogue used here is stated in "What was built"; a different
  choice could change the S11 variable→factor row, though not the S13 result,
  since the single-step rule uses no backward pass at all.
* Budgets of 4, 6 and 8 against unreduced references of up to 729 components.
  Behaviour at larger budgets, where reduction is gentler, is not measured.
* "Beats Runnalls in N% of configurations" counts configurations, not
  independent problems; six seeds are shared across regimes and budgets.
