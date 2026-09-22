"""The composite triage score.

This is the one place in Sorbent where judgement enters, so it is also the one
place that must be fully transparent: every score ships with its breakdown, and
the weights are caller-supplied.

**What this is not.** It is not a predicted activity, a predicted affinity, or
a probability of anything. Nothing here was fitted to an assay. It is a
weighted mean of four computed quantities, and its only job is to impose a
defensible order on a list so a chemist looks at the top of it first. Two
molecules a tenth of a point apart are not meaningfully different.

Components, each normalised to [0, 1] before weighting:

  rule_compliance      fraction of requested rule sets passed. 1.0 when none
                       were requested - no evidence against is not evidence
                       for, but penalising a molecule for a question nobody
                       asked would be worse.
  alert_penalty        1 / (1 + n_alerts). Sharp drop for the first alert,
                       diminishing after, which matches how a chemist reads an
                       alert list: one PAINS hit changes your mind, the sixth
                       does not change it further.
  property_centrality  product of Gaussians on MW / clogp / TPSA around
                       lead-like targets. Rewards the middle of the range
                       rather than merely being inside it - a molecule at
                       MW 499 passes Lipinski and is still a bad starting
                       point, because there is nowhere left to grow.
  complexity_penalty   unassigned stereocentres and ring count. Both are
                       purchasing and synthesis risk rather than statements
                       about the chemistry.

    score = exp( sum(weight_i * ln(component_i)) / sum(weight_i) )

a weighted *geometric* mean, which stays in [0, 1] for any non-negative
weights since every component does. Evaluated in log space rather than as
prod(c**w)**(1/sum w), because the product underflows to 0.0 once a few
components are small and the logs do not.

Geometric, not arithmetic, and deliberately: a component near zero should sink
the total rather than be averaged away. Under an arithmetic mean water scored
0.821 - it passes Lipinski and Veber (both upper bounds only), trips no alert,
and has no stereocentre or ring, so three of four components read 1.00 and the
fourth was outvoted. Geometrically it scores 0.465, benzene 0.596, and the six
highest-scoring molecules in a mixed test set are all real drugs.

**A component of exactly zero is close to a veto, which is why
``rule_compliance`` is not weighted by default.** It is 0.0 whenever every
requested rule set fails, and being a fraction its resolution is set by how
many rule sets were asked for - two give {0, 0.5, 1}, one gives {0, 1} with no
middle ground. Weighted, that buried marketed drugs:

                        weighted    unweighted (the default)
    diazepam              0.8318        0.7487
    aspirin               0.7357        0.6174
    atorvastatin          0.0022        0.1744
    erythromycin          0.0007        0.0295
    benzene               0.3547        0.1962
    water                 0.2164        0.0902

It cuts both ways. Dropping the component lifts molecules that fail their
rules, and *lowers* trivially small ones - water and benzene were being handed
a free 1.00 for passing rules they cannot fail. Roughly a third of marketed
oral drugs violate Ro5 and every macrolide does; none of them belong below
water.

Note what this does NOT rest on. An earlier version of this file argued that
most peptidomimetics fail Veber on TPSA or rotatable bonds. Measured over
206,922 of them, 96.9% pass. The default is right; that reasoning for it was
not.

**Rule results are still computed and reported on every molecule.** Dropping
the weight drops the ranking influence, not the evidence: a chemist still sees
which rules a compound broke and by how much. Put ``rule_compliance`` back into
``score_weights`` when compliance genuinely is the ranking criterion - a
permeability-focused oral series, say - and know that you are accepting the
veto when you do.

Note also that *trimming* rule_sets does not lift a molecule that fails the
ones that remain: rule_compliance is a fraction, so 0/2 and 0/1 are both 0.0.
Erythromycin scored 0.0021 under [lipinski, veber] and 0.0021 under [veber].
Only removing the component helps.

Flooring the components before the log was tried as a fix and does not work.
Measured across floors of 1e-6, 0.02, 0.05, 0.10 and 0.20, raising the floor
lifts the rule-failing drugs but lifts water further, because water passes
everything except property_centrality and so benefits from every floor:

    floor     water    erythromycin
    1e-6      0.465    0.002
    0.05      0.580    0.195
    0.20      0.746    0.416

At no floor does a Ro5-failing real drug outrank water. So EPSILON below is
set just high enough to keep ln() finite and nothing more.
Only the components the caller weights are computed, so the breakdown explains
exactly the number beside it and no descriptor is demanded for a component
nobody asked for.

**property_centrality carries the weight, because it is the only component
that discriminates.** Measured over 15,000 real compounds:

                          p10     p25  median     p75     p90   at 1.00
    property_centrality  0.426   0.639   0.799   0.900   0.955     0.1%
    alert_penalty        0.500   1.000   1.000   1.000   1.000    87.7%
    complexity_penalty   0.741   0.741   1.000   1.000   1.000    59.1%

Two of the three sit at 1.00 for most molecules and contribute nothing except
to inflate sum(w), which shrinks the exponent on the one component that does
vary. Underweighted at 0.5 of 1.75, the score collapsed to centrality**0.286
and put 53.5% of a 206,922-compound library above 0.90 - not wrong, but
unreadable, and impossible to threshold on.

Weighting centrality above the other two together also fixed something
flooring could not: water fell from 0.465 to 0.090, below atorvastatin at
0.174. Be clear about what that did and did not change, though - the reordering
is small. Spearman against the old weights is 0.971 and the top hundred is
identical. This makes the number legible; it does not materially change who
comes first.

Trivially small molecules are still best removed before scoring rather than
ranked low, with a ``descriptor_windows`` minimum on ``molecular_weight``
around 150-200, which ``finalize`` applies as a hard filter. The score orders a
list; it does not filter one.

The constants below are declared rather than buried because they are the
judgement, and somebody tuning this service for a fragment campaign rather
than a lead-optimisation one should change them and say so.
"""

from __future__ import annotations

import math

#: Weights live in TriageConfig.score_weights, not here; these are the targets
#: the centrality component measures distance from.
#:
#: Targets for property_centrality. Lead-like, deliberately: Teague's argument
#: is that screening hits grow during optimisation, so a good starting point
#: sits below the Lipinski ceiling rather than against it.
TARGET_MW = 350.0
TARGET_CLOGP = 2.5
TARGET_TPSA = 75.0

#: Gaussian widths. Chosen so that a molecule at the Lipinski/Veber limit
#: scores roughly 0.4-0.6 on that axis - clearly penalised, not zeroed.
SIGMA_MW = 150.0
SIGMA_CLOGP = 2.0
SIGMA_TPSA = 50.0

#: complexity_penalty. Four rings is unremarkable in a drug, so only the fifth
#: and beyond count. An unassigned stereocentre costs more than an extra ring
#: because it is a procurement problem, not just a synthesis one: you cannot
#: order a defined single enantiomer that the vendor has not defined.
COMFORTABLE_RINGS = 4
STEREO_PENALTY = 0.35
RING_PENALTY = 0.20

#: Floor applied to a component before taking its log. Only exact zeros need
#: it - rule_compliance is 0.0 whenever every requested rule fails, and ln(0)
#: is -inf. At 1e-6 it sits three orders of magnitude below the smallest value
#: any component produces for a real molecule (property_centrality bottoms out
#: near 1e-3), so it distorts nothing and merely bounds how far one component
#: can veto the rest.
EPSILON = 1e-6

#: Every component this module knows how to compute. A weight naming anything
#: else is a typo, and a typo that silently changed a ranking would be the
#: worst kind of bug in a file like this.
COMPONENTS: tuple[str, ...] = (
    "rule_compliance",
    "alert_penalty",
    "property_centrality",
    "complexity_penalty",
)


def _gaussian(value: float, target: float, sigma: float) -> float:
    """Unit-height Gaussian: 1.0 at the target, decaying either side."""
    return math.exp(-(((value - target) / sigma) ** 2) / 2.0)


def _require(desc: dict[str, float | int], key: str, component: str) -> float:
    if key not in desc:
        raise KeyError(f"{component} needs descriptor {key!r}; got {sorted(desc)}")
    value = float(desc[key])
    if not math.isfinite(value):
        raise ValueError(f"{component}: descriptor {key!r} is {value!r}, not a finite number")
    return value


def rule_compliance(rules: list[dict[str, object]]) -> float:
    """Fraction of rule sets passed, in [0, 1]. Returns 1.0 if none requested."""
    if not rules:
        return 1.0
    return sum(1 for rule in rules if rule["passed"]) / len(rules)


def alert_penalty(n_alerts: int) -> float:
    """1.0 for a clean molecule, decaying with the number of alerts.

    Reciprocal rather than exponential on purpose: 1 -> 0.50, 2 -> 0.33,
    3 -> 0.25, 6 -> 0.14. The first hit costs half, and a molecule already
    carrying five is barely punished further, which is how an alert list
    actually reads.
    """
    if n_alerts < 0:
        raise ValueError(f"n_alerts must not be negative, got {n_alerts}")
    return 1.0 / (1.0 + n_alerts)


def property_centrality(desc: dict[str, float | int]) -> float:
    """Product of Gaussians on MW / clogp / TPSA around the lead-like targets.

    A product, not a mean: being badly wrong on one axis should sink the
    component regardless of the other two, the way a chemist would read it.
    """
    return (
        _gaussian(_require(desc, "molecular_weight", "property_centrality"), TARGET_MW, SIGMA_MW)
        * _gaussian(_require(desc, "clogp", "property_centrality"), TARGET_CLOGP, SIGMA_CLOGP)
        * _gaussian(_require(desc, "tpsa", "property_centrality"), TARGET_TPSA, SIGMA_TPSA)
    )


def complexity_penalty(desc: dict[str, float | int]) -> float:
    """1.0 for a tractable molecule, lower for stereo-heavy or ring-heavy ones.

    Only rings beyond COMFORTABLE_RINGS count, so ordinary drugs are not
    penalised for having a scaffold.
    """
    unassigned = _require(desc, "unassigned_stereocentres", "complexity_penalty")
    rings = _require(desc, "rings", "complexity_penalty")
    excess_rings = max(0.0, rings - COMFORTABLE_RINGS)
    return 1.0 / (1.0 + STEREO_PENALTY * unassigned + RING_PENALTY * excess_rings)


def composite_score(
    desc: dict[str, float | int],
    rules: list[dict[str, object]],
    n_alerts: int,
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Return (score, breakdown).

    ``breakdown`` holds the *unweighted* value of every component that was
    weighted, so a caller can see why a molecule scored the way it did and
    reproduce the arithmetic from the weights in the job config.

    Raises on an unknown component name, a negative weight, or weights summing
    to zero. All three are caller mistakes that would otherwise quietly change
    a ranking: a typo drops a component, a negative weight inverts its meaning
    and breaks the [0, 1] guarantee, and a zero total has no defined answer.
    """
    unknown = set(weights) - set(COMPONENTS)
    if unknown:
        raise KeyError(f"unknown score component(s) {sorted(unknown)}; known: {list(COMPONENTS)}")
    negative = {name: w for name, w in weights.items() if w < 0}
    if negative:
        raise ValueError(f"score weights must not be negative, got {negative}")

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("score weights must include at least one positive value")

    # Computed lazily so a component nobody weighted costs nothing and demands
    # no descriptor - the weights decide what this molecule is asked for.
    available = {
        "rule_compliance": lambda: rule_compliance(rules),
        "alert_penalty": lambda: alert_penalty(n_alerts),
        "property_centrality": lambda: property_centrality(desc),
        "complexity_penalty": lambda: complexity_penalty(desc),
    }

    breakdown = {name: available[name]() for name in COMPONENTS if name in weights}

    # Weighted GEOMETRIC mean, evaluated in log space. exp(sum(w*ln c)/sum w)
    # rather than prod(c**w)**(1/sum w): the product underflows to 0.0 once a
    # few components are small, and the logs do not.
    log_total = sum(
        weights[name] * math.log(max(value, EPSILON)) for name, value in breakdown.items()
    )
    score = math.exp(log_total / total_weight)
    # Guard against float drift at the boundaries rather than emitting
    # 1.0000000000000002 into a response model annotated as a probability-like
    # quantity.
    return min(1.0, max(0.0, score)), breakdown
