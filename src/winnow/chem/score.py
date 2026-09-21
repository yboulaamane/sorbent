"""The composite triage score.

This is the one place in Winnow where judgement enters, so it is also the one
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

    score = sum(weight_i * component_i) / sum(weight_i)

which stays in [0, 1] for any non-negative weights, since every component does.
Only the components the caller weights are computed, so the breakdown explains
exactly the number beside it and no descriptor is demanded for a component
nobody asked for.

**Known limitation: the score has a high floor, and that floor is structural.**
Three of the four components measure the *absence* of problems, and something
trivially small has no problems to measure. Water passes Lipinski and Veber
(both are upper bounds only), trips no structural alert, and has neither a
stereocentre nor a ring - so it scores 1.00, 1.00 and 1.00 on three components
and is rescued only by property_centrality, which at the default weight of 0.5
out of 2.75 is 18% of the total. Measured with the default weights and the
default rule sets:

    diazepam     0.928        <- a real drug
    ibuprofen    0.878
    caffeine     0.840
    benzene      0.829        <- not a lead
    water        0.821        <- not even a molecule anyone submitted
    aspirin      0.715
    erythromycin 0.455

Raising property_centrality to 2.0 moves water to 0.536, and adding Ghose -
the one rule set with lower bounds - moves it to 0.700. Neither fixes it,
because a weighted mean of "nothing is wrong" cannot go low for a molecule
with nothing wrong.

The fix is not to tune the weights. **This score orders a list; it does not
filter one.** Anything too small or too polar to be a starting point should be
removed before scoring, by ``descriptor_windows`` (a ``molecular_weight``
minimum of 150-200 is the usual choice) or by requesting a rule set with lower
bounds. ``finalize`` applies those windows as hard filters, and what reaches
this function is meant to be a set of plausible candidates already.

If a future version wants the score itself to be the filter, the aggregation
has to change from a weighted arithmetic mean to a weighted geometric one, so
that a near-zero component sinks the total instead of being averaged away.
That would put water at ~0.20 against diazepam's ~0.91. It is a deliberate
design change affecting every score the service emits, not a tuning tweak, so
it is not made here.

The constants below are declared rather than buried because they are the
judgement, and somebody tuning this service for a fragment campaign rather
than a lead-optimisation one should change them and say so.
"""

from __future__ import annotations

import math

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
    score = sum(weights[name] * value for name, value in breakdown.items()) / total_weight
    # Guard against float drift at the boundaries rather than emitting
    # 1.0000000000000002 into a response model annotated as a probability-like
    # quantity.
    return min(1.0, max(0.0, score)), breakdown
