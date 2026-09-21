"""The composite triage score.

This is the one place in Winnow where judgement enters, so it is also the one
place that must be fully transparent: every score ships with its breakdown, and
the weights are caller-supplied.

Suggested components, each normalised to [0, 1] before weighting:

  rule_compliance      fraction of requested rule sets passed.
  alert_penalty        1 / (1 + n_alerts). Sharp drop for the first alert,
                       diminishing after - which matches how a chemist reads
                       an alert list.
  property_centrality  how central the molecule sits in lead-like space.
                       A Gaussian on each of MW / clogp / tpsa around a target
                       (say 350 / 2.5 / 75), multiplied together. Rewards the
                       middle of the range rather than merely being inside it.
  complexity_penalty   penalise many unassigned stereocentres and very high
                       ring counts - both are synthesis and purchasing risk.

Score = sum(weight_i * component_i) / sum(weight_i), so it stays in [0, 1]
regardless of what weights the caller passes.

Be honest in the docs about what this is NOT: it is not a predicted activity,
a predicted affinity, or a probability of anything. It is a transparent
weighted sum of computed properties, useful for ordering a list.
"""

from __future__ import annotations

#: Defaults for property_centrality. Lead-like, deliberately.
TARGET_MW = 350.0
TARGET_CLOGP = 2.5
TARGET_TPSA = 75.0


def rule_compliance(rules: list[dict[str, object]]) -> float:
    """Fraction of rule sets passed, in [0, 1]. Returns 1.0 if none requested."""
    raise NotImplementedError


def alert_penalty(n_alerts: int) -> float:
    """1.0 for a clean molecule, decaying with the number of alerts."""
    raise NotImplementedError


def property_centrality(desc: dict[str, float | int]) -> float:
    """Product of Gaussians on MW / clogp / tpsa around the lead-like targets."""
    raise NotImplementedError


def complexity_penalty(desc: dict[str, float | int]) -> float:
    """1.0 for a tractable molecule, lower for stereo/ring-heavy ones."""
    raise NotImplementedError


def composite_score(
    desc: dict[str, float | int],
    rules: list[dict[str, object]],
    n_alerts: int,
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Return (score, breakdown).

    The breakdown must contain every component by name with its *unweighted*
    value, so a caller can see why a molecule scored the way it did.
    """
    raise NotImplementedError
