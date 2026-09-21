"""Drug-likeness rule sets.

Each rule is a pure function of the descriptor dict - no molecule needed, which
means they are trivially testable and you can run them over a CSV of
precomputed properties without RDKit at all. This module imports no RDKit, and
it should stay that way.

The canonical definitions, so you do not have to look them up:

  Lipinski Ro5 (Adv Drug Deliv Rev 1997;23:3-25) - at most one violation of:
      MW <= 500, clogp <= 5, hbd <= 5, hba <= 10
      NB: the rule permits ONE violation. A strict four-of-four check is a
      common and wrong implementation.

  Veber (J Med Chem 2002;45:2615) - both required:
      rotatable_bonds <= 10, tpsa <= 140

  Egan (J Med Chem 2000;43:3867) - both required:
      tpsa <= 131.6, clogp between -1 and 5.88

  Ghose (J Comb Chem 1999;1:55) - all required:
      160 <= MW <= 480, -0.4 <= clogp <= 5.6, 20 <= total_atoms <= 70

  Lead-like (Teague, Angew Chem 1999;38:3743):
      MW <= 350, clogp <= 3.5, rotatable_bonds <= 7

  Fragment / Ro3 (Congreve, Drug Discov Today 2003;8:876):
      MW <= 300, clogp <= 3, hbd <= 3, hba <= 3, rotatable_bonds <= 3

Ghose's atom count INCLUDES HYDROGENS, and reading it as heavy atoms - which
is easy to do and common in the wild - inverts the filter. The two criteria
pin each other: a 160 Da molecule with 20 heavy atoms would need an average
heavy-atom mass of 8 Da, lighter than carbon, so the count cannot be
heavy-atoms-only. Measured, on heavy atoms the filter rejects aspirin (13),
paracetamol (11) and caffeine (14) while accepting atorvastatin (41) and
erythromycin (51); on total atoms it accepts the first three (21, 20, 24) and
rejects the last two (76, 118). The second is the drug-like set. Hence
``total_atoms`` in ``descriptors.py``.

Two deviations from the papers, both deliberate:

  Ghose additionally constrains molar refractivity to 40-130. Winnow does not
  compute MR, so that clause is silently absent - a molecule passing `ghose`
  here has met three of the four original criteria. Add ``Crippen.MolMR`` to
  ``descriptors.py`` and a bound below if you need the full filter.

  Congreve suggests TPSA <= 60 alongside the five Ro3 criteria. It is usually
  quoted as an optional addition rather than part of the rule, so it is left
  out; the five above are what `fragment` checks.

Boundaries are inclusive throughout: MW of exactly 500 passes Lipinski. That is
the conventional reading, and it matters more often than you would think, since
round numbers are exactly where vendor libraries cluster.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

from winnow.schemas.filters import RuleSet

#: Lipinski's one-violation allowance, named rather than buried in a comparison.
#: This is the clause that gets implemented wrong.
LIPINSKI_ALLOWED_VIOLATIONS = 1


class _Bound(NamedTuple):
    """One numeric constraint on one descriptor. Both bounds inclusive."""

    key: str
    label: str
    minimum: float | None = None
    maximum: float | None = None


def _fmt(value: float) -> str:
    """Render a number for a violation string without trailing noise."""
    if isinstance(value, int):
        return str(value)
    rounded = round(value, 2)
    return str(int(rounded)) if rounded == int(rounded) else f"{rounded:g}"


def _violations(desc: dict[str, float | int], bounds: tuple[_Bound, ...], rule: str) -> list[str]:
    """Collect every bound ``desc`` breaks, as human-readable strings.

    Raises rather than guessing when a descriptor is missing or non-finite.
    A NaN compares False against everything, so a silent pass would look like
    a clean molecule - the single worst outcome for a triage tool, and a real
    risk given this module is meant to run over caller-supplied CSVs.
    """
    out: list[str] = []
    for bound in bounds:
        if bound.key not in desc:
            raise KeyError(f"{rule} needs descriptor {bound.key!r}; got {sorted(desc)}")
        value = desc[bound.key]
        if not math.isfinite(value):
            raise ValueError(f"{rule}: descriptor {bound.key!r} is {value!r}, not a finite number")
        if bound.maximum is not None and value > bound.maximum:
            out.append(f"{bound.label} {_fmt(value)} > {_fmt(bound.maximum)}")
        if bound.minimum is not None and value < bound.minimum:
            out.append(f"{bound.label} {_fmt(value)} < {_fmt(bound.minimum)}")
    return out


# --- the rules ---------------------------------------------------------------

_LIPINSKI = (
    _Bound("molecular_weight", "MW", maximum=500),
    _Bound("clogp", "clogp", maximum=5),
    _Bound("hbd", "HBD", maximum=5),
    _Bound("hba", "HBA", maximum=10),
)

_VEBER = (
    _Bound("rotatable_bonds", "rotatable bonds", maximum=10),
    _Bound("tpsa", "TPSA", maximum=140),
)

_EGAN = (
    _Bound("tpsa", "TPSA", maximum=131.6),
    _Bound("clogp", "clogp", minimum=-1.0, maximum=5.88),
)

_GHOSE = (
    _Bound("molecular_weight", "MW", minimum=160, maximum=480),
    _Bound("clogp", "clogp", minimum=-0.4, maximum=5.6),
    _Bound("total_atoms", "atoms", minimum=20, maximum=70),
)

_LEAD_LIKE = (
    _Bound("molecular_weight", "MW", maximum=350),
    _Bound("clogp", "clogp", maximum=3.5),
    _Bound("rotatable_bonds", "rotatable bonds", maximum=7),
)

_FRAGMENT = (
    _Bound("molecular_weight", "MW", maximum=300),
    _Bound("clogp", "clogp", maximum=3),
    _Bound("hbd", "HBD", maximum=3),
    _Bound("hba", "HBA", maximum=3),
    _Bound("rotatable_bonds", "rotatable bonds", maximum=3),
)


def lipinski(desc: dict[str, float | int]) -> tuple[bool, list[str]]:
    """Rule of Five. Passes with up to LIPINSKI_ALLOWED_VIOLATIONS breaches.

    Violations are reported even when the molecule passes, so a caller can see
    that a compound is borderline rather than comfortably inside.
    """
    violations = _violations(desc, _LIPINSKI, "lipinski")
    return len(violations) <= LIPINSKI_ALLOWED_VIOLATIONS, violations


def veber(desc: dict[str, float | int]) -> tuple[bool, list[str]]:
    violations = _violations(desc, _VEBER, "veber")
    return not violations, violations


def egan(desc: dict[str, float | int]) -> tuple[bool, list[str]]:
    violations = _violations(desc, _EGAN, "egan")
    return not violations, violations


def ghose(desc: dict[str, float | int]) -> tuple[bool, list[str]]:
    violations = _violations(desc, _GHOSE, "ghose")
    return not violations, violations


def lead_like(desc: dict[str, float | int]) -> tuple[bool, list[str]]:
    violations = _violations(desc, _LEAD_LIKE, "lead_like")
    return not violations, violations


def fragment(desc: dict[str, float | int]) -> tuple[bool, list[str]]:
    violations = _violations(desc, _FRAGMENT, "fragment")
    return not violations, violations


#: Dispatch table. Defined after the functions so it cannot drift out of sync
#: with them; the test suite asserts it covers every member of RuleSet.
RULE_FUNCTIONS: dict[RuleSet, Callable[[dict[str, float | int]], tuple[bool, list[str]]]] = {
    RuleSet.LIPINSKI: lipinski,
    RuleSet.VEBER: veber,
    RuleSet.EGAN: egan,
    RuleSet.GHOSE: ghose,
    RuleSet.LEAD_LIKE: lead_like,
    RuleSet.FRAGMENT: fragment,
}


def evaluate(desc: dict[str, float | int], rule_sets: list[RuleSet]) -> list[dict[str, object]]:
    """Run each requested rule set, returning dicts shaped like RuleResult.

    Duplicates in ``rule_sets`` are collapsed - evaluating a rule twice yields
    the same answer and would only put a confusing repeated row in the report.
    Order follows first appearance in the request.
    """
    results: list[dict[str, object]] = []
    for rule_set in dict.fromkeys(rule_sets):
        try:
            rule = RULE_FUNCTIONS[rule_set]
        except KeyError:
            raise KeyError(f"no implementation registered for rule set {rule_set!r}") from None
        passed, violations = rule(desc)
        # name is the enum *value*, so a client can correlate a result against
        # the rule_sets it asked for without a lookup table.
        results.append({"name": rule_set.value, "passed": passed, "violations": violations})
    return results
