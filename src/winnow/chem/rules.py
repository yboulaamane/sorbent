"""Drug-likeness rule sets.

Each rule is a pure function of the descriptor dict - no molecule needed, which
means they are trivially testable and you can run them over a CSV of
precomputed properties without RDKit at all.

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
      160 <= MW <= 480, -0.4 <= clogp <= 5.6, 20 <= heavy_atoms <= 70

  Lead-like (Teague, Angew Chem 1999;38:3743):
      MW <= 350, clogp <= 3.5, rotatable_bonds <= 7

  Fragment / Ro3 (Congreve, Drug Discov Today 2003;8:876):
      MW <= 300, clogp <= 3, hbd <= 3, hba <= 3, rotatable_bonds <= 3
"""

from __future__ import annotations

from collections.abc import Callable

from winnow.schemas.filters import RuleSet

#: Populate this as you implement each rule. The pipeline dispatches through it,
#: so an unimplemented rule set raises a clear KeyError naming the gap.
RULE_FUNCTIONS: dict[RuleSet, Callable[[dict[str, float | int]], tuple[bool, list[str]]]] = {}


def lipinski(desc: dict[str, float | int]) -> tuple[bool, list[str]]:
    """Return (passed, violations). Remember: one violation is still a pass.

    Violations are reported even when the molecule passes, so a caller can see
    that a compound is borderline.
    """
    raise NotImplementedError


def veber(desc: dict[str, float | int]) -> tuple[bool, list[str]]:
    raise NotImplementedError


def egan(desc: dict[str, float | int]) -> tuple[bool, list[str]]:
    raise NotImplementedError


def ghose(desc: dict[str, float | int]) -> tuple[bool, list[str]]:
    raise NotImplementedError


def lead_like(desc: dict[str, float | int]) -> tuple[bool, list[str]]:
    raise NotImplementedError


def fragment(desc: dict[str, float | int]) -> tuple[bool, list[str]]:
    raise NotImplementedError


def evaluate(desc: dict[str, float | int], rule_sets: list[RuleSet]) -> list[dict[str, object]]:
    """Run each requested rule set, returning dicts shaped like RuleResult.

    Dispatch through RULE_FUNCTIONS so adding a rule set is one entry, not an
    if-chain.
    """
    raise NotImplementedError
