"""Bemis-Murcko scaffolds.

``from rdkit.Chem.Scaffolds import MurckoScaffold`` gives you both variants:

    MurckoScaffold.GetScaffoldForMol(mol)        # ring systems + linkers
    MurckoScaffold.MakeScaffoldGeneric(scaffold) # atoms -> C, bonds -> single

The generic form is what you want for clustering by topology; the plain form
keeps atom identity and is what a chemist expects to see. Winnow reports the
plain form and leaves the generic one as an extension.

Edge case worth handling explicitly: an acyclic molecule has an *empty*
scaffold. Return None rather than the empty string, so "no scaffold" is
distinguishable from a parse failure downstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit.Chem import Mol


def murcko_scaffold(mol: Mol) -> str | None:
    """Bemis-Murcko scaffold as canonical SMILES, or None if acyclic."""
    raise NotImplementedError


def generic_scaffold(mol: Mol) -> str | None:
    """Topology-only scaffold (all atoms carbon, all bonds single)."""
    raise NotImplementedError
