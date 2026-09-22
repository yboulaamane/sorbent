"""Bemis-Murcko scaffolds.

``from rdkit.Chem.Scaffolds import MurckoScaffold`` gives you both variants:

    MurckoScaffold.GetScaffoldForMol(mol)        # ring systems + linkers
    MurckoScaffold.MakeScaffoldGeneric(scaffold) # atoms -> C, bonds -> single

The generic form is for clustering by topology; the plain form keeps atom
identity and is what a chemist expects to see. Sorbent reports the plain form
on every molecule and offers the generic one for callers who want it.

Three things worth knowing:

**An acyclic molecule has an empty scaffold, and RDKit says so with an empty
Mol, not None.** ``Chem.MolToSmiles`` then returns ``''``, which is
indistinguishable from a failure once it reaches a report. Both functions here
return None instead, so "no scaffold" reads as a distinct state.

**Scaffold SMILES are written without stereochemistry.** RDKit's own
``MurckoScaffoldSmiles`` defaults to ``includeChirality=False``, and for good
reason: the Bemis-Murcko framework is a statement about ring systems and
linkers, not configuration. Kept, the two enantiomers of nicotine produce
``c1cncc([C@@H]2CCCN2)c1`` and ``c1cncc([C@H]2CCCN2)c1`` and land in different
scaffold groups, which is not what anybody counting chemotypes means. Dropped,
both give ``c1cncc(C2CCCN2)c1``. Stereochemistry is preserved on the molecule
itself - see ``standard_smiles`` - so nothing is lost, only moved to where it
belongs.

**Exocyclic double bonds on ring atoms are retained.** ``O=C1CCCCC1`` scaffolds
to ``O=C1CCCCC1``, not ``C1CCCCC1``, because the carbonyl carbon is itself a
ring atom. A side-chain carbonyl is stripped as normal: ``O=C(c1ccccc1)C``
scaffolds to ``c1ccccc1``. So a ring ketone and its parent cyclohexane are
*different* scaffolds. That is genuine Bemis-Murcko behaviour rather than an
RDKit quirk, but it surprises people counting chemotypes.

Note that the generic form does *not* rescue you here: it turns the exocyclic
oxygen into a carbon rather than removing it, so ``O=C1CCCCC1`` becomes
``CC1CCCCC1`` and stays distinct from ``C1CCCCC1``. What the generic form does
collapse is heteroatom identity and aromaticity - benzene, cyclohexane and
pyridine all become ``C1CCCCC1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

if TYPE_CHECKING:
    from rdkit.Chem import Mol


def _scaffold_mol(mol: Mol) -> Any | None:
    """Murcko scaffold as a Mol, or None when the molecule is acyclic.

    ``GetScaffoldForMol`` does not mutate its argument, so no defensive copy.
    """
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    # An acyclic input yields a real Mol object with zero atoms, not None.
    return scaffold if scaffold.GetNumAtoms() else None


def murcko_scaffold(mol: Mol) -> str | None:
    """Bemis-Murcko scaffold as canonical SMILES, or None if acyclic.

    Written without stereochemistry, so the string works as a grouping key -
    see the module docstring.
    """
    scaffold = _scaffold_mol(mol)
    if scaffold is None:
        return None
    return Chem.MolToSmiles(scaffold, isomericSmiles=False)


def generic_scaffold(mol: Mol) -> str | None:
    """Topology-only scaffold (all atoms carbon, all bonds single).

    Collapses heteroatom identity and aromaticity, so pyridine and benzene
    share a generic scaffold and a ring ketone collapses onto its parent ring.
    Useful for grouping by shape when the plain scaffold splits too finely.
    """
    scaffold = _scaffold_mol(mol)
    if scaffold is None:
        return None
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    if not generic.GetNumAtoms():
        return None
    return Chem.MolToSmiles(generic, isomericSmiles=False)
