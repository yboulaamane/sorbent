"""Physicochemical descriptors.

All of these come straight out of ``rdkit.Chem.Descriptors`` /
``rdkit.Chem.rdMolDescriptors``. Every value is deterministic and reproducible
from the connection table alone.

One of them is nonetheless a *model*, and the distinction matters for a service
whose whole pitch is that its numbers are defensible:

  - Counts and sums - MW, heavy atoms, rings, aromatic rings, rotatable bonds,
    Fsp3, formal charge, stereocentres - are graph arithmetic. They are exact.
  - TPSA is Ertl's additive fragment sum: a lookup table applied to the graph.
    Exact given the table.
  - HBD/HBA are SMARTS match counts. Exact given the patterns, but the patterns
    are a judgement call - see below.
  - **clogp is a fitted QSPR model.** Wildman & Crippen (J Chem Inf Comput Sci
    1999;39:868) regressed atom-type contributions against experimental logP.
    It is deterministic and citable, but it is an estimate of a physical
    property and it is routinely off by half a log unit or more - paracetamol
    computes 1.35 against an experimental 0.46. Treat it as a coordinate for
    ranking, never as a measurement.

Four further choices worth knowing about, because a reviewer will eventually
ask:

**Average mass, not monoisotopic.** ``Descriptors.MolWt`` gives 180.159 for
aspirin; ``ExactMolWt`` gives 180.042. Drug-likeness rules are written against
the average mass, so that is what ``molecular_weight`` holds. Anyone comparing
against an MS result wants the other one.

**HBD/HBA follow RDKit's SMARTS, not Lipinski's literal wording.** The 1997
paper counts donors as "OH plus NH" and acceptors as "all N and O" - which for
aspirin is 4 acceptors. ``Descriptors.NumHAcceptors`` applies a refined pattern
that excludes, among others, the ester and amide oxygens that cannot really
accept, and gives 3. RDKit's definition is the more chemically sensible one and
is what most software reports, so it is what we use - but a borderline compound
can pass here and fail against a tool using the literal N+O count. The
equivalents, if you ever need them, are ``Lipinski.NHOHCount`` and
``Lipinski.NOCount``.

**Crippen logP is parameterised for organic elements.** It does not raise on a
metal complex - cisplatin quietly returns 1.70 - so a logP on anything
inorganic is a number, not information. Nothing here filters those out; see
``parse.standardize`` for why that policy belongs to the caller.

**``stereocentres`` counts atoms only.** ``FindMolChiralCenters`` does not see
double-bond geometry, so an undefined E/Z alkene is not counted as unassigned
here even though it is exactly the same purchasing problem. Extending this
means ``Chem.FindPotentialStereo``, which reports atom and bond stereo
together, and a matching field on the ``Descriptors`` schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

if TYPE_CHECKING:
    from rdkit.Chem import Mol

#: Descriptor keys this module promises to return. Must stay in sync with the
#: fields of ``winnow.schemas.molecule.Descriptors`` - the test suite asserts it.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "molecular_weight",
    "heavy_atoms",
    "clogp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "aromatic_rings",
    "rings",
    "fraction_csp3",
    "formal_charge",
    "stereocentres",
    "unassigned_stereocentres",
)


def compute_descriptors(mol: Mol) -> dict[str, float | int]:
    """Return every descriptor in DESCRIPTOR_NAMES for ``mol``.

    Keys are exactly DESCRIPTOR_NAMES - no more, no fewer.

    Deliberately does not catch anything. These calls are robust in practice
    (they survive boron, platinum, selenium, isotopes, radicals and
    carbon-free molecules without complaint), so a raise here means something
    genuinely odd arrived, and the pipeline should record that molecule as
    failed rather than publish a row of plausible-looking fallback numbers.
    """
    # One call, reused for both counts. Returns (atom_index, code) pairs where
    # code is "R", "S" or "?" - "?" being a centre RDKit recognises as a
    # stereocentre but the input never specified.
    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)

    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "clogp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "stereocentres": len(centres),
        "unassigned_stereocentres": sum(1 for _, code in centres if code == "?"),
    }
