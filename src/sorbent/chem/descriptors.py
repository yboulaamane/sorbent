"""Physicochemical descriptors.

All of these come straight out of ``rdkit.Chem.Descriptors`` /
``rdkit.Chem.rdMolDescriptors``. Every value is deterministic and reproducible
from the connection table alone.

One of them is nonetheless a *model*, and the distinction matters for a service
whose whole pitch is that its numbers are defensible:

  - Counts and sums - MW, atom counts, rings, aromatic rings, rotatable bonds,
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

**HBA is Lipinski's literal N+O count; HBD is RDKit's refined pattern.** The
two are not treated the same way, and the reason is measurement rather than
taste. Checked against Molport's own published descriptors for 206,922
peptidomimetics:

    HBA   Lipinski.NOCount, the 1997 paper's "all N and O"   97.5% agreement
    HBA   Descriptors.NumHAcceptors, RDKit's refined SMARTS  11.1% agreement
    HBD   Descriptors.NumHDonors                             95.6% agreement
    HBD   Lipinski.NHOHCount, the paper's "OH plus NH"       91.5% agreement

So each field uses whichever definition agrees, and they do not agree on the
same one.

The primary argument is the paper, not the catalogue: a rule that cites
Lipinski 1997 should count acceptors the way Lipinski 1997 did, and that paper
says N+O. Molport agreeing is corroboration, not the reason - see the TPSA note
below for what happens when a vendor is treated as ground truth.

Nor is this an artefact of one chemotype. The two definitions disagree across
chemistry generally: on 30,000 diverse PubChem structures they differ for 74%
of molecules with a median gap of one acceptor. Amide isosteres merely widen
the gap to two, because RDKit's refined pattern excludes amide nitrogens and
such a library is made of them.

**TPSA excludes sulphur and phosphorus, which is Ertl's own convention.**
Molport includes them, and switching to ``includeSandP=True`` raises agreement
with their catalogue from 58.4% to 85.0% - which is exactly the wrong reason to
do it. RDKit ships reference values in ``Data/NCI/first_5k.tpsa.csv``, computed
with Daylight tools and Ertl's own contrib ``tpsa.c``. Over the 1,028 of those
molecules that contain S or P:

    RDKit default, S and P excluded    100.0% agreement with Ertl's program
    includeSandP=True                    0.4% agreement

So the canonical implementation excludes them and Molport is the outlier. This
was briefly changed to chase that vendor's numbers and reverted once the
reference data was checked. Agreeing with one catalogue is not the same as
being right.

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
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

if TYPE_CHECKING:
    from rdkit.Chem import Mol

#: Descriptor keys this module promises to return. Must stay in sync with the
#: fields of ``sorbent.schemas.molecule.Descriptors`` - the test suite asserts it.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "molecular_weight",
    "heavy_atoms",
    "total_atoms",
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
        # Including hydrogens, which are implicit on a SMILES-derived molecule.
        # Summing GetTotalNumHs matches Chem.AddHs(mol).GetNumAtoms() exactly
        # and avoids copying the molecule to find out.
        "total_atoms": mol.GetNumHeavyAtoms()
        + sum(atom.GetTotalNumHs() for atom in mol.GetAtoms()),
        "clogp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Lipinski.NOCount(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "stereocentres": len(centres),
        "unassigned_stereocentres": sum(1 for _, code in centres if code == "?"),
    }
