"""Physicochemical descriptors.

All of these come straight out of ``rdkit.Chem.Descriptors`` /
``rdkit.Chem.rdMolDescriptors``. The only judgement calls:

  - Use ``Descriptors.NumHDonors`` / ``NumHAcceptors`` (Lipinski's definitions,
    which is what the rule expects) rather than the Lipinski-module variants.
  - ``fraction_csp3`` is ``rdMolDescriptors.CalcFractionCSP3``.
  - Stereocentres: ``Chem.FindMolChiralCenters(mol, includeUnassigned=True,
    useLegacyImplementation=False)``. Count unassigned separately - a library
    full of undefined stereo is a purchasing problem, and flagging it is one of
    the more useful things this service can do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    Keys must be exactly DESCRIPTOR_NAMES - no more, no fewer.
    """
    raise NotImplementedError
