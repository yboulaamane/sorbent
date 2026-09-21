"""ECFP fingerprints and similarity.

Use the modern generator API - ``AllChem.GetMorganFingerprintAsBitVect`` is
deprecated in recent RDKit:

    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = gen.GetFingerprint(mol)

Radius 2 with 2048 bits is ECFP4 - the default for a reason. Note the naming
trap: ECFP*4* is radius *2* (the number is the diameter).

Pickling: RDKit ExplicitBitVect pickles fine, which matters because these cross
a process boundary on their way back from the pool. They are also the bulk of
the payload - 2048 bits x 200k molecules is ~50 MB. If that becomes a problem,
return ``fp.ToBinary()`` and rebuild in the parent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rdkit.Chem import Mol


def get_generator(radius: int = 2, n_bits: int = 2048) -> Any:
    """Cached Morgan generator. Building one per molecule is pure waste."""
    raise NotImplementedError


def compute_fingerprint(mol: Mol, radius: int = 2, n_bits: int = 2048) -> Any:
    """Return the ECFP bit vector for ``mol``."""
    raise NotImplementedError


def tanimoto(fp_a: Any, fp_b: Any) -> float:
    """Tanimoto similarity in [0, 1]. Use DataStructs.TanimotoSimilarity."""
    raise NotImplementedError


def bulk_tanimoto(query: Any, targets: list[Any]) -> list[float]:
    """One-against-many similarity.

    Use ``DataStructs.BulkTanimotoSimilarity`` - it is a C++ loop and roughly
    an order of magnitude faster than a Python comprehension over
    ``TanimotoSimilarity``. On a 200k library that difference is the whole job.
    """
    raise NotImplementedError
