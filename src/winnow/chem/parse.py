"""SMILES in, clean RDKit molecule out.

Standardisation order - it matters, and getting it wrong is the classic source
of "why do my InChIKeys not match ChEMBL":

    1. Parse without sanitising, then sanitise explicitly, so parse failure and
       sanitisation failure stay distinguishable and neither escapes as an
       exception.
    2. ``Cleanup`` - normalise functional groups, disconnect metals, reionise.
    3. Strip salts via ``LargestFragmentChooser``. (A curated salt list is the
       rigorous alternative; this is the pragmatic one.)
    4. Neutralise with ``Uncharger``.
    5. Canonical tautomer. The expensive step - it costs roughly 40% of total
       runtime here - so it is switchable via ``canonical_tautomer``.

Two things found the hard way, both worth keeping:

``rdMolStandardize`` lives at ``rdkit.Chem.MolStandardize.rdMolStandardize``.
``from rdkit.Chem import rdMolStandardize`` does not work.

**The tautomer canonicaliser destroys defined stereochemistry by default.**
``CleanupParameters.tautomerRemoveSp3Stereo`` is True out of the box, which
strips sp3 stereo from any centre adjacent to a tautomerisable system. That is
the alpha carbon of every amino acid:

    C[C@H](N)C(=O)O                          -> CC(N)C(=O)O
    C[C@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)O   -> CC(N)C(=O)NC(Cc1ccccc1)C(=O)O

RDKit is not exactly wrong - such a centre is epimerisable in principle - but
silently racemising a peptidomimetic library is not triage, it is data loss.
We set the flag False and keep the stereo the submitter gave us.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, NamedTuple

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

if TYPE_CHECKING:
    from rdkit.Chem import Mol

# RDKit writes parse failures straight to stderr. A vendor file with 3% junk
# would otherwise bury the application log under a quarter of a million lines.
RDLogger.DisableLog("rdApp.*")

#: Longest input echoed back in an error message. Garbage input is sometimes a
#: whole mangled spreadsheet row, and it should not end up in the response.
_MAX_ECHO = 60


def _echo(smiles: str) -> str:
    text = smiles.strip()
    return repr(text if len(text) <= _MAX_ECHO else text[:_MAX_ECHO] + "...")


# --- cached standardiser components -----------------------------------------
#
# Each of these compiles transform tables on construction, so they are built
# once per process and reused. Under the spawn start method every worker builds
# its own set at import - that is correct, and it is why the pool is created in
# the app lifespan rather than per request.
#
# They are stateless once built, and the only callers are a worker process or
# the event-loop thread, one molecule at a time. Treat them as read-only.


@functools.lru_cache(maxsize=1)
def _cleanup_params() -> Any:
    params = rdMolStandardize.CleanupParameters()
    # See the module docstring. Without this, stereo silently disappears.
    params.tautomerRemoveSp3Stereo = False
    return params


@functools.lru_cache(maxsize=1)
def _fragment_chooser() -> Any:
    return rdMolStandardize.LargestFragmentChooser()


@functools.lru_cache(maxsize=1)
def _uncharger() -> Any:
    return rdMolStandardize.Uncharger()


@functools.lru_cache(maxsize=1)
def _tautomer_enumerator() -> Any:
    return rdMolStandardize.TautomerEnumerator(_cleanup_params())


# --- public API --------------------------------------------------------------


class ParsedMolecule(NamedTuple):
    """Result of parsing one input record.

    ``mol`` is None exactly when ``error`` is set; callers rely on that.

    ``inchikey`` is the one field that may be None without ``error`` being set:
    a molecule can be perfectly valid and still defeat RDKit's InChI layer. It
    is then simply unkeyable, and deduplication has to fall back to the
    canonical SMILES for that record.
    """

    mol: Mol | None
    standard_smiles: str | None
    inchikey: str | None
    error: str | None


def parse_smiles(smiles: str) -> Mol | None:
    """Parse a SMILES string into a sanitised molecule, or None if invalid.

    Never raises. A 200k-compound vendor file contains empty lines, stray
    quotes and Excel-mangled strings, and one bad record must not take down a
    chunk of 2000.
    """
    if not smiles or not smiles.strip():
        return None

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    except Exception:  # noqa: BLE001 - malformed input is data, not a bug
        return None

    if mol is None:
        return None

    # MolFromSmiles("") returns an EMPTY Mol, not None - so a blank line parses
    # "successfully" into a molecule with no atoms and sails through
    # sanitisation. Catch it here or it becomes a zero-MW row in the report.
    if mol.GetNumAtoms() == 0:
        return None

    try:
        problems = Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:  # noqa: BLE001
        return None

    if problems != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    # Same failure class as the empty Mol above: "*" and "[*]" are valid SMILES
    # for a dummy atom, so an R-group placeholder in a vendor SMILES column
    # parses and sanitises cleanly, then reports MW 0. A molecule made only of
    # dummy atoms is not a compound. A *mixture* of dummy and real atoms is
    # kept - that is an ordinary fragment-library attachment point.
    if not any(atom.GetAtomicNum() > 0 for atom in mol.GetAtoms()):
        return None

    return mol


def standardize(mol: Mol, *, canonical_tautomer: bool = True) -> Mol:
    """Return a standardised copy of ``mol``. Does not mutate the input.

    None of the RDKit steps below mutate their argument today, but the copy
    makes that a guarantee of this function rather than an implementation
    detail of whichever RDKit is installed.

    Set ``canonical_tautomer=False`` to skip the tautomer pass - roughly 40%
    faster, at the cost of keto and enol forms of one compound no longer
    deduplicating against each other.

    One degenerate case is left alone deliberately: a record that is nothing
    but counterions (``[Na+].[Cl-]``) standardises to HCl rather than being
    rejected. Filtering all-inorganic records would be easy and would also
    throw out cisplatin, so that policy belongs to the caller, not here.
    """
    work = Chem.Mol(mol)
    work = rdMolStandardize.Cleanup(work)
    work = _fragment_chooser().choose(work)
    work = _uncharger().uncharge(work)
    if canonical_tautomer:
        work = _tautomer_enumerator().Canonicalize(work)
    return work


def to_inchikey(mol: Mol) -> str | None:
    """Standard InChIKey, or None if RDKit's InChI layer refuses the molecule.

    This is the deduplication key, so it must be computed on the *standardised*
    molecule or two spellings of one compound will not collapse.

    Note what InChI deliberately ignores: it normalises some tautomers and
    drops certain stereo detail. Two records that differ only there share a key
    and will be deduplicated. If a project needs them distinguished, key on
    canonical SMILES instead.
    """
    if mol.GetNumAtoms() == 0:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:  # noqa: BLE001 - InChI is fussy about exotic valences
        return None
    # Failure is an empty string here, not an exception.
    return key or None


def process_record(smiles: str, standardize_mol: bool = True) -> ParsedMolecule:
    """Parse, optionally standardise, and key one record.

    The single per-molecule entry point for the pipeline. Never raises; a
    failure comes back as a ParsedMolecule with ``error`` set.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return ParsedMolecule(None, None, None, f"could not parse SMILES {_echo(smiles)}")

    if standardize_mol:
        try:
            mol = standardize(mol)
        except Exception as exc:  # noqa: BLE001
            return ParsedMolecule(
                None, None, None, f"standardisation failed: {type(exc).__name__}: {exc}"
            )
        # Salt stripping on a record that was *only* counterions leaves nothing.
        if mol is None or mol.GetNumAtoms() == 0:
            return ParsedMolecule(None, None, None, "standardisation left no parent fragment")

    try:
        standard_smiles = Chem.MolToSmiles(mol)
    except Exception as exc:  # noqa: BLE001
        return ParsedMolecule(
            None, None, None, f"could not canonicalise: {type(exc).__name__}: {exc}"
        )

    return ParsedMolecule(mol, standard_smiles, to_inchikey(mol), None)
