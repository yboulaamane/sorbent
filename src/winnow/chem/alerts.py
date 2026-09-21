"""Structural alerts (PAINS, BRENK, NIH, ZINC).

RDKit ships these as ``FilterCatalog``, so do NOT hand-write SMARTS:

    from rdkit.Chem import FilterCatalog
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    entries = catalog.GetMatches(mol)   # all hits, not just the first

Four things that will bite you, all measured on this codebase:

**Building a catalog is expensive.** PAINS compiles 480 SMARTS in ~17 ms, NIH
180 in ~7 ms, BRENK 105 in ~4 ms. Rebuilding per molecule instead of caching
runs at 57 mol/s against 1455 mol/s - **25x slower**, which makes this stage
dominate the entire pipeline. Hence ``_CATALOG_CACHE``. Under the spawn start
method each worker process builds its own set once, on first use.

**An entry's name does not name its family.** The PAINS alert
``catechol_A(92)`` belongs to ``FilterSet='PAINS_B'`` - the ``_A`` is part of
Baell's own numbering, not the family letter. So the ``catalog`` field on each
Alert reports the entry's own ``FilterSet`` property, which is the truth, and
not the catalog the caller happened to request.

**PAINS is exactly the union of PAINS_A, PAINS_B and PAINS_C** (480 = 16 + 55
+ 409). Requesting ``[PAINS, PAINS_B]`` therefore reports catechol twice for
one structural liability - and since ``n_alerts`` feeds ``alert_penalty`` in
``score.py``, the molecule would be penalised twice over. ``find_alerts``
deduplicates on (filter set, alert name, matched atoms), which collapses that
exactly while keeping genuinely independent flags: PAINS calls catechol
``catechol_A(92)`` and BRENK calls it ``catechol``, and those are two catalogs
agreeing, which is worth seeing.

**``GetMatches`` gives the entry but not the matched atoms.** For highlighting
you need ``entry.GetFilterMatches(mol)``, then read ``.target`` off each
``atomPairs`` entry - ``.query`` indexes the SMARTS, ``.target`` indexes your
molecule.

A PAINS hit is information, not a verdict. Report it; let the caller decide.
Each entry also carries a ``Reference`` property with the full literature
citation (Baell & Holloway 2010 for PAINS); it is not copied onto every Alert
because a 200-character citation repeated across a 200k-molecule report is
bulk, not information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rdkit.Chem import FilterCatalog

from winnow.schemas.filters import AlertCatalog

if TYPE_CHECKING:
    from rdkit.Chem import Mol

_FILTER_CATALOGS = FilterCatalog.FilterCatalogParams.FilterCatalogs

#: Our enum -> RDKit's catalog enum. Written out rather than derived from
#: ``member.name`` so that adding a member to AlertCatalog without a mapping
#: fails loudly here instead of with an AttributeError deep in a worker.
_RDKIT_CATALOG: dict[AlertCatalog, Any] = {
    AlertCatalog.PAINS: _FILTER_CATALOGS.PAINS,
    AlertCatalog.PAINS_A: _FILTER_CATALOGS.PAINS_A,
    AlertCatalog.PAINS_B: _FILTER_CATALOGS.PAINS_B,
    AlertCatalog.PAINS_C: _FILTER_CATALOGS.PAINS_C,
    AlertCatalog.BRENK: _FILTER_CATALOGS.BRENK,
    AlertCatalog.NIH: _FILTER_CATALOGS.NIH,
    AlertCatalog.ZINC: _FILTER_CATALOGS.ZINC,
}

#: Lazily built, one per catalog per process. Keep it - rebuilding per molecule
#: makes this the slowest stage in the pipeline by 25x.
_CATALOG_CACHE: dict[AlertCatalog, Any] = {}


def get_catalog(catalog: AlertCatalog) -> Any:
    """Return a cached RDKit FilterCatalog for the requested alert set.

    Not locked. Two threads racing here both build a catalog and one wins;
    the loser is discarded and the result is identical either way, which is
    cheaper than serialising every lookup on the hot path.
    """
    cached = _CATALOG_CACHE.get(catalog)
    if cached is not None:
        return cached

    try:
        rdkit_catalog = _RDKIT_CATALOG[catalog]
    except KeyError:
        raise KeyError(
            f"no RDKit catalog mapped for {catalog!r}; add it to _RDKIT_CATALOG"
        ) from None

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(rdkit_catalog)
    built = FilterCatalog.FilterCatalog(params)
    _CATALOG_CACHE[catalog] = built
    return built


def _matched_atoms(entry: Any, mol: Mol) -> list[int]:
    """Every atom in ``mol`` covered by this alert, across all occurrences.

    ``GetFilterMatches`` returns ONE match per entry - the catalog's question
    is "does this molecule trip the alert", not "where, exhaustively". On
    O=[N+]([O-])c1ccc(cc1)[N+](=O)[O-] it reports atoms [0, 1, 2]: the first
    nitro group only, though the molecule has two. A client highlighting that
    would show one nitro lit and the other apparently clean, which is a poor
    thing for a liability flag to do.

    So: take the SMARTS off the match and re-run it for all occurrences, which
    gives [0, 1, 2, 9, 10, 11]. Only paid when an alert actually fires, and
    every entry across PAINS, BRENK, NIH and ZINC exposes a usable pattern -
    but a compound matcher (And/Or/Not) would not, so fall back to the single
    match rather than losing the alert.
    """
    atoms: set[int] = set()
    for match in entry.GetFilterMatches(mol):
        # .query indexes the SMARTS pattern; .target indexes our molecule.
        single = {pair.target for pair in match.atomPairs}
        try:
            pattern = match.filterMatch.GetPattern()
            occurrences = mol.GetSubstructMatches(pattern, uniquify=True)
        except Exception:  # noqa: BLE001 - not a simple SMARTS matcher
            occurrences = ()
        if occurrences:
            atoms.update(index for hit in occurrences for index in hit)
        else:
            atoms.update(single)
    return sorted(atoms)


def find_alerts(mol: Mol, catalogs: list[AlertCatalog]) -> list[dict[str, object]]:
    """Return every alert match across the requested catalogs.

    Each dict is shaped like ``winnow.schemas.molecule.Alert``. All matches are
    returned, not the first: a compound tripping six BRENK alerts is a
    different proposition from one tripping a single borderline alert, and the
    count is what ``score.alert_penalty`` consumes.

    Duplicates arising from overlapping requests are collapsed - see the module
    docstring on PAINS being the union of its three families.
    """
    found: list[dict[str, object]] = []
    seen: set[tuple[str, str, tuple[int, ...]]] = set()

    for catalog in dict.fromkeys(catalogs):
        rdkit_catalog = get_catalog(catalog)
        for entry in rdkit_catalog.GetMatches(mol):
            name = entry.GetDescription()
            props = set(entry.GetPropList())
            # The entry's own family, not the catalog asked for. Falls back to
            # the request only if RDKit ever ships an entry without the prop.
            filter_set = entry.GetProp("FilterSet") if "FilterSet" in props else catalog.value
            atoms = _matched_atoms(entry, mol)

            key = (filter_set, name, tuple(atoms))
            if key in seen:
                continue
            seen.add(key)

            found.append(
                {
                    "catalog": filter_set,
                    "name": name,
                    # Scope is a short human description of what the family is
                    # for. The full citation lives on the Reference prop.
                    "description": entry.GetProp("Scope") if "Scope" in props else None,
                    "atom_indices": atoms,
                }
            )
    return found
