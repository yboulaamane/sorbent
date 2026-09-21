"""Structural alerts (PAINS, BRENK, NIH, ZINC).

RDKit already ships these as ``FilterCatalog``, so do NOT hand-write SMARTS:

    from rdkit.Chem import FilterCatalog
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    entries = catalog.GetMatches(mol)          # all hits, not just the first
    entry.GetDescription()                      # the alert's name

Two things that will bite you:

  1. Building a FilterCatalog is slow (hundreds of SMARTS get compiled). Build
     each one ONCE at module level and cache it - see ``get_catalog``. In a
     process pool each worker builds its own, which is correct and why a warm
     pool beats spawning per request.
  2. ``GetMatches`` gives the entry but not the matched atoms. For atom
     highlighting you need ``entry.GetFilterMatches(mol)`` and to read the
     atom pairs off each match.

A PAINS hit is information, not a verdict. Report it; let the caller decide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from winnow.schemas.filters import AlertCatalog

if TYPE_CHECKING:
    from rdkit.Chem import Mol

#: Lazily built, one per catalog per process. Keep it - rebuilding per molecule
#: makes this the slowest stage in the pipeline by an order of magnitude.
_CATALOG_CACHE: dict[AlertCatalog, Any] = {}


def get_catalog(catalog: AlertCatalog) -> Any:
    """Return a cached RDKit FilterCatalog for the requested alert set."""
    raise NotImplementedError


def find_alerts(mol: Mol, catalogs: list[AlertCatalog]) -> list[dict[str, object]]:
    """Return every alert match across the requested catalogs.

    Each dict is shaped like ``winnow.schemas.molecule.Alert``. Return all
    matches, not the first - a compound hitting six BRENK alerts is a different
    proposition from one hitting a single borderline alert.
    """
    raise NotImplementedError
