"""The triage pipeline: how the stages above compose.

Split into two phases, and the split is the whole design:

  ``process_chunk``  - embarrassingly parallel. Everything that depends on one
                       molecule only: parse, standardise, descriptors, rules,
                       alerts, scaffold, fingerprint. Runs in a worker process.

  ``finalize``       - inherently global. Everything that needs the whole
                       library: deduplication, clustering, ranking, top-N.
                       Runs in the parent, once, after all chunks land.

Anything that looks per-molecule but needs library context (dedup needs to know
which InChIKeys were seen; cluster ids are only meaningful across the set)
belongs in phase two. Getting this boundary wrong is the standard way these
pipelines end up subtly non-deterministic under parallelism.

``process_chunk`` MUST be importable at module level and take only picklable
arguments - it is the process-pool target.
"""

from __future__ import annotations

from typing import Any

from winnow.schemas.filters import TriageConfig


def process_chunk(
    records: list[tuple[str, str]], config_json: dict[str, Any]
) -> list[dict[str, Any]]:
    """Phase one. Per-molecule work for one chunk of the library.

    Args:
        records: (identifier, smiles) pairs. Plain tuples, not Pydantic models,
            because every element crosses a pickle boundary.
        config_json: ``TriageConfig.model_dump()``. Passed as a dict for the
            same reason; rebuild it inside with ``TriageConfig(**config_json)``.

    Returns:
        Dicts shaped like ``TriagedMolecule``, with ``cluster_id``,
        ``duplicate_of`` and ``is_cluster_representative`` left unset - those
        are phase two's job.

    Must never raise. A molecule that blows up gets ``parse_ok=False`` and an
    error string; one bad record in a 200k library must not kill the chunk, let
    alone the job.

    Return the fingerprint under a private key (``_fingerprint``) so finalize
    can cluster without recomputing, and strip it before serialising - it is
    not part of the public response model.
    """
    raise NotImplementedError


def finalize(
    molecules: list[dict[str, Any]], config: TriageConfig
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Phase two. Library-wide work, in this order:

        1. Partition out parse failures - they are reported, never dropped.
        2. Apply descriptor windows and (if configured) rule/alert drops.
        3. Deduplicate on InChIKey. Keep the FIRST occurrence and point the
           others at it via ``duplicate_of``; do not silently discard, the
           caller wants to know their library had 12% redundancy.
        4. Score everything retained.
        5. Cluster retained molecules, mark the highest scorer in each cluster
           as the representative.
        6. Sort by score descending, apply representatives_only, then top_n.
           That order matters - top_n after representative selection, or you
           will drop whole clusters.

    Returns:
        (molecules, counts) where counts has the keys of
        ``winnow.schemas.job.JobCounts``. The counts must reconcile:
        parsed + parse_failed == submitted.
    """
    raise NotImplementedError
