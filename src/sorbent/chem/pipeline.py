"""The triage pipeline: how the stages above compose.

Split into two phases, and the split is the whole design:

  ``process_chunk``  - embarrassingly parallel. Everything that depends on one
                       molecule only: parse, standardise, descriptors, rules,
                       alerts, scaffold, fingerprint. Runs in a worker process.

  ``finalize``       - inherently global. Everything that needs the whole
                       library: deduplication, clustering, ranking, top-N.
                       Runs once, after all chunks land.

Anything that looks per-molecule but needs library context (dedup needs to know
which InChIKeys were seen; cluster ids are only meaningful across the set)
belongs in phase two. Getting this boundary wrong is the standard way these
pipelines end up subtly non-deterministic under parallelism.

``process_chunk`` MUST be importable at module level and take only picklable
arguments - it is the process-pool target.

**What comes back, and what does not.** The output is an audit trail, not just
a shortlist, but it is not indiscriminate either:

  - Parse failures, duplicates and filter drops are ALWAYS reported. These are
    data-quality findings about the submitted library - that 12% of it was
    redundant, that a tenth of it would not parse - and a triage tool that
    swallowed them would be hiding the most actionable thing it knows.
  - Molecules that passed everything but fell outside ``top_n`` or
    ``representatives_only`` are NOT reported. They are not problems, merely
    surplus to what was asked for, and ``counts.retained`` says how many there
    were. The alternative - returning 200,000 records when the caller asked
    for the best 100 - serves nobody.

So ``score is not None`` identifies exactly the shortlist, and everything else
in the output carries a reason: ``parse_error``, ``duplicate_of``, or a
descriptor outside its window.

Ordering is fully deterministic and independent of the order chunks arrive in:
the shortlist sorts by descending score with the identifier as tie-break, and
each reported group sorts by identifier. ``asyncio.as_completed`` in the runner
destroys chunk order, so nothing here may depend on it.
"""

from __future__ import annotations

from typing import Any

from sorbent.chem.alerts import find_alerts
from sorbent.chem.cluster import assign_clusters, butina_cluster
from sorbent.chem.descriptors import compute_descriptors
from sorbent.chem.fingerprints import compute_fingerprint
from sorbent.chem.parse import process_record
from sorbent.chem.rules import evaluate
from sorbent.chem.scaffolds import murcko_scaffold
from sorbent.chem.score import composite_score
from sorbent.schemas.filters import TriageConfig
from sorbent.schemas.job import JobCounts

#: Where phase one parks the fingerprint for phase two. Private, and stripped
#: before the result leaves finalize - it is not part of TriagedMolecule.
FINGERPRINT_KEY = "_fingerprint"


def _blank(identifier: str, smiles: str) -> dict[str, Any]:
    """Every field of TriagedMolecule, so a record is always fully shaped."""
    return {
        "identifier": identifier,
        "input_smiles": smiles,
        "parse_ok": False,
        "parse_error": None,
        "standard_smiles": None,
        "inchikey": None,
        "descriptors": None,
        "rules": [],
        "alerts": [],
        "murcko_scaffold": None,
        "cluster_id": None,
        "is_cluster_representative": False,
        "duplicate_of": None,
        "score": None,
        "score_breakdown": {},
    }


def process_chunk(
    records: list[tuple[str, str]], config_json: dict[str, Any]
) -> list[dict[str, Any]]:
    """Phase one. Per-molecule work for one chunk of the library.

    Args:
        records: (identifier, smiles) pairs. Plain tuples, not Pydantic models,
            because every element crosses a pickle boundary.
        config_json: ``TriageConfig.model_dump()``. Passed as a dict for the
            same reason; rebuilt inside.

    Returns:
        Dicts shaped like ``TriagedMolecule``, with ``cluster_id``,
        ``duplicate_of`` and ``is_cluster_representative`` left unset - those
        are phase two's job, because a worker cannot see the other chunks.

    Never raises. A molecule that blows up gets ``parse_ok=False`` and an error
    string; one bad record in a 200k library must not kill the chunk of 2000,
    let alone the job.
    """
    config = TriageConfig.model_validate(config_json)
    results: list[dict[str, Any]] = []

    for identifier, smiles in records:
        entry = _blank(identifier, smiles)
        try:
            parsed = process_record(smiles, standardize_mol=config.standardize)
            if parsed.error is not None or parsed.mol is None:
                entry["parse_error"] = parsed.error or "molecule could not be built"
                results.append(entry)
                continue

            entry["parse_ok"] = True
            entry["standard_smiles"] = parsed.standard_smiles
            entry["inchikey"] = parsed.inchikey

            descriptors = compute_descriptors(parsed.mol)
            entry["descriptors"] = descriptors
            entry["rules"] = evaluate(descriptors, config.rule_sets)
            entry["alerts"] = find_alerts(parsed.mol, config.alert_catalogs)
            entry["murcko_scaffold"] = murcko_scaffold(parsed.mol)
            entry[FINGERPRINT_KEY] = compute_fingerprint(
                parsed.mol, config.fingerprint_radius, config.fingerprint_bits
            )
        except Exception as exc:  # noqa: BLE001 - a bad record is data, not a bug
            # Demote rather than discard: whatever was computed before the
            # failure stays visible, but parse_ok=False keeps it out of the
            # scored set in phase two.
            entry["parse_ok"] = False
            entry["parse_error"] = f"{type(exc).__name__}: {exc}"
            entry.pop(FINGERPRINT_KEY, None)

        results.append(entry)

    return results


def _drop_reason(molecule: dict[str, Any], config: TriageConfig) -> str | None:
    """Which filter excludes this molecule, if any.

    Returns the JobCounts key to increment. First match wins, so a molecule
    failing on several counts is counted once - otherwise the counts would sum
    to more than the library.
    """
    descriptors = molecule["descriptors"] or {}
    for name, window in config.descriptor_windows.items():
        if name in descriptors and not window.contains(float(descriptors[name])):
            return "dropped_by_window"

    if config.drop_rule_failures and any(not rule["passed"] for rule in molecule["rules"]):
        return "dropped_by_rule"

    if config.drop_alert_hits and molecule["alerts"]:
        return "dropped_by_alert"

    return None


def _deduplicate(
    molecules: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into (kept, duplicates), keeping the first of each key.

    "First" is submission order, which the runner restores before calling
    finalize - as_completed returns chunks in completion order, and a report
    whose survivor depended on which worker finished first would not be
    reproducible.

    Keyed on standard InChIKey, falling back to canonical SMILES for the
    molecules where RDKit's InChI layer declined - see ``parse.to_inchikey``.
    """
    seen: dict[str, str] = {}
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []

    for molecule in molecules:
        key = molecule["inchikey"] or molecule["standard_smiles"]
        if key is None:
            kept.append(molecule)
            continue
        if key in seen:
            molecule["duplicate_of"] = seen[key]
            duplicates.append(molecule)
        else:
            seen[key] = molecule["identifier"]
            kept.append(molecule)

    return kept, duplicates


def finalize(
    molecules: list[dict[str, Any]], config: TriageConfig
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Phase two. Library-wide work.

    Returns (molecules, counts). The counts reconcile by construction:
    ``parsed + parse_failed == submitted``, and every parsed molecule is
    either retained, a duplicate, or attributed to exactly one drop reason.
    """
    counts = dict.fromkeys(JobCounts.model_fields, 0)
    counts["submitted"] = len(molecules)

    # Work on copies. Only top-level keys are ever assigned, so a shallow copy
    # per molecule is enough, and it costs microseconds against the clustering.
    # Without it finalize mutates the caller's dicts - setting scores, and
    # stripping the fingerprint the next call would need - which makes calling
    # it twice on one chunk's output fail, and makes a retry after a transient
    # error impossible.
    molecules = [dict(molecule) for molecule in molecules]

    # 1. Parse failures are reported, never dropped.
    parsed = [m for m in molecules if m["parse_ok"]]
    failed = [m for m in molecules if not m["parse_ok"]]
    counts["parsed"] = len(parsed)
    counts["parse_failed"] = len(failed)

    # 2. Hard filters.
    survivors: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for molecule in parsed:
        reason = _drop_reason(molecule, config)
        if reason is None:
            survivors.append(molecule)
        else:
            counts[reason] += 1
            dropped.append(molecule)

    # 3. Deduplicate.
    if config.deduplicate:
        retained, duplicates = _deduplicate(survivors)
    else:
        retained, duplicates = survivors, []
    counts["duplicates_removed"] = len(duplicates)
    counts["retained"] = len(retained)

    # 4. Score everything retained.
    for molecule in retained:
        score, breakdown = composite_score(
            molecule["descriptors"],
            molecule["rules"],
            len(molecule["alerts"]),
            config.score_weights,
        )
        molecule["score"] = score
        molecule["score_breakdown"] = breakdown

    # 5. Cluster, and mark the highest scorer in each cluster.
    if config.representatives_only and not config.cluster:
        raise ValueError(
            "representatives_only requires cluster=True: there are no clusters "
            "to take a representative from"
        )
    if config.cluster and retained:
        # Raises above MAX_EXACT_CLUSTER_SIZE with a message naming the
        # alternatives. Deliberately not caught - silently skipping would make
        # representatives_only quietly meaningless.
        missing = [m["identifier"] for m in retained if FINGERPRINT_KEY not in m]
        if missing:
            raise KeyError(
                f"{len(missing)} molecule(s) reached clustering without a "
                f"fingerprint, first {missing[0]!r}. finalize needs the output of "
                f"process_chunk, which carries it under {FINGERPRINT_KEY!r}."
            )
        clusters = butina_cluster(
            [m[FINGERPRINT_KEY] for m in retained], cutoff=config.cluster_cutoff
        )
        for molecule, cluster_id in zip(
            retained, assign_clusters(clusters, len(retained)), strict=True
        ):
            molecule["cluster_id"] = cluster_id
        counts["clusters"] = len(clusters)

        for members in clusters:
            # The representative is the highest scorer, which need not be the
            # Butina centroid - the centroid is most central, not most useful.
            best = max(members, key=lambda i: (retained[i]["score"], retained[i]["identifier"]))
            retained[best]["is_cluster_representative"] = True

    # 6. Rank, then take representatives, then top_n. That order matters:
    #    top_n first would drop whole clusters before they were represented.
    shortlist = sorted(retained, key=lambda m: (-m["score"], m["identifier"]))
    if config.representatives_only:
        shortlist = [m for m in shortlist if m["is_cluster_representative"]]
    if config.top_n is not None:
        shortlist = shortlist[: config.top_n]

    # Retained molecules that did not make the shortlist are surplus, not
    # findings, so they are not reported - see the module docstring. Their
    # score is cleared so that "has a score" means "is in the shortlist".
    in_shortlist = {id(m) for m in shortlist}
    for molecule in retained:
        if id(molecule) not in in_shortlist:
            molecule["score"] = None
            molecule["score_breakdown"] = {}

    by_identifier = lambda group: sorted(group, key=lambda m: m["identifier"])  # noqa: E731
    output = shortlist + by_identifier(duplicates) + by_identifier(dropped) + by_identifier(failed)

    # The fingerprint is an internal carrier between phases, not part of the
    # response model.
    for molecule in output:
        molecule.pop(FINGERPRINT_KEY, None)

    return output, counts
