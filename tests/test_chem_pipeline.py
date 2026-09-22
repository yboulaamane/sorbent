"""Pipeline contract: the phase-one / phase-two boundary.

The subtle bugs in a parallel triage service all live here. These tests pin the
invariants that parallelism most easily breaks.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
pytestmark = pytest.mark.chem


def _records(smiles):
    return [(str(i), s) for i, s in enumerate(smiles)]


def test_process_chunk_never_raises_on_bad_input():
    """One malformed record must not kill the chunk."""
    from sorbent.chem.pipeline import process_chunk
    from sorbent.schemas.filters import TriageConfig

    records = _records(["CCO", "C(((", "", "not_a_smiles", "c1ccccc1"])
    out = process_chunk(records, TriageConfig().model_dump(mode="json"))
    assert len(out) == 5
    assert sum(m["parse_ok"] for m in out) == 2
    for m in out:
        if not m["parse_ok"]:
            assert m["parse_error"]


def test_process_chunk_output_validates_against_schema(small_library):
    from sorbent.chem.pipeline import process_chunk
    from sorbent.schemas.filters import TriageConfig
    from sorbent.schemas.molecule import TriagedMolecule

    out = process_chunk(_records(small_library), TriageConfig().model_dump(mode="json"))
    for m in out:
        TriagedMolecule.model_validate(m)


def test_process_chunk_leaves_global_fields_unset(small_library):
    """cluster_id and duplicate_of are phase two's job. A worker that sets them
    is a worker that has guessed, because it cannot see the other chunks."""
    from sorbent.chem.pipeline import process_chunk
    from sorbent.schemas.filters import TriageConfig

    out = process_chunk(_records(small_library), TriageConfig().model_dump(mode="json"))
    for m in out:
        assert m.get("cluster_id") is None
        assert m.get("duplicate_of") is None


def test_process_chunk_is_picklable_and_deterministic(small_library):
    """It crosses a process boundary, and two runs must agree."""
    import pickle

    from sorbent.chem.pipeline import process_chunk
    from sorbent.schemas.filters import TriageConfig

    cfg = TriageConfig().model_dump(mode="json")
    a = process_chunk(_records(small_library), cfg)
    b = process_chunk(_records(small_library), cfg)
    assert [m["standard_smiles"] for m in a] == [m["standard_smiles"] for m in b]
    pickle.dumps(a)


def test_finalize_deduplicates_and_points_at_the_survivor():
    """Charged and neutral aspirin are the same compound. The duplicate is
    reported, not silently dropped - a 12% redundant library is news."""
    from sorbent.chem.pipeline import finalize, process_chunk
    from sorbent.schemas.filters import TriageConfig

    cfg = TriageConfig()
    records = _records(
        [
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]",
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        ]
    )
    out = process_chunk(records, cfg.model_dump(mode="json"))
    molecules, counts = finalize(out, cfg)

    assert counts["duplicates_removed"] == 1
    dupes = [m for m in molecules if m.get("duplicate_of") is not None]
    assert len(dupes) == 1
    assert dupes[0]["duplicate_of"] == "0"


def test_finalize_counts_reconcile(small_library):
    from sorbent.chem.pipeline import finalize, process_chunk
    from sorbent.schemas.filters import TriageConfig

    cfg = TriageConfig()
    records = _records([*small_library, "C((("])
    out = process_chunk(records, cfg.model_dump(mode="json"))
    _, counts = finalize(out, cfg)
    assert counts["parsed"] + counts["parse_failed"] == counts["submitted"]
    assert counts["submitted"] == len(records)
    assert counts["parse_failed"] == 1


def test_finalize_is_order_independent(small_library):
    """as_completed returns chunks in whatever order they finish. The report
    must not depend on that."""
    from sorbent.chem.pipeline import finalize, process_chunk
    from sorbent.schemas.filters import TriageConfig

    cfg = TriageConfig()
    records = _records(small_library)
    out = process_chunk(records, cfg.model_dump(mode="json"))

    forward, counts_a = finalize(list(out), cfg)
    reverse, counts_b = finalize(list(reversed(out)), cfg)

    assert counts_a == counts_b
    key = lambda ms: sorted((m["identifier"], m.get("score")) for m in ms)  # noqa: E731
    assert key(forward) == key(reverse)


def test_descriptor_window_drops_and_is_counted(small_library):
    from sorbent.chem.pipeline import finalize, process_chunk
    from sorbent.schemas.filters import DescriptorWindow, TriageConfig

    cfg = TriageConfig(descriptor_windows={"molecular_weight": DescriptorWindow(maximum=150.0)})
    out = process_chunk(_records(small_library), cfg.model_dump(mode="json"))
    molecules, counts = finalize(out, cfg)
    assert counts["dropped_by_window"] > 0
    assert counts["retained"] < counts["parsed"]
    for m in molecules:
        if m.get("descriptors") and m.get("score") is not None:
            assert m["descriptors"]["molecular_weight"] <= 150.0


def test_top_n_applied_after_representative_selection(small_library):
    """Order matters: top_n first would drop whole clusters."""
    from sorbent.chem.pipeline import finalize, process_chunk
    from sorbent.schemas.filters import TriageConfig

    cfg = TriageConfig(cluster=True, representatives_only=True, top_n=2)
    out = process_chunk(_records(small_library), cfg.model_dump(mode="json"))
    molecules, _ = finalize(out, cfg)
    scored = [m for m in molecules if m.get("score") is not None]
    assert len(scored) <= 2
    assert all(m["is_cluster_representative"] for m in scored)


def test_results_are_sorted_by_score_descending(small_library):
    from sorbent.chem.pipeline import finalize, process_chunk
    from sorbent.schemas.filters import TriageConfig

    cfg = TriageConfig()
    out = process_chunk(_records(small_library), cfg.model_dump(mode="json"))
    molecules, _ = finalize(out, cfg)
    scores = [m["score"] for m in molecules if m.get("score") is not None]
    assert scores == sorted(scores, reverse=True)


def test_fingerprints_are_stripped_from_output(small_library):
    """_fingerprint is an internal carrier between phases, not part of the
    response model."""
    from sorbent.chem.pipeline import finalize, process_chunk
    from sorbent.schemas.filters import TriageConfig

    cfg = TriageConfig()
    out = process_chunk(_records(small_library), cfg.model_dump(mode="json"))
    molecules, _ = finalize(out, cfg)
    for m in molecules:
        assert "_fingerprint" not in m
