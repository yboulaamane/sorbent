"""Contract tests for the chem layer.

These are the specification for what you are implementing. They are marked
``chem`` and skipped when RDKit is absent; every one of them fails against the
stubs. Run just these while you work:

    pytest -m chem -x

The point of each assertion is a real property of the science, not a
tautology - read them before implementing the function they cover.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
pytestmark = pytest.mark.chem


# --- parse ------------------------------------------------------------------


def test_parse_valid_smiles(aspirin):
    from winnow.chem.parse import parse_smiles

    assert parse_smiles(aspirin) is not None


@pytest.mark.parametrize("bad", ["", "   ", "not_a_smiles", "C(((", "[Xx]", "\t"])
def test_parse_returns_none_and_never_raises(bad):
    """A 200k-compound vendor file WILL contain these. One bad line must not
    take down a chunk."""
    from winnow.chem.parse import parse_smiles

    assert parse_smiles(bad) is None


def test_standardize_strips_salt():
    """The sodium must go, and the parent must survive intact."""
    from winnow.chem.parse import process_record

    salt = process_record("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
    free = process_record("CC(=O)Oc1ccccc1C(=O)O")
    assert salt.error is None and free.error is None
    assert salt.inchikey == free.inchikey


def test_standardize_is_idempotent(aspirin):
    """Standardising twice must not keep changing the answer - otherwise your
    InChIKeys depend on how many times a record went through the pipeline."""
    from winnow.chem.parse import process_record

    once = process_record(aspirin).standard_smiles
    twice = process_record(once).standard_smiles
    assert once == twice


def test_standardize_does_not_mutate_input(aspirin):
    from rdkit import Chem

    from winnow.chem.parse import parse_smiles, standardize

    mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
    before = Chem.MolToSmiles(mol)
    standardize(mol)
    assert Chem.MolToSmiles(mol) == before


def test_inchikey_shape(aspirin):
    from winnow.chem.parse import parse_smiles, to_inchikey

    key = to_inchikey(parse_smiles(aspirin))
    assert key is not None
    assert len(key) == 27 and key.count("-") == 2


def test_process_record_reports_error_not_raises():
    from winnow.chem.parse import process_record

    result = process_record("C(((")
    assert result.mol is None
    assert result.error is not None
    assert result.standard_smiles is None


@pytest.mark.parametrize(
    "smiles",
    [
        "C[C@H](N)C(=O)O",  # L-alanine
        "C[C@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",  # a dipeptide
        "CN1CCC[C@H]1c1cccnc1",  # nicotine
    ],
)
def test_standardisation_preserves_defined_stereo(smiles):
    """Regression: RDKit's TautomerEnumerator strips sp3 stereo by DEFAULT.

    CleanupParameters.tautomerRemoveSp3Stereo is True out of the box, which
    removes stereo from any centre adjacent to a tautomerisable system - the
    alpha carbon of every amino acid. Without the flag set False, this test
    sees L-alanine come back racemic.
    """
    from winnow.chem.parse import process_record

    result = process_record(smiles)
    assert result.error is None
    assert "@" in result.standard_smiles, f"stereo lost: {smiles} -> {result.standard_smiles}"


def test_standardisation_preserves_double_bond_geometry():
    from winnow.chem.parse import process_record

    assert "/" in process_record("C/C=C/C(=O)O").standard_smiles


@pytest.mark.parametrize("placeholder", ["*", "[*]", "[*][*]"])
def test_dummy_atom_only_records_are_rejected(placeholder):
    """R-group placeholders parse and sanitise cleanly, then report MW 0.

    Same failure class as MolFromSmiles("") returning an empty Mol: valid to
    RDKit, not a compound.
    """
    from winnow.chem.parse import parse_smiles

    assert parse_smiles(placeholder) is None


def test_attachment_points_on_a_real_fragment_are_kept():
    """A dummy atom alongside real atoms is an ordinary fragment-library
    attachment point, and must survive."""
    from winnow.chem.parse import parse_smiles

    assert parse_smiles("*c1ccccc1") is not None


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("CC(=O)CC(=O)C", "CC(O)=CC(=O)C"),  # keto / enol
        ("Oc1ccccn1", "O=c1cccc[nH]1"),  # 2-pyridone / 2-hydroxypyridine
    ],
)
def test_tautomers_collapse_to_one_key(a, b):
    """What the expensive tautomer pass buys: these dedupe against each other."""
    from winnow.chem.parse import process_record

    assert process_record(a).inchikey == process_record(b).inchikey


def test_skipping_tautomer_canonicalisation_is_allowed(aspirin):
    from winnow.chem.parse import parse_smiles, standardize

    assert standardize(parse_smiles(aspirin), canonical_tautomer=False) is not None


@pytest.mark.parametrize(
    "junk",
    ["", "   ", "\t", "\n", "nan", "None", "N/A", "smiles", '"CCO"', "CCO,extra,cols"],
)
def test_spreadsheet_junk_is_rejected_cleanly(junk):
    """The contents of a real vendor SMILES column."""
    from winnow.chem.parse import process_record

    result = process_record(junk)
    assert result.mol is None
    assert result.error is not None


def test_mol_is_none_exactly_when_error_is_set():
    """The ParsedMolecule contract the pipeline relies on."""
    from winnow.chem.parse import process_record

    for smiles in ["CCO", "", "C(((", "c1ccccc1", "garbage", "*", "[Na+].[Cl-]"]:
        result = process_record(smiles)
        assert (result.mol is None) == (result.error is not None), smiles


# --- descriptors ------------------------------------------------------------


def test_descriptor_keys_match_schema(aspirin):
    """The dict must line up with the response model exactly, or the route
    layer will 500 on validation."""
    from winnow.chem.descriptors import DESCRIPTOR_NAMES, compute_descriptors
    from winnow.chem.parse import parse_smiles
    from winnow.schemas.molecule import Descriptors

    desc = compute_descriptors(parse_smiles(aspirin))
    assert set(desc) == set(DESCRIPTOR_NAMES)
    assert set(desc) == set(Descriptors.model_fields)
    Descriptors(**desc)


def test_aspirin_descriptors_are_right(aspirin):
    """Known values. If these drift, something in standardisation changed."""
    from winnow.chem.descriptors import compute_descriptors
    from winnow.chem.parse import parse_smiles

    d = compute_descriptors(parse_smiles(aspirin))
    assert d["molecular_weight"] == pytest.approx(180.16, abs=0.1)
    assert d["heavy_atoms"] == 13
    assert d["hbd"] == 1
    assert d["hba"] == 3
    assert d["aromatic_rings"] == 1
    assert d["tpsa"] == pytest.approx(63.6, abs=0.5)


def test_unassigned_stereocentres_are_counted():
    """Undefined stereo is a purchasing problem worth flagging."""
    from winnow.chem.descriptors import compute_descriptors
    from winnow.chem.parse import parse_smiles

    defined = compute_descriptors(parse_smiles("C[C@H](N)C(=O)O"))
    undefined = compute_descriptors(parse_smiles("CC(N)C(=O)O"))
    assert defined["unassigned_stereocentres"] == 0
    assert undefined["unassigned_stereocentres"] == 1


@pytest.mark.parametrize(
    ("smiles", "total", "unassigned"),
    [
        ("C[C@H](O)[C@@H](N)C(=O)O", 2, 0),  # both defined
        ("C[C@H](O)C(N)C(=O)O", 2, 1),  # one of two defined
        ("CC(O)C(N)C(=O)O", 2, 2),  # neither
        ("CCCCO", 0, 0),  # no centres at all
    ],
)
def test_partial_stereo_assignment_is_counted_correctly(smiles, total, unassigned):
    from winnow.chem.descriptors import compute_descriptors
    from winnow.chem.parse import parse_smiles

    desc = compute_descriptors(parse_smiles(smiles))
    assert desc["stereocentres"] == total
    assert desc["unassigned_stereocentres"] == unassigned


def test_double_bond_geometry_is_not_counted_as_a_stereocentre():
    """Documents a known limitation rather than asserting it is correct.

    FindMolChiralCenters sees atoms only, so an undefined E/Z alkene is not
    flagged even though it is the same purchasing problem. If this test ever
    starts failing, someone has moved to FindPotentialStereo - update the
    Descriptors schema to match.
    """
    from winnow.chem.descriptors import compute_descriptors
    from winnow.chem.parse import parse_smiles

    assert compute_descriptors(parse_smiles("CC=CC(=O)O"))["unassigned_stereocentres"] == 0


@pytest.mark.parametrize(
    ("name", "smiles", "mw"),
    [
        ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", 194.19),
        ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", 206.29),
        ("paracetamol", "CC(=O)Nc1ccc(O)cc1", 151.16),
        ("nicotine", "CN1CCC[C@H]1c1cccnc1", 162.24),
    ],
)
def test_molecular_weight_matches_literature(name, smiles, mw):
    """Average mass, not monoisotopic - ExactMolWt would fail every one."""
    from winnow.chem.descriptors import compute_descriptors
    from winnow.chem.parse import parse_smiles

    desc = compute_descriptors(parse_smiles(smiles))
    assert desc["molecular_weight"] == pytest.approx(mw, abs=0.05)


def test_counts_are_python_ints_not_floats():
    """Pydantic declares these as int. A numpy scalar or a float would either
    fail validation or serialise as 13.0, and only show up in production."""
    from winnow.chem.descriptors import compute_descriptors
    from winnow.chem.parse import parse_smiles

    desc = compute_descriptors(parse_smiles("CC(=O)Oc1ccccc1C(=O)O"))
    integral = (
        "heavy_atoms",
        "hbd",
        "hba",
        "rotatable_bonds",
        "aromatic_rings",
        "rings",
        "formal_charge",
        "stereocentres",
        "unassigned_stereocentres",
    )
    for key in integral:
        assert type(desc[key]) is int, f"{key} is {type(desc[key]).__name__}"
    for key in ("molecular_weight", "clogp", "tpsa", "fraction_csp3"):
        assert type(desc[key]) is float, f"{key} is {type(desc[key]).__name__}"


def test_formal_charge_is_reported():
    from rdkit import Chem

    from winnow.chem.descriptors import compute_descriptors

    # Not via parse_smiles: standardisation would neutralise it.
    assert compute_descriptors(Chem.MolFromSmiles("CC(=O)[O-]"))["formal_charge"] == -1
    assert compute_descriptors(Chem.MolFromSmiles("C[NH3+]"))["formal_charge"] == 1


@pytest.mark.parametrize(
    "smiles",
    [
        "O",  # no carbon at all - Fsp3 must not divide by zero
        "OB(O)c1ccccc1",  # boron
        "N.N.Cl[Pt]Cl",  # cisplatin - Crippen has no Pt parameter
        "C[Se]C",  # selenium
        "C[Si](C)(C)C",  # silicon
        "[13CH4]",  # isotope
        "C[CH2]",  # radical
        "*c1ccccc1",  # attachment point
    ],
)
def test_awkward_chemistry_does_not_raise(smiles):
    """A vendor library contains all of these. None may kill a chunk."""
    from rdkit import Chem

    from winnow.chem.descriptors import DESCRIPTOR_NAMES, compute_descriptors

    desc = compute_descriptors(Chem.MolFromSmiles(smiles))
    assert set(desc) == set(DESCRIPTOR_NAMES)


def test_carbon_free_molecule_has_zero_fsp3():
    from rdkit import Chem

    from winnow.chem.descriptors import compute_descriptors

    assert compute_descriptors(Chem.MolFromSmiles("O"))["fraction_csp3"] == 0.0


# --- rules ------------------------------------------------------------------


def test_lipinski_allows_one_violation():
    """The single most commonly mis-implemented rule in cheminformatics."""
    from winnow.chem.rules import lipinski

    one_violation = {"molecular_weight": 520, "clogp": 3.0, "hbd": 2, "hba": 5}
    passed, violations = lipinski(one_violation)
    assert passed is True
    assert len(violations) == 1


def test_lipinski_fails_on_two_violations():
    from winnow.chem.rules import lipinski

    passed, violations = lipinski({"molecular_weight": 520, "clogp": 6.0, "hbd": 2, "hba": 5})
    assert passed is False
    assert len(violations) == 2


def test_veber_requires_both():
    from winnow.chem.rules import veber

    assert veber({"rotatable_bonds": 5, "tpsa": 90})[0] is True
    assert veber({"rotatable_bonds": 12, "tpsa": 90})[0] is False
    assert veber({"rotatable_bonds": 5, "tpsa": 160})[0] is False


def test_rule_functions_registry_is_complete():
    from winnow.chem.rules import RULE_FUNCTIONS
    from winnow.schemas.filters import RuleSet

    assert set(RULE_FUNCTIONS) == set(RuleSet)


def test_evaluate_shape_matches_schema(aspirin):
    from winnow.chem.descriptors import compute_descriptors
    from winnow.chem.parse import parse_smiles
    from winnow.chem.rules import evaluate
    from winnow.schemas.filters import RuleSet
    from winnow.schemas.molecule import RuleResult

    desc = compute_descriptors(parse_smiles(aspirin))
    results = evaluate(desc, [RuleSet.LIPINSKI, RuleSet.VEBER])
    assert len(results) == 2
    for r in results:
        RuleResult(**r)


# --- alerts -----------------------------------------------------------------


def test_catalog_is_cached():
    """Rebuilding a FilterCatalog per molecule makes this the slowest stage in
    the pipeline by roughly an order of magnitude."""
    from winnow.chem.alerts import get_catalog
    from winnow.schemas.filters import AlertCatalog

    assert get_catalog(AlertCatalog.PAINS) is get_catalog(AlertCatalog.PAINS)


def test_clean_molecule_has_no_pains(aspirin):
    from winnow.chem.alerts import find_alerts
    from winnow.chem.parse import parse_smiles
    from winnow.schemas.filters import AlertCatalog

    assert find_alerts(parse_smiles(aspirin), [AlertCatalog.PAINS]) == []


def test_known_pains_is_flagged():
    """A catechol / quinone-forming motif - a textbook PAINS hit."""
    from winnow.chem.alerts import find_alerts
    from winnow.chem.parse import parse_smiles
    from winnow.schemas.filters import AlertCatalog
    from winnow.schemas.molecule import Alert

    mol = parse_smiles("Oc1ccccc1O")
    hits = find_alerts(mol, [AlertCatalog.PAINS, AlertCatalog.BRENK])
    assert hits, "expected catechol to trip at least one alert catalog"
    for h in hits:
        Alert(**h)


# --- scaffolds --------------------------------------------------------------


def test_murcko_scaffold_of_aspirin(aspirin):
    from winnow.chem.parse import parse_smiles
    from winnow.chem.scaffolds import murcko_scaffold

    assert murcko_scaffold(parse_smiles(aspirin)) == "c1ccccc1"


def test_acyclic_molecule_has_no_scaffold():
    """None, not '' - so a client can tell 'acyclic' from 'failed'."""
    from winnow.chem.parse import parse_smiles
    from winnow.chem.scaffolds import murcko_scaffold

    assert murcko_scaffold(parse_smiles("CCCCO")) is None


# --- fingerprints -----------------------------------------------------------


def test_identical_molecules_have_tanimoto_one(aspirin):
    from winnow.chem.fingerprints import compute_fingerprint, tanimoto
    from winnow.chem.parse import parse_smiles

    fp = compute_fingerprint(parse_smiles(aspirin))
    assert tanimoto(fp, fp) == pytest.approx(1.0)


def test_similar_beats_dissimilar(aspirin, caffeine):
    """Aspirin is closer to salicylic acid than to caffeine. If this fails your
    fingerprint is wired up wrong."""
    from winnow.chem.fingerprints import compute_fingerprint, tanimoto
    from winnow.chem.parse import parse_smiles

    a = compute_fingerprint(parse_smiles(aspirin))
    salicylic = compute_fingerprint(parse_smiles("OC(=O)c1ccccc1O"))
    caf = compute_fingerprint(parse_smiles(caffeine))
    assert tanimoto(a, salicylic) > tanimoto(a, caf)


def test_bulk_matches_pairwise(small_library):
    from winnow.chem.fingerprints import bulk_tanimoto, compute_fingerprint, tanimoto
    from winnow.chem.parse import parse_smiles

    fps = [compute_fingerprint(parse_smiles(s)) for s in small_library]
    assert bulk_tanimoto(fps[0], fps[1:]) == pytest.approx([tanimoto(fps[0], f) for f in fps[1:]])


# --- clustering -------------------------------------------------------------


def test_distance_matrix_is_lower_triangle(small_library):
    from winnow.chem.cluster import build_distance_matrix
    from winnow.chem.fingerprints import compute_fingerprint
    from winnow.chem.parse import parse_smiles

    n = len(small_library)
    fps = [compute_fingerprint(parse_smiles(s)) for s in small_library]
    dists = build_distance_matrix(fps)
    assert len(dists) == n * (n - 1) // 2
    assert all(0.0 <= d <= 1.0 for d in dists)


def test_every_molecule_lands_in_exactly_one_cluster(small_library):
    from winnow.chem.cluster import assign_clusters, butina_cluster
    from winnow.chem.fingerprints import compute_fingerprint
    from winnow.chem.parse import parse_smiles

    fps = [compute_fingerprint(parse_smiles(s)) for s in small_library]
    clusters = butina_cluster(fps, cutoff=0.4)
    flat = [i for c in clusters for i in c]
    assert sorted(flat) == list(range(len(small_library)))

    ids = assign_clusters(clusters, len(small_library))
    assert len(ids) == len(small_library)
    assert all(i is not None for i in ids)


def test_clusters_are_ordered_largest_first(small_library):
    from winnow.chem.cluster import butina_cluster
    from winnow.chem.fingerprints import compute_fingerprint
    from winnow.chem.parse import parse_smiles

    fps = [compute_fingerprint(parse_smiles(s)) for s in small_library]
    sizes = [len(c) for c in butina_cluster(fps, cutoff=0.6)]
    assert sizes == sorted(sizes, reverse=True)


def test_oversized_input_is_refused_clearly():
    from winnow.chem.cluster import MAX_EXACT_CLUSTER_SIZE, butina_cluster

    with pytest.raises(ValueError, match="(?i)cluster"):
        butina_cluster([None] * (MAX_EXACT_CLUSTER_SIZE + 1))


# --- score ------------------------------------------------------------------


def test_score_is_bounded_regardless_of_weights(aspirin):
    from winnow.chem.descriptors import compute_descriptors
    from winnow.chem.parse import parse_smiles
    from winnow.chem.score import composite_score

    desc = compute_descriptors(parse_smiles(aspirin))
    for weights in (
        {"rule_compliance": 1.0},
        {"rule_compliance": 100.0, "alert_penalty": 0.001},
        {
            "rule_compliance": 1,
            "alert_penalty": 1,
            "property_centrality": 1,
            "complexity_penalty": 1,
        },
    ):
        score, breakdown = composite_score(desc, [], 0, weights)
        assert 0.0 <= score <= 1.0
        assert breakdown


def test_alerts_lower_the_score(aspirin):
    from winnow.chem.descriptors import compute_descriptors
    from winnow.chem.parse import parse_smiles
    from winnow.chem.score import composite_score

    desc = compute_descriptors(parse_smiles(aspirin))
    weights = {"rule_compliance": 1.0, "alert_penalty": 1.0}
    clean, _ = composite_score(desc, [], 0, weights)
    dirty, _ = composite_score(desc, [], 3, weights)
    assert clean > dirty


def test_breakdown_reports_unweighted_components(aspirin):
    """The caller must be able to see why a molecule scored what it did."""
    from winnow.chem.descriptors import compute_descriptors
    from winnow.chem.parse import parse_smiles
    from winnow.chem.score import composite_score

    desc = compute_descriptors(parse_smiles(aspirin))
    _, breakdown = composite_score(desc, [], 0, {"rule_compliance": 1.0, "alert_penalty": 1.0})
    assert "rule_compliance" in breakdown and "alert_penalty" in breakdown
    assert breakdown["alert_penalty"] == pytest.approx(1.0)
