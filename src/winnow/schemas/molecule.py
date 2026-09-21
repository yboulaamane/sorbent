"""Per-molecule payloads.

These are the response contract. The ``chem`` layer returns plain dataclass-ish
dicts; the routes validate them into these models on the way out, so a bug in
the science layer surfaces as a 500 with a clear validation error rather than
as malformed JSON.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InputRecord(BaseModel):
    """One line of the submitted library, before any chemistry happens."""

    model_config = ConfigDict(frozen=True)

    smiles: str = Field(description="SMILES exactly as submitted, unmodified.")
    identifier: str | None = Field(
        default=None,
        description="Caller's own ID. Winnow never invents one; if this is "
        "None the index in the submitted list is used.",
    )


class Descriptors(BaseModel):
    """Physicochemical descriptors. All computed, none predicted."""

    molecular_weight: float
    heavy_atoms: int
    total_atoms: int = Field(
        description="Heavy atoms plus hydrogens. Ghose's atom-count criterion "
        "is defined on this, not on heavy atoms - see chem/rules.py."
    )
    clogp: float = Field(description="Crippen logP (RDKit MolLogP).")
    tpsa: float = Field(description="Topological polar surface area, Ertl.")
    hbd: int = Field(description="Lipinski H-bond donors.")
    hba: int = Field(description="Lipinski H-bond acceptors.")
    rotatable_bonds: int
    aromatic_rings: int
    rings: int
    fraction_csp3: float
    formal_charge: int
    stereocentres: int = Field(description="Total, including unassigned.")
    unassigned_stereocentres: int


class RuleResult(BaseModel):
    """Outcome of one drug-likeness rule set."""

    name: str
    passed: bool
    violations: list[str] = Field(
        default_factory=list,
        description="Human-readable violations, e.g. 'MW 512.3 > 500'.",
    )


class Alert(BaseModel):
    """A matched structural-alert substructure."""

    catalog: str = Field(description="PAINS_A, BRENK, NIH, ...")
    name: str = Field(description="Catalog's own name for the alert.")
    description: str | None = None
    atom_indices: list[int] = Field(
        default_factory=list,
        description="Atoms of the match, so a client can highlight it.",
    )


class TriagedMolecule(BaseModel):
    """Everything Winnow has to say about one molecule."""

    identifier: str
    input_smiles: str
    parse_ok: bool
    parse_error: str | None = None

    standard_smiles: str | None = None
    inchikey: str | None = None

    descriptors: Descriptors | None = None
    rules: list[RuleResult] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)

    murcko_scaffold: str | None = Field(
        default=None, description="Bemis-Murcko scaffold as SMILES."
    )
    cluster_id: int | None = Field(
        default=None,
        description="Assigned in the global phase, so it is None until the "
        "whole library has been processed.",
    )
    is_cluster_representative: bool = False

    duplicate_of: str | None = Field(
        default=None,
        description="Identifier of the retained molecule this one duplicates "
        "(same standard InChIKey). None if this record is the one retained.",
    )

    score: float | None = Field(
        default=None, description="Composite triage score; higher is better."
    )
    score_breakdown: dict[str, float] = Field(default_factory=dict)

    @property
    def is_retained(self) -> bool:
        return self.parse_ok and self.duplicate_of is None
