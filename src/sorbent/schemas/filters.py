"""Request payloads: what the caller wants done to their library."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuleSet(StrEnum):
    LIPINSKI = "lipinski"
    VEBER = "veber"
    EGAN = "egan"
    GHOSE = "ghose"
    LEAD_LIKE = "lead_like"
    FRAGMENT = "fragment"


class AlertCatalog(StrEnum):
    PAINS = "pains"
    PAINS_A = "pains_a"
    PAINS_B = "pains_b"
    PAINS_C = "pains_c"
    BRENK = "brenk"
    NIH = "nih"
    ZINC = "zinc"


class DescriptorWindow(BaseModel):
    """An explicit numeric window. Overrides anything a rule set implies.

    Both bounds are inclusive; either may be omitted for a one-sided window.
    """

    model_config = ConfigDict(frozen=True)

    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def _check_order(self) -> DescriptorWindow:
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError(f"minimum {self.minimum} exceeds maximum {self.maximum}")
        return self

    def contains(self, value: float) -> bool:
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True


class TriageConfig(BaseModel):
    """How to triage. Every field has a defensible default."""

    # --- standardisation ----------------------------------------------------
    standardize: bool = Field(
        default=True,
        description="Strip salts, neutralise, canonicalise tautomer, reionise.",
    )
    deduplicate: bool = Field(
        default=True, description="Collapse records sharing a standard InChIKey."
    )

    # --- hard filters -------------------------------------------------------
    rule_sets: list[RuleSet] = Field(
        default=[RuleSet.VEBER],
        description="Evaluated and reported for every molecule. Whether a "
        "failure removes the molecule is controlled by drop_rule_failures. "
        "Lipinski is deliberately NOT a default: the composite score "
        "aggregates geometrically, so failing every requested rule set is "
        "close to disqualifying, and roughly a third of marketed oral drugs "
        "violate Ro5. Request it explicitly when it suits the chemotype.",
    )
    drop_rule_failures: bool = False

    descriptor_windows: dict[str, DescriptorWindow] = Field(
        default={},
        description="Keyed by a field name of Descriptors, e.g. "
        '{"molecular_weight": {"minimum": 250, "maximum": 500}}. '
        "A molecule outside any window is always dropped.",
    )

    alert_catalogs: list[AlertCatalog] = Field(default=[AlertCatalog.PAINS, AlertCatalog.BRENK])
    drop_alert_hits: bool = Field(
        default=False,
        description="False by default on purpose: PAINS matches are a reason "
        "to look closely, not automatic grounds for deletion.",
    )

    # --- clustering ---------------------------------------------------------
    cluster: bool = True
    cluster_cutoff: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Butina distance cutoff (1 - Tanimoto) on ECFP4.",
    )
    fingerprint_radius: int = Field(default=2, ge=1, le=4)
    fingerprint_bits: int = Field(default=2048, ge=512, le=8192)

    # --- output -------------------------------------------------------------
    top_n: int | None = Field(
        default=None,
        ge=1,
        description="Keep only the N highest-scoring retained molecules. None keeps everything.",
    )
    representatives_only: bool = Field(
        default=False,
        description="Return one molecule per cluster (the highest scorer).",
    )

    score_weights: dict[str, float] = Field(
        default={
            "property_centrality": 1.0,
            "alert_penalty": 0.5,
            "complexity_penalty": 0.25,
        },
        description="Weights for the composite score. See chem/score.py. "
        "rule_compliance is deliberately absent: the score aggregates "
        "geometrically, so a component of zero is close to a veto, and "
        "rule_compliance is zero whenever every requested rule set fails. "
        "Rule results are still computed and reported on every molecule - "
        "they just do not move the ranking unless you weight them.",
    )

    @model_validator(mode="after")
    def _check_windows(self) -> TriageConfig:
        from sorbent.schemas.molecule import Descriptors

        allowed = set(Descriptors.model_fields)
        unknown = set(self.descriptor_windows) - allowed
        if unknown:
            raise ValueError(f"unknown descriptor(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
        return self


class TriageRequest(BaseModel):
    """Submit a library inline. For anything large, use the upload endpoint."""

    name: str | None = Field(default=None, max_length=200)
    smiles: list[str] = Field(min_length=1)
    identifiers: list[str] | None = Field(
        default=None,
        description="Parallel to smiles. Must match in length if given.",
    )
    config: TriageConfig = Field(default_factory=TriageConfig)

    @model_validator(mode="after")
    def _check_identifiers(self) -> TriageRequest:
        if self.identifiers is not None and len(self.identifiers) != len(self.smiles):
            raise ValueError(
                f"identifiers has {len(self.identifiers)} entries but smiles has {len(self.smiles)}"
            )
        return self
