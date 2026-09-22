"""Synchronous, small-batch endpoints.

These run the chemistry inline, which is fine for a handful of molecules and
catastrophic for a library - hence the ``max_sync_batch`` cap. The cap is the
whole reason the jobs API exists; keep it low and make the error message point
at the alternative.

Even at 100 molecules this blocks the event loop for tens of milliseconds. If
that shows up in your latency percentiles, push these through the pool too via
``run_in_executor`` - the runner already owns one.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from sorbent.api.deps import SettingsDep
from sorbent.schemas.filters import TriageConfig
from sorbent.schemas.molecule import TriagedMolecule

router = APIRouter(prefix="/v1/molecules", tags=["molecules"])


class DescribeRequest(BaseModel):
    smiles: list[str] = Field(min_length=1)
    identifiers: list[str] | None = None
    config: TriageConfig = Field(default_factory=TriageConfig)


class DescribeResponse(BaseModel):
    molecules: list[TriagedMolecule]


@router.post(
    "/describe",
    response_model=DescribeResponse,
    summary="Descriptors, rules and alerts for a small batch, synchronously",
)
async def describe(request: DescribeRequest, settings: SettingsDep) -> DescribeResponse:
    if len(request.smiles) > settings.max_sync_batch:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"{len(request.smiles)} molecules exceeds the synchronous cap of "
                f"{settings.max_sync_batch}. POST /v1/jobs instead."
            ),
        )

    from sorbent.chem.pipeline import process_chunk

    identifiers = request.identifiers or [str(i) for i in range(len(request.smiles))]
    records = list(zip(identifiers, request.smiles, strict=True))

    try:
        results = process_chunk(records, request.config.model_dump(mode="json"))
    except NotImplementedError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"chem layer is still a stub: {exc}. See src/sorbent/chem/.",
        ) from exc

    return DescribeResponse(molecules=[TriagedMolecule.model_validate(r) for r in results])


class StandardizeRequest(BaseModel):
    smiles: list[str] = Field(min_length=1)


class StandardizedMolecule(BaseModel):
    input_smiles: str
    standard_smiles: str | None = None
    inchikey: str | None = None
    error: str | None = None


class StandardizeResponse(BaseModel):
    molecules: list[StandardizedMolecule]


@router.post(
    "/standardize",
    response_model=StandardizeResponse,
    summary="Salt-strip, neutralise and canonicalise a small batch",
)
async def standardize(request: StandardizeRequest, settings: SettingsDep) -> StandardizeResponse:
    if len(request.smiles) > settings.max_sync_batch:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"{len(request.smiles)} exceeds the synchronous cap of {settings.max_sync_batch}",
        )

    from sorbent.chem.parse import process_record

    out: list[StandardizedMolecule] = []
    for smi in request.smiles:
        try:
            parsed = process_record(smi, standardize_mol=True)
        except NotImplementedError as exc:
            raise HTTPException(
                status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"chem layer is still a stub: {exc}. See src/sorbent/chem/parse.py.",
            ) from exc
        out.append(
            StandardizedMolecule(
                input_smiles=smi,
                standard_smiles=parsed.standard_smiles,
                inchikey=parsed.inchikey,
                error=parsed.error,
            )
        )
    return StandardizeResponse(molecules=out)
