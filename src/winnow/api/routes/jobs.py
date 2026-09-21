"""Triage jobs: submit, poll, page, stream, cancel.

The shape is the standard async-job REST pattern, and it is worth knowing why
each piece is there:

  POST   /v1/jobs            -> 202 Accepted + Location header. Not 200: the
                                work has not happened yet, and saying 200 with
                                an empty result teaches clients to ignore
                                status.
  GET    /v1/jobs/{id}       -> status and progress. Cheap; safe to poll.
  GET    /v1/jobs/{id}/results         -> paginated JSON.
  GET    /v1/jobs/{id}/results.ndjson  -> streamed, one object per line, for
                                result sets too large to hold in a client's
                                memory (or in this process's, as a single
                                serialised body).
  DELETE /v1/jobs/{id}       -> cancel if running, delete if terminal.
"""

from __future__ import annotations

import csv
import io
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from winnow.api.deps import RunnerDep, SettingsDep, StoreDep
from winnow.jobs.store import JobNotFoundError
from winnow.schemas.filters import TriageConfig, TriageRequest
from winnow.schemas.job import Job, JobList, JobStatus, ResultPage
from winnow.schemas.molecule import TriagedMolecule

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _pair_records(smiles: list[str], identifiers: list[str] | None) -> list[tuple[str, str]]:
    """(identifier, smiles) pairs, generating identifiers where absent."""
    if identifiers is None:
        return [(str(i), smi) for i, smi in enumerate(smiles)]
    return list(zip(identifiers, smiles, strict=True))


async def _create_job(
    store: StoreDep,
    runner: RunnerDep,
    settings: SettingsDep,
    *,
    name: str | None,
    records: list[tuple[str, str]],
    config: TriageConfig,
) -> Job:
    if len(records) > settings.max_library_size:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"library has {len(records)} molecules; the cap is "
                f"{settings.max_library_size}. Split it or raise "
                f"WINNOW_MAX_LIBRARY_SIZE."
            ),
        )
    if not records:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "library is empty")

    job = Job(
        id=uuid.uuid4().hex,
        name=name,
        config=config,
        created_at=datetime.now(UTC),
    )
    job.progress.molecules_total = len(records)
    job.counts.submitted = len(records)
    await store.create(job)
    runner.submit(job.id, records, config)
    return job


@router.post(
    "",
    response_model=Job,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a library inline",
)
async def submit_job(
    request: TriageRequest,
    store: StoreDep,
    runner: RunnerDep,
    settings: SettingsDep,
    response: Response,
) -> Job:
    job = await _create_job(
        store,
        runner,
        settings,
        name=request.name,
        records=_pair_records(request.smiles, request.identifiers),
        config=request.config,
    )
    response.headers["Location"] = f"/v1/jobs/{job.id}"
    return job


@router.post(
    "/upload",
    response_model=Job,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a library as a .smi or .csv file",
)
async def upload_job(
    store: StoreDep,
    runner: RunnerDep,
    settings: SettingsDep,
    response: Response,
    file: Annotated[UploadFile, File(description=".smi (SMILES [id] per line) or .csv")],
    name: Annotated[str | None, Form()] = None,
    config_json: Annotated[
        str | None, Form(description="TriageConfig as a JSON string. Defaults if absent.")
    ] = None,
    smiles_column: Annotated[str, Form()] = "smiles",
    id_column: Annotated[str | None, Form()] = None,
) -> Job:
    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"upload is {len(raw)} bytes; the cap is {settings.max_upload_bytes}",
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"file is not UTF-8: {exc}"
        ) from exc

    config = TriageConfig.model_validate_json(config_json) if config_json else TriageConfig()
    filename = (file.filename or "").lower()

    records: list[tuple[str, str]] = []
    if filename.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None or smiles_column not in reader.fieldnames:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"column {smiles_column!r} not found; header is {reader.fieldnames}",
            )
        for i, row in enumerate(reader):
            smi = (row.get(smiles_column) or "").strip()
            if not smi:
                continue
            ident = (row.get(id_column) or "").strip() if id_column else ""
            records.append((ident or str(i), smi))
    else:
        # .smi / .txt: SMILES first, optional whitespace-separated identifier.
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            records.append((parts[1] if len(parts) > 1 else str(i), parts[0]))

    job = await _create_job(
        store, runner, settings, name=name or file.filename, records=records, config=config
    )
    response.headers["Location"] = f"/v1/jobs/{job.id}"
    return job


@router.get("", response_model=JobList, summary="List jobs, newest first")
async def list_jobs(
    store: StoreDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> JobList:
    jobs, total = await store.list_jobs(limit=limit, offset=offset)
    return JobList(jobs=jobs, total=total)


@router.get("/{job_id}", response_model=Job, summary="Job status and progress")
async def get_job(job_id: str, store: StoreDep) -> Job:
    try:
        return await store.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no job {job_id}") from None


@router.get(
    "/{job_id}/results",
    response_model=ResultPage,
    summary="Triaged molecules, paginated",
)
async def get_results(
    job_id: str,
    store: StoreDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> ResultPage:
    try:
        job = await store.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no job {job_id}") from None

    if not job.status.is_terminal:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"job is {job.status}; results are not available until it finishes",
        )
    if job.status is JobStatus.FAILED:
        raise HTTPException(status.HTTP_409_CONFLICT, f"job failed: {job.error}")

    items, total = await store.get_results(job_id, offset=offset, limit=limit)
    end = offset + len(items)
    return ResultPage(
        job_id=job_id,
        items=[TriagedMolecule.model_validate(i) for i in items],
        total=total,
        offset=offset,
        limit=limit,
        next_offset=end if end < total else None,
    )


@router.get(
    "/{job_id}/results.ndjson",
    summary="Triaged molecules as newline-delimited JSON",
    response_class=StreamingResponse,
)
async def stream_results(job_id: str, store: StoreDep) -> StreamingResponse:
    """Stream the full result set.

    NDJSON rather than a JSON array because a client can begin parsing at the
    first newline instead of waiting for the closing bracket, and neither side
    has to hold 200k records in memory at once.
    """
    try:
        job = await store.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no job {job_id}") from None
    if job.status is not JobStatus.SUCCEEDED:
        raise HTTPException(status.HTTP_409_CONFLICT, f"job is {job.status}")

    async def lines() -> AsyncIterator[bytes]:
        async for item in store.iter_results(job_id):
            yield TriagedMolecule.model_validate(item).model_dump_json().encode() + b"\n"

    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.ndjson"'},
    )


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a running job, or delete a finished one",
)
async def delete_job(job_id: str, store: StoreDep, runner: RunnerDep) -> None:
    try:
        job = await store.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no job {job_id}") from None

    if not job.status.is_terminal:
        # Chunks already dispatched to the pool cannot be recalled; see the
        # cancellation note in jobs/runner.py.
        runner.cancel(job_id)
        return
    await store.delete(job_id)
