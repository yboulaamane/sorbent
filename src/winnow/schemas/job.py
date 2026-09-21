"""Job lifecycle payloads."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from winnow.schemas.filters import TriageConfig
from winnow.schemas.molecule import TriagedMolecule


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}


class JobProgress(BaseModel):
    """Coarse progress. Chunk-level, because that is the unit of work."""

    phase: str = Field(
        default="queued",
        description="queued | per_molecule | deduplicate | cluster | score | done",
    )
    molecules_total: int = 0
    molecules_processed: int = 0
    chunks_total: int = 0
    chunks_completed: int = 0

    @property
    def fraction(self) -> float:
        if not self.chunks_total:
            return 0.0
        return self.chunks_completed / self.chunks_total


class JobCounts(BaseModel):
    """Where the library went. These should always reconcile against submitted."""

    submitted: int = 0
    parsed: int = 0
    parse_failed: int = 0
    duplicates_removed: int = 0
    dropped_by_window: int = 0
    dropped_by_rule: int = 0
    dropped_by_alert: int = 0
    retained: int = 0
    clusters: int = 0


class Job(BaseModel):
    id: str
    name: str | None = None
    status: JobStatus = JobStatus.PENDING
    config: TriageConfig

    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    progress: JobProgress = Field(default_factory=JobProgress)
    counts: JobCounts = Field(default_factory=JobCounts)

    error: str | None = Field(
        default=None, description="Set when status is failed. Never a traceback."
    )

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or datetime.now(tz=self.started_at.tzinfo)
        return (end - self.started_at).total_seconds()


class JobList(BaseModel):
    jobs: list[Job]
    total: int


class ResultPage(BaseModel):
    """One page of triaged molecules.

    ``next_offset`` is None at the end of the collection, so a client loops
    until it is None rather than comparing counts.
    """

    job_id: str
    items: list[TriagedMolecule]
    total: int
    offset: int
    limit: int
    next_offset: int | None = None
