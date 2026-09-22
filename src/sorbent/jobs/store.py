"""Job persistence.

``JobStore`` is an abstract base class on purpose. The in-memory implementation
below is enough to develop against, but it dies with the process and does not
survive more than one uvicorn worker. When that becomes a problem, write a
RedisJobStore against this same interface and change one line in the lifespan -
nothing in the routes knows which implementation it is talking to.

Every method is async even though the in-memory one never awaits anything. That
is deliberate: if the interface were sync, swapping in Redis later would mean
changing every call site.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from sorbent.schemas.job import Job, JobCounts, JobProgress, JobStatus


class JobNotFoundError(KeyError):
    """Raised by the store; translated to a 404 by the route layer."""

    def __init__(self, job_id: str) -> None:
        super().__init__(job_id)
        self.job_id = job_id


class JobStore(ABC):
    @abstractmethod
    async def create(self, job: Job) -> Job: ...

    @abstractmethod
    async def get(self, job_id: str) -> Job: ...

    # Named list_jobs, not list: a method called `list` shadows the builtin
    # inside the class body, so every `list[Job]` annotation below it would
    # resolve to the method instead of the type.
    @abstractmethod
    async def list_jobs(self, limit: int = 50, offset: int = 0) -> tuple[list[Job], int]: ...

    @abstractmethod
    async def update_progress(self, job_id: str, progress: JobProgress) -> None: ...

    @abstractmethod
    async def set_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
        counts: JobCounts | None = None,
    ) -> None: ...

    @abstractmethod
    async def set_results(self, job_id: str, results: list[dict[str, Any]]) -> None: ...

    @abstractmethod
    async def get_results(
        self, job_id: str, offset: int = 0, limit: int = 100
    ) -> tuple[list[dict[str, Any]], int]: ...

    @abstractmethod
    def iter_results(self, job_id: str) -> AsyncIterator[dict[str, Any]]: ...

    @abstractmethod
    async def delete(self, job_id: str) -> None: ...


class InMemoryJobStore(JobStore):
    """Dict-backed store, guarded by a lock.

    The lock is not paranoia: a job's runner task and an inbound status request
    touch the same Job object from different coroutines. Without it, a progress
    update interleaved with a status read can serve a half-updated record.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._results: dict[str, list[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def create(self, job: Job) -> Job:
        async with self._lock:
            self._jobs[job.id] = job
            self._results[job.id] = []
            return job

    async def get(self, job_id: str) -> Job:
        async with self._lock:
            try:
                return self._jobs[job_id].model_copy(deep=True)
            except KeyError:
                raise JobNotFoundError(job_id) from None

    async def list_jobs(self, limit: int = 50, offset: int = 0) -> tuple[list[Job], int]:
        async with self._lock:
            ordered = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.model_copy(deep=True) for j in ordered[offset : offset + limit]], len(ordered)

    async def update_progress(self, job_id: str, progress: JobProgress) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            job.progress = progress

    async def set_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
        counts: JobCounts | None = None,
    ) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            job.status = status
            if error is not None:
                job.error = error
            if counts is not None:
                job.counts = counts
            if status is JobStatus.RUNNING and job.started_at is None:
                job.started_at = datetime.now(UTC)
            if status.is_terminal:
                job.finished_at = datetime.now(UTC)

    async def set_results(self, job_id: str, results: list[dict[str, Any]]) -> None:
        async with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            self._results[job_id] = results

    async def get_results(
        self, job_id: str, offset: int = 0, limit: int = 100
    ) -> tuple[list[dict[str, Any]], int]:
        async with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            items = self._results[job_id]
            return items[offset : offset + limit], len(items)

    async def iter_results(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        """Stream results without holding the lock for the whole walk.

        Snapshot under the lock, then yield outside it - otherwise a slow
        client streaming 200k rows blocks every other request for the duration.

        This is an async *generator*, so the JobNotFoundError surfaces on first
        iteration rather than at call time. The route checks existence first.
        """
        async with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            snapshot = list(self._results[job_id])
        for item in snapshot:
            yield item

    async def delete(self, job_id: str) -> None:
        async with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            del self._jobs[job_id]
            self._results.pop(job_id, None)
