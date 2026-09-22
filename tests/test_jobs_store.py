"""The store contract. Write these same tests against a RedisJobStore later and
you will know the swap is safe."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sorbent.jobs.store import InMemoryJobStore, JobNotFoundError
from sorbent.schemas.filters import TriageConfig
from sorbent.schemas.job import Job, JobCounts, JobProgress, JobStatus


def _job(job_id: str = "j1") -> Job:
    return Job(id=job_id, config=TriageConfig(), created_at=datetime.now(UTC))


async def test_create_and_get_roundtrip():
    store = InMemoryJobStore()
    await store.create(_job())
    assert (await store.get("j1")).id == "j1"


async def test_get_unknown_raises():
    with pytest.raises(JobNotFoundError):
        await InMemoryJobStore().get("nope")


async def test_get_returns_a_copy():
    """Callers must not be able to mutate stored state by holding the result."""
    store = InMemoryJobStore()
    await store.create(_job())
    fetched = await store.get("j1")
    fetched.status = JobStatus.SUCCEEDED
    assert (await store.get("j1")).status is JobStatus.PENDING


async def test_running_sets_started_at_once():
    store = InMemoryJobStore()
    await store.create(_job())
    await store.set_status("j1", JobStatus.RUNNING)
    first = (await store.get("j1")).started_at
    await store.set_status("j1", JobStatus.RUNNING)
    assert (await store.get("j1")).started_at == first


async def test_terminal_sets_finished_at():
    store = InMemoryJobStore()
    await store.create(_job())
    await store.set_status("j1", JobStatus.RUNNING)
    await store.set_status("j1", JobStatus.SUCCEEDED, counts=JobCounts(submitted=5))
    job = await store.get("j1")
    assert job.finished_at is not None
    assert job.counts.submitted == 5
    assert job.duration_seconds is not None


async def test_results_pagination():
    store = InMemoryJobStore()
    await store.create(_job())
    await store.set_results("j1", [{"i": i} for i in range(10)])

    page, total = await store.get_results("j1", offset=0, limit=4)
    assert total == 10 and [p["i"] for p in page] == [0, 1, 2, 3]

    page, _ = await store.get_results("j1", offset=8, limit=4)
    assert [p["i"] for p in page] == [8, 9]

    page, _ = await store.get_results("j1", offset=99, limit=4)
    assert page == []


async def test_iter_results_yields_everything():
    store = InMemoryJobStore()
    await store.create(_job())
    await store.set_results("j1", [{"i": i} for i in range(5)])
    assert [x["i"] async for x in store.iter_results("j1")] == [0, 1, 2, 3, 4]


async def test_list_is_newest_first():
    store = InMemoryJobStore()
    for i in range(3):
        job = _job(f"j{i}")
        job.created_at = datetime(2026, 1, i + 1, tzinfo=UTC)
        await store.create(job)
    jobs, total = await store.list_jobs()
    assert total == 3
    assert [j.id for j in jobs] == ["j2", "j1", "j0"]


async def test_delete_removes_job_and_results():
    store = InMemoryJobStore()
    await store.create(_job())
    await store.set_results("j1", [{"i": 1}])
    await store.delete("j1")
    with pytest.raises(JobNotFoundError):
        await store.get("j1")


async def test_progress_fraction():
    p = JobProgress(chunks_total=4, chunks_completed=1)
    assert p.fraction == 0.25
    assert JobProgress().fraction == 0.0
