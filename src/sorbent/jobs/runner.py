"""Executing a triage job off the event loop.

The central problem this file solves:

    RDKit is CPU-bound C++ called from Python. A FastAPI endpoint that parses
    50k molecules inline blocks the event loop for the whole duration - not
    just for that caller, for *every* connection the process is serving. The
    health check times out. The load balancer removes the node.

``BackgroundTasks`` does not fix this. It runs the task on the same event loop
after the response is sent, so the blocking is merely deferred, not removed.

A thread pool is a partial fix - RDKit releases the GIL inside its C++ calls,
so threads do overlap - but the Python-level glue between calls does not, and
you are still capped by one interpreter.

So: a ProcessPoolExecutor, driven from an async supervisor task. The parent
stays responsive, the workers do the chemistry, and results come back over a
pickle boundary. That last part constrains the design - see ``chem.pipeline``.

Cancellation note: a ProcessPoolExecutor future cannot be cancelled once it has
started running. Cancelling a job therefore stops the supervisor from
submitting *further* chunks and marks the job cancelled; chunks already in
flight run to completion and their results are discarded. Say so in the API
docs rather than pretending otherwise.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import pathlib
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from sorbent.jobs.store import JobStore
from sorbent.schemas.filters import TriageConfig
from sorbent.schemas.job import JobCounts, JobProgress, JobStatus

logger = logging.getLogger(__name__)


def _origin_of(exc: BaseException) -> str:
    """Best-effort "file:line in function" for where ``exc`` was actually raised.

    Not as simple as reading ``exc.__traceback__``: when a worker process
    raises, ``concurrent.futures`` rebuilds the exception in the parent and the
    local traceback only contains parent frames. The real one is preserved as a
    ``_RemoteTraceback`` hanging off ``__cause__``, as text. So: parse that when
    it is there, and otherwise walk the local traceback to its deepest frame.
    """
    cause = exc.__cause__
    if cause is not None and type(cause).__name__ == "_RemoteTraceback":
        frames = re.findall(r'File "([^"]+)", line (\d+), in (\S+)', str(cause))
        if frames:
            path, line, fn = frames[-1]
            return f"{pathlib.Path(path).name}:{line} in {fn}()"

    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is None:
        return "the chem layer"
    code = tb.tb_frame.f_code
    return f"{pathlib.Path(code.co_filename).name}:{tb.tb_lineno} in {code.co_qualname}()"


def _chunked(items: list[tuple[str, str]], size: int) -> list[list[tuple[str, str]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class TriageRunner:
    """Owns the process pool and the set of in-flight supervisor tasks.

    One instance per application, created in the lifespan. Creating a pool per
    request would pay the fork/spawn cost - and, worse, rebuild every RDKit
    FilterCatalog - on every call.
    """

    def __init__(self, store: JobStore, *, workers: int, chunk_size: int) -> None:
        self._store = store
        self._chunk_size = chunk_size
        # "spawn", not the Linux default "fork". Forking a process that
        # already has threads (which any ASGI server does) is unsafe - a lock
        # held by a thread that does not exist in the child never unlocks, and
        # you get a deadlock that only shows up under load. CPython warns about
        # this today and changes the default in 3.14.
        #
        # The cost is that each worker re-imports the world at startup. Paid
        # once, at boot, and it is why the pool lives in the lifespan rather
        # than being created per request.
        self._pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancelled: set[str] = set()

    # -- lifecycle -----------------------------------------------------------

    def shutdown(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def submit(self, job_id: str, records: list[tuple[str, str]], config: TriageConfig) -> None:
        """Start a job. Returns immediately; the work happens in the background."""
        task = asyncio.create_task(self._run(job_id, records, config))
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))

    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)
        task = self._tasks.get(job_id)
        if task is not None:
            task.cancel()

    # -- execution -----------------------------------------------------------

    async def _run(self, job_id: str, records: list[tuple[str, str]], config: TriageConfig) -> None:
        loop = asyncio.get_running_loop()
        chunks = _chunked(records, self._chunk_size)

        progress = JobProgress(
            phase="per_molecule",
            molecules_total=len(records),
            chunks_total=len(chunks),
        )

        try:
            await self._store.set_status(job_id, JobStatus.RUNNING)
            await self._store.update_progress(job_id, progress)

            # Import here, not at module scope: it pulls in RDKit, and the API
            # should boot and serve /health even if the chem layer is broken
            # or RDKit is not installed yet.
            from sorbent.chem.pipeline import finalize, process_chunk

            config_json = config.model_dump(mode="json")

            futures = [
                loop.run_in_executor(self._pool, process_chunk, chunk, config_json)
                for chunk in chunks
            ]

            molecules: list[dict[str, Any]] = []
            for future in asyncio.as_completed(futures):
                if job_id in self._cancelled:
                    raise asyncio.CancelledError
                chunk_result = await future
                molecules.extend(chunk_result)
                progress.chunks_completed += 1
                progress.molecules_processed = len(molecules)
                await self._store.update_progress(job_id, progress)

            # as_completed destroys submission order, and a triage report that
            # reshuffles the caller's library on every run is maddening to
            # diff. Restore it before the global phase.
            order = {ident: i for i, (ident, _) in enumerate(records)}
            molecules.sort(key=lambda m: order.get(m.get("identifier", ""), 0))

            progress.phase = "deduplicate"
            await self._store.update_progress(job_id, progress)

            # finalize is global and single-threaded by nature. It runs in a
            # worker too, so a 200k-molecule clustering pass does not block the
            # loop either.
            molecules, counts = await loop.run_in_executor(self._pool, finalize, molecules, config)

            progress.phase = "done"
            progress.chunks_completed = progress.chunks_total
            await self._store.update_progress(job_id, progress)

            await self._store.set_results(job_id, molecules)
            await self._store.set_status(job_id, JobStatus.SUCCEEDED, counts=JobCounts(**counts))

        except asyncio.CancelledError:
            self._cancelled.discard(job_id)
            await self._store.set_status(job_id, JobStatus.CANCELLED, error="cancelled by request")
            raise

        except NotImplementedError as exc:
            # The expected failure while the chem layer is still stubs. Make it
            # legible rather than a generic 500 with a traceback.
            await self._store.set_status(
                job_id,
                JobStatus.FAILED,
                error=f"not implemented yet: {_origin_of(exc)}",
            )

        except Exception as exc:  # noqa: BLE001 - a job failure is not a crash
            logger.exception("job %s failed", job_id)
            await self._store.set_status(
                job_id, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}"
            )
