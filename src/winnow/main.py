"""Application factory and lifespan.

``create_app()`` is a function, not a module-level ``app = FastAPI()``, for one
practical reason: tests need to build an app with a different store and a
different settings object, and a module-level instance makes that a
monkeypatching exercise.

The lifespan owns the two expensive objects - the job store and the process
pool - and tears them down on shutdown. A ProcessPoolExecutor that is never
shut down leaves orphaned children behind on reload, which during development
quietly eats your cores.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from winnow import __version__
from winnow.api.routes import health, jobs, molecules
from winnow.config import Settings, get_settings
from winnow.jobs.runner import TriageRunner
from winnow.jobs.store import InMemoryJobStore, JobNotFoundError

logger = logging.getLogger(__name__)

DESCRIPTION = """
Submit a compound library, get back a ranked, deduplicated, liability-flagged
shortlist.

**Everything Winnow reports is computed, not predicted.** Descriptors come from
RDKit, rule sets are the published literature definitions, structural alerts are
RDKit's bundled catalogs, and the composite score is a transparent weighted sum
whose breakdown ships with every molecule. There is no trained model anywhere in
this service, and therefore nothing here is an affinity, an activity, or a
probability of success.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    store = InMemoryJobStore()
    runner = TriageRunner(
        store,
        workers=settings.effective_workers,
        chunk_size=settings.chunk_size,
    )

    app.state.store = store
    app.state.runner = runner

    logger.info(
        "%s %s up: %d worker processes, chunk size %d",
        settings.app_name,
        __version__,
        settings.effective_workers,
        settings.chunk_size,
    )
    try:
        yield
    finally:
        runner.shutdown()
        logger.info("shut down")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Liveness and readiness probes."},
            {"name": "jobs", "description": "Asynchronous library triage."},
            {"name": "molecules", "description": "Synchronous small-batch chemistry."},
        ],
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(JobNotFoundError)
    async def _job_not_found(_: Request, exc: JobNotFoundError) -> JSONResponse:
        """Catch the store's own exception at the edge.

        Without this, a JobNotFoundError raised inside a streaming generator -
        after the response has started - becomes a truncated body rather than a
        404. Handling it here keeps the store free of HTTP concerns.
        """
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": f"no job {exc.job_id}"},
        )

    app.include_router(health.router)
    app.include_router(jobs.router)
    app.include_router(molecules.router)
    return app


app = create_app()
