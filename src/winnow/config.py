"""Application settings.

Everything is overridable from the environment with a ``WINNOW_`` prefix, or
from a ``.env`` file. Settings are consumed through the ``get_settings``
dependency rather than imported as a module-level singleton, so tests can
override them per-request.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WINNOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Winnow"
    log_level: str = "INFO"

    # --- limits -------------------------------------------------------------
    max_library_size: int = Field(
        default=250_000,
        description="Hard cap on molecules accepted in a single triage job.",
    )
    max_sync_batch: int = Field(
        default=100,
        description="Cap for the synchronous /molecules endpoints. Anything "
        "larger must go through a job, or it will block the request.",
    )
    max_upload_bytes: int = 64 * 1024 * 1024

    # --- execution ----------------------------------------------------------
    chunk_size: int = Field(
        default=2_000,
        description="Molecules per unit of work handed to a worker process. "
        "Too small and pickling dominates; too large and progress is lumpy.",
    )
    worker_processes: int = Field(
        default=0,
        description="Process-pool size. 0 means os.cpu_count().",
    )

    # --- retention ----------------------------------------------------------
    job_ttl_seconds: int = 86_400

    @property
    def effective_workers(self) -> int:
        return self.worker_processes or (os.cpu_count() or 1)


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is parsed once per process."""
    return Settings()
