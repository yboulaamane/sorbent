"""Dependency providers.

Nothing here reaches for a module-level global. The store and runner live on
``app.state``, put there by the lifespan, and are handed to routes through
``Depends``. That is what makes the routes testable: a test overrides the
dependency and gets a different store with no monkeypatching.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from sorbent.config import Settings, get_settings
from sorbent.jobs.runner import TriageRunner
from sorbent.jobs.store import JobStore


def get_store(request: Request) -> JobStore:
    return request.app.state.store


def get_runner(request: Request) -> TriageRunner:
    return request.app.state.runner


def get_request_settings(request: Request) -> Settings:
    """Settings for THIS app, not the process-wide cached singleton.

    ``get_settings`` is lru_cached against the environment, so a route that
    depended on it directly would ignore whatever settings ``create_app`` was
    handed - and a test that builds an app with a 1000-molecule cap would
    silently get the 250k default. Read from app.state instead; fall back to
    the cached settings only for an app that never set them.
    """
    return getattr(request.app.state, "settings", None) or get_settings()


StoreDep = Annotated[JobStore, Depends(get_store)]
RunnerDep = Annotated[TriageRunner, Depends(get_runner)]
SettingsDep = Annotated[Settings, Depends(get_request_settings)]
