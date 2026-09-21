"""Liveness and readiness.

Two endpoints, not one, because they answer different questions:

  /health  - is the process up? Must never touch RDKit, the pool, or the store.
             A health check that does real work is a health check that fails
             under load, which is exactly when you need it to be honest.

  /ready   - can this process actually do the job? Checks that the chem layer
             imports and that the pool is alive. A 503 here should take the
             node out of rotation.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from winnow import __version__

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/ready", summary="Readiness probe")
async def ready(response: Response) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    try:
        import rdkit

        checks["rdkit"] = rdkit.__version__
    except ImportError as exc:
        checks["rdkit"] = f"unavailable: {exc}"

    try:
        from winnow.chem import pipeline  # noqa: F401

        checks["chem"] = "importable"
    except Exception as exc:  # noqa: BLE001
        checks["chem"] = f"unavailable: {type(exc).__name__}: {exc}"

    ok = all(not str(v).startswith("unavailable") for v in checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ok else "degraded", "checks": checks}
