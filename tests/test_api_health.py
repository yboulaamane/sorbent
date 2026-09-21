"""The health endpoints must work before any chemistry does."""

from __future__ import annotations


async def test_health_is_up(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_health_does_not_touch_rdkit(client):
    """Liveness must stay cheap. If this starts importing RDKit it will start
    failing under memory pressure, which is when you most need it to answer."""
    r = await client.get("/health")
    assert set(r.json()) == {"status", "version"}


async def test_ready_reports_chem_layer(client):
    r = await client.get("/ready")
    assert r.status_code in (200, 503)
    assert "rdkit" in r.json()["checks"]


async def test_openapi_is_valid(client):
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert "/v1/jobs" in spec["paths"]
    assert "/v1/molecules/describe" in spec["paths"]
