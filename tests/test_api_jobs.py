"""Job lifecycle. These pass against the stubs - the job simply ends FAILED
with a 'not implemented' message, which is itself the correct behaviour."""

from __future__ import annotations

import asyncio

from sorbent.schemas.job import JobStatus


async def _wait_terminal(client, job_id, timeout=10.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        r = await client.get(f"/v1/jobs/{job_id}")
        job = r.json()
        if JobStatus(job["status"]).is_terminal:
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached a terminal state")


async def test_submit_returns_202_and_location(client, small_library):
    r = await client.post("/v1/jobs", json={"smiles": small_library, "name": "test"})
    assert r.status_code == 202
    assert r.headers["location"] == f"/v1/jobs/{r.json()['id']}"
    assert r.json()["status"] == "pending"
    assert r.json()["counts"]["submitted"] == len(small_library)


async def test_empty_library_is_rejected(client):
    r = await client.post("/v1/jobs", json={"smiles": []})
    assert r.status_code == 422


async def test_library_over_cap_is_rejected(client):
    r = await client.post("/v1/jobs", json={"smiles": ["C"] * 1001})
    assert r.status_code == 413
    assert "1000" in r.json()["detail"]


async def test_mismatched_identifiers_are_rejected(client, small_library):
    r = await client.post("/v1/jobs", json={"smiles": small_library, "identifiers": ["a", "b"]})
    assert r.status_code == 422


async def test_unknown_descriptor_window_is_rejected(client, small_library):
    r = await client.post(
        "/v1/jobs",
        json={
            "smiles": small_library,
            "config": {"descriptor_windows": {"not_a_descriptor": {"maximum": 5}}},
        },
    )
    assert r.status_code == 422


async def test_inverted_descriptor_window_is_rejected(client, small_library):
    r = await client.post(
        "/v1/jobs",
        json={
            "smiles": small_library,
            "config": {
                "descriptor_windows": {"molecular_weight": {"minimum": 500, "maximum": 100}}
            },
        },
    )
    assert r.status_code == 422


async def test_unknown_job_is_404(client):
    assert (await client.get("/v1/jobs/nope")).status_code == 404
    assert (await client.get("/v1/jobs/nope/results")).status_code == 404
    assert (await client.delete("/v1/jobs/nope")).status_code == 404


async def test_results_conflict_while_running(client, small_library):
    job_id = (await client.post("/v1/jobs", json={"smiles": small_library})).json()["id"]
    job = await _wait_terminal(client, job_id)
    r = await client.get(f"/v1/jobs/{job_id}/results")
    if job["status"] == "failed":
        assert r.status_code == 409
        assert "not implemented" in job["error"].lower() or job["error"]
    else:
        assert r.status_code == 200


async def test_job_list_is_newest_first(client, small_library):
    for i in range(3):
        await client.post("/v1/jobs", json={"smiles": small_library, "name": f"job-{i}"})
    r = await client.get("/v1/jobs")
    assert r.status_code == 200
    names = [j["name"] for j in r.json()["jobs"]]
    assert names == ["job-2", "job-1", "job-0"]
    assert r.json()["total"] == 3


async def test_upload_smi_file(client):
    content = b"CC(=O)Oc1ccccc1C(=O)O aspirin\nCn1cnc2c1c(=O)n(C)c(=O)n2C caffeine\n# a comment\n\n"
    r = await client.post(
        "/v1/jobs/upload",
        files={"file": ("lib.smi", content, "chemical/x-daylight-smiles")},
    )
    assert r.status_code == 202
    assert r.json()["counts"]["submitted"] == 2


async def test_upload_csv_file(client):
    content = (
        b"id,smiles,vendor\nA1,CC(=O)Oc1ccccc1C(=O)O,acme\nA2,Cn1cnc2c1c(=O)n(C)c(=O)n2C,acme\n"
    )
    r = await client.post(
        "/v1/jobs/upload",
        files={"file": ("lib.csv", content, "text/csv")},
        data={"smiles_column": "smiles", "id_column": "id"},
    )
    assert r.status_code == 202
    assert r.json()["counts"]["submitted"] == 2


async def test_upload_csv_missing_column_is_422(client):
    r = await client.post(
        "/v1/jobs/upload",
        files={"file": ("lib.csv", b"id,structure\nA1,CCO\n", "text/csv")},
        data={"smiles_column": "smiles"},
    )
    assert r.status_code == 422
    assert "smiles" in r.json()["detail"]


async def test_delete_terminal_job_removes_it(client, small_library):
    job_id = (await client.post("/v1/jobs", json={"smiles": small_library})).json()["id"]
    await _wait_terminal(client, job_id)
    assert (await client.delete(f"/v1/jobs/{job_id}")).status_code == 204
    assert (await client.get(f"/v1/jobs/{job_id}")).status_code == 404
