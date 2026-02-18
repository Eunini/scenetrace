import time

import pytest
from fastapi.testclient import TestClient
from scenetrace.media import probe
from scenetrace.tools import CATALOG


def wait_job(client, job_id):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in ("ready", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("Tiny local media job exceeded its deadline")


def indexed(workspace, tiny_video):
    app, alice, bob = workspace
    response = alice.post(
        "/api/videos", content=tiny_video, headers={"X-Filename": "../../private-<clip>.mp4"}
    )
    assert response.status_code == 202, response.text
    result = response.json()
    job = wait_job(alice, result["job"]["id"])
    assert job["state"] == "ready", job.get("error")
    frames = alice.post("/api/agent", json={"query": "search:a scene"}).json()["matches"]
    return result, frames


def test_sessions_require_signed_cookie_and_cross_origin_writes_fail(workspace):
    app, alice, bob = workspace
    anonymous = TestClient(app)
    assert anonymous.get("/api/videos").status_code == 401
    anonymous.cookies.set("scenetrace_workspace", "0" * 32 + ".9999999999.invalid")
    assert anonymous.get("/api/videos").status_code == 401
    assert (
        alice.post(
            "/api/agent", json={"query": "library:"}, headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert (
        alice.post(
            "/api/agent", json={"query": "library:"}, headers={"Origin": "http://testserver"}
        ).status_code
        == 200
    )


def test_streamed_upload_indexes_real_video_and_limits_inputs(workspace, tiny_video):
    app, alice, bob = workspace
    result, frames = indexed(workspace, tiny_video)
    videos = alice.get("/api/videos").json()["videos"]
    assert videos[0]["filename"] == "private-<clip>.mp4"
    assert videos[0]["state"] == "ready" and 1.9 < videos[0]["duration"] < 2.1
    assert frames and len(frames) <= 32
    assert alice.get(frames[0]["image_url"]).headers["content-type"] == "image/jpeg"
    assert alice.get(frames[0]["video_url"], headers={"Range": "bytes=0-99"}).status_code == 206
    assert alice.post("/api/videos", content=b"").status_code == 422
    assert (
        alice.post(
            "/api/videos", content=b"small", headers={"content-length": str(51 * 1024 * 1024)}
        ).status_code
        == 413
    )


def test_all_media_jobs_frames_exports_and_feedback_are_owner_scoped(workspace, tiny_video):
    app, alice, bob = workspace
    result, frames = indexed(workspace, tiny_video)
    frame = frames[0]
    assert bob.get("/api/videos").json()["videos"] == []
    for path in [f"/api/jobs/{result['job']['id']}", frame["image_url"], frame["video_url"]]:
        assert bob.get(path).status_code == 404
    assert bob.post("/api/frames/detect", json={"frame_id": frame["id"]}).status_code == 404
    assert (
        bob.post(
            "/api/frames/mask", json={"frame_id": frame["id"], "box": [1, 1, 50, 50]}
        ).status_code
        == 404
    )
    assert bob.delete(f"/api/videos/{result['video_id']}").status_code == 404
    assert (
        bob.post(
            "/api/exports/prepare", json={"video_id": result["video_id"], "start": 0, "end": 1}
        ).status_code
        == 422
    )
    turn = alice.post("/api/agent", json={"query": "library:"}).json()
    assert (
        bob.post("/api/feedback", json={"turn_id": turn["turn_id"], "vote": 1}).status_code == 404
    )


def test_no_export_runs_until_approved_and_approval_cannot_be_replayed(
    workspace, tiny_video, tmp_path
):
    app, alice, bob = workspace
    result, frames = indexed(workspace, tiny_video)
    response = alice.post(
        "/api/exports/prepare", json={"video_id": result["video_id"], "start": 0.2, "end": 1.4}
    )
    assert response.status_code == 200
    approval = response.json()
    assert approval["requires_approval"] and approval["state"] == "pending"
    assert app.state.store.one("SELECT COUNT(*) AS n FROM jobs WHERE kind='export'")["n"] == 0
    path = f"/api/exports/{approval['id']}"
    assert alice.get(path + "/download").status_code == 404
    assert bob.post(path + "/approve", json={"approved": True}).status_code == 404
    accepted = alice.post(path + "/approve", json={"approved": True})
    assert accepted.status_code == 202
    assert wait_job(alice, accepted.json()["job"]["id"])["state"] == "ready"
    assert alice.post(path + "/approve", json={"approved": True}).status_code == 409
    download = alice.get(path + "/download")
    assert download.status_code == 200 and download.headers["content-type"] == "video/mp4"
    assert bob.get(path + "/download").status_code == 404
    target = tmp_path / "export.mp4"
    target.write_bytes(download.content)
    assert 1.1 <= probe(target)["duration"] <= 1.4


@pytest.mark.parametrize("start,end", [(1, 0), (0, 31), (-1, 1), (0, 1000)])
def test_export_bounds_are_enforced(workspace, tiny_video, start, end):
    result, frames = indexed(workspace, tiny_video)
    alice = workspace[1]
    assert (
        alice.post(
            "/api/exports/prepare",
            json={"video_id": result["video_id"], "start": start, "end": end},
        ).status_code
        == 422
    )


def test_invalid_media_fails_visibly_without_claiming_it_is_indexed(workspace):
    app, alice, bob = workspace
    result = alice.post("/api/videos", content=b"not a real video").json()
    assert wait_job(alice, result["job"]["id"])["state"] == "failed"
    video = alice.get("/api/videos").json()["videos"][0]
    assert video["state"] == "failed" and video["error"]
    assert alice.post("/api/agent", json={"query": "search:a rabbit"}).json()["matches"] == []


def test_catalog_schemas_reject_injection_unknown_fields_and_nonfinite_numbers(workspace):
    app, alice, bob = workspace
    catalog = alice.get("/api/tools").json()
    assert set(catalog) == set(CATALOG)
    assert catalog["request_export"]["requires_approval"]
    assert all(tool["schema"]["additionalProperties"] is False for tool in catalog.values())
    assert (
        alice.post("/api/agent", json={"query": "count:", "sql": "DROP TABLE videos"}).status_code
        == 422
    )
    assert alice.get("/api/frames/../../etc/passwd/image").status_code == 404
    assert (
        alice.post("/api/frames/detect", json={"frame_id": "../../etc/passwd"}).status_code == 422
    )
    assert (
        alice.post(
            "/api/exports/prepare",
            content='{"video_id":"' + "0" * 32 + '","start":NaN,"end":2}',
            headers={"content-type": "application/json"},
        ).status_code
        == 422
    )


def test_rate_limits_apply_and_workspace_deletion_removes_files(workspace, tiny_video):
    app, alice, bob = workspace
    result, frames = indexed(workspace, tiny_video)
    directory = app.state.store.root / result["video_id"]
    assert directory.exists()
    assert alice.delete("/api/workspace").status_code == 200
    assert not directory.exists() and alice.get("/api/videos").json()["videos"] == []
    statuses = [alice.post("/api/agent", json={"query": "library:"}).status_code for _ in range(16)]
    assert 429 in statuses


def test_agent_budget_export_policy_and_no_selected_frame_handling(workspace, tiny_video):
    result, frames = indexed(workspace, tiny_video)
    alice = workspace[1]
    empty = alice.post("/api/agent", json={"query": "export:"}).json()
    assert empty["budget"]["used"] == 0 and empty["approval"] is None
    response = alice.post(
        "/api/agent", json={"query": "export:", "frame_id": frames[0]["id"]}
    ).json()
    assert response["approval"]["requires_approval"]
    assert response["budget"]["used"] <= response["budget"]["maximum"] == 4
    assert [r["tool"] for r in response["trace"]] == ["retrieve_frame", "request_export"]
    assert all(r["status"] == "ok" for r in response["trace"])
