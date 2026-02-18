import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scenetrace.app import create_app


class TestVision:
    """Deliberate test double; real model evidence is verified separately."""

    def images(self, paths):
        return np.array([[1, 0] for _ in paths], dtype=np.float32)

    def text(self, query):
        return np.array([1, 0], dtype=np.float32)

    def detect(self, path):
        return {"width": 128, "height": 72, "objects": []}


class TestPlanner:
    def choose(self, query, selected=False, previous=""):
        return {"plan": query.split(":", 1)[0], "stage": "test-double"}


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x72:rate=4",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
        timeout=15,
    )
    return path.read_bytes()


@pytest.fixture
def workspace(tmp_path):
    app = create_app(tmp_path, vision=TestVision(), planner=TestPlanner())
    with TestClient(app) as alice:
        bob = TestClient(app)
        alice.get("/api/bootstrap")
        bob.get("/api/bootstrap")
        yield app, alice, bob
        bob.close()
