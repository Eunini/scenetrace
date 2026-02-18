"""One bounded worker; job state survives requests and failures remain visible."""

import json
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from .media import MediaError, export_clip, extract_frames, probe, transcode
from .vision import sharpness


class BusyError(ValueError):
    pass


class Jobs:
    def __init__(self, store, vision):
        self.store, self.vision = store, vision
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-worker")
        self.slots = threading.BoundedSemaphore(6)
        self.disk_lock = threading.RLock()
        self.active_videos = {}
        self.state_lock = threading.RLock()

    def submit(self, owner, video_id, kind, function, *args):
        if not self.slots.acquire(blocking=False):
            raise BusyError("The video queue is full. Please try again shortly.")
        job_id = uuid.uuid4().hex
        now = time.time()
        with self.state_lock:
            self.active_videos[video_id] = self.active_videos.get(video_id, 0) + 1
        self.store.execute(
            "INSERT INTO jobs (id,owner,video_id,kind,state,created,updated) VALUES (?,?,?,?,?,?,?)",
            (job_id, owner, video_id, kind, "queued", now, now),
        )

        def task():
            try:
                self.store.update_job(job_id, "running", 0.05)
                result = function(job_id, owner, video_id, *args)
                self.store.update_job(job_id, "ready", 1, result=result)
            except (MediaError, ValueError) as exc:
                self.store.update_job(job_id, "failed", error=str(exc)[:220])
                if kind == "index":
                    self.store.execute(
                        "UPDATE videos SET state='failed',error=? WHERE id=? AND owner=?",
                        (str(exc)[:220], video_id, owner),
                    )
            except Exception:
                # Never expose subprocess stderr, local paths, model internals or uploaded data.
                self.store.update_job(
                    job_id,
                    "failed",
                    error="Processing failed. Please retry; shorter videos are faster.",
                )
                if kind == "index":
                    self.store.execute(
                        "UPDATE videos SET state='failed',error='Indexing failed; please retry.' "
                        "WHERE id=? AND owner=?",
                        (video_id, owner),
                    )
            finally:
                with self.state_lock:
                    remaining = self.active_videos[video_id] - 1
                    if remaining:
                        self.active_videos[video_id] = remaining
                    else:
                        del self.active_videos[video_id]
                self.slots.release()

        self.executor.submit(task)
        return self.store.one("SELECT * FROM jobs WHERE id=? AND owner=?", (job_id, owner))

    def index(self, job_id, owner, video_id):
        directory = self.store.root / video_id
        source = directory / "upload"
        metadata = probe(source)
        self.store.update_job(job_id, "running", 0.15)
        proxy = directory / "video.mp4"
        transcode(source, proxy)
        source.unlink(missing_ok=True)
        frames = extract_frames(proxy, directory / "frames", metadata["duration"])
        self.store.update_job(job_id, "running", 0.4)
        embeddings = self.vision.images([p for p, _ in frames])
        with self.store.connect() as db:
            for (path, seconds), embedding in zip(frames, embeddings, strict=True):
                frame_id = uuid.uuid4().hex
                path.rename(directory / "frames" / f"{frame_id}.jpg")
                final_path = directory / "frames" / f"{frame_id}.jpg"
                db.execute(
                    "INSERT INTO frames VALUES (?,?,?,?,?,?)",
                    (
                        frame_id,
                        owner,
                        video_id,
                        seconds,
                        sharpness(final_path),
                        embedding.tobytes(),
                    ),
                )
            db.execute(
                "UPDATE videos SET state='ready',duration=?,width=?,height=?,bytes=? "
                "WHERE id=? AND owner=?",
                (
                    metadata["duration"],
                    metadata["width"],
                    metadata["height"],
                    proxy.stat().st_size,
                    video_id,
                    owner,
                ),
            )
        return {"video_id": video_id, "frames": len(frames), **metadata}

    def export(self, job_id, owner, video_id, approval_id):
        approval = self.store.one(
            "SELECT * FROM approvals WHERE id=? AND owner=? AND state='approved'",
            (approval_id, owner),
        )
        if not approval:
            raise ValueError("Export needs your approval before it can run.")
        directory = self.store.root / video_id
        export_clip(
            directory / "video.mp4",
            directory / f"export-{approval_id}.mp4",
            approval["start"],
            approval["end"],
        )
        self.store.execute(
            "UPDATE approvals SET state='ready',job_id=? WHERE id=? AND owner=?",
            (job_id, approval_id, owner),
        )
        return {"approval_id": approval_id, "download_url": f"/api/exports/{approval_id}/download"}

    def delete_video(self, owner, video_id):
        with self.disk_lock, self.state_lock:
            if video_id in self.active_videos:
                raise BusyError("Wait for this video's current job to finish before deleting it.")
            self.store.execute("DELETE FROM videos WHERE id=? AND owner=?", (video_id, owner))
            self.store.execute("DELETE FROM jobs WHERE video_id=? AND owner=?", (video_id, owner))
            shutil.rmtree(self.store.root / video_id, ignore_errors=True)

    def cleanup(self):
        cutoff = time.time() - 24 * 3600
        for video in self.store.all("SELECT id,owner FROM videos WHERE created<?", (cutoff,)):
            try:
                self.delete_video(video["owner"], video["id"])
            except BusyError:
                continue
        self.store.execute("DELETE FROM turns WHERE created<?", (cutoff,))
        self.store.execute(
            "DELETE FROM jobs WHERE created<? AND state NOT IN ('queued','running')", (cutoff,)
        )

    def job(self, owner, job_id):
        row = self.store.one("SELECT * FROM jobs WHERE id=? AND owner=?", (job_id, owner))
        if row and row["result"]:
            row["result"] = json.loads(row["result"])
        return row
