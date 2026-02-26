"""Production API: visitor isolation, bounded tools, background media and explicit exports."""

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import shutil
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from .jobs import BusyError, Jobs
from .media import MAX_BYTES, MediaError
from .policy import Planner
from .store import Store
from .tools import (
    CATALOG,
    ExportArgs,
    FrameArgs,
    Identifier,
    MaskArgs,
    SearchArgs,
    StrictModel,
    ToolBudget,
    ToolError,
    retrieve_frame,
    search_frames,
    validate_export,
)
from .vision import Vision, object_mask

ROOT = Path(__file__).resolve().parents[2]


class AgentRequest(StrictModel):
    query: str = Field(min_length=1, max_length=500)
    frame_id: Identifier | None = None
    video_id: Identifier | None = None
    min_sharpness: float = Field(default=0, ge=0, le=10000)


class ApprovalRequest(StrictModel):
    approved: bool


class FeedbackRequest(StrictModel):
    turn_id: Identifier
    vote: int = Field(ge=-1, le=1)
    comment: str = Field(default="", max_length=500)


def create_app(data_dir=None, vision=None, planner=None):
    store = Store(Path(data_dir or os.environ.get("SCENETRACE_DATA", ROOT / ".runtime")))
    vision, planner = vision or Vision(), planner or Planner()
    jobs = Jobs(store, vision)
    secret = os.environ.get("SCENETRACE_SESSION_SECRET", "").encode() or secrets.token_bytes(32)
    rate_lock = threading.Lock()
    rates = defaultdict(deque)
    agent_slot = threading.BoundedSemaphore(2)

    @asynccontextmanager
    async def lifespan(app):
        async def cleanup_loop():
            while True:
                await asyncio.sleep(60)
                await run_in_threadpool(jobs.cleanup)

        jobs.cleanup()
        task = asyncio.create_task(cleanup_loop())
        yield
        task.cancel()
        jobs.executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="SceneTrace", version="1.0.0", lifespan=lifespan)
    app.state.store, app.state.jobs = store, jobs

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not reflect arbitrary uploaded input or non-finite values into JSON errors.
        return JSONResponse(
            {
                "detail": "Invalid request fields.",
                "errors": [
                    {"field": list(error["loc"]), "type": error["type"], "message": error["msg"]}
                    for error in exc.errors()
                ],
            },
            status_code=422,
        )

    def sign(value):
        return hmac.new(secret, value.encode(), hashlib.sha256).hexdigest()

    def current_owner(request, required=True):
        token = request.cookies.get("scenetrace_workspace", "")
        try:
            owner, expiry, signature = token.split(".")
            if len(owner) != 32 or any(c not in "0123456789abcdef" for c in owner):
                raise ValueError
            if int(expiry) < time.time() or not hmac.compare_digest(
                signature, sign(f"{owner}.{expiry}")
            ):
                raise ValueError
            return owner
        except (ValueError, TypeError):
            if required:
                raise HTTPException(401, "Open a new workspace and retry.") from None
            return None

    def rate(owner, operation, maximum):
        now = time.time()
        with rate_lock:
            if len(rates) > 3000:
                for key in list(rates):
                    if not rates[key] or rates[key][-1] < now - 60:
                        del rates[key]
            history = rates[(owner, operation)]
            while history and history[0] < now - 60:
                history.popleft()
            if len(history) >= maximum:
                raise HTTPException(
                    429,
                    "This workspace is busy. Wait a minute and retry.",
                    headers={"Retry-After": "60"},
                )
            history.append(now)

    def video_for(owner, video_id):
        row = store.one("SELECT * FROM videos WHERE id=? AND owner=?", (video_id, owner))
        if not row:
            raise HTTPException(404, "Video not found in this workspace.")
        return row

    def frame_for(owner, frame_id):
        try:
            return retrieve_frame(store, owner, frame_id)
        except ToolError:
            raise HTTPException(404, "Frame not found in this workspace.") from None

    @app.middleware("http")
    async def boundaries(request, call_next):
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            expected = f"{request.url.scheme}://{request.url.netloc}"
            public = os.environ.get("SCENETRACE_PUBLIC_ORIGIN", "")
            if origin not in {expected, public}:
                return JSONResponse(
                    {"detail": "Requests must come from this workspace."}, status_code=403
                )
        if request.method not in ("GET", "HEAD"):
            if (
                request.method in ("POST", "PUT", "PATCH")
                and "content-length" not in request.headers
            ):
                return JSONResponse(
                    {"detail": "Send a bounded request with a content length."}, status_code=411
                )
            try:
                length = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse({"detail": "Invalid content length."}, status_code=400)
            limit = MAX_BYTES if request.url.path == "/api/videos" else 16384
            if length < 0 or length > limit:
                return JSONResponse(
                    {"detail": "Request exceeds this endpoint's size limit."}, status_code=413
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self'; "
            "connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; "
            "frame-ancestors 'self' https://huggingface.co"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.exception_handler(ToolError)
    @app.exception_handler(MediaError)
    @app.exception_handler(BusyError)
    async def tool_error(request, exc):
        return JSONResponse(
            {"detail": str(exc)}, status_code=409 if isinstance(exc, BusyError) else 422
        )

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "version": "1.0.0",
            "planner_installed": (planner.adapter / "adapter_config.json").exists()
            if isinstance(planner, Planner)
            else True,
        }

    @app.get("/api/bootstrap")
    def bootstrap(request: Request, response: Response):
        owner = current_owner(request, required=False)
        if owner is None:
            owner = uuid.uuid4().hex
            value = f"{owner}.{int(time.time() + 86400)}"
            secure = request.url.scheme == "https" or request.url.hostname.endswith(".hf.space")
            response.set_cookie(
                "scenetrace_workspace",
                f"{value}.{sign(value)}",
                max_age=86400,
                httponly=True,
                secure=secure,
                samesite="none" if secure else "lax",
            )
        return {
            "limits": {
                "video_mb": 50,
                "duration_seconds": 180,
                "videos": 5,
                "clip_seconds": 30,
                "retention_hours": 24,
                "tool_calls": 4,
            },
            "library": store.library(owner),
            "privacy": "Session storage; expires after 24 hours or a host restart.",
            "model": {
                "planner": "SmolLM2-135M + evaluated SceneTrace LoRA",
                "retriever": "CLIP ViT-B/32",
            },
        }

    @app.get("/api/tools")
    def catalog():
        return {
            name: {
                "description": v["description"],
                "requires_approval": v["approval"],
                "schema": v["schema"].model_json_schema(),
            }
            for name, v in CATALOG.items()
        }

    @app.get("/api/evaluation")
    def evaluation():
        path = ROOT / "artifacts/evaluation.json"
        return (
            json.loads(path.read_text()) if path.exists() else {"status": "training", "metrics": {}}
        )

    @app.get("/api/videos")
    def library(request: Request):
        return {"videos": store.library(current_owner(request))}

    async def ingest(request, sample=False):
        owner = current_owner(request)
        rate(owner, "upload", 5)
        with jobs.disk_lock:
            if len(store.library(owner)) >= 5:
                raise HTTPException(
                    409, "Your workspace holds five videos. Delete one before uploading another."
                )
            used = store.one("SELECT COALESCE(SUM(bytes),0) AS n FROM videos")["n"]
            if used > 1_800_000_000 or shutil.disk_usage(store.root).free < 500_000_000:
                raise HTTPException(
                    507, "The public workspace is at capacity. Please try again later."
                )
            video_id = uuid.uuid4().hex
            directory = store.root / video_id
            directory.mkdir(mode=0o700)
            filename = (
                "Big Buck Bunny · open-film sample.mp4"
                if sample
                else unquote(request.headers.get("x-filename", "video.mp4"))
            )
            filename = filename.replace("\\", "/").rsplit("/", 1)[-1][:160]
            filename = "".join(c for c in filename if c.isprintable()) or "video.mp4"
            store.execute(
                "INSERT INTO videos (id,owner,filename,state,bytes,created) VALUES (?,?,?,?,?,?)",
                (video_id, owner, filename, "indexing", MAX_BYTES, time.time()),
            )
        path = directory / "upload"
        try:
            if sample:
                shutil.copyfile(ROOT / "assets/open-film.mp4", path)
                size = path.stat().st_size
            else:
                size = 0
                with path.open("wb") as stream:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise HTTPException(413, "Videos must be 50 MB or smaller.")
                        stream.write(chunk)
            if size == 0:
                raise HTTPException(422, "Choose a non-empty video file.")
            store.execute(
                "UPDATE videos SET bytes=? WHERE id=? AND owner=?", (size, video_id, owner)
            )
            return {"video_id": video_id, "job": jobs.submit(owner, video_id, "index", jobs.index)}
        except Exception:
            store.execute("DELETE FROM videos WHERE id=? AND owner=?", (video_id, owner))
            shutil.rmtree(directory, ignore_errors=True)
            raise

    @app.post("/api/videos", status_code=202)
    async def upload(request: Request):
        return await ingest(request)

    @app.post("/api/sample", status_code=202)
    async def sample(request: Request):
        return await ingest(request, sample=True)

    @app.get("/api/jobs/{job_id}")
    def job(request: Request, job_id: Identifier):
        row = jobs.job(current_owner(request), job_id)
        if not row:
            raise HTTPException(404, "Job not found in this workspace.")
        return row

    @app.get("/api/videos/{video_id}/stream")
    def stream(request: Request, video_id: Identifier):
        owner = current_owner(request)
        video = video_for(owner, video_id)
        if video["state"] != "ready":
            raise HTTPException(409, "This video is not ready yet.")
        return FileResponse(
            store.root / video_id / "video.mp4",
            media_type="video/mp4",
            content_disposition_type="inline",
        )

    @app.delete("/api/videos/{video_id}")
    def remove(request: Request, video_id: Identifier):
        owner = current_owner(request)
        video_for(owner, video_id)
        jobs.delete_video(owner, video_id)
        return {"deleted": True}

    @app.get("/api/frames/{frame_id}/image")
    def image(request: Request, frame_id: Identifier):
        frame = frame_for(current_owner(request), frame_id)
        return FileResponse(
            store.root / frame["video_id"] / "frames" / f"{frame_id}.jpg", media_type="image/jpeg"
        )

    @app.post("/api/frames/detect")
    def detect(request: Request, args: FrameArgs):
        owner = current_owner(request)
        rate(owner, "detect", 5)
        frame = frame_for(owner, args.frame_id)
        return vision.detect(store.root / frame["video_id"] / "frames" / f"{args.frame_id}.jpg")

    @app.post("/api/search")
    def search(request: Request, args: SearchArgs):
        owner = current_owner(request)
        rate(owner, "agent", 15)
        return {
            "matches": search_frames(store, vision, owner, args),
            "method": "CLIP cosine similarity",
        }

    @app.post("/api/frames/mask")
    def mask(request: Request, args: MaskArgs):
        owner = current_owner(request)
        rate(owner, "mask", 8)
        frame = frame_for(owner, args.frame_id)
        directory = store.root / frame["video_id"] / "frames"
        try:
            result = object_mask(
                directory / f"{args.frame_id}.jpg", directory / f"{args.frame_id}.png", args.box
            )
        except Exception:
            raise ToolError(
                "A foreground mask could not be created for this box. Try a tighter selection."
            ) from None
        return {**result, "image_url": f"/api/frames/{args.frame_id}/mask"}

    @app.get("/api/frames/{frame_id}/mask")
    def mask_image(request: Request, frame_id: Identifier):
        frame = frame_for(current_owner(request), frame_id)
        path = store.root / frame["video_id"] / "frames" / f"{frame_id}.png"
        if not path.exists():
            raise HTTPException(404, "Create a mask for this frame first.")
        return FileResponse(path, media_type="image/png")

    def prepare(owner, args):
        video = validate_export(store, owner, args)
        count = store.one(
            "SELECT COUNT(*) AS n FROM approvals WHERE video_id=? AND owner=?",
            (args.video_id, owner),
        )["n"]
        if count >= 8:
            raise ToolError(
                "This video's export limit has been reached. Download an existing export."
            )
        approval_id = uuid.uuid4().hex
        store.execute(
            "INSERT INTO approvals (id,owner,video_id,start,end,state,created) VALUES (?,?,?,?,?,?,?)",
            (approval_id, owner, args.video_id, args.start, args.end, "pending", time.time()),
        )
        return {
            "id": approval_id,
            "video_id": args.video_id,
            "filename": video["filename"],
            "start": args.start,
            "end": args.end,
            "state": "pending",
            "requires_approval": True,
        }

    @app.post("/api/exports/prepare")
    def prepare_export(request: Request, args: ExportArgs):
        owner = current_owner(request)
        rate(owner, "export", 10)
        return prepare(owner, args)

    @app.post("/api/exports/{approval_id}/approve", status_code=202)
    def approve(request: Request, approval_id: Identifier, args: ApprovalRequest):
        owner = current_owner(request)
        rate(owner, "export", 10)
        if not args.approved:
            store.execute(
                "UPDATE approvals SET state='cancelled' WHERE id=? AND owner=? AND state='pending'",
                (approval_id, owner),
            )
            return {"state": "cancelled"}
        with jobs.disk_lock:
            approval = store.one(
                "SELECT * FROM approvals WHERE id=? AND owner=?", (approval_id, owner)
            )
            if not approval:
                raise HTTPException(404, "Export not found in this workspace.")
            if approval["state"] != "pending":
                raise HTTPException(409, "This export has already been approved or cancelled.")
            store.execute(
                "UPDATE approvals SET state='approved' WHERE id=? AND owner=?", (approval_id, owner)
            )
            try:
                job = jobs.submit(owner, approval["video_id"], "export", jobs.export, approval_id)
            except Exception:
                store.execute(
                    "UPDATE approvals SET state='pending' WHERE id=? AND owner=?",
                    (approval_id, owner),
                )
                raise
            return {"state": "queued", "job": job}

    @app.get("/api/exports/{approval_id}/download")
    def download(request: Request, approval_id: Identifier):
        owner = current_owner(request)
        approval = store.one(
            "SELECT * FROM approvals WHERE id=? AND owner=? AND state='ready'", (approval_id, owner)
        )
        if not approval:
            raise HTTPException(404, "This export is not available in this workspace.")
        return FileResponse(
            store.root / approval["video_id"] / f"export-{approval_id}.mp4",
            media_type="video/mp4",
            filename=f"scenetrace-{approval_id[:8]}.mp4",
        )

    @app.post("/api/agent")
    def agent(request: Request, args: AgentRequest):
        owner = current_owner(request)
        rate(owner, "agent", 15)
        if not agent_slot.acquire(blocking=False):
            raise BusyError("The planner is handling another request. Please retry in a moment.")
        started = time.perf_counter()
        try:
            previous = store.one(
                "SELECT query FROM turns WHERE owner=? AND plan='search' ORDER BY created DESC LIMIT 1",
                (owner,),
            )
            previous = previous["query"] if previous else ""
            choice = planner.choose(args.query, bool(args.frame_id), previous)
            plan, budget = choice["plan"], ToolBudget(4)
            result = {
                "plan": plan,
                "planner": choice,
                "query": args.query,
                "matches": [],
                "approval": None,
            }
            if plan == "search":
                query = args.query
                if previous and query.lower().startswith(
                    ("only ", "same ", "without ", "and ", "now ", "what about ")
                ):
                    query = f"{previous[:250]}; {query}"
                rows = budget.call(
                    "search_frames",
                    search_frames,
                    store,
                    vision,
                    owner,
                    SearchArgs(
                        query=query[:500], video_id=args.video_id, min_sharpness=args.min_sharpness
                    ),
                )
                if rows:
                    budget.call("retrieve_frame", retrieve_frame, store, owner, rows[0]["id"])
                result.update(
                    matches=rows,
                    answer="Ranked the closest sampled frames. Play a result to verify the scene."
                    if rows
                    else "No indexed frames match these filters yet. Add a video or lower the sharpness filter.",
                )
            elif plan in ("library", "count"):
                library_result = budget.call(
                    "query_library", store.library, owner, "count" if plan == "count" else "list"
                )
                result["library"] = library_result
                result["answer"] = (
                    f"Your ready footage contains {library_result['videos']} videos, totaling "
                    f"{library_result['duration']:.1f} seconds."
                    if plan == "count"
                    else f"Your workspace contains {len(library_result)} videos. Select one from the library to explore it."
                )
            elif plan in ("export", "inspect"):
                if not args.frame_id:
                    result["answer"] = "Select a search result first, then ask again."
                else:
                    frame = budget.call(
                        "retrieve_frame", retrieve_frame, store, owner, args.frame_id
                    )
                    if plan == "export":
                        args_export = ExportArgs(
                            video_id=frame["video_id"],
                            start=max(0, frame["seconds"] - 3),
                            end=min(frame["duration"], frame["seconds"] + 3),
                        )
                        result["approval"] = budget.call(
                            "request_export", prepare, owner, args_export
                        )
                        result["answer"] = (
                            "Review the clip range, then approve the export. No clip has been rendered yet."
                        )
                    else:
                        rate(owner, "detect", 5)
                        result["detection"] = budget.call(
                            "detect_objects",
                            vision.detect,
                            store.root / frame["video_id"] / "frames" / f"{args.frame_id}.jpg",
                        )
                        result["answer"] = (
                            "Ran the object detector on the selected frame. Boxes are predictions; check them against the image."
                        )
            else:
                raise ToolError("The planner returned an unsupported tool plan.")
            result["trace"] = budget.trace
            result["budget"] = {"used": budget.used, "maximum": budget.maximum}
            result["seconds"] = round(time.perf_counter() - started, 3)
            result["turn_id"] = uuid.uuid4().hex
            store.execute(
                "INSERT INTO turns VALUES (?,?,?,?,?,?)",
                (result["turn_id"], owner, args.query, plan, json.dumps(result), time.time()),
            )
            store.execute(
                "DELETE FROM turns WHERE owner=? AND id NOT IN "
                "(SELECT id FROM turns WHERE owner=? ORDER BY created DESC LIMIT 8)",
                (owner, owner),
            )
            return result
        finally:
            agent_slot.release()

    @app.post("/api/feedback")
    def feedback(request: Request, args: FeedbackRequest):
        owner = current_owner(request)
        rate(owner, "feedback", 20)
        if not store.one("SELECT id FROM turns WHERE id=? AND owner=?", (args.turn_id, owner)):
            raise HTTPException(404, "Search not found in this workspace.")
        store.execute(
            "INSERT INTO feedback VALUES (?,?,?,?,?,?)",
            (uuid.uuid4().hex, owner, args.turn_id, args.vote, args.comment, time.time()),
        )
        return {"saved": True}

    @app.delete("/api/workspace")
    def clear(request: Request):
        owner = current_owner(request)
        with jobs.disk_lock, jobs.state_lock:
            videos = store.library(owner)
            if any(v["id"] in jobs.active_videos for v in videos):
                raise BusyError("Wait for current jobs to finish before clearing the workspace.")
            for video in videos:
                jobs.delete_video(owner, video["id"])
            store.execute("DELETE FROM turns WHERE owner=?", (owner,))
        return {"deleted": True}

    dist = ROOT / "web/dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app


app = create_app()
