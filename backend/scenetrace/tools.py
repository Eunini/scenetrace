"""Typed tool catalog and a bounded dispatcher. No model output becomes executable code."""

import time
from enum import StrEnum
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Identifier = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SearchArgs(StrictModel):
    query: str = Field(min_length=1, max_length=500)
    video_id: Identifier | None = None
    limit: int = Field(default=6, ge=1, le=12)
    min_sharpness: float = Field(default=0, ge=0, le=10000)


class LibraryMode(StrEnum):
    LIST = "list"
    COUNT = "count"


class LibraryArgs(StrictModel):
    mode: LibraryMode = LibraryMode.LIST
    contains: str = Field(default="", max_length=100)


class FrameArgs(StrictModel):
    frame_id: Identifier


class ExportArgs(StrictModel):
    video_id: Identifier
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class MaskArgs(FrameArgs):
    box: tuple[float, float, float, float]


CATALOG = {
    "search_frames": {"schema": SearchArgs, "description": "Rank sampled frames using CLIP cosine similarity.", "approval": False},
    "retrieve_frame": {"schema": FrameArgs, "description": "Resolve a frame to its source video and timestamp.", "approval": False},
    "query_library": {"schema": LibraryArgs, "description": "Run a parameterized video inventory or aggregate SQL query.", "approval": False},
    "detect_objects": {"schema": FrameArgs, "description": "Run DETR on one selected frame; returns labeled bounding boxes.", "approval": False},
    "request_export": {"schema": ExportArgs, "description": "Prepare a bounded clip export for explicit visitor approval.", "approval": True},
    "mask_object": {"schema": MaskArgs, "description": "Run GrabCut using a selected bounding box; not semantic segmentation.", "approval": False},
}


class ToolError(ValueError):
    pass


class ToolBudget:
    def __init__(self, maximum=4):
        self.maximum, self.used, self.trace = maximum, 0, []

    def call(self, name, function, *args):
        if self.used >= self.maximum:
            raise ToolError("The tool-call budget was reached. Narrow the request and retry.")
        self.used += 1
        start = time.perf_counter()
        try:
            result = function(*args)
            self.trace.append({"tool": name, "status": "ok", "milliseconds": round((time.perf_counter() - start) * 1000)})
            return result
        except (ValueError, RuntimeError) as exc:
            self.trace.append({"tool": name, "status": "error", "milliseconds": round((time.perf_counter() - start) * 1000)})
            raise ToolError(str(exc)) from None


def retrieve_frame(store, owner, frame_id):
    row = store.one("SELECT f.id,f.video_id,f.seconds,f.sharpness,v.filename,v.duration "
                    "FROM frames f JOIN videos v ON v.id=f.video_id WHERE f.id=? AND f.owner=?",
                    (frame_id, owner))
    if not row:
        raise ToolError("That frame is not available in this workspace.")
    row["image_url"] = f"/api/frames/{frame_id}/image"
    row["video_url"] = f"/api/videos/{row['video_id']}/stream"
    return row


def search_frames(store, vision, owner, args: SearchArgs):
    sql = ("SELECT f.*,v.filename,v.duration FROM frames f JOIN videos v ON f.video_id=v.id "
           "WHERE f.owner=? AND v.state='ready' AND f.sharpness>=?")
    parameters = [owner, args.min_sharpness]
    if args.video_id:
        sql += " AND f.video_id=?"
        parameters.append(args.video_id)
    rows = store.all(sql, tuple(parameters))
    if not rows:
        return []
    query = vision.text(args.query)
    matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    scores = matrix @ query
    candidates = sorted(zip(rows, scores, strict=True), key=lambda x: -x[1])
    output = []
    for row, score in candidates:
        if any(r["video_id"] == row["video_id"] and abs(r["seconds"] - row["seconds"]) < 3 for r in output):
            continue
        item = retrieve_frame(store, owner, row["id"])
        item["similarity"] = round(float(score), 4)
        output.append(item)
        if len(output) >= args.limit:
            break
    return output


def validate_export(store, owner, args: ExportArgs):
    row = store.one("SELECT * FROM videos WHERE id=? AND owner=? AND state='ready'",
                    (args.video_id, owner))
    if not row:
        raise ToolError("Choose a ready video in this workspace.")
    if args.end <= args.start or args.end - args.start > 30 or args.end > row["duration"] + 0.01:
        raise ToolError("Choose a clip between zero and thirty seconds, within the source video.")
    return row
