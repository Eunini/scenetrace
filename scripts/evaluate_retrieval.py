"""Illustrative actual-model retrieval checks on the openly licensed sample film."""

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scenetrace.media import extract_frames
from scenetrace.vision import CLIP_MODEL, CLIP_REVISION, Vision

ROOT = Path(__file__).resolve().parents[1]
# Intervals were inspected against evidence/contact-sheet.jpg during implementation.
# These are illustrative author-provided annotations, not an independent human benchmark.
CASES = [
    ("a bird sitting on a tree branch", [(16.5, 21)]),
    ("a large rabbit", [(44.5, 55), (57, 60)]),
    ("pink clouds above green hills", [(1.5, 12)]),
    ("flowers next to a stream", [(12.5, 15.5)]),
    ("a dark hole in a tree", [(33, 44)]),
    ("the Big Buck Bunny title", [(0, 0.5), (25.5, 29)]),
]


def main():
    model = Vision()
    with tempfile.TemporaryDirectory(prefix="scenetrace-retrieval-") as directory:
        frames = extract_frames(ROOT / "assets/open-film.mp4", Path(directory), 60)
        embeddings = model.images([p for p, _ in frames])
        results = []
        for query, intervals in CASES:
            start = time.perf_counter()
            scores = embeddings @ model.text(query)
            ranked = np.argsort(-scores).tolist()
            hits = []
            for i in ranked[:3]:
                seconds = frames[i][1]
                hits.append(
                    {
                        "seconds": seconds,
                        "similarity": float(scores[i]),
                        "relevant": any(a <= seconds <= b for a, b in intervals),
                    }
                )
            results.append(
                {
                    "query": query,
                    "relevant_intervals": intervals,
                    "top3": hits,
                    "top1_correct": hits[0]["relevant"],
                    "hit_at_3": any(r["relevant"] for r in hits),
                    "query_seconds": time.perf_counter() - start,
                }
            )
        report = {
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "model": CLIP_MODEL,
            "revision": CLIP_REVISION,
            "frames": len(frames),
            "video": "Big Buck Bunny first sixty seconds",
            "annotation_source": "Contact-sheet inspection during implementation; no independent human review",
            "scope": "Six illustrative queries on one public film; not a held-out real-user quality estimate",
            "top1_accuracy": sum(r["top1_correct"] for r in results) / len(results),
            "hit_at_3": sum(r["hit_at_3"] for r in results) / len(results),
            "queries": results,
        }
        (ROOT / "artifacts/retrieval-evaluation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
