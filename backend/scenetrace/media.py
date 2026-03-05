"""Bounded local media processing. User supplied URLs and command fragments are never accepted."""

import json
import subprocess
from pathlib import Path

MAX_BYTES = 50 * 1024 * 1024
MAX_DURATION = 180
FRAME_COUNT = 32


class MediaError(ValueError):
    pass


def run_media(args, timeout=90):
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise MediaError(
            "This video took too long to process. Try a shorter or smaller file."
        ) from None
    if result.returncode != 0:
        raise MediaError("The file could not be decoded. Use an MP4, MOV or WebM video.")
    return result.stdout


def probe(path: Path):
    output = run_media(
        [
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-format_whitelist",
            "mov,matroska,webm",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        timeout=15,
    )
    try:
        data = json.loads(output)
        stream = next(s for s in data["streams"] if s.get("codec_type") == "video")
        duration = float(data["format"]["duration"])
        width, height = int(stream["width"]), int(stream["height"])
        if not 0 < duration <= MAX_DURATION:
            raise MediaError("Videos must be between zero and three minutes long.")
        if not 16 <= width <= 3840 or not 16 <= height <= 2160:
            raise MediaError("Video dimensions must be no larger than 3840 × 2160.")
        return {"duration": duration, "width": width, "height": height}
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        if isinstance(exc, MediaError):
            raise
        raise MediaError("The video has no valid duration or video stream.") from None


def transcode(source: Path, destination: Path):
    run_media(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-map_metadata",
            "-1",
            "-sn",
            "-dn",
            "-vf",
            "scale=w='min(960,iw)':h='min(540,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-r",
            "24",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "25",
            "-maxrate",
            "2M",
            "-bufsize",
            "4M",
            "-threads",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-t",
            str(MAX_DURATION),
            "-movflags",
            "+faststart",
            "-y",
            str(destination),
        ],
        timeout=150,
    )


def extract_frames(video: Path, directory: Path, duration: float):
    directory.mkdir(exist_ok=True)
    # Select the first decoded frame at or after each bin boundary. Selecting against
    # absolute time avoids drift and also produces a frame for very short videos.
    count = min(FRAME_COUNT, max(1, int(duration / 1.5)))
    interval = duration / count
    times = [interval * i for i in range(count)]
    # A single ffmpeg pass is substantially faster than seeking once per frame.
    run_media(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(video),
            "-vf",
            f"select=gte(t\\,selected_n*{interval:.8f}),scale=384:-2",
            "-vsync",
            "vfr",
            "-frames:v",
            str(count),
            "-threads",
            "1",
            "-q:v",
            "3",
            "-y",
            str(directory / "%03d.jpg"),
        ],
        timeout=60,
    )
    files = sorted(directory.glob("*.jpg"))
    if not files:
        raise MediaError("No visual frames could be extracted from this video.")
    return list(zip(files, times[: len(files)], strict=True))


def export_clip(video: Path, destination: Path, start: float, end: float):
    run_media(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video),
            "-t",
            f"{end - start:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-map_metadata",
            "-1",
            "-sn",
            "-dn",
            "-vf",
            "scale=w='min(1280,iw)':h='min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-r",
            "24",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-maxrate",
            "4M",
            "-bufsize",
            "8M",
            "-preset",
            "veryfast",
            "-threads",
            "1",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-y",
            str(destination),
        ],
        timeout=60,
    )
