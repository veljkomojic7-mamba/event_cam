"""FFmpeg and ffprobe helpers. Commands are argument lists, never shell strings."""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from traffic_analysis.schemas import StrictModel, VideoInfo

logger = logging.getLogger(__name__)

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


class SampleRequest(BaseModel):
    """Validated parameters for cutting a scene-setup sample."""

    source: Path
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    output: Path


class SampleManifest(StrictModel):
    """Sidecar written next to a sample so times can be mapped back to the source."""

    source: str
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)


def ensure_ffmpeg_available() -> None:
    """Raise if ffmpeg or ffprobe is not on PATH."""
    for tool in (FFMPEG, FFPROBE):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found on PATH; install FFmpeg first")


def build_sample_command(request: SampleRequest) -> list[str]:
    """Build a re-encoded, audio-free sample cut.

    Re-encoding (not stream copy) keeps the cut frame-accurate.
    """
    return [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{request.start_s:.3f}",
        "-i",
        str(request.source),
        "-t",
        f"{request.duration_s:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(request.output),
    ]


def build_frame_command(source: Path, time_s: float, output: Path) -> list[str]:
    """Build a command that writes one frame at ``time_s`` to ``output``."""
    return [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{time_s:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        str(output),
    ]


def parse_ffprobe_output(raw: str, video_id: str) -> VideoInfo:
    """Parse ffprobe JSON output into a VideoInfo model."""
    data: dict[str, Any] = json.loads(raw)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError("no video stream found")
    stream = streams[0]
    fps = _parse_frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    duration_raw = stream.get("duration") or data.get("format", {}).get("duration")
    if duration_raw is None:
        raise ValueError("ffprobe did not report a duration")
    return VideoInfo(
        id=video_id,
        fps=fps,
        duration_s=float(duration_raw),
        width=int(stream["width"]),
        height=int(stream["height"]),
    )


def _parse_frame_rate(value: str | None) -> float:
    if not value:
        raise ValueError("ffprobe did not report a frame rate")
    numerator, _, denominator = value.partition("/")
    if not denominator:
        return float(numerator)
    if float(denominator) == 0:
        raise ValueError(f"invalid frame rate: {value}")
    return float(numerator) / float(denominator)


def run_ffmpeg(command: list[str]) -> None:
    """Run an FFmpeg command, logging stderr and raising on failure."""
    ensure_ffmpeg_available()
    logger.info("running: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("ffmpeg failed: %s", result.stderr.strip())
        raise RuntimeError(f"ffmpeg exited with code {result.returncode}")


def probe_video(path: Path) -> VideoInfo:
    """Read fps, duration and resolution of a video with ffprobe."""
    if not path.is_file():
        raise FileNotFoundError(f"video not found: {path}")
    ensure_ffmpeg_available()
    command = [
        FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    return parse_ffprobe_output(result.stdout, video_id=path.stem)


def cut_sample(request: SampleRequest) -> Path:
    """Cut a sample clip and write its manifest. Returns the sample path."""
    if not request.source.is_file():
        raise FileNotFoundError(f"video not found: {request.source}")
    request.output.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(build_sample_command(request))
    manifest = SampleManifest(
        source=str(request.source),
        start_s=request.start_s,
        duration_s=request.duration_s,
    )
    manifest_path = request.output.with_suffix(".json")
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    logger.info("wrote sample %s (manifest %s)", request.output, manifest_path)
    return request.output
