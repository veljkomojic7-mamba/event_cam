"""Experimental Gemini counting from timestamped, sampled image frames."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from google import genai
from google.genai import types
from pydantic import Field

from traffic_analysis.detection import iter_sampled_frames
from traffic_analysis.schemas import SceneConfig, StrictModel, save_model

MAX_DURATION_S = 60.0
MAX_FPS = 10.0
BATCH_SECONDS = 5.0
MAX_INLINE_BYTES = 12_000_000
DEFAULT_MODEL = "gemini-2.5-flash"
OUTPUT = "gemini-count.json"


class Candidate(StrictModel):
    kind: Literal["entry", "exit", "wrong_way_entry"]
    time_s: float = Field(ge=0)
    arm: str
    description: str


class BatchAnswer(StrictModel):
    events: list[Candidate]
    notes: str


class BatchResult(StrictModel):
    start_s: float
    end_s: float
    frame_count: int
    answer: BatchAnswer


class GeminiCountReport(StrictModel):
    model: str
    start_s: float
    end_s: float
    frames_sent: int
    batches: list[BatchResult]


def run_gemini_count(
    video: Path,
    scene: SceneConfig,
    out_dir: Path,
    start_s: float,
    end_s: float,
    fps: float,
    progress: Callable[[float, str], None],
) -> GeminiCountReport:
    """Send only sampled JPEGs to Gemini, in small independent time windows."""
    if not 0 < end_s - start_s <= MAX_DURATION_S:
        raise ValueError("Gemini runs must be at most 60 seconds long")
    if not 0 < fps <= MAX_FPS:
        raise ValueError("Gemini frame rate must be at most 10 fps")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured on the server")
    model = os.environ.get("GEMINI_COUNT_MODEL", DEFAULT_MODEL)
    client = genai.Client(api_key=key)
    batches: list[BatchResult] = []
    frames_sent = 0
    contents: list[types.Part] = []
    frame_count = 0
    byte_count = 0
    current_index: int | None = None

    def submit(index: int) -> None:
        nonlocal contents, frame_count, byte_count, frames_sent
        begin = start_s + index * BATCH_SECONDS
        end = min(begin + BATCH_SECONDS, end_s)
        prompt = (
            "Analyze the timestamped frames in chronological order. Identify only "
            "vehicle entry, exit, or entry from an exit lane that is visible in this "
            "window. Use the scene zones and frame coordinates below. Do not invent "
            "events; describe uncertainty in notes. Times must be original video "
            f"seconds in [{begin}, {end}). Each window is independent; do not infer "
            "movements outside it. Give one event per observed movement. "
            "The image is resized to 1280 pixels wide or less; scale original scene "
            "coordinates to image size. Scene JSON: " + scene.model_dump_json()
        )
        response = client.models.generate_content(
            model=model,
            contents=types.Content(
                role="user", parts=[types.Part.from_text(text=prompt), *contents]
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=BatchAnswer
            ),
        )
        if not response.text:
            raise RuntimeError(f"Gemini returned an empty response for {begin:.1f}s")
        answer = BatchAnswer.model_validate_json(response.text)
        if any(not begin <= item.time_s < end for item in answer.events):
            raise ValueError(f"Gemini returned an event outside {begin:.1f}-{end:.1f}s")
        batches.append(
            BatchResult(
                start_s=begin, end_s=end, frame_count=frame_count, answer=answer
            )
        )
        frames_sent += frame_count
        progress(min((end - start_s) / (end_s - start_s), 1.0), f"Reviewed {end:.1f}s")
        contents, frame_count, byte_count = [], 0, 0

    for _, t_s, frame, _ in iter_sampled_frames(video, fps, start_s, end_s, 0.0):
        if t_s >= end_s:
            break
        index = int((t_s - start_s) // BATCH_SECONDS)
        if current_index is not None and index != current_index:
            submit(current_index)
        current_index = index
        height, width = frame.shape[:2]
        if width > 1280:
            frame = np.asarray(
                cv2.resize(frame, (1280, round(height * 1280 / width))),
                dtype=np.uint8,
            )
        ok, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise RuntimeError(f"Could not encode frame at {t_s:.2f}s")
        data = image.tobytes()
        byte_count += len(data)
        if byte_count > MAX_INLINE_BYTES:
            raise ValueError("Gemini image batch exceeds the inline request limit")
        contents.extend(
            [
                types.Part.from_text(text=f"Video time: {t_s:.3f} seconds"),
                types.Part.from_bytes(data=data, mime_type="image/jpeg"),
            ]
        )
        frame_count += 1
    if current_index is None:
        raise RuntimeError("No sampled frames were read from the selected video")
    submit(current_index)
    report = GeminiCountReport(
        model=model,
        start_s=start_s,
        end_s=end_s,
        frames_sent=frames_sent,
        batches=batches,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(report, out_dir / OUTPUT)
    return report
