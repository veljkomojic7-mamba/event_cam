"""Detection and tracking stage: stream frames, track, store observations."""

import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from traffic_analysis.detector import Tracker
from traffic_analysis.schemas import AnalysisSettings
from traffic_analysis.store import Observation, insert_observations

logger = logging.getLogger(__name__)

DIFF_SIZE = (160, 90)


@dataclass(frozen=True)
class DetectionSummary:
    """Counts describing one detection run."""

    frames_read: int
    frames_processed: int
    frames_skipped_static: int
    observations: int


def _gray_small(frame: NDArray[np.uint8]) -> NDArray[np.float32]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, DIFF_SIZE, interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def iter_sampled_frames(
    video: Path,
    detect_fps: float,
    start_s: float,
    end_s: float | None,
    static_threshold: float,
) -> Iterator[tuple[int, float, NDArray[np.uint8], bool]]:
    """Yield (frame_idx, t_s, frame, is_static) at the requested sample rate.

    Frames between samples are grabbed without decoding to keep reads cheap.
    ``is_static`` is True when the frame barely changed from the previous
    sample; the caller may skip tracking those frames.
    """
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    cap = cv2.VideoCapture(str(video))
    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0:
            raise RuntimeError(f"could not read fps from {video}")
        stride = max(1, round(native_fps / detect_fps))
        first = round(start_s * native_fps)
        last = None if end_s is None else round(end_s * native_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        previous: NDArray[np.float32] | None = None
        frame_idx = first
        while last is None or frame_idx <= last:
            if (frame_idx - first) % stride != 0:
                if not cap.grab():
                    break
                frame_idx += 1
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_u8 = cast("NDArray[np.uint8]", frame)
            small = _gray_small(frame_u8)
            is_static = (
                previous is not None
                and float(np.mean(np.abs(small - previous))) < static_threshold
            )
            previous = small
            yield frame_idx, frame_idx / native_fps, frame_u8, is_static
            frame_idx += 1
    finally:
        cap.release()


def run_detection(
    video: Path,
    conn: sqlite3.Connection,
    tracker: Tracker,
    settings: AnalysisSettings,
    pass_no: int,
    start_s: float = 0.0,
    end_s: float | None = None,
    detect_fps: float | None = None,
    reset_tracker: bool = True,
) -> DetectionSummary:
    """Track one time window of a video and store the observations.

    Pass 1 uses ``settings.detect_fps`` over the whole video. Pass 2 calls this
    with the native fps for flagged windows and ``reset_tracker=True`` so each
    window starts with fresh track IDs.
    """
    rate = settings.detect_fps if detect_fps is None else detect_fps
    if reset_tracker:
        tracker.reset()
    read = processed = skipped = written = 0
    batch: list[Observation] = []
    for frame_idx, t_s, frame, is_static in iter_sampled_frames(
        video, rate, start_s, end_s, settings.static_diff_threshold
    ):
        read += 1
        if is_static:
            skipped += 1
            continue
        processed += 1
        for box in tracker.update(frame):
            batch.append(
                Observation(
                    pass_no=pass_no,
                    track_id=box.track_id,
                    class_name=box.class_name,
                    conf=box.conf,
                    frame_idx=frame_idx,
                    t_s=t_s,
                    x1=box.x1,
                    y1=box.y1,
                    x2=box.x2,
                    y2=box.y2,
                    ground_x_px=(box.x1 + box.x2) / 2.0,
                    ground_y_px=box.y2,
                )
            )
        if len(batch) >= 2000:
            written += insert_observations(conn, batch)
            batch.clear()
        if processed % 200 == 0:
            logger.info(
                "pass %d: %d frames processed (t=%.1fs)", pass_no, processed, t_s
            )
    written += insert_observations(conn, batch)
    logger.info(
        "pass %d done: read=%d processed=%d static_skipped=%d observations=%d",
        pass_no,
        read,
        processed,
        skipped,
        written,
    )
    return DetectionSummary(read, processed, skipped, written)
