"""Ground-plane calibration: pixel-to-metre homography, reprojection error, speeds."""

import hashlib
from collections.abc import Sequence

import cv2
import numpy as np

from traffic_analysis.geometry import FloatArray, apply_homography
from traffic_analysis.schemas import (
    CalibrationFile,
    CalibrationPoint,
    SpeedEstimate,
)

MIN_POINTS = 4
KMH_PER_MPS = 3.6


def build_calibration(
    camera_id: str, points: Sequence[CalibrationPoint]
) -> CalibrationFile:
    """Fit an image-to-metres homography and report its reprojection error."""
    if len(points) < MIN_POINTS:
        raise ValueError(f"need at least {MIN_POINTS} calibration points")
    image = np.array([[p.image.x, p.image.y] for p in points], dtype=np.float64)
    world = np.array([[p.world.x, p.world.y] for p in points], dtype=np.float64)
    fitted, _ = cv2.findHomography(image, world, method=0)
    if fitted is None:
        raise ValueError("homography could not be fitted; check for collinear points")
    homography: FloatArray = np.asarray(fitted, dtype=np.float64)
    projected = apply_homography(image, homography)
    errors = np.linalg.norm(projected - world, axis=1)
    digest = hashlib.sha256(
        (camera_id + repr(np.round(homography, 9).tolist())).encode()
    ).hexdigest()[:12]
    return CalibrationFile(
        calibration_id=f"{camera_id}-{digest}",
        camera_id=camera_id,
        points=list(points),
        homography=homography.tolist(),
        rms_error_m=round(float(np.sqrt(np.mean(errors**2))), 4),
        max_error_m=round(float(np.max(errors)), 4),
    )


def homography_matrix(calibration: CalibrationFile) -> FloatArray:
    """Return the 3x3 homography as a float64 array."""
    return np.array(calibration.homography, dtype=np.float64)


def estimate_speed(
    metres: FloatArray, times: FloatArray, window_s: float
) -> SpeedEstimate | None:
    """Windowed speed from ground positions in metres.

    Each sample is paired with the first later sample at least ``window_s`` away,
    which smooths per-frame jitter. Returns None when the track is too short.
    """
    n = len(times)
    speeds: list[float] = []
    j = 0
    for i in range(n):
        if j <= i:
            j = i + 1
        while j < n and times[j] - times[i] < window_s:
            j += 1
        if j >= n:
            break
        dt = float(times[j] - times[i])
        if dt <= 0:
            continue
        dist = float(np.hypot(*(metres[j] - metres[i])))
        speeds.append(dist / dt * KMH_PER_MPS)
    if not speeds:
        return None
    arr = np.array(speeds, dtype=np.float64)
    return SpeedEstimate(
        mean=round(float(arr.mean()), 2),
        median=round(float(np.median(arr)), 2),
        max=round(float(arr.max()), 2),
    )
