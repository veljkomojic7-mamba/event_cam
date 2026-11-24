"""Plain geometry helpers: polygon containment, line crossings, homography."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from traffic_analysis.schemas import CountingLine, Point2D, Zone

FloatArray = NDArray[np.float64]


def point_in_polygon(x: float, y: float, polygon: Sequence[Point2D]) -> bool:
    """Return True when (x, y) lies inside or on the polygon boundary."""
    contour = np.array([[p.x, p.y] for p in polygon], dtype=np.float32)
    return bool(cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0)


def zones_containing(x: float, y: float, zones: Sequence[Zone]) -> list[str]:
    """Return the names of all zones that contain the point."""
    return [zone.name for zone in zones if point_in_polygon(x, y, zone.polygon)]


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def segment_crossing_sign(
    p0: tuple[float, float],
    p1: tuple[float, float],
    line: CountingLine,
) -> int:
    """Return +1 or -1 when segment p0->p1 crosses the line, else 0.

    The sign is the side of the line the segment moves to: +1 when the motion
    goes from the left of ``line.start -> line.end`` to the right, -1 for the
    reverse. Zero means no crossing.
    """
    lx, ly = line.end.x - line.start.x, line.end.y - line.start.y
    s0 = _cross(lx, ly, p0[0] - line.start.x, p0[1] - line.start.y)
    s1 = _cross(lx, ly, p1[0] - line.start.x, p1[1] - line.start.y)
    if s0 == 0 or s1 == 0 or (s0 > 0) == (s1 > 0):
        return 0
    # Both endpoints of the line must lie on opposite sides of the segment.
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0 = _cross(dx, dy, line.start.x - p0[0], line.start.y - p0[1])
    t1 = _cross(dx, dy, line.end.x - p0[0], line.end.y - p0[1])
    if (t0 > 0) == (t1 > 0):
        return 0
    return 1 if s0 > 0 else -1


def apply_homography(points: FloatArray, homography: FloatArray) -> FloatArray:
    """Map an (N, 2) array of pixel points to road-plane metres."""
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    reshaped = points.reshape(-1, 1, 2).astype(np.float64)
    mapped = cv2.perspectiveTransform(reshaped, homography)
    return np.asarray(mapped.reshape(-1, 2), dtype=np.float64)
