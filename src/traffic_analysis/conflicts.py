"""Near-miss candidates and hard-deceleration events, ranked by a transparent score.

Metrics are geometric approximations on the ground plane (2D distances, constant
velocity). They are candidates for review, not confirmed conflicts.
"""

from dataclasses import dataclass

import numpy as np

from traffic_analysis.analytics import VEHICLE_CLASSES, Track
from traffic_analysis.geometry import FloatArray, apply_homography
from traffic_analysis.schemas import AnalysisSettings, EventMetrics, EventRecord

EVENT_HALF_WINDOW_S = 1.0


@dataclass(frozen=True)
class MotionPath:
    """A vehicle's ground path in metres with a centred-window speed per sample."""

    track_id: int
    class_name: str
    t: FloatArray
    xy: FloatArray
    speed: FloatArray


def _windowed_speed(t: FloatArray, xy: FloatArray, window_s: float) -> FloatArray:
    half = window_s / 2.0
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right") - 1
    speed = np.zeros(len(t), dtype=np.float64)
    for i in range(len(t)):
        a, b = int(lo[i]), int(hi[i])
        dt = t[b] - t[a]
        if b > a and dt > 0:
            speed[i] = float(np.hypot(*(xy[b] - xy[a]))) / float(dt)
    return speed


def build_paths(
    tracks: list[Track], homography: FloatArray, settings: AnalysisSettings
) -> list[MotionPath]:
    """Convert vehicle tracks into metre paths with windowed speeds."""
    paths: list[MotionPath] = []
    for track in tracks:
        if track.class_name not in VEHICLE_CLASSES or len(track.observations) < 3:
            continue
        pts = np.array(
            [[o.ground_x_px, o.ground_y_px] for o in track.observations],
            dtype=np.float64,
        )
        t = np.array([o.t_s for o in track.observations], dtype=np.float64)
        # Stitched fragments can share a frame time; keep one sample per time
        # so interpolation and gradients see strictly increasing times.
        t, keep = np.unique(t, return_index=True)
        pts = pts[keep]
        xy = apply_homography(pts, homography)
        speed = _windowed_speed(t, xy, settings.speed_window_s)
        paths.append(
            MotionPath(
                track_id=track.track_id,
                class_name=track.class_name,
                t=t,
                xy=xy,
                speed=speed,
            )
        )
    return paths


@dataclass(frozen=True)
class _Candidate:
    score: float
    event: EventRecord


def _pair_candidate(
    a: MotionPath, b: MotionPath, settings: AnalysisSettings
) -> _Candidate | None:
    lo = max(a.t[0], b.t[0])
    hi = min(a.t[-1], b.t[-1])
    if hi <= lo:
        return None
    mask = (a.t >= lo) & (a.t <= hi)
    ta = a.t[mask]
    if len(ta) < 3:
        return None
    xa = a.xy[mask]
    xb = np.stack(
        [np.interp(ta, b.t, b.xy[:, 0]), np.interp(ta, b.t, b.xy[:, 1])], axis=1
    )
    va = a.speed[mask]
    vb = np.interp(ta, b.t, b.speed)
    sep = np.hypot(*(xa - xb).T)
    k = int(np.argmin(sep))
    min_sep = float(sep[k])
    if min_sep > settings.sep_m:
        return None
    sep_rate = np.gradient(sep, ta)
    closing = -sep_rate
    approaching = closing > settings.conflict_min_speed_mps
    approaching &= sep > 0
    if not np.any(approaching):
        return None
    ttc_values = sep[approaching] / closing[approaching]
    ttc = float(np.min(ttc_values))
    drac = float(np.max(closing[approaching] ** 2 / (2.0 * sep[approaching])))
    if (
        va[k] < settings.conflict_min_speed_mps
        or vb[k] < settings.conflict_min_speed_mps
    ):
        return None
    midpoint = (xa[k] + xb[k]) / 2.0
    near_a = float(a.t[int(np.argmin(np.hypot(*(a.xy - midpoint).T)))])
    near_b = float(b.t[int(np.argmin(np.hypot(*(b.xy - midpoint).T)))])
    pet = abs(near_a - near_b)
    score = 0.6 * max(0.0, 1.0 - min_sep / settings.sep_m) + 0.4 * max(
        0.0, 1.0 - ttc / settings.ttc_s
    )
    t_conflict = float(ta[k])
    event = EventRecord(
        event_id="pending",
        type="near_miss_candidate",
        t_start_s=max(0.0, t_conflict - EVENT_HALF_WINDOW_S),
        t_end_s=t_conflict + EVENT_HALF_WINDOW_S,
        tracks=[a.track_id, b.track_id],
        metrics=EventMetrics(
            min_sep_m=round(min_sep, 3),
            ttc_min_s=round(ttc, 3),
            pet_s=round(pet, 3),
            drac=round(drac, 3),
            score=round(score, 4),
        ),
    )
    return _Candidate(score=score, event=event)


def _hard_decel(path: MotionPath, settings: AnalysisSettings) -> _Candidate | None:
    t, speed = path.t, path.speed
    best = 0.0
    best_i = best_j = -1
    for i in range(len(t)):
        j = int(np.searchsorted(t, t[i] + settings.speed_window_s, side="left"))
        if j >= len(t):
            break
        dt = t[j] - t[i]
        if dt <= 0:
            continue
        decel = (speed[i] - speed[j]) / dt
        if decel > best:
            best, best_i, best_j = decel, i, j
    if best <= settings.decel_mps2:
        return None
    event = EventRecord(
        event_id="pending",
        type="hard_deceleration",
        t_start_s=float(t[best_i]),
        t_end_s=float(t[best_j]),
        tracks=[path.track_id],
        metrics=EventMetrics(
            decel_mps2=round(best, 3),
            score=round(best / settings.decel_mps2, 4),
        ),
    )
    return _Candidate(score=best / settings.decel_mps2, event=event)


def detect_events(
    paths: list[MotionPath], settings: AnalysisSettings
) -> list[EventRecord]:
    """Score every pair and every track, keep the top events, assign stable IDs."""
    candidates: list[_Candidate] = []
    for i, a in enumerate(paths):
        decel = _hard_decel(a, settings)
        if decel is not None:
            candidates.append(decel)
        for b in paths[i + 1 :]:
            pair = _pair_candidate(a, b, settings)
            if pair is not None:
                candidates.append(pair)
    candidates.sort(key=lambda c: c.score, reverse=True)
    kept = candidates[: settings.max_events]
    events: list[EventRecord] = []
    for index, candidate in enumerate(kept, start=1):
        events.append(
            candidate.event.model_copy(update={"event_id": f"ev-{index:03d}"})
        )
    return events


def plan_pass2_windows(
    events: list[EventRecord], settings: AnalysisSettings, duration_s: float
) -> list[tuple[float, float]]:
    """Merge padded event windows, highest score first, capped at the window limit."""
    ordered = sorted(events, key=lambda e: e.metrics.score, reverse=True)
    windows: list[tuple[float, float]] = []
    for event in ordered:
        start = max(0.0, event.t_start_s - settings.pass2_pad_s)
        end = min(duration_s, event.t_end_s + settings.pass2_pad_s)
        if end <= start:
            continue
        merged = False
        for index, (s, e) in enumerate(windows):
            if start <= e and end >= s:
                windows[index] = (min(s, start), max(e, end))
                merged = True
                break
        if not merged:
            windows.append((start, end))
        if len(windows) >= settings.pass2_max_windows:
            break
    return sorted(windows)
