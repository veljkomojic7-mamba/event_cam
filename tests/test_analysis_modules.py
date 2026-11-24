"""Unit tests for geometry, storage, calibration, track stitching, review and clips."""

from pathlib import Path

import numpy as np
import pytest

from traffic_analysis.analytics import build_tracks
from traffic_analysis.calibration import build_calibration, estimate_speed
from traffic_analysis.clips import clip_window
from traffic_analysis.geometry import (
    point_in_polygon,
    segment_crossing_sign,
    zones_containing,
)
from traffic_analysis.review import ReviewBudget, validate_requests
from traffic_analysis.schemas import (
    AnalysisSettings,
    CalibrationPoint,
    CountingLine,
    EventRecord,
    Point2D,
    ReviewRequest,
    Zone,
)
from traffic_analysis.store import (
    Observation,
    clear_pass,
    insert_observations,
    load_observations,
    open_store,
)

SQUARE = [
    Point2D(x=0, y=0),
    Point2D(x=10, y=0),
    Point2D(x=10, y=10),
    Point2D(x=0, y=10),
]


def test_point_in_polygon_and_zones() -> None:
    assert point_in_polygon(5, 5, SQUARE)
    assert not point_in_polygon(15, 5, SQUARE)
    zones = [
        Zone(name="north", polygon=SQUARE),
        Zone(
            name="far",
            polygon=[
                Point2D(x=100, y=100),
                Point2D(x=110, y=100),
                Point2D(x=110, y=110),
            ],
        ),
    ]
    assert zones_containing(5, 5, zones) == ["north"]


def test_segment_crossing_sign_direction() -> None:
    line = CountingLine(name="l", start=Point2D(x=0, y=5), end=Point2D(x=10, y=5))
    # Moving from y=0 to y=10 crosses the horizontal line in one direction.
    forward = segment_crossing_sign((5.0, 0.0), (5.0, 10.0), line)
    backward = segment_crossing_sign((5.0, 10.0), (5.0, 0.0), line)
    assert forward != 0
    assert backward == -forward
    assert segment_crossing_sign((0.0, 0.0), (0.0, 2.0), line) == 0


def test_store_round_trip_and_clear(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "obs.sqlite")
    rows = [
        Observation(
            pass_no=1,
            track_id=7,
            class_name="car",
            conf=0.9,
            frame_idx=i,
            t_s=i * 0.1,
            x1=0,
            y1=0,
            x2=1,
            y2=1,
            ground_x_px=0.5,
            ground_y_px=1.0,
        )
        for i in range(3)
    ]
    assert insert_observations(conn, rows) == 3
    loaded = load_observations(conn, 1)
    assert [o.frame_idx for o in loaded] == [0, 1, 2]
    clear_pass(conn, 1)
    assert load_observations(conn, 1) == []
    conn.close()


def test_calibration_recovers_scale() -> None:
    # Known mapping: 1 px = 0.05 m, with a shifted origin.
    image = [(0, 0), (1000, 0), (1000, 800), (0, 800), (500, 400)]
    points = [
        CalibrationPoint(
            image=Point2D(x=float(x), y=float(y)),
            world=Point2D(x=x * 0.05 + 2.0, y=y * 0.05 - 1.0),
        )
        for x, y in image
    ]
    calibration = build_calibration("cam-test", points)
    assert calibration.rms_error_m < 1e-3
    assert calibration.calibration_id.startswith("cam-test-")


def test_estimate_speed_constant_velocity() -> None:
    times = np.arange(0.0, 5.0, 0.1)
    metres = np.column_stack([times * 10.0, np.zeros_like(times)])
    speed = estimate_speed(metres, times, window_s=2.0)
    assert speed is not None
    assert speed.mean == pytest.approx(36.0, rel=0.05)


def test_build_tracks_stitches_close_fragments() -> None:
    settings = AnalysisSettings(stitch_gap_s=1.0, stitch_max_px=20.0, min_track_s=0.5)

    def obs(track_id: int, t: float, x: float) -> Observation:
        return Observation(
            pass_no=1,
            track_id=track_id,
            class_name="car",
            conf=0.8,
            frame_idx=0,
            t_s=t,
            x1=0,
            y1=0,
            x2=1,
            y2=1,
            ground_x_px=x,
            ground_y_px=50.0,
        )

    first = [obs(1, t * 0.1, 100.0) for t in range(0, 21)]
    second = [obs(2, 2.5 + t * 0.1, 105.0) for t in range(0, 21)]
    tracks = build_tracks(first + second, settings)
    assert len(tracks) == 1


def test_validate_requests_filters_bad_ranges() -> None:
    settings = AnalysisSettings()
    requests = [
        ReviewRequest(start_s=1.0, end_s=3.0),
        ReviewRequest(start_s=3.0, end_s=2.0),  # reversed
        ReviewRequest(start_s=0.0, end_s=500.0),  # beyond the clip
        ReviewRequest(start_s=0.0, end_s=settings.review_request_max_s + 5),  # too long
    ]
    valid = validate_requests(requests, clip_duration_s=20.0, settings=settings)
    assert valid == [ReviewRequest(start_s=1.0, end_s=3.0)]


def test_review_budget_caps() -> None:
    budget = ReviewBudget(max_calls=1, max_requests=2)
    budget.take_call()
    with pytest.raises(RuntimeError, match="call budget"):
        budget.take_call()
    budget.take_requests(2)
    with pytest.raises(RuntimeError, match="request budget"):
        budget.take_requests(1)


def test_clip_window_is_padded_and_clamped() -> None:
    event = EventRecord(
        event_id="ev-001",
        type="near_miss_candidate",
        t_start_s=1.0,
        t_end_s=4.0,
        tracks=[1],
    )
    assert clip_window(event, duration_s=10.0, pad_s=2.0) == (0.0, 6.0)
    assert clip_window(event, duration_s=5.0, pad_s=2.0) == (0.0, 5.0)
