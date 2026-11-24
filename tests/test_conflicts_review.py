"""Tests for conflict scoring and the review budget and request validation."""

import numpy as np
import pytest

from traffic_analysis.conflicts import MotionPath, detect_events
from traffic_analysis.review import ReviewBudget, validate_requests
from traffic_analysis.schemas import AnalysisSettings, ReviewRequest


def _straight(
    track_id: int, start: tuple[float, float], velocity: tuple[float, float]
) -> MotionPath:
    t = np.linspace(0.0, 10.0, 101)
    x = start[0] + velocity[0] * t
    y = start[1] + velocity[1] * t
    speed = np.full_like(t, float(np.hypot(*velocity)))
    return MotionPath(
        track_id=track_id,
        class_name="car",
        t=t,
        xy=np.stack([x, y], axis=1),
        speed=speed,
    )


def test_converging_pair_yields_candidate() -> None:
    # Both reach (0, 0) at t = 5 s, one heading east and one heading north.
    a = _straight(1, (-50.0, 0.0), (10.0, 0.0))
    b = _straight(2, (0.0, -50.0), (0.0, 10.0))
    events = detect_events([a, b], AnalysisSettings())
    assert len(events) == 1
    assert events[0].event_id == "ev-001"
    assert sorted(events[0].tracks) == [1, 2]
    min_sep = events[0].metrics.min_sep_m
    assert min_sep is not None and min_sep < 1.0


def test_distant_parallel_pair_yields_no_event() -> None:
    a = _straight(1, (0.0, 0.0), (10.0, 0.0))
    b = _straight(2, (0.0, 100.0), (10.0, 0.0))
    assert detect_events([a, b], AnalysisSettings()) == []


def test_validate_requests_drops_bad_ranges() -> None:
    settings = AnalysisSettings()
    requests = [
        ReviewRequest(start_s=1.0, end_s=4.0),
        ReviewRequest(start_s=5.0, end_s=5.0),  # empty
        ReviewRequest(start_s=0.0, end_s=40.0),  # longer than 30 s
        ReviewRequest(start_s=8.0, end_s=12.0),  # runs past a 10 s clip
    ]
    valid = validate_requests(requests, clip_duration_s=10.0, settings=settings)
    assert valid == [ReviewRequest(start_s=1.0, end_s=4.0)]


def test_validate_requests_caps_count() -> None:
    settings = AnalysisSettings(review_max_requests=2)
    requests = [ReviewRequest(start_s=float(i), end_s=float(i) + 1.0) for i in range(5)]
    assert len(validate_requests(requests, 100.0, settings)) == 2


def test_budget_blocks_calls_past_cap() -> None:
    budget = ReviewBudget(max_calls=2, max_requests=3)
    budget.take_call()
    budget.take_call()
    with pytest.raises(RuntimeError):
        budget.take_call()


def test_budget_blocks_requests_past_cap() -> None:
    budget = ReviewBudget(max_calls=5, max_requests=3)
    budget.take_requests(2)
    with pytest.raises(RuntimeError):
        budget.take_requests(2)
