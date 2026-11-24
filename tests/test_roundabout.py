"""Roundabout entry, exit and wrong-way counting, independent of lane."""

from traffic_analysis.analytics import (
    Track,
    build_stats,
    build_vehicle_record,
    roundabout_passage,
    wrong_way_events,
)
from traffic_analysis.report import render_report
from traffic_analysis.schemas import AnalysisSettings, Point2D, SceneConfig, Zone
from traffic_analysis.store import Observation


def _box(x0: float, y0: float, x1: float, y1: float) -> list[Point2D]:
    return [
        Point2D(x=x0, y=y0),
        Point2D(x=x1, y=y0),
        Point2D(x=x1, y=y1),
        Point2D(x=x0, y=y1),
    ]


SCENE = SceneConfig(
    camera_id="cam",
    frame_width=300,
    frame_height=300,
    zones=[
        Zone(name="n_in", role="approach", polygon=_box(0, 0, 100, 80)),
        Zone(name="n_out", role="exit", polygon=_box(100, 0, 200, 80)),
        Zone(name="ring", role="ring", polygon=_box(0, 60, 300, 300)),
    ],
)


def _track(track_id: int, points: list[tuple[float, float]]) -> Track:
    observations = [
        Observation(
            pass_no=1,
            track_id=track_id,
            class_name="car",
            conf=0.9,
            frame_idx=i,
            t_s=i * 0.1,
            x1=0,
            y1=0,
            x2=1,
            y2=1,
            ground_x_px=x,
            ground_y_px=y,
        )
        for i, (x, y) in enumerate(points)
    ]
    return Track(
        track_id=track_id, class_name="car", class_conf=0.9, observations=observations
    )


def _drive(x: float, ys: list[float]) -> list[tuple[float, float]]:
    return [(x, y) for y in ys]


NORMAL_IN = _track(1, _drive(50, [10, 20, 30, 40, 50, 100, 120]))
WRONG_WAY_IN = _track(2, _drive(150, [10, 20, 30, 40, 50, 100, 120]))
NORMAL_OUT = _track(3, _drive(150, [120, 100, 50, 40, 30, 20, 10]))
WRONG_WAY_OUT = _track(4, _drive(50, [120, 100, 50, 40, 30, 20, 10]))
FLICKER_OUT = _track(5, _drive(150, [120, 100, 30, 100, 30, 20, 10]))


def test_normal_entry_is_counted_once_regardless_of_lane() -> None:
    passage = roundabout_passage(NORMAL_IN, SCENE, 2)
    assert passage is not None
    assert passage.entry_arm == "n_in"
    assert not passage.entry_wrong_way
    assert passage.exit_arm is None


def test_entry_from_exit_lane_is_wrong_way() -> None:
    passage = roundabout_passage(WRONG_WAY_IN, SCENE, 2)
    assert passage is not None
    assert passage.entry_arm == "n_out"
    assert passage.entry_wrong_way


def test_exits_and_wrong_way_exits() -> None:
    normal = roundabout_passage(NORMAL_OUT, SCENE, 2)
    wrong = roundabout_passage(WRONG_WAY_OUT, SCENE, 2)
    assert normal is not None and wrong is not None
    assert normal.entry_arm is None
    assert normal.exit_arm == "n_out" and not normal.exit_wrong_way
    assert wrong.exit_arm == "n_in" and wrong.exit_wrong_way


def test_boundary_flicker_does_not_create_wrong_way_entry() -> None:
    passage = roundabout_passage(FLICKER_OUT, SCENE, 2)
    assert passage is not None
    assert passage.entry_arm is None
    assert not passage.entry_wrong_way


def test_exit_counts_when_track_ends_right_after_crossing() -> None:
    track = _track(6, _drive(150, [120, 100, 50]))
    passage = roundabout_passage(track, SCENE, 2)
    assert passage is not None
    assert passage.exit_arm == "n_out"


def test_no_ring_zone_gives_no_passage() -> None:
    scene = SCENE.model_copy(update={"zones": SCENE.zones[:2]})
    assert roundabout_passage(NORMAL_IN, scene, 2) is None


def test_wrong_way_events_and_stats_and_report() -> None:
    settings = AnalysisSettings()
    tracks = [NORMAL_IN, WRONG_WAY_IN, NORMAL_OUT, WRONG_WAY_OUT]
    events = wrong_way_events(tracks, SCENE, settings)
    assert [e.event_id for e in events] == ["ww-001"]
    assert events[0].type == "wrong_way_entry"
    assert events[0].tracks == [2]

    records = [build_vehicle_record(t, SCENE, settings, 10.0, None) for t in tracks]
    stats = build_stats("v", SCENE, settings, tracks, records, None)
    ring = stats.roundabout
    assert ring is not None
    assert (ring.entered, ring.exited) == (2, 2)
    assert (ring.wrong_way_entries, ring.wrong_way_exits) == (1, 1)
    assert ring.wrong_way_examples[0].track_id == 2
    report = render_report(stats, events)
    assert "Entered the roundabout: 2" in report
    assert "Entered from an exit lane (wrong way): 1" in report


def test_first_seen_ring_is_separate_and_excludes_island_and_arms() -> None:
    scene = SCENE.model_copy(update={"ring_exclusion": _box(200, 150, 260, 210)})
    tracks = [
        NORMAL_IN,
        NORMAL_OUT,
        _track(10, [(230, 180), (230, 220)]),
        _track(11, [(250, 20), (250, 100), (250, 120)]),
        _track(12, [(50, 60), (50, 70), (50, 100), (50, 120)]),
    ]
    settings = AnalysisSettings()
    records = [build_vehicle_record(t, scene, settings, 10.0, None) for t in tracks]
    stats = build_stats("v", scene, settings, tracks, records, None)
    assert stats.roundabout is not None
    ring = stats.roundabout
    assert ring.first_seen_on_ring is not None
    assert [item.track_id for item in ring.first_seen_on_ring] == [NORMAL_OUT.track_id]
    assert ring.entered == 2
    assert ring.exited == 1
    report = render_report(stats, [])
    assert "Tracks first seen on ring: 1" in report
    assert "Track 3: 0.0 s" in report


def test_first_seen_ring_unavailable_without_exclusion() -> None:
    settings = AnalysisSettings()
    record = build_vehicle_record(NORMAL_OUT, SCENE, settings, 10.0, None)
    stats = build_stats("v", SCENE, settings, [NORMAL_OUT], [record], None)
    assert stats.roundabout is not None
    assert stats.roundabout.first_seen_on_ring is None


def test_first_seen_ring_empty_when_no_vehicles() -> None:
    scene = SCENE.model_copy(update={"ring_exclusion": _box(200, 150, 260, 210)})
    stats = build_stats("v", scene, AnalysisSettings(), [], [], None)
    assert stats.roundabout is not None
    assert stats.roundabout.first_seen_on_ring == []
