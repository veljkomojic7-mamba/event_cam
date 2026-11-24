from pathlib import Path

import pytest
from pydantic import ValidationError

from traffic_analysis.schemas import (
    EventRecord,
    Point2D,
    SceneConfig,
    VehicleRecord,
    Zone,
    load_scene_config,
    save_scene_config,
)

EXAMPLE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "example.json"


def _square(size: float = 100.0) -> list[Point2D]:
    return [
        Point2D(x=0, y=0),
        Point2D(x=size, y=0),
        Point2D(x=size, y=size),
        Point2D(x=0, y=size),
    ]


def _config(**overrides: object) -> SceneConfig:
    data: dict[str, object] = {
        "camera_id": "cam01",
        "frame_width": 200,
        "frame_height": 200,
        "zones": [{"name": "a", "role": "approach", "polygon": _square()}],
    }
    data.update(overrides)
    return SceneConfig.model_validate(data)


def test_example_config_loads() -> None:
    config = load_scene_config(EXAMPLE_CONFIG)
    assert [zone.name for zone in config.zones] == ["north_approach", "east_exit"]


def test_scene_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "cam01.json"
    save_scene_config(_config(), path)
    assert load_scene_config(path) == _config()


def test_duplicate_zone_names_rejected() -> None:
    zone = {"name": "a", "polygon": [p.model_dump() for p in _square()]}
    with pytest.raises(ValidationError, match="unique"):
        _config(zones=[zone, zone])


def test_point_outside_frame_rejected() -> None:
    polygon = [p.model_dump() for p in _square(size=300)]
    with pytest.raises(ValidationError, match="outside the frame"):
        _config(zones=[{"name": "a", "polygon": polygon}])


def test_polygon_needs_three_vertices() -> None:
    with pytest.raises(ValidationError):
        Zone(name="a", polygon=_square()[:2])


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        _config(unexpected=True)


def test_vehicle_record_uses_class_key() -> None:
    record = VehicleRecord.model_validate(
        {
            "track_id": 17,
            "class": "car",
            "class_conf": 0.91,
            "quality": {"id_switches": 0, "occluded_frac": 0.05},
        }
    )
    dumped = record.model_dump(by_alias=True)
    assert dumped["class"] == "car"
    assert record.vehicle_class == "car"


def test_event_end_before_start_rejected() -> None:
    with pytest.raises(ValidationError, match="t_end_s"):
        EventRecord(
            event_id="E-1",
            type="near_miss_candidate",
            t_start_s=5.0,
            t_end_s=4.0,
            tracks=[1],
        )
