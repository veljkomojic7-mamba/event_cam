import pytest

from traffic_analysis.overlay import (
    EditorState,
    apply_key,
    cycle_role,
    display_scale,
    hit_test,
    to_display,
    to_native,
)
from traffic_analysis.schemas import Point2D, Zone

ENTER = 13
ESC = 27


def _state(scale: float = 1.0, zones: list[Zone] | None = None) -> EditorState:
    return EditorState(width=1000, height=500, scale=scale, zones=zones or [])


def _triangle() -> list[Point2D]:
    return [Point2D(x=10, y=10), Point2D(x=100, y=10), Point2D(x=50, y=80)]


def test_display_scale_never_upscales_and_fits_width() -> None:
    assert display_scale(800) == 1.0
    assert display_scale(2560) == pytest.approx(0.5)


def test_coordinate_round_trip_with_scale() -> None:
    state = _state(scale=0.5)
    point = to_native(100, 50, state)
    assert (point.x, point.y) == (200.0, 100.0)
    assert to_display(point, state.scale) == (100, 50)


def test_to_native_clamps_to_frame() -> None:
    state = _state()
    point = to_native(5000, -20, state)
    assert (point.x, point.y) == (1000.0, 0.0)


def test_enter_needs_three_vertices() -> None:
    state = _state()
    state.draft = _triangle()[:2]
    assert apply_key(state, ENTER) == "continue"
    assert state.zones == []
    assert len(state.draft) == 2


def test_enter_closes_draft_into_named_zone() -> None:
    state = _state()
    state.draft = _triangle()
    apply_key(state, ENTER)
    assert [zone.name for zone in state.zones] == ["zone_1"]
    assert state.draft == []
    assert state.dirty


def test_zone_names_do_not_collide_after_delete() -> None:
    state = _state(zones=[Zone(name="zone_2", polygon=_triangle())])
    state.draft = _triangle()
    apply_key(state, ENTER)
    assert [zone.name for zone in state.zones] == ["zone_2", "zone_3"]


def test_undo_cancel_and_delete() -> None:
    state = _state(zones=[Zone(name="z", polygon=_triangle())])
    state.draft = _triangle()
    apply_key(state, ord("u"))
    assert len(state.draft) == 2
    apply_key(state, ord("c"))
    assert state.draft == []
    apply_key(state, ord("x"))
    assert state.zones == []


def test_cycle_role_wraps_around() -> None:
    zone = Zone(name="z", polygon=_triangle())
    seen: list[str] = []
    for _ in range(5):
        cycle_role(zone)
        seen.append(zone.role)
    assert seen == ["approach", "exit", "both", "ring", "other"]


def test_hit_test_finds_vertex_within_radius() -> None:
    state = _state(zones=[Zone(name="z", polygon=_triangle())])
    assert hit_test(state, 12, 11) == (0, 0)
    assert hit_test(state, 500, 400) is None


def test_hit_test_addresses_draft_after_zones() -> None:
    state = _state(zones=[Zone(name="z", polygon=_triangle())])
    state.draft = [Point2D(x=300, y=300)]
    assert hit_test(state, 300, 300) == (1, 0)


def test_save_and_quit_keys() -> None:
    state = _state()
    assert apply_key(state, ord("s")) == "save"
    assert apply_key(state, ord("q")) == "quit"
    assert apply_key(state, ESC) == "quit"
