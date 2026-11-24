"""Interactive zone editor: draw and adjust polygons on a frame, then save.

Controls: left-click adds a vertex; drag a vertex to move it; Enter closes the
current polygon into a zone; c cancels it; u undoes its last vertex; r cycles
the role of the last zone; x deletes the last zone; s saves; q or Esc quits.
Clicking on an existing vertex grabs it instead of adding a new one.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from traffic_analysis.schemas import (
    Point2D,
    Role,
    SceneConfig,
    Zone,
    load_scene_config,
    save_scene_config,
)

logger = logging.getLogger(__name__)

WINDOW_NAME = "Zone editor"
MAX_DISPLAY_WIDTH = 1280
VERTEX_HIT_RADIUS_PX = 12
MIN_POLYGON_VERTICES = 3
ROLE_CYCLE: tuple[Role, ...] = ("other", "approach", "exit", "both", "ring")
KEY_ENTER = 13
KEY_ESC = 27
KEY_NONE = 255
FILL_ALPHA = 0.25
ZONE_COLOR = (0, 200, 0)
OUTLINE_COLOR = (0, 255, 0)
DRAFT_COLOR = (0, 255, 255)
TEXT_COLOR = (255, 255, 255)
HELP_LINES = (
    "Click: add vertex | Drag vertex: move | Enter: close zone | c: cancel draft",
    "u: undo vertex | r: cycle role of last zone | x: delete last zone",
    "s: save | q / Esc: quit",
)

Command = Literal["continue", "save", "quit"]


@dataclass
class EditorState:
    """Mutable editor state. Polygon coordinates are always in native pixels."""

    width: int
    height: int
    scale: float
    zones: list[Zone]
    draft: list[Point2D] = field(default_factory=list)
    drag: tuple[int, int] | None = None
    dirty: bool = False


def display_scale(width: int) -> float:
    """Scale factor so the frame fits the display width (never upscales)."""
    return min(1.0, MAX_DISPLAY_WIDTH / width)


def to_display(point: Point2D, scale: float) -> tuple[int, int]:
    """Convert a native-pixel point to display pixels."""
    return round(point.x * scale), round(point.y * scale)


def to_native(x: int, y: int, state: EditorState) -> Point2D:
    """Convert display pixels to native pixels, clamped to the frame."""
    native_x = min(max(x / state.scale, 0.0), float(state.width))
    native_y = min(max(y / state.scale, 0.0), float(state.height))
    return Point2D(x=round(native_x, 1), y=round(native_y, 1))


def polygon_at(state: EditorState, index: int) -> list[Point2D]:
    """Return the polygon at ``index``; ``len(zones)`` addresses the draft."""
    if index < len(state.zones):
        return state.zones[index].polygon
    return state.draft


def hit_test(state: EditorState, x: int, y: int) -> tuple[int, int] | None:
    """Return (polygon index, vertex index) of the vertex under the cursor."""
    best: tuple[float, tuple[int, int]] | None = None
    polygons = [zone.polygon for zone in state.zones] + [state.draft]
    for polygon_index, polygon in enumerate(polygons):
        for vertex_index, point in enumerate(polygon):
            px, py = to_display(point, state.scale)
            distance = math.hypot(px - x, py - y)
            if distance <= VERTEX_HIT_RADIUS_PX and (
                best is None or distance < best[0]
            ):
                best = (distance, (polygon_index, vertex_index))
    return None if best is None else best[1]


def next_zone_name(zones: list[Zone]) -> str:
    """Return an unused name of the form ``zone_N``."""
    existing = {zone.name for zone in zones}
    number = len(zones) + 1
    while f"zone_{number}" in existing:
        number += 1
    return f"zone_{number}"


def finish_draft(state: EditorState) -> bool:
    """Turn the draft into a zone. Returns False if it has too few vertices."""
    if len(state.draft) < MIN_POLYGON_VERTICES:
        logger.warning("a zone needs at least %d vertices", MIN_POLYGON_VERTICES)
        return False
    state.zones.append(
        Zone(
            name=next_zone_name(state.zones),
            role="other",
            polygon=list(state.draft),
        )
    )
    state.draft.clear()
    state.dirty = True
    return True


def cycle_role(zone: Zone) -> None:
    """Advance a zone's role through ROLE_CYCLE."""
    position = ROLE_CYCLE.index(zone.role)
    zone.role = ROLE_CYCLE[(position + 1) % len(ROLE_CYCLE)]


def apply_key(state: EditorState, key: int) -> Command:
    """Apply one key press to the editor state and return the next command."""
    if key == KEY_ENTER:
        state.dirty = finish_draft(state) or state.dirty
    elif key == ord("u"):
        if state.draft:
            state.draft.pop()
    elif key == ord("c"):
        state.draft.clear()
    elif key == ord("r"):
        if state.zones:
            cycle_role(state.zones[-1])
            state.dirty = True
    elif key == ord("x"):
        if state.zones:
            state.zones.pop()
            state.dirty = True
    elif key == ord("s"):
        return "save"
    elif key in (ord("q"), KEY_ESC):
        return "quit"
    return "continue"


def _handle_mouse(event: int, x: int, y: int, _flags: int, state: EditorState) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        hit = hit_test(state, x, y)
        if hit is not None:
            state.drag = hit
        else:
            state.draft.append(to_native(x, y, state))
    elif event == cv2.EVENT_MOUSEMOVE and state.drag is not None:
        polygon_index, vertex_index = state.drag
        polygon_at(state, polygon_index)[vertex_index] = to_native(x, y, state)
    elif event == cv2.EVENT_LBUTTONUP and state.drag is not None:
        state.drag = None
        state.dirty = True


def _display_points(polygon: list[Point2D], scale: float) -> NDArray[np.int32]:
    return np.array(
        [to_display(point, scale) for point in polygon], dtype=np.int32
    ).reshape(-1, 1, 2)


def render(base: NDArray[np.uint8], state: EditorState) -> NDArray[np.uint8]:
    """Draw zones, the draft polygon and help text on a copy of the frame."""
    canvas: NDArray[np.uint8] = base.copy()
    fill_layer = canvas.copy()
    for zone in state.zones:
        cv2.fillPoly(
            fill_layer, [_display_points(zone.polygon, state.scale)], ZONE_COLOR
        )
    cv2.addWeighted(fill_layer, FILL_ALPHA, canvas, 1 - FILL_ALPHA, 0, canvas)

    for zone in state.zones:
        points = _display_points(zone.polygon, state.scale)
        cv2.polylines(canvas, [points], True, OUTLINE_COLOR, 2)
        centre = points.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(
            canvas,
            f"{zone.name} ({zone.role})",
            (int(centre[0]), int(centre[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            2,
        )

    if state.draft:
        draft_points = _display_points(state.draft, state.scale)
        cv2.polylines(canvas, [draft_points], False, DRAFT_COLOR, 2)
        for x, y in draft_points.reshape(-1, 2):
            cv2.circle(canvas, (int(x), int(y)), 4, DRAFT_COLOR, -1)

    for line_number, text in enumerate(HELP_LINES):
        cv2.putText(
            canvas,
            text,
            (10, 25 + 25 * line_number),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            TEXT_COLOR,
            2,
        )
    return canvas


def run_editor(frame_path: Path, scene_path: Path) -> None:
    """Open the editor on a frame, loading or creating the scene config."""
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(f"could not read frame image: {frame_path}")
    height, width = frame.shape[:2]

    if scene_path.is_file():
        config = load_scene_config(scene_path)
        if (config.frame_width, config.frame_height) != (width, height):
            raise ValueError(
                f"scene {scene_path} was made for "
                f"{config.frame_width}x{config.frame_height}, frame is {width}x{height}"
            )
    else:
        config = SceneConfig(
            camera_id=scene_path.stem, frame_width=width, frame_height=height
        )

    scale = display_scale(width)
    display_frame: NDArray[Any] = frame
    if scale < 1.0:
        display_frame = cv2.resize(
            frame,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    state = EditorState(
        width=width, height=height, scale=scale, zones=list(config.zones)
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event: int, x: int, y: int, flags: int, _param: object) -> None:
        _handle_mouse(event, x, y, flags, state)

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    logger.info("editing %s; saving to %s", frame_path, scene_path)
    while True:
        cv2.imshow(WINDOW_NAME, render(display_frame, state))
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == KEY_NONE:
            continue
        command = apply_key(state, key)
        if command == "save":
            config.zones = list(state.zones)
            save_scene_config(config, scene_path)
            state.dirty = False
            logger.info("saved %d zone(s) to %s", len(state.zones), scene_path)
        elif command == "quit":
            break
    if state.dirty:
        logger.warning("quit with unsaved changes; they were discarded")
    cv2.destroyWindow(WINDOW_NAME)
