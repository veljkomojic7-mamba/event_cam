"""Detector and tracker behind a small interface so the model can be swapped.

The Ultralytics backend runs detection and ByteTrack in one ``model.track``
call, so the detector and tracker share one interface here.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from traffic_analysis.schemas import AnalysisSettings

KEPT_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle", "person"})
AERIAL_CLASSES = {"small vehicle": "car", "large vehicle": "car"}


@dataclass(frozen=True)
class TrackedBox:
    """One tracked detection in frame pixels."""

    track_id: int
    class_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


class Tracker(Protocol):
    """Detects and tracks objects frame by frame."""

    def update(self, frame: NDArray[np.uint8]) -> list[TrackedBox]:
        """Process one frame and return the boxes that have a track ID."""
        ...

    def reset(self) -> None:
        """Forget all tracks, e.g. before a new time window."""
        ...


class YoloTracker:
    """Ultralytics YOLO detection with its built-in ByteTrack tracker."""

    def __init__(self, settings: AnalysisSettings) -> None:
        # Imported here so that tests and tools without torch still import.
        from ultralytics.models.yolo.model import YOLO

        self._settings = settings
        self._model: Any = YOLO(settings.model)

    def update(self, frame: NDArray[np.uint8]) -> list[TrackedBox]:
        options: dict[str, Any] = {}
        if self._settings.device is not None:
            options["device"] = self._settings.device
        if self._settings.half:
            options["half"] = True
        results = self._model.track(
            frame,
            persist=True,
            tracker=self._settings.tracker,
            conf=self._settings.conf,
            imgsz=self._settings.imgsz,
            verbose=False,
            **options,
        )
        result = results[0]
        boxes = result.obb if result.obb is not None else result.boxes
        if boxes is None or boxes.id is None:
            return []
        names: dict[int, str] = dict(result.names)
        ids = boxes.id.int().cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.float().cpu().tolist()
        coords = boxes.xyxy.float().cpu().tolist()
        found: list[TrackedBox] = []
        for track_id, cls_id, conf, (x1, y1, x2, y2) in zip(
            ids, classes, confs, coords, strict=True
        ):
            raw_class = names.get(cls_id, str(cls_id))
            class_name = AERIAL_CLASSES.get(raw_class, raw_class)
            if class_name not in KEPT_CLASSES:
                continue
            found.append(
                TrackedBox(
                    track_id=int(track_id),
                    class_name=class_name,
                    conf=float(conf),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
        return found

    def reset(self) -> None:
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []):
            tracker.reset()
