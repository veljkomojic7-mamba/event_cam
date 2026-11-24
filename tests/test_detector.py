"""Tests for standard and aerial tracker output."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from traffic_analysis.detector import YoloTracker
from traffic_analysis.schemas import AnalysisSettings


@pytest.mark.parametrize("oriented", [False, True])
def test_tracker_extracts_supported_boxes(
    monkeypatch: pytest.MonkeyPatch, oriented: bool
) -> None:
    from ultralytics.models.yolo import model as yolo_module

    boxes = SimpleNamespace(
        id=torch.tensor([1, 2, 3]),
        cls=torch.tensor([0, 1, 2]),
        conf=torch.tensor([0.8, 0.7, 0.6]),
        xyxy=torch.tensor([[10, 20, 30, 40], [40, 50, 60, 70], [0, 0, 5, 5]]),
    )
    names = (
        {0: "small vehicle", 1: "large vehicle", 2: "plane"}
        if oriented
        else {0: "car", 1: "truck", 2: "traffic light"}
    )
    result = SimpleNamespace(
        obb=boxes if oriented else None,
        boxes=None if oriented else boxes,
        names=names,
    )
    model = Mock()
    model.track.return_value = [result]
    monkeypatch.setattr(yolo_module, "YOLO", Mock(return_value=model))

    tracker = YoloTracker(AnalysisSettings())
    found = tracker.update(np.zeros((10, 10, 3), dtype=np.uint8))

    assert [box.track_id for box in found] == [1, 2]
    assert [box.class_name for box in found] == (
        ["car", "car"] if oriented else ["car", "truck"]
    )
    assert (found[0].x1, found[0].y1, found[0].x2, found[0].y2) == (
        10,
        20,
        30,
        40,
    )


def test_tracker_ignores_untracked_aerial_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ultralytics.models.yolo import model as yolo_module

    model = Mock()
    model.track.return_value = [
        SimpleNamespace(obb=SimpleNamespace(id=None), boxes=None)
    ]
    monkeypatch.setattr(yolo_module, "YOLO", Mock(return_value=model))
    tracker = YoloTracker(AnalysisSettings(model="yolo11m-obb.pt"))
    assert tracker.update(np.zeros((10, 10, 3), dtype=np.uint8)) == []
