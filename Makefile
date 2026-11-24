# Traffic analysis pipeline. Override on the command line, e.g.:
#   make sample VIDEO=data/input/cam01.mp4 START=600 DURATION=300
VIDEO ?= data/input/cam01.mp4
START ?= 0
DURATION ?= 240
FRAME_TIME ?= 10

CAM := $(basename $(notdir $(VIDEO)))
VIDEO_INFO := data/$(CAM)/video.json
SAMPLE := data/$(CAM)/sample.mp4
FRAME := data/$(CAM)/frame.png
SCENE := configs/$(CAM).json

.PHONY: setup lint format typecheck test check probe sample frame zones

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

probe:
	uv run traffic-analysis probe --video $(VIDEO) --output $(VIDEO_INFO)

sample: $(SAMPLE)

$(SAMPLE): $(VIDEO)
	uv run traffic-analysis sample --video $(VIDEO) --start $(START) --duration $(DURATION) --output $(SAMPLE)

frame: $(FRAME)

$(FRAME): $(VIDEO)
	uv run traffic-analysis frame --video $(VIDEO) --time $(FRAME_TIME) --output $(FRAME)

zones: $(FRAME)
	uv run traffic-analysis zones --frame $(FRAME) --scene $(SCENE)

# Pipeline stages. Output goes to data/<cam>/out. Cloud steps need CLOUD=1.
OUT := data/$(CAM)/out
CALIB ?= configs/$(CAM)_calibration.json
CLOUD_FLAG := $(if $(CLOUD),--allow-cloud,)

.PHONY: detect calibrate analyze events clips review report run

detect:
	uv run traffic-analysis detect --video $(VIDEO) --out-dir $(OUT)

calibrate:
	uv run traffic-analysis calibrate --camera-id $(CAM) --points $(POINTS) --out $(CALIB)

analyze:
	uv run traffic-analysis analyze --video $(VIDEO) --out-dir $(OUT) --scene $(SCENE) --calibration $(CALIB)

events:
	uv run traffic-analysis events --video $(VIDEO) --out-dir $(OUT) --calibration $(CALIB) --scene $(SCENE)

clips:
	uv run traffic-analysis clips --video $(VIDEO) --out-dir $(OUT)

review:
	uv run traffic-analysis review --video $(VIDEO) --out-dir $(OUT) $(CLOUD_FLAG)

report:
	uv run traffic-analysis report --out-dir $(OUT) $(CLOUD_FLAG)

run:
	uv run traffic-analysis run --video $(VIDEO) --out-dir $(OUT) --scene $(SCENE) --calibration $(CALIB) $(CLOUD_FLAG)
