from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project_name": "real_time_object_detection_tracking",
    "source": "0",
    "model": {
        "weights": "yolov8n.pt",
        "tracker": "botsort.yaml",
        "confidence": 0.35,
        "iou": 0.45,
        "image_size": 640,
        "device": "cpu",
        "classes": None,
    },
    "output": {
        "show_window": True,
        "save_video": True,
        "output_dir": "artifacts/runs",
        "window_name": "Real-Time Object Detection and Tracking",
        "video_name_prefix": "tracking_run",
        "display_scale": 1.0,
    },
    "features": {
        "draw_trails": True,
        "trail_length": 30,
        "enable_line_counting": False,
        "count_line": [[200, 300], [1000, 300]],
    },
}


@dataclass(slots=True)
class ModelConfig:
    weights: str
    tracker: str
    confidence: float
    iou: float
    image_size: int
    device: str
    classes: list[int] | None = None


@dataclass(slots=True)
class OutputConfig:
    show_window: bool
    save_video: bool
    output_dir: str
    window_name: str
    video_name_prefix: str
    display_scale: float = 1.0


@dataclass(slots=True)
class FeatureConfig:
    draw_trails: bool
    trail_length: int
    enable_line_counting: bool
    count_line: tuple[tuple[int, int], tuple[int, int]] | None = None


@dataclass(slots=True)
class AppConfig:
    project_name: str
    source: str | int
    model: ModelConfig
    output: OutputConfig
    features: FeatureConfig
    config_path: Path | None = None


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = {**base}
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_source(source: Any) -> str | int:
    if isinstance(source, int):
        return source

    source_text = str(source).strip()
    return int(source_text) if source_text.isdigit() else source_text


def _normalize_count_line(raw_line: Any) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if raw_line in (None, "", []):
        return None

    if not isinstance(raw_line, list) or len(raw_line) != 2:
        raise ValueError("count_line must contain exactly two points, for example [[200, 300], [1000, 300]].")

    start, end = raw_line
    if len(start) != 2 or len(end) != 2:
        raise ValueError("Each count_line point must contain exactly two numbers.")

    return (int(start[0]), int(start[1])), (int(end[0]), int(end[1]))


def _build_config(data: dict[str, Any], config_path: Path | None = None) -> AppConfig:
    model = data["model"]
    output = data["output"]
    features = data["features"]

    return AppConfig(
        project_name=str(data["project_name"]),
        source=_normalize_source(data["source"]),
        model=ModelConfig(
            weights=str(model["weights"]),
            tracker=str(model["tracker"]),
            confidence=float(model["confidence"]),
            iou=float(model["iou"]),
            image_size=int(model["image_size"]),
            device=str(model["device"]),
            classes=[int(value) for value in model["classes"]] if model.get("classes") else None,
        ),
        output=OutputConfig(
            show_window=bool(output["show_window"]),
            save_video=bool(output["save_video"]),
            output_dir=str(output["output_dir"]),
            window_name=str(output["window_name"]),
            video_name_prefix=str(output["video_name_prefix"]),
            display_scale=float(output.get("display_scale", 1.0)),
        ),
        features=FeatureConfig(
            draw_trails=bool(features["draw_trails"]),
            trail_length=max(1, int(features["trail_length"])),
            enable_line_counting=bool(features["enable_line_counting"]),
            count_line=_normalize_count_line(features.get("count_line")),
        ),
        config_path=config_path,
    )


def load_config(config_path: str | None, overrides: dict[str, Any] | None = None) -> AppConfig:
    merged = DEFAULT_CONFIG
    path_obj: Path | None = None

    if config_path:
        path_obj = Path(config_path).expanduser().resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path_obj}")
        with path_obj.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        merged = _deep_merge(merged, loaded)

    if overrides:
        merged = _deep_merge(merged, overrides)

    return _build_config(merged, config_path=path_obj)
