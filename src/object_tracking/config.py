from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


Point = tuple[int, int]


DEFAULT_CONFIG: dict[str, Any] = {
    "project_name": "codealpha_object_detection_pro",
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
        "save_analytics_json": True,
        "output_dir": "artifacts/runs",
        "window_name": "CodeAlpha Object Detection Pro",
        "video_name_prefix": "tracking_run",
        "analytics_name_prefix": "analytics_run",
        "display_scale": 1.0,
    },
    "features": {
        "draw_trails": True,
        "trail_length": 30,
        "analytics": {
            "enable_line_counting": False,
            "count_line": [[200, 300], [1000, 300]],
            "enable_zone_counting": False,
            "zones": [
                {
                    "name": "Main Zone",
                    "polygon": [[100, 100], [1180, 100], [1180, 620], [100, 620]],
                    "color": [0, 255, 0],
                    "count_classes": ["person", "car", "truck", "bus", "motorbike", "bicycle"],
                    "restricted": False,
                },
                {
                    "name": "Restricted Area",
                    "polygon": [[850, 160], [1180, 160], [1180, 620], [850, 620]],
                    "color": [0, 80, 255],
                    "count_classes": ["person"],
                    "restricted": True,
                },
            ],
            "intrusion": {
                "enabled": False,
                "classes": ["person"],
                "cooldown_frames": 45,
            },
            "speed": {
                "enabled": False,
                "pixels_per_meter": 12.0,
                "smoothing_window": 8,
                "classes": ["person", "car", "truck", "bus", "motorbike", "bicycle"],
            },
        },
        "privacy": {
            "enabled": False,
            "mode": "face",
            "classes": ["person"],
            "blur_kernel_size": 31,
        },
        "pose": {
            "enabled": False,
            "weights": "yolov8n-pose.pt",
            "confidence": 0.35,
            "draw_labels": False,
            "classes": [0],
        },
    },
}


@dataclass(slots=True)
class ZoneConfig:
    name: str
    polygon: tuple[Point, ...]
    color: tuple[int, int, int]
    count_classes: list[str] | None = None
    restricted: bool = False


@dataclass(slots=True)
class IntrusionConfig:
    enabled: bool
    classes: list[str]
    cooldown_frames: int


@dataclass(slots=True)
class SpeedConfig:
    enabled: bool
    pixels_per_meter: float
    smoothing_window: int
    classes: list[str] | None = None


@dataclass(slots=True)
class AnalyticsConfig:
    enable_line_counting: bool
    count_line: tuple[Point, Point] | None
    enable_zone_counting: bool
    zones: list[ZoneConfig] = field(default_factory=list)
    intrusion: IntrusionConfig = field(
        default_factory=lambda: IntrusionConfig(enabled=False, classes=["person"], cooldown_frames=45)
    )
    speed: SpeedConfig = field(
        default_factory=lambda: SpeedConfig(enabled=False, pixels_per_meter=12.0, smoothing_window=8, classes=None)
    )


@dataclass(slots=True)
class PrivacyConfig:
    enabled: bool
    mode: str
    classes: list[str] | None
    blur_kernel_size: int


@dataclass(slots=True)
class PoseConfig:
    enabled: bool
    weights: str
    confidence: float
    draw_labels: bool
    classes: list[int] | None = None


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
    save_analytics_json: bool
    output_dir: str
    window_name: str
    video_name_prefix: str
    analytics_name_prefix: str
    display_scale: float = 1.0


@dataclass(slots=True)
class FeatureConfig:
    draw_trails: bool
    trail_length: int
    analytics: AnalyticsConfig
    privacy: PrivacyConfig
    pose: PoseConfig


@dataclass(slots=True)
class AppConfig:
    project_name: str
    source: str | int
    model: ModelConfig
    output: OutputConfig
    features: FeatureConfig
    config_path: Path | None = None


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
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


def _normalize_point(raw_point: Any) -> Point:
    if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
        raise ValueError("A point must contain exactly two numeric values.")
    return int(raw_point[0]), int(raw_point[1])


def _normalize_count_line(raw_line: Any) -> tuple[Point, Point] | None:
    if raw_line in (None, "", []):
        return None

    if not isinstance(raw_line, list) or len(raw_line) != 2:
        raise ValueError("count_line must contain exactly two points, for example [[200, 300], [1000, 300]].")

    return _normalize_point(raw_line[0]), _normalize_point(raw_line[1])


def _normalize_zone(raw_zone: dict[str, Any]) -> ZoneConfig:
    polygon = raw_zone.get("polygon", [])
    if len(polygon) < 3:
        raise ValueError("Each zone polygon must contain at least three points.")

    color_values = raw_zone.get("color", [0, 255, 0])
    if not isinstance(color_values, list) or len(color_values) != 3:
        raise ValueError("Zone color must contain exactly three values, for example [0, 255, 0].")

    return ZoneConfig(
        name=str(raw_zone["name"]),
        polygon=tuple(_normalize_point(point) for point in polygon),
        color=(int(color_values[0]), int(color_values[1]), int(color_values[2])),
        count_classes=[str(item) for item in raw_zone["count_classes"]] if raw_zone.get("count_classes") else None,
        restricted=bool(raw_zone.get("restricted", False)),
    )


def _build_config(data: dict[str, Any], config_path: Path | None = None) -> AppConfig:
    model = data["model"]
    output = data["output"]
    features = data["features"]
    analytics = features["analytics"]
    intrusion = analytics["intrusion"]
    speed = analytics["speed"]
    privacy = features["privacy"]
    pose = features["pose"]

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
            save_analytics_json=bool(output.get("save_analytics_json", True)),
            output_dir=str(output["output_dir"]),
            window_name=str(output["window_name"]),
            video_name_prefix=str(output["video_name_prefix"]),
            analytics_name_prefix=str(output.get("analytics_name_prefix", "analytics_run")),
            display_scale=float(output.get("display_scale", 1.0)),
        ),
        features=FeatureConfig(
            draw_trails=bool(features["draw_trails"]),
            trail_length=max(1, int(features["trail_length"])),
            analytics=AnalyticsConfig(
                enable_line_counting=bool(analytics["enable_line_counting"]),
                count_line=_normalize_count_line(analytics.get("count_line")),
                enable_zone_counting=bool(analytics.get("enable_zone_counting", False)),
                zones=[_normalize_zone(zone) for zone in analytics.get("zones", [])],
                intrusion=IntrusionConfig(
                    enabled=bool(intrusion.get("enabled", False)),
                    classes=[str(item) for item in intrusion.get("classes", ["person"])],
                    cooldown_frames=max(1, int(intrusion.get("cooldown_frames", 45))),
                ),
                speed=SpeedConfig(
                    enabled=bool(speed.get("enabled", False)),
                    pixels_per_meter=max(0.0001, float(speed.get("pixels_per_meter", 12.0))),
                    smoothing_window=max(1, int(speed.get("smoothing_window", 8))),
                    classes=[str(item) for item in speed["classes"]] if speed.get("classes") else None,
                ),
            ),
            privacy=PrivacyConfig(
                enabled=bool(privacy.get("enabled", False)),
                mode=str(privacy.get("mode", "face")).lower(),
                classes=[str(item) for item in privacy["classes"]] if privacy.get("classes") else None,
                blur_kernel_size=max(3, int(privacy.get("blur_kernel_size", 31))),
            ),
            pose=PoseConfig(
                enabled=bool(pose.get("enabled", False)),
                weights=str(pose.get("weights", "yolov8n-pose.pt")),
                confidence=float(pose.get("confidence", 0.35)),
                draw_labels=bool(pose.get("draw_labels", False)),
                classes=[int(value) for value in pose["classes"]] if pose.get("classes") else None,
            ),
        ),
        config_path=config_path,
    )


def load_config(config_path: str | None, overrides: dict[str, Any] | None = None) -> AppConfig:
    merged = copy.deepcopy(DEFAULT_CONFIG)
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
