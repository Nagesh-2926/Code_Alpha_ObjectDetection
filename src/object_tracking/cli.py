from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import run_tracking


def _parse_classes(value: str | None) -> list[int] | None:
    if value in (None, ""):
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_labels(value: str | None) -> list[str] | None:
    if value in (None, ""):
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _ensure_section(container: dict[str, object], *keys: str) -> dict[str, object]:
    current = container
    for key in keys:
        nested = current.get(key)
        if not isinstance(nested, dict):
            nested = {}
            current[key] = nested
        current = nested
    return current


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run advanced real-time object detection, tracking, analytics, privacy masking, and pose overlays."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to a YAML config file.")
    parser.add_argument("--source", type=str, help="Video source. Use 0 for webcam or a path for video files.")
    parser.add_argument("--model", type=str, help="YOLO model weights, for example yolov8n.pt.")
    parser.add_argument("--tracker", type=str, choices=["botsort.yaml", "bytetrack.yaml"], help="Tracker config.")
    parser.add_argument("--conf", type=float, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, help="IoU threshold.")
    parser.add_argument("--imgsz", type=int, help="Inference image size.")
    parser.add_argument("--device", type=str, help="cpu, 0, 0,1, and so on.")
    parser.add_argument("--classes", type=str, help="Comma-separated class IDs, for example 0,2,3.")
    parser.add_argument("--no-show", action="store_true", help="Disable the OpenCV display window.")
    parser.add_argument("--no-save", action="store_true", help="Disable saving the processed video.")
    parser.add_argument("--display-scale", type=float, help="Resize the shown frame, for example 0.8.")
    parser.add_argument("--line-counting", action="store_true", help="Enable line crossing counts.")
    parser.add_argument("--zone-counting", action="store_true", help="Enable polygon-based zone counting.")
    parser.add_argument("--intrusion", action="store_true", help="Enable intrusion detection for restricted zones.")
    parser.add_argument("--speed", action="store_true", help="Enable speed estimation.")
    parser.add_argument("--pixels-per-meter", type=float, help="Calibration value for speed estimation.")
    parser.add_argument("--pose", action="store_true", help="Enable body-pose and keypoint estimation.")
    parser.add_argument("--pose-labels", action="store_true", help="Draw keypoint labels on pose landmarks.")
    parser.add_argument("--privacy-mode", type=str, choices=["face", "person"], help="Enable privacy masking mode.")
    parser.add_argument("--privacy-classes", type=str, help="Comma-separated class labels for privacy masking.")
    parser.add_argument(
        "--line",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Custom line for counting. Example: --line 200 300 1000 300",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    overrides: dict[str, object] = {}

    if args.source is not None:
        overrides["source"] = args.source

    if any(
        [
            args.model,
            args.tracker,
            args.conf is not None,
            args.iou is not None,
            args.imgsz is not None,
            args.device,
            args.classes,
        ]
    ):
        model_section = _ensure_section(overrides, "model")
        if args.model:
            model_section["weights"] = args.model
        if args.tracker:
            model_section["tracker"] = args.tracker
        if args.conf is not None:
            model_section["confidence"] = args.conf
        if args.iou is not None:
            model_section["iou"] = args.iou
        if args.imgsz is not None:
            model_section["image_size"] = args.imgsz
        if args.device:
            model_section["device"] = args.device
        if args.classes:
            model_section["classes"] = _parse_classes(args.classes)

    if args.no_show or args.no_save or args.display_scale is not None:
        output_section = _ensure_section(overrides, "output")
        if args.no_show:
            output_section["show_window"] = False
        if args.no_save:
            output_section["save_video"] = False
        if args.display_scale is not None:
            output_section["display_scale"] = args.display_scale

    analytics_section = _ensure_section(overrides, "features", "analytics")
    if args.line_counting:
        analytics_section["enable_line_counting"] = True
    if args.zone_counting:
        analytics_section["enable_zone_counting"] = True
    if args.intrusion:
        _ensure_section(overrides, "features", "analytics", "intrusion")["enabled"] = True
    if args.speed:
        _ensure_section(overrides, "features", "analytics", "speed")["enabled"] = True
    if args.pixels_per_meter is not None:
        speed_section = _ensure_section(overrides, "features", "analytics", "speed")
        speed_section["enabled"] = True
        speed_section["pixels_per_meter"] = args.pixels_per_meter
    if args.line:
        analytics_section["enable_line_counting"] = True
        x1, y1, x2, y2 = args.line
        analytics_section["count_line"] = [[x1, y1], [x2, y2]]

    if args.pose or args.pose_labels:
        pose_section = _ensure_section(overrides, "features", "pose")
        if args.pose:
            pose_section["enabled"] = True
        if args.pose_labels:
            pose_section["enabled"] = True
            pose_section["draw_labels"] = True

    if args.privacy_mode or args.privacy_classes:
        privacy_section = _ensure_section(overrides, "features", "privacy")
        privacy_section["enabled"] = True
        if args.privacy_mode:
            privacy_section["mode"] = args.privacy_mode
        if args.privacy_classes:
            privacy_section["classes"] = _parse_labels(args.privacy_classes)

    config_path = None if args.config == "" else args.config
    if config_path and not Path(config_path).exists():
        parser.error(f"Config file does not exist: {config_path}")

    config = load_config(config_path, overrides=overrides)
    run_tracking(config)
    return 0
