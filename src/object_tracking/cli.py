from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import run_tracking


def _parse_classes(value: str | None) -> list[int] | None:
    if value in (None, ""):
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real-time object detection and tracking with YOLO, OpenCV, and BoT-SORT or ByteTrack."
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
    if args.model or args.tracker or args.conf is not None or args.iou is not None or args.imgsz is not None or args.device or args.classes:
        overrides["model"] = {}
    if args.model:
        overrides["model"]["weights"] = args.model
    if args.tracker:
        overrides["model"]["tracker"] = args.tracker
    if args.conf is not None:
        overrides["model"]["confidence"] = args.conf
    if args.iou is not None:
        overrides["model"]["iou"] = args.iou
    if args.imgsz is not None:
        overrides["model"]["image_size"] = args.imgsz
    if args.device:
        overrides["model"]["device"] = args.device
    if args.classes:
        overrides["model"]["classes"] = _parse_classes(args.classes)

    if args.no_show or args.no_save or args.display_scale is not None:
        overrides["output"] = {}
    if args.no_show:
        overrides["output"]["show_window"] = False
    if args.no_save:
        overrides["output"]["save_video"] = False
    if args.display_scale is not None:
        overrides["output"]["display_scale"] = args.display_scale

    if args.line_counting or args.line:
        overrides["features"] = {}
    if args.line_counting:
        overrides["features"]["enable_line_counting"] = True
    if args.line:
        overrides["features"]["enable_line_counting"] = True
        x1, y1, x2, y2 = args.line
        overrides["features"]["count_line"] = [[x1, y1], [x2, y2]]

    config_path = None if args.config == "" else args.config
    if config_path and not Path(config_path).exists():
        parser.error(f"Config file does not exist: {config_path}")

    config = load_config(config_path, overrides=overrides)
    run_tracking(config)
    return 0
