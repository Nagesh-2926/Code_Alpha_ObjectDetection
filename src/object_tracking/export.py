from __future__ import annotations

import argparse

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a YOLO model to ONNX for deployment.")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO weights file.")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes.")
    parser.add_argument("--simplify", action="store_true", help="Simplify the ONNX graph when possible.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = YOLO(args.weights)
    model.export(format="onnx", imgsz=args.imgsz, dynamic=args.dynamic, simplify=args.simplify)
    return 0
