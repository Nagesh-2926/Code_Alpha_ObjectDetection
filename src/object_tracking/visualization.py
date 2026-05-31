from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .config import FeatureConfig

if TYPE_CHECKING:
    from .pipeline import Detection


@dataclass(slots=True)
class OverlayStats:
    fps: float
    frame_index: int
    total_detections: int
    tracked_objects: int
    in_count: int = 0
    out_count: int = 0


def color_for_track(track_id: int | None, class_id: int) -> tuple[int, int, int]:
    seed = track_id if track_id is not None else class_id * 97
    return (
        (37 * seed + 53) % 255,
        (17 * seed + 109) % 255,
        (29 * seed + 191) % 255,
    )


def draw_frame(
    frame: np.ndarray,
    detections: list[Detection],
    track_history: dict[int, deque[tuple[int, int]]],
    features: FeatureConfig,
    stats: OverlayStats,
) -> np.ndarray:
    annotated = frame.copy()

    if features.enable_line_counting and features.count_line:
        cv2.line(annotated, features.count_line[0], features.count_line[1], (0, 255, 255), 2)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = color_for_track(detection.track_id, detection.class_id)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        if detection.track_id is not None:
            label = f"ID {detection.track_id} | {label}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, detection.center, 4, color, -1)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if features.draw_trails:
        for history in track_history.values():
            if len(history) < 2:
                continue
            points = list(history)
            for start, end in zip(points[:-1], points[1:]):
                cv2.line(annotated, start, end, (255, 255, 0), 2)

    panel_color = (20, 20, 20)
    cv2.rectangle(annotated, (15, 15), (360, 165), panel_color, -1)
    cv2.rectangle(annotated, (15, 15), (360, 165), (0, 200, 255), 2)

    overlay_lines = [
        f"Frame: {stats.frame_index}",
        f"FPS: {stats.fps:.2f}",
        f"Detections: {stats.total_detections}",
        f"Tracked IDs: {stats.tracked_objects}",
        f"Line IN: {stats.in_count}",
        f"Line OUT: {stats.out_count}",
        "Press Q to quit",
    ]

    for index, line in enumerate(overlay_lines):
        cv2.putText(
            annotated,
            line,
            (30, 40 + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated
