from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .config import FeatureConfig
from .geometry import polygon_label_anchor
from .pose import POSE_CONNECTIONS, POSE_KEYPOINT_NAMES

if TYPE_CHECKING:
    from .analytics import FrameAnalyticsSnapshot
    from .pipeline import Detection
    from .pose import PosePerson


@dataclass(slots=True)
class OverlayStats:
    fps: float
    frame_index: int
    total_detections: int
    tracked_objects: int
    unique_classes: int
    in_count: int = 0
    out_count: int = 0


def color_for_track(track_id: int | None, class_id: int) -> tuple[int, int, int]:
    seed = track_id if track_id is not None else class_id * 97
    return (
        (37 * seed + 53) % 255,
        (17 * seed + 109) % 255,
        (29 * seed + 191) % 255,
    )


def _draw_pose_annotations(frame: np.ndarray, pose_people: list[PosePerson], draw_labels: bool) -> None:
    for pose_person in pose_people:
        for start_index, end_index in POSE_CONNECTIONS:
            if pose_person.confidences[start_index] < 0.25 or pose_person.confidences[end_index] < 0.25:
                continue
            cv2.line(frame, pose_person.keypoints[start_index], pose_person.keypoints[end_index], (0, 255, 255), 2)

        for index, (point, confidence) in enumerate(zip(pose_person.keypoints, pose_person.confidences)):
            if confidence < 0.25:
                continue
            cv2.circle(frame, point, 3, (255, 255, 0), -1)
            if draw_labels:
                cv2.putText(
                    frame,
                    POSE_KEYPOINT_NAMES[index],
                    (point[0] + 4, point[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )


def draw_frame(
    frame: np.ndarray,
    detections: list[Detection],
    track_history: dict[int, deque[tuple[int, int]]],
    features: FeatureConfig,
    stats: OverlayStats,
    analytics: FrameAnalyticsSnapshot,
    pose_people: list[PosePerson] | None = None,
) -> np.ndarray:
    annotated = frame.copy()

    if features.analytics.enable_zone_counting:
        for zone in features.analytics.zones:
            polygon = np.array(zone.polygon, dtype=np.int32)
            cv2.polylines(annotated, [polygon], isClosed=True, color=zone.color, thickness=2)
            label_x, label_y = polygon_label_anchor(zone.polygon)
            zone_snapshot = next((item for item in analytics.zone_stats if item.name == zone.name), None)
            zone_text = zone.name
            if zone_snapshot:
                zone_text = f"{zone.name}: {zone_snapshot.current_count}"
            cv2.putText(
                annotated,
                zone_text,
                (label_x - 60, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                zone.color,
                2,
                cv2.LINE_AA,
            )

    if features.analytics.enable_line_counting and features.analytics.count_line:
        cv2.line(annotated, features.analytics.count_line[0], features.analytics.count_line[1], (0, 255, 255), 2)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = (0, 0, 255) if detection.is_intrusion else color_for_track(detection.track_id, detection.class_id)
        label_parts = [detection.class_name, f"{detection.confidence:.2f}"]
        if detection.track_id is not None:
            label_parts.insert(0, f"ID {detection.track_id}")
        if detection.speed_kph is not None:
            label_parts.append(f"{detection.speed_kph:.1f} km/h")
        if detection.zone_names:
            label_parts.append("/".join(detection.zone_names))
        label = " | ".join(label_parts)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, detection.center, 4, color, -1)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
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

    if pose_people:
        _draw_pose_annotations(annotated, pose_people, draw_labels=features.pose.draw_labels)

    panel_lines = [
        f"Frame: {stats.frame_index}",
        f"FPS: {stats.fps:.2f}",
        f"Detections: {stats.total_detections}",
        f"Tracked IDs: {stats.tracked_objects}",
        f"Visible classes: {stats.unique_classes}",
        f"Line IN: {stats.in_count}",
        f"Line OUT: {stats.out_count}",
        f"Intrusions: {analytics.total_intrusions}",
        f"Avg speed: {analytics.average_speed_kph:.1f} km/h",
        f"Max speed: {analytics.max_speed_kph:.1f} km/h",
        "Press Q to quit",
    ]

    zone_lines = [
        f"{zone.name}: {zone.current_count} now / {zone.cumulative_entries} entries"
        for zone in analytics.zone_stats[:4]
    ]
    alert_lines = analytics.active_alerts[:2]

    all_lines = panel_lines + zone_lines + alert_lines
    panel_height = max(120, 20 + len(all_lines) * 14)
    panel_width = 280
    cv2.rectangle(annotated, (15, 15), (15 + panel_width, 15 + panel_height), (20, 20, 20), -1)
    cv2.rectangle(annotated, (15, 15), (15 + panel_width, 15 + panel_height), (0, 200, 255), 2)

    for index, line in enumerate(all_lines):
        text_color = (0, 120, 255) if line.startswith("Intrusion:") else (255, 255, 255)
        cv2.putText(
            annotated,
            line,
            (25, 35 + index * 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return annotated
