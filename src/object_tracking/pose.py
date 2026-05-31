from __future__ import annotations

from dataclasses import dataclass


POSE_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


POSE_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass(slots=True)
class PosePerson:
    bbox: tuple[int, int, int, int]
    keypoints: list[tuple[int, int]]
    confidences: list[float]


def extract_pose_people(result) -> list[PosePerson]:
    if result is None or result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return []

    boxes = result.boxes.xyxy.int().cpu().tolist()
    keypoints_xy = result.keypoints.xy.int().cpu().tolist()
    if result.keypoints.conf is not None:
        confidences = result.keypoints.conf.float().cpu().tolist()
    else:
        confidences = [[1.0] * len(keypoint_set) for keypoint_set in keypoints_xy]

    pose_people: list[PosePerson] = []
    for bbox, keypoint_set, confidence_set in zip(boxes, keypoints_xy, confidences):
        pose_people.append(
            PosePerson(
                bbox=tuple(int(value) for value in bbox),
                keypoints=[(int(point[0]), int(point[1])) for point in keypoint_set],
                confidences=[float(value) for value in confidence_set],
            )
        )
    return pose_people
