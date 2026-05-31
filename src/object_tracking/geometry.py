from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np


Point = tuple[int, int]


def bbox_center(bbox: Iterable[int]) -> Point:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return (x1 + x2) // 2, (y1 + y2) // 2


def point_line_side(point: Point, line_start: Point, line_end: Point) -> int:
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if cross_product == 0:
        return 0
    return 1 if cross_product > 0 else -1


def crossing_direction(previous_side: int, current_side: int) -> str | None:
    if previous_side == current_side or 0 in (previous_side, current_side):
        return None
    return "in" if previous_side < current_side else "out"


def point_in_polygon(point: Point, polygon: Iterable[Point]) -> bool:
    polygon_array = np.array(list(polygon), dtype=np.int32)
    return cv2.pointPolygonTest(polygon_array, point, False) >= 0


def polygon_label_anchor(polygon: Iterable[Point]) -> Point:
    points = list(polygon)
    x = int(sum(point[0] for point in points) / len(points))
    y = int(sum(point[1] for point in points) / len(points))
    return x, y


def euclidean_distance(point_a: Point, point_b: Point) -> float:
    return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
