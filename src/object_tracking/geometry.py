from __future__ import annotations

from typing import Iterable


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
