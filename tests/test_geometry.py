from object_tracking.geometry import bbox_center, crossing_direction, point_in_polygon, point_line_side


def test_bbox_center_returns_middle_point():
    assert bbox_center((10, 20, 30, 40)) == (20, 30)


def test_point_line_side_detects_positive_and_negative_side():
    line_start = (0, 0)
    line_end = (10, 0)
    assert point_line_side((5, -2), line_start, line_end) == 1
    assert point_line_side((5, 2), line_start, line_end) == -1


def test_crossing_direction_returns_in_for_negative_to_positive():
    assert crossing_direction(-1, 1) == "in"


def test_crossing_direction_returns_out_for_positive_to_negative():
    assert crossing_direction(1, -1) == "out"


def test_crossing_direction_returns_none_for_same_side():
    assert crossing_direction(1, 1) is None


def test_point_in_polygon_accepts_inner_points():
    polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), polygon) is True
    assert point_in_polygon((15, 5), polygon) is False
