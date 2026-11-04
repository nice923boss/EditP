from editp.geometry import clamp_box, from_norm, line_at_point, pad_box, region_name


def test_clamp_box_keeps_box_inside_image():
    assert clamp_box([-5, -5, 300, 300], 100, 50) == [0, 0, 100, 50]


def test_pad_box_grows_then_clamps():
    assert pad_box([10, 10, 20, 20], 5, 100, 100) == [5, 5, 25, 25]
    assert pad_box([2, 2, 98, 98], 5, 100, 100) == [0, 0, 100, 100]


def test_from_norm_sorts_swapped_corners():
    assert from_norm([500, 600, 100, 200], 1000, 500) == [100, 100, 500, 300]


def test_line_at_point_prefers_containing_then_nearest():
    lines = [{"id": 0, "box": [0, 0, 100, 20]}, {"id": 1, "box": [0, 60, 100, 80]}]
    assert line_at_point(lines, 50, 10)["id"] == 0
    assert line_at_point(lines, 50, 50)["id"] == 1  # 10 px from line 1, 30 px from line 0
    assert line_at_point(lines, 50, 200) is None


def test_region_name_corners_and_center():
    assert region_name([0, 0, 10, 10], 300, 300) == "左上角"
    assert region_name([140, 140, 160, 160], 300, 300) == "正中央"
    assert region_name([290, 290, 300, 300], 300, 300) == "右下角"
