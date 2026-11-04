"""Pure box helpers. A box is [x1, y1, x2, y2] in image pixels (x2/y2 exclusive)."""
from collections.abc import Sequence

Box = list[int]


def quad_to_box(quad: Sequence[Sequence[float]]) -> Box:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [int(min(xs)), int(min(ys)), int(round(max(xs))), int(round(max(ys)))]


def clamp_box(box: Sequence[float], w: int, h: int) -> Box:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return [min(max(x1, 0), w), min(max(y1, 0), h), min(max(x2, 0), w), min(max(y2, 0), h)]


def pad_box(box: Sequence[float], pad: float, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box
    return clamp_box([x1 - pad, y1 - pad, x2 + pad, y2 + pad], w, h)


def union_box(a: Sequence[int], b: Sequence[int]) -> Box:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def to_norm(box: Sequence[float], w: int, h: int) -> Box:
    """Pixel box -> Qwen-VL grounding grid (0-1000)."""
    x1, y1, x2, y2 = box
    return [round(x1 * 1000 / w), round(y1 * 1000 / h), round(x2 * 1000 / w), round(y2 * 1000 / h)]


def from_norm(box: Sequence[float], w: int, h: int) -> Box:
    """Qwen-VL grounding grid (0-1000) -> pixel box."""
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return clamp_box([x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000], w, h)


def line_at_point(lines: list[dict], x: float, y: float, max_dist: float = 30) -> dict | None:
    """Line whose box contains (x, y); otherwise the nearest within max_dist; otherwise None."""
    best, best_d = None, max_dist
    for line in lines:
        x1, y1, x2, y2 = line["box"]
        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)
        d = (dx * dx + dy * dy) ** 0.5
        if d == 0:
            return line
        if d <= best_d:
            best, best_d = line, d
    return best


_ROWS = ("上", "中", "下")
_COLS = ("左", "中", "右")
_REGION = {
    ("上", "左"): "左上角", ("上", "中"): "上方中央", ("上", "右"): "右上角",
    ("中", "左"): "左側中段", ("中", "中"): "正中央", ("中", "右"): "右側中段",
    ("下", "左"): "左下角", ("下", "中"): "下方中央", ("下", "右"): "右下角",
}


def region_name(box: Sequence[float], w: int, h: int) -> str:
    """Coarse 3x3 position label, used to help the LLM resolve '右上角'-style references."""
    cx = (box[0] + box[2]) / 2 / w
    cy = (box[1] + box[3]) / 2 / h
    row = _ROWS[min(int(cy * 3), 2)]
    col = _COLS[min(int(cx * 3), 2)]
    return _REGION[(row, col)]
