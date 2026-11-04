"""Text line detection with RapidOCR (PP-OCRv4 mobile, onnxruntime)."""
import logging
from functools import lru_cache

import cv2
import numpy as np

from .geometry import Box, pad_box, quad_to_box

logging.getLogger("RapidOCR").setLevel(logging.WARNING)


@lru_cache(maxsize=1)
def _engine():
    from rapidocr import RapidOCR

    return RapidOCR()


def _run(rgb: np.ndarray) -> list[tuple[Box, str, float]]:
    res = _engine()(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))  # ndarray input is read as BGR
    if res.boxes is None:
        return []
    return [(quad_to_box(b), t, float(s)) for b, t, s in zip(res.boxes, res.txts, res.scores)]


def detect(rgb: np.ndarray) -> list[dict]:
    """Text lines in reading order: [{id, text, score, box}]."""
    return [
        {"id": i, "text": text, "score": round(score, 3), "box": box}
        for i, (box, text, score) in enumerate(_run(rgb))
    ]


def read_region(rgb: np.ndarray, box: Box) -> str:
    """OCR text inside a user-drawn box (small regions are upscaled first)."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = pad_box(box, 4, w, h)
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    scale = max(1.0, 64 / max(1, crop.shape[0]))
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    crop = np.ascontiguousarray(crop)
    return "".join(text for _, text, _ in _run(crop))
