"""Erase text regions with LaMa (big-lama): on CUDA when there is a usable GPU, else on the CPU.
Without torch or simple-lama-inpainting it falls back to OpenCV's Telea fill, which is visibly worse."""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from .geometry import Box, pad_box

_engine: dict = {}  # name (lama | opencv), device (cuda | cpu), note shown to the user; set on first load


def _use(name: str, device: str, note: str = "") -> None:
    _engine.update(name=name, device=device, note=note)
    if note:
        print(f"[inpaint] {note}", flush=True)


@lru_cache(maxsize=1)
def _lama():
    """The LaMa fill function, or None when it cannot load (then erase() uses OpenCV)."""
    try:
        import torch
        from simple_lama_inpainting import SimpleLama
    except ImportError as e:
        _use("opencv", "cpu", f"沒有安裝 LaMa（缺少 {e.name}），改用 OpenCV 擦除，邊緣較糊；重跑 install.bat 可補裝")
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        lama = SimpleLama(device=torch.device(device))
    except Exception as e:  # model download or driver failure: keep the app usable, say why
        _use("opencv", "cpu", f"LaMa 載入失敗（{e}），改用 OpenCV 擦除，邊緣較糊")
        return None
    _use("lama", device, "" if device == "cuda" else "沒有可用的 NVIDIA 顯示卡，LaMa 改用 CPU，每次套用會慢幾秒到幾十秒")
    return lama


def warm_up() -> None:
    _lama()


def status() -> dict:
    """{name, device, note} of the eraser in use; {} until warm_up() or the first erase has loaded it."""
    return dict(_engine)


def _fill(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lama = _lama()
    if lama is None:
        return cv2.inpaint(crop, mask, 5, cv2.INPAINT_TELEA)
    filled = np.array(lama(Image.fromarray(crop), Image.fromarray(mask)))
    return filled[: crop.shape[0], : crop.shape[1]]  # LaMa pads to a multiple of 8


def erase(rgb: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Fill every box, one local crop per box; pixels outside the boxes stay untouched.

    Every box reaching into a crop is masked there too: text still waiting to be erased must not
    be the context the fill copies from, or its strokes come back as ghosts in the fill.
    """
    out = rgb.copy()
    h, w = rgb.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        ctx = max(64, 2 * (y2 - y1))
        cx1, cy1, cx2, cy2 = pad_box(box, ctx, w, h)
        crop = np.ascontiguousarray(out[cy1:cy2, cx1:cx2])
        mask = np.zeros(crop.shape[:2], np.uint8)
        for bx1, by1, bx2, by2 in boxes:
            mask[max(by1 - cy1, 0):max(by2 - cy1, 0), max(bx1 - cx1, 0):max(bx2 - cx1, 0)] = 255
        filled = _fill(crop, mask)
        out[cy1:cy2, cx1:cx2] = np.where(mask[..., None] > 0, filled, crop)
    return out
