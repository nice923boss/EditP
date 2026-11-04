"""Measure how existing text looks inside a line box: color, ink extent, stroke width, glow."""
import cv2
import numpy as np

from .geometry import Box, pad_box

GLOW_MAX = 14  # px searched outside the strokes


def to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb)


def text_mask(crop: np.ndarray) -> np.ndarray:
    """2-means in LAB; the cluster owning most of the crop border is background."""
    cv2.setRNGSeed(0)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, _ = cv2.kmeans(lab, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(crop.shape[:2])
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    bg = np.bincount(border, minlength=2).argmax()
    mask = (labels != bg).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 3  # drop specks
    return keep[cc].astype(np.uint8)


def _glow(rgb: np.ndarray, mask: np.ndarray, origin: tuple[int, int], color: np.ndarray) -> tuple[int, float]:
    """Outer glow as (radius px, opacity 3 px outside the strokes); (0, 0) when there is none."""
    h, w = rgb.shape[:2]
    ox, oy = origin
    mh, mw = mask.shape
    big = pad_box([ox, oy, ox + mw, oy + mh], GLOW_MAX + 6, w, h)
    region = rgb[big[1]:big[3], big[0]:big[2]].astype(np.float32)
    ink = np.zeros(region.shape[:2], np.uint8)
    ink[oy - big[1]:oy - big[1] + mh, ox - big[0]:ox - big[0] + mw] = mask
    dist = cv2.distanceTransform(1 - ink, cv2.DIST_L2, 3)
    far = dist > GLOW_MAX
    if far.sum() < 20:
        return 0, 0.0
    bg = np.median(region[far], axis=0)
    axis = color - bg
    denom = float(axis @ axis)
    if denom < 100:  # text colour too close to the background to tell
        return 0, 0.0
    proj = ((region - bg) @ axis) / denom  # 0 = background, 1 = text colour
    prof = [float(np.median(proj[(dist > r - 1) & (dist <= r)])) for r in range(1, GLOW_MAX + 1)]
    strength = prof[2]
    if strength < 0.08:
        return 0, 0.0
    radius = next((r + 1 for r, p in enumerate(prof) if r >= 2 and p < strength * 0.2), None)
    if radius is None:  # never fades out: background shading or a neighbouring line, not a glow
        return 0, 0.0
    return radius, round(min(strength, 1.0), 3)


def measure(rgb: np.ndarray, box: Box) -> dict | None:
    """Text colour, ink box, stroke width, glow and glyph mask inside box; None if no ink is found."""
    h, w = rgb.shape[:2]
    bx = pad_box(box, 4, w, h)
    crop = rgb[bx[1]:bx[3], bx[0]:bx[2]]
    if min(crop.shape[:2]) < 4:
        return None
    mask = text_mask(crop)
    if mask.sum() < 10:
        return None
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    inner = dt[dt > 0]
    core = (mask > 0) & (dt >= np.percentile(inner, 50))  # skip anti-aliased edges
    color = np.median(crop[core].astype(np.float32), axis=0)
    ys, xs = np.nonzero(mask)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    radius, strength = _glow(rgb, mask, (bx[0], bx[1]), color)
    return {
        "color": to_hex(color),
        "ink": [bx[0] + x1, bx[1] + y1, bx[0] + x2, bx[1] + y2],
        "stroke": round(float(2 * np.percentile(inner, 90)), 2),
        "glow": {"radius": radius, "strength": strength},
        "mask": mask[y1:y2, x1:x2],
    }
