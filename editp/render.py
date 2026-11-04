"""Fit a font to measured text and draw replacement text with the same look."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import ROOT

FONT_DIRS = (
    ROOT / "fonts",  # downloaded by install.py when the system has none
    Path(r"C:\Windows\Fonts"),
    Path.home() / r"AppData\Local\Microsoft\Windows\Fonts",  # fonts installed for one user
)
FONT_FILES = {"sans": ("NotoSansTC-VF.ttf", True), "serif": ("NotoSerifTC-VF.ttf", True), "kai": ("kaiu.ttf", False)}


def find_font(name: str) -> Path | None:
    return next((d / name for d in FONT_DIRS if (d / name).is_file()), None)


FONTS = {family: (str(p), variable) for family, (name, variable) in FONT_FILES.items() if (p := find_font(name))}
WEIGHTS = (300, 400, 500, 600, 700, 800, 900)
REF_SIZE = 100


def from_hex(color: str) -> np.ndarray:
    c = color.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32)


@lru_cache(maxsize=512)
def get_font(family: str, size: float, weight: int) -> ImageFont.FreeTypeFont:
    path, variable = FONTS.get(family) or FONTS["sans"]  # kaiu.ttf ships only with Traditional Chinese Windows
    font = ImageFont.truetype(path, max(1.0, size))
    if variable:
        font.set_variation_by_axes([weight])
    return font


def glyphs(text: str, font: ImageFont.FreeTypeFont, tracking: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Alpha mask (0-255) of text, and where the text origin sits inside it."""
    size = font.size
    advance = sum(font.getlength(c) for c in text) + tracking * max(0, len(text) - 1)
    ox, oy = int(size) + 2, int(size * 0.5) + 2
    img = Image.new("L", (int(ox + max(advance, size) + size) + 4, int(size * 2.5) + 4), 0)
    d = ImageDraw.Draw(img)
    if abs(tracking) < 0.25:
        d.text((ox, oy), text, font=font, fill=255)
    else:
        x = float(ox)
        for c in text:
            d.text((x, oy), c, font=font, fill=255)
            x += font.getlength(c) + tracking
    return np.array(img), (ox, oy)


def ink_box(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(alpha > 127)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


SHORTLIST = 5  # family/weight candidates searched over size
SIZE_STEPS = np.arange(0.88, 1.101, 0.03)  # multiples of the size estimated from ink height
SIZE_REFINE = (-0.015, -0.0075, 0.0075, 0.015)  # finer steps around the best multiple


def _shape(text: str, family: str, size: float, weight: int, tracking: float) -> np.ndarray | None:
    """Boolean ink of text cropped to its ink box."""
    a = glyphs(text, get_font(family, size, weight), tracking)[0]
    b = ink_box(a)
    return None if b is None else a[b[1]:b[3], b[0]:b[2]] > 127


def _tracked(text: str, family: str, weight: int, size: float, natural_w: float, ink_w: int,
             correct: bool = True) -> tuple[np.ndarray | None, float]:
    """Ink of text at size, with the tracking that makes it ink_w wide. natural_w is its width at REF_SIZE.

    Glyph advances do not scale linearly with size, so with correct=True the tracking is fixed once after rendering.
    """
    n = len(text)
    lo, hi = -0.15 * size, 0.8 * size
    tracking = float(np.clip((ink_w - natural_w * size / REF_SIZE) / (n - 1), lo, hi)) if n > 1 else 0.0
    shape = _shape(text, family, size, weight, tracking)
    if correct and shape is not None and n > 1:
        tracking = float(np.clip(tracking + (ink_w - shape.shape[1]) / (n - 1), lo, hi))
        shape = _shape(text, family, size, weight, tracking)
    return shape, tracking


def _soft_overlap(shape: np.ndarray, target: np.ndarray, sigma: float) -> float:
    """Overlap of the blurred masks centred on one canvas, best over +-1 px shifts. Nothing is stretched,
    so a wrong size or stroke weight (and with it the gaps between glyphs) lowers the score."""
    h = max(shape.shape[0], target.shape[0]) + 8
    w = max(shape.shape[1], target.shape[1]) + 8

    def soft(m: np.ndarray) -> np.ndarray:
        canvas = np.zeros((h, w), np.float32)
        y, x = (h - m.shape[0]) // 2, (w - m.shape[1]) // 2
        canvas[y:y + m.shape[0], x:x + m.shape[1]] = m
        return cv2.GaussianBlur(canvas, (0, 0), sigma)

    a, t = soft(shape), soft(target)
    best = 0.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            s = np.roll(a, (dy, dx), axis=(0, 1))
            best = max(best, float(np.minimum(s, t).sum() / max(np.maximum(s, t).sum(), 1e-6)))
    return best


def fit(meas: dict, ref_text: str) -> dict:
    """Family/weight/size/tracking whose rendering of ref_text overlaps the measured glyphs best.

    Every family/weight is first ranked by glyph shape (stretched onto the ink box). The best few are then searched
    over size, each with the tracking that keeps the original line width, and scored without stretching so glyph
    size, stroke weight and the gaps between glyphs all have to match the original.
    """
    x1, y1, x2, y2 = meas["ink"]
    ink_w, ink_h = x2 - x1, y2 - y1
    target = meas["mask"] > 0
    sigma = max(1.0, meas["stroke"] / 2)
    ranked = []
    for family, (_, variable) in FONTS.items():
        for weight in WEIGHTS if variable else (400,):
            b = ink_box(glyphs(ref_text, get_font(family, REF_SIZE, weight), 0)[0])
            if b is None:
                continue
            s0 = REF_SIZE * ink_h / (b[3] - b[1])
            shape, _ = _tracked(ref_text, family, weight, s0, b[2] - b[0], ink_w, correct=False)
            if shape is None:
                continue
            shape = cv2.resize(shape.astype(np.uint8), (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            ranked.append((_iou(shape, target), family, weight, s0, b[2] - b[0]))
    ranked.sort(key=lambda c: -c[0])

    best = None

    def attempt(family: str, weight: int, s0: float, natural_w: float, k: float) -> None:
        nonlocal best
        size = round(s0 * k, 1)
        shape, tracking = _tracked(ref_text, family, weight, size, natural_w, ink_w)
        if shape is None:
            return
        score = _soft_overlap(shape, target, sigma)
        if best is None or score > best[0]:
            best = (score, family, weight, size, tracking, s0, natural_w, k)

    for _, family, weight, s0, natural_w in ranked[:SHORTLIST]:
        for k in SIZE_STEPS:
            attempt(family, weight, s0, natural_w, k)
    _, family, weight, _, _, s0, natural_w, k = best
    for d in SIZE_REFINE:
        attempt(family, weight, s0, natural_w, k + d)
    score, family, weight, size, tracking = best[:5]
    return {"font": family, "weight": weight, "size": float(size), "tracking": round(tracking, 2), "fit_score": round(score, 3)}


def _glow_alpha(alpha: np.ndarray, radius: int, strength: float) -> np.ndarray:
    """Blurred halo scaled so its opacity 3 px outside the strokes equals strength."""
    blur = cv2.GaussianBlur(alpha, (0, 0), max(radius / 2.0, 0.8))
    dist = cv2.distanceTransform((alpha <= 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    ring = (dist > 2) & (dist <= 3)
    level = float(blur[ring].mean()) if ring.any() else 0.0
    if level <= 0:
        return np.zeros_like(alpha)
    return np.clip(blur * (strength / level), 0, 1)


def _composite(rgb: np.ndarray, layers: list[tuple[str, np.ndarray]], left: int, top: int) -> np.ndarray:
    out = rgb.copy()
    h, w = rgb.shape[:2]
    ah, aw = layers[0][1].shape
    x1, y1, x2, y2 = max(left, 0), max(top, 0), min(left + aw, w), min(top + ah, h)
    if x1 >= x2 or y1 >= y2:
        return out
    region = out[y1:y2, x1:x2].astype(np.float32)
    for color, alpha in layers:
        a = alpha[y1 - top:y2 - top, x1 - left:x2 - left][..., None]
        region = region * (1 - a) + from_hex(color) * a
    out[y1:y2, x1:x2] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
    return out


def draw(rgb: np.ndarray, style: dict, text: str) -> np.ndarray:
    """Draw text anchored to style['ink'] (the original ink box) with the style's font and effects."""
    font = get_font(style["font"], float(style["size"]), int(style["weight"]))
    tracking = float(style.get("tracking", 0))
    a, (ox, oy) = glyphs(text, font, tracking)
    b = ink_box(a)
    if b is None:
        return rgb
    ix1, iy1, ix2, iy2 = style["ink"]
    # vertical: the reference text's ink is centred on the original ink, so every glyph shares its baseline
    ra, (rox, roy) = glyphs(style.get("ref_text") or text, font, tracking)
    rb = ink_box(ra) or b
    origin_y = (iy1 + iy2) / 2 - ((rb[1] + rb[3]) / 2 - roy)
    left_edge, right_edge = b[0] - ox, b[2] - ox
    align = style.get("align", "left")
    if align == "center":
        origin_x = (ix1 + ix2) / 2 - (left_edge + right_edge) / 2
    elif align == "right":
        origin_x = ix2 - right_edge
    else:
        origin_x = ix1 - left_edge
    origin_x += float(style.get("dx", 0))
    origin_y += float(style.get("dy", 0))

    glow = style.get("glow") or {}
    outline = style.get("outline") or {}
    radius, strength = int(glow.get("radius", 0)), float(glow.get("strength", 0))
    width = int(outline.get("width", 0))
    m = 2 * radius + width + 2
    alpha = np.pad(a.astype(np.float32) / 255, m)
    layers = []
    if radius > 0 and strength > 0:
        layers.append((glow.get("color") or style["color"], _glow_alpha(alpha, radius, strength)))
    if width > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
        layers.append((outline.get("color", "#000000"), cv2.dilate(alpha, k)))
    layers.append((style["color"], alpha))
    return _composite(rgb, layers, int(round(origin_x - ox)) - m, int(round(origin_y - oy)) - m)
