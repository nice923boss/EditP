"""Erase-and-redraw pipeline shared by the web UI and the HTTP API."""
import numpy as np

from . import inpaint, render, style
from .geometry import Box, pad_box, union_box

ERASE_PAD = 6  # a padded rectangle erases glyph halos that a glyph-shaped mask leaves behind


def auto_style(rgb: np.ndarray, box: Box, ref_text: str, new_text: str) -> dict:
    """Measure the original text in box and fit a font to it. ref_text is what the box currently says.

    ref_text can be a raw OCR misreading and new_text differs from the original, so the font is fitted
    against both and the one whose glyphs overlap the original best wins (the true text scores highest).
    """
    h, w = rgb.shape[:2]
    refs = [t for t in dict.fromkeys((ref_text.strip(), new_text.strip())) if t]
    meas = style.measure(rgb, box)
    base = {"align": "left", "dx": 0, "dy": 0, "outline": {"width": 0, "color": "#000000"},
            "ref_text": refs[0] if refs else ""}
    if meas is None:
        return {
            **base, "font": "sans", "weight": 400, "size": max(8.0, (box[3] - box[1]) * 0.8), "tracking": 0,
            "color": "#ffffff", "glow": {"radius": 0, "strength": 0}, "ink": list(box),
            "erase": pad_box(box, ERASE_PAD, w, h), "fit_score": 0,
        }
    fits = [{**render.fit(meas, r), "ref_text": r} for r in refs]
    fitted = max(fits, key=lambda f: f["fit_score"]) if fits else {"font": "sans", "weight": 400, "size": 12, "tracking": 0, "fit_score": 0}
    erase = union_box(pad_box(box, ERASE_PAD, w, h), pad_box(meas["ink"], meas["glow"]["radius"] + 3, w, h))
    return {**base, **fitted, "color": meas["color"], "glow": meas["glow"], "ink": meas["ink"], "erase": erase}


def apply(rgb: np.ndarray, edits: list[dict], erase_cache: dict | None = None) -> tuple[np.ndarray, list[dict]]:
    """edits: [{box, old_text, new_text, style?}]. A style without 'ink' is merged over the measured one.

    Returns the result image and the full style used for each edit (send it back to re-render with tweaks).
    """
    styles = []
    for e in edits:
        given = e.get("style") or {}
        if "ink" not in given:
            given = {**auto_style(rgb, e["box"], e.get("old_text", ""), e["new_text"]), **given}
        styles.append(given)
    key = tuple(tuple(s["erase"]) for s in styles)
    if erase_cache is not None and key in erase_cache:
        out = erase_cache[key]
    else:
        out = inpaint.erase(rgb, [s["erase"] for s in styles])
        if erase_cache is not None:
            erase_cache.clear()
            erase_cache[key] = out
    for e, s in zip(edits, styles):
        if e["new_text"].strip():
            out = render.draw(out, s, e["new_text"])
    return out, styles
