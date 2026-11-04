import numpy as np
import pytest
from PIL import Image, ImageDraw

from editp import pipeline, render

if "sans" not in render.FONTS:
    pytest.skip("Noto Sans TC 未安裝", allow_module_level=True)

TEXT = "如何破解"


@pytest.fixture(scope="module")
def poster():
    img = Image.new("RGB", (400, 120), (40, 50, 70))
    ImageDraw.Draw(img).text((40, 30), TEXT, font=render.get_font("sans", 50, 600), fill=(170, 230, 240))
    return np.array(img), [30, 20, 290, 100]


@pytest.mark.parametrize("ref_text, new_text", [
    ("鼠何破解", TEXT),  # OCR misread the box; the new text happens to match the original glyphs better
    (TEXT, "如何化解"),  # the usual case: the box's true text is the better reference
])
def test_auto_style_fits_against_the_better_reference(poster, ref_text, new_text):
    rgb, box = poster
    st = pipeline.auto_style(rgb, box, ref_text, new_text)
    assert st["ref_text"] == TEXT
    assert st["font"] == "sans" and st["weight"] == 600


@pytest.mark.parametrize("text, size, weight", [("美食市集", 30, 500), ("煙火秀", 30, 500), (TEXT, 50, 600)])
def test_auto_style_matches_original_size_weight_and_spacing(text, size, weight):
    # drawn with tracking 0: a fit that is too big or too bold has to squeeze the gaps with negative tracking
    img = Image.new("RGB", (400, 120), (40, 50, 70))
    ImageDraw.Draw(img).text((40, 30), text, font=render.get_font("sans", size, weight), fill=(240, 240, 240))
    st = pipeline.auto_style(np.array(img), [30, 20, 290, 100], text, text)
    assert st["font"] == "sans" and st["weight"] == weight
    assert abs(st["size"] - size) <= 0.5
    assert abs(st["tracking"]) <= 0.6
