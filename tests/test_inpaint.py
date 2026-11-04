import numpy as np
from PIL import Image

from editp import inpaint

FILL = 7


def fake_lama(masks: list):
    def run(crop: Image.Image, mask: Image.Image) -> Image.Image:
        m = np.array(mask)
        masks.append(m)
        out = np.array(crop)
        out[m > 0] = FILL
        return Image.fromarray(out)

    return run


def test_erase_masks_every_pending_box_in_each_crop(monkeypatch):
    """Erasing one line must not let LaMa copy the next, still-unerased line into the fill."""
    masks: list = []
    monkeypatch.setattr(inpaint, "_lama", lambda: fake_lama(masks))
    rgb = np.full((100, 200, 3), 200, np.uint8)
    lower, upper = [50, 45, 150, 60], [50, 20, 150, 35]

    out = inpaint.erase(rgb, [lower, upper])

    assert masks[0][20:35, 50:150].all()  # the upper box is masked while the lower one is filled
    assert (out[20:35, 50:150] == FILL).all() and (out[45:60, 50:150] == FILL).all()
    outside = np.ones((100, 200), bool)
    outside[20:35, 50:150] = outside[45:60, 50:150] = False
    assert (out[outside] == 200).all()


def test_erase_falls_back_to_opencv_without_lama(monkeypatch):
    monkeypatch.setattr(inpaint, "_lama", lambda: None)
    rgb = np.full((80, 120, 3), 200, np.uint8)
    rgb[30:40, 40:80] = 0  # "text" to erase
    out = inpaint.erase(rgb, [[38, 28, 82, 42]])
    assert out[30:40, 40:80].min() > 150  # filled from the grey around it
    outside = np.ones((80, 120), bool)
    outside[28:42, 38:82] = False
    assert (out[outside] == 200).all()
