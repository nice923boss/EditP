import io

import numpy as np
import pytest
from PIL import Image

from editp import images


def _bytes(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def test_load_keeps_transparency():
    img = Image.new("RGBA", (6, 4), (255, 0, 0, 255))
    img.putpixel((0, 0), (255, 0, 0, 0))
    rgb, alpha, fmt = images.load(_bytes(img, "PNG"))
    assert fmt == "PNG" and rgb.shape == (4, 6, 3)
    assert alpha[0, 0] == 0 and alpha[1, 1] == 255


def test_load_drops_an_all_opaque_alpha():
    assert images.load(_bytes(Image.new("RGBA", (4, 4), (1, 2, 3, 255)), "PNG"))[1] is None


def test_load_turns_phone_photos_upright():
    img = Image.new("RGB", (6, 4), (9, 9, 9))
    exif = img.getexif()
    exif[0x0112] = 6  # stored sideways: rotate 90 degrees to view
    rgb, _, _ = images.load(_bytes(img, "JPEG", exif=exif))
    assert rgb.shape == (6, 4, 3)


def test_load_refuses_huge_and_broken_files(monkeypatch):
    monkeypatch.setattr(images, "MAX_PIXELS", 10)
    with pytest.raises(images.ImageError, match="圖片太大"):
        images.load(_bytes(Image.new("RGB", (4, 4)), "PNG"))
    with pytest.raises(images.ImageError, match="無法讀取圖片"):
        images.load(b"\x89PNG broken")


def test_jpg_flattens_transparency_onto_white():
    rgb = np.zeros((2, 2, 3), np.uint8)
    alpha = np.array([[0, 255], [255, 255]], np.uint8)
    out = np.array(Image.open(io.BytesIO(images.encode(rgb, "jpg", alpha))))
    assert out[0, 0].min() > 200 and out[1, 1].max() < 60


def test_png_keeps_transparency():
    alpha = np.array([[0, 255], [255, 255]], np.uint8)
    out = Image.open(io.BytesIO(images.encode(np.zeros((2, 2, 3), np.uint8), "png", alpha)))
    assert out.mode == "RGBA" and out.getpixel((0, 0))[3] == 0


def test_output_path_falls_back_to_png(tmp_path):
    assert images.output_path(tmp_path / "a.webp") == (tmp_path / "a.webp", "webp")
    assert images.output_path(tmp_path / "a.JPEG") == (tmp_path / "a.JPEG", "jpg")
    assert images.output_path(tmp_path / "a.bmp") == (tmp_path / "a.png", "png")


def test_write_reports_a_missing_folder(tmp_path):
    with pytest.raises(images.ImageError, match="無法寫入"):
        images.write(np.zeros((2, 2, 3), np.uint8), tmp_path / "no" / "a.png")
