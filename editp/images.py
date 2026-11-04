"""Reading and writing image files: EXIF rotation, transparency, size limits and output formats."""
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MAX_PIXELS = 50_000_000  # about 8K x 6K; past this LaMa and the browser run out of memory
SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif")
FORMATS = {  # extension -> (PIL format, media type, save options)
    "png": ("PNG", "image/png", {}),
    "jpg": ("JPEG", "image/jpeg", {"quality": 95}),
    "webp": ("WEBP", "image/webp", {"quality": 95}),
}
EXTENSION = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif", "GIF": ".gif"}


class ImageError(ValueError):
    """The file is not an image this app can open; the message is shown to the user."""


def load(src: Path | bytes) -> tuple[np.ndarray, np.ndarray | None, str]:
    """RGB pixels turned upright per EXIF, the alpha channel (None when fully opaque), and the file format."""
    try:
        with Image.open(src if isinstance(src, Path) else io.BytesIO(src)) as img:
            fmt = img.format or ""
            if img.width * img.height > MAX_PIXELS:
                raise ImageError(f"圖片太大（{img.width}×{img.height}），上限約 {MAX_PIXELS // 10_000} 萬像素，請先縮小")
            img = ImageOps.exif_transpose(img)  # phone photos store rotation as a tag; OCR needs it upright
            alpha = None
            if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
                a = np.array(img.convert("RGBA"))[..., 3]
                alpha = a if (a < 255).any() else None
            return np.array(img.convert("RGB")), alpha, fmt
    except ImageError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        raise ImageError(f"無法讀取圖片：{e}") from e


def _with_alpha(rgb: np.ndarray, alpha: np.ndarray | None, fmt: str) -> Image.Image:
    if alpha is None:
        return Image.fromarray(rgb)
    if fmt == "jpg":  # JPEG has no transparency: flatten onto white rather than the hidden colours
        a = alpha[..., None].astype(np.float32) / 255
        return Image.fromarray((rgb * a + 255 * (1 - a) + 0.5).astype(np.uint8))
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def encode(rgb: np.ndarray, fmt: str, alpha: np.ndarray | None = None, fast: bool = False) -> bytes:
    """fmt is a FORMATS key; fast trades PNG size for speed (previews)."""
    kind, _, opts = FORMATS[fmt]
    buf = io.BytesIO()
    _with_alpha(rgb, alpha, fmt).save(buf, kind, **({"compress_level": 1} if fast and fmt == "png" else opts))
    return buf.getvalue()


def output_path(out: Path) -> tuple[Path, str]:
    """out with a supported suffix (anything else becomes .png), and its FORMATS key."""
    fmt = {".jpeg": "jpg"}.get(out.suffix.lower(), out.suffix.lower().lstrip("."))
    return (out, fmt) if fmt in FORMATS else (out.with_suffix(".png"), "png")


def write(rgb: np.ndarray, out: Path, alpha: np.ndarray | None = None) -> Path:
    """Save as PNG, JPG or WebP by out's suffix, via a temp file so a failed write never leaves half an image."""
    out, fmt = output_path(out)
    part = out.with_name(out.name + ".part")
    try:
        part.write_bytes(encode(rgb, fmt, alpha))
        part.replace(out)
    except OSError as e:
        part.unlink(missing_ok=True)
        raise ImageError(f"無法寫入 {out}：{e}") from e
    return out
