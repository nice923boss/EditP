"""HTTP API and the web UI. Single-user local app: one image is open in the UI at a time."""
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated
from urllib.parse import quote, urlparse

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import __version__, config, explorer, images, inpaint, llm, ocr, pipeline, render
from .geometry import clamp_box, line_at_point

STATIC = config.ROOT / "static"
IMAGE_TYPES = [("圖片", " ".join("*" + s for s in images.SUFFIXES)), ("所有檔案", "*.*")]
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
RESULT_CACHE_BYTES = 600 * 2**20  # renders kept in memory so undo/redo can show them again without redoing them
CACHED_IMAGES = 16  # OCR and LLM readings kept per image content, so reopening an image skips both
UPLOAD_DIR = Path.home() / "Pictures" / "EditP"  # dropped or pasted images are saved here, results beside them
MAX_UPLOAD = 200 * 2**20
FORMATS = images.FORMATS

app = FastAPI(title="EditP")
_lock = threading.Lock()  # one GPU, one LLM: requests that touch them run one at a time
_state: dict = {}  # path, rgb, alpha, key, lines, read, erase_cache, result, version, results {version: result}, saved
_ocr_cache: OrderedDict = OrderedDict()  # image key -> OCR lines
_read_cache: OrderedDict = OrderedDict()  # (image key, base_url, model) -> LLM-proofread lines
_tk = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tk")  # Tcl objects must stay on the thread that made them
on_exit = None  # set by app.py: stops the server and the tray icon

Box = Annotated[list[float], Field(min_length=4, max_length=4)]
Text = Annotated[str, Field(max_length=500)]


class EditIn(BaseModel):
    box: Box
    old_text: Text = ""
    new_text: Text
    style: dict | None = None


class ConfigIn(BaseModel):
    base_url: str
    model: str
    api_key: str | None = None  # empty keeps the stored key


class OpenIn(BaseModel):
    path: str | None = None  # None opens a file dialog


class BoxIn(BaseModel):
    box: Box


class PlanIn(BaseModel):
    instruction: Annotated[str, Field(min_length=1, max_length=2000)]


class RenderIn(BaseModel):
    edits: list[EditIn]


class SaveIn(BaseModel):
    path: str | None = None
    dialog: bool = False


class RestoreIn(BaseModel):
    version: int  # 0 = no result (the original)


class RevealIn(BaseModel):
    path: str


class EditRequest(BaseModel):
    path: str
    instruction: Annotated[str, Field(max_length=2000)] | None = None
    edits: list[EditIn] | None = None
    output: str | None = None
    dry_run: bool = False


def warm_up() -> None:
    """Load OCR and LaMa ahead of the first request (under the lock, so a request never loads them twice)."""
    with _lock:
        ocr.detect(np.zeros((32, 32, 3), np.uint8))
        inpaint.warm_up()


@app.exception_handler(llm.LLMError)
def _llm_error(_: Request, e: llm.LLMError) -> JSONResponse:
    return JSONResponse({"detail": str(e)}, status_code=502)


@app.exception_handler(images.ImageError)
def _image_error(_: Request, e: images.ImageError) -> JSONResponse:
    return JSONResponse({"detail": str(e)}, status_code=400)


@app.exception_handler(RequestValidationError)
def _bad_input(_: Request, e: RequestValidationError) -> JSONResponse:
    """One readable sentence instead of pydantic's list, so the page can show it as is."""
    first = e.errors()[0] if e.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    return JSONResponse({"detail": f"輸入格式不正確（{where}）：{first.get('msg', '')}"}, status_code=422)


@app.middleware("http")
async def _local_only(request: Request, call_next):
    """Only this app's own page may call it: blocks DNS rebinding and other sites' cross-site POSTs."""
    host = (request.headers.get("host") or "").rsplit(":", 1)[0]
    origin = request.headers.get("origin")
    if host not in ALLOWED_HOSTS or (origin and urlparse(origin).hostname not in ALLOWED_HOSTS):
        return JSONResponse({"detail": "只接受本機的 EditP 頁面存取"}, status_code=403)
    return await call_next(request)


def _jsonable(obj):
    """Style dicts carry numpy scalars; round-trip them into plain JSON types."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else o.tolist()))


def _ask(kind: str, kw: dict) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    root.lift()
    root.focus_force()
    try:
        ask = filedialog.askopenfilename if kind == "open" else filedialog.asksaveasfilename
        return ask(parent=root, **kw) or ""
    finally:
        root.destroy()


def _dialog(kind: str, **kw) -> str:
    """Native file dialog on top of the browser, always run on the one tkinter thread."""
    return _tk.submit(_ask, kind, kw).result()


def _remember(s: dict, version: int, result: np.ndarray | None) -> None:
    """Keep each render for undo/redo, dropping the oldest once they pass RESULT_CACHE_BYTES."""
    s["results"][version] = result
    while len(s["results"]) > 1 and sum(r.nbytes for r in s["results"].values() if r is not None) > RESULT_CACHE_BYTES:
        del s["results"][next(iter(s["results"]))]


def _cache_get(cache: OrderedDict, key):
    if key in cache:
        cache.move_to_end(key)
    return cache.get(key)


def _cache_put(cache: OrderedDict, key, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > CACHED_IMAGES:
        cache.popitem(last=False)


def _key(rgb: np.ndarray) -> str:
    return hashlib.sha1(rgb.tobytes() + str(rgb.shape).encode()).hexdigest()


def _detect(rgb: np.ndarray, key: str) -> list[dict]:
    """OCR lines, from the cache when this image content was opened before."""
    lines = _cache_get(_ocr_cache, key)
    if lines is None:
        lines = [{**ln, "ocr_text": ln["text"], "garbled": False} for ln in ocr.detect(rgb)]
        _cache_put(_ocr_cache, key, lines)
    return lines


def _proofread(rgb: np.ndarray, lines: list[dict], key: str) -> list[dict]:
    """LLM reading of the lines, cached per image content and LLM (a new model reads again)."""
    cfg = config.load()
    ck = (key, cfg["base_url"], cfg["model"])
    out = _cache_get(_read_cache, ck)
    if out is None:
        out = llm.read(cfg, rgb, lines)
        _cache_put(_read_cache, ck, out)
    return out


def _load(path: str) -> tuple[Path, np.ndarray, np.ndarray | None]:
    p = Path(path.strip().strip('"'))
    if not p.is_file():
        raise HTTPException(404, f"找不到檔案：{p}")
    rgb, alpha, _ = images.load(p)
    return p, rgb, alpha


def _boxes(edits: list[EditIn], w: int, h: int) -> list[dict]:
    """Edits with boxes clamped to the image; a box left with no area is refused, not silently drawn nowhere."""
    out = []
    for e in edits:
        box = clamp_box(e.box, w, h)
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            raise HTTPException(400, f"框 {[round(v) for v in e.box]} 太小或在圖片外，請重新框選")
        out.append({**e.model_dump(), "box": box})
    return out


def default_output(src: Path) -> Path:
    """<name>_edited.png next to the original; _edited_2, _edited_3 ... when taken."""
    out = src.with_name(f"{src.stem}_edited.png")
    n = 2
    while out.exists():
        out = src.with_name(f"{src.stem}_edited_{n}.png")
        n += 1
    return out


def _text_in(lines: list[dict], box: list[int]) -> str:
    """Text of the line under the box centre, '' when none: a reference for font fitting when an API
    caller gives no old_text (fitted against the new text alone, a real poster came out 800 instead of 600)."""
    line = line_at_point(lines, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, max_dist=0)
    return line["text"] if line else ""


def _write_image(rgb: np.ndarray, out: Path, src: Path, alpha: np.ndarray | None = None) -> Path:
    """PNG, JPG or WebP by out's suffix; never over the original."""
    if images.output_path(out)[0].resolve() == src.resolve():
        raise HTTPException(400, "不能覆蓋原圖，請換一個檔名")
    if not out.parent.is_dir():
        raise HTTPException(400, f"資料夾不存在：{out.parent}")
    return images.write(rgb, out, alpha)


def _image() -> dict:
    if "rgb" not in _state:
        raise HTTPException(400, "還沒有開啟圖片")
    return _state


def _ensure_read(s: dict) -> None:
    if not s["read"]:
        s["lines"] = _proofread(s["rgb"], s["lines"], s["key"])
        s["read"] = True


def _summary(s: dict) -> dict:
    h, w = s["rgb"].shape[:2]
    p = Path(s["path"])
    return {"path": str(p), "name": p.name, "width": w, "height": h, "lines": s["lines"], "read": s["read"],
            "key": s["key"]}  # the page keeps its unsaved list per image under this key


@app.get("/api/health")
def health() -> dict:
    """engine is {} until the eraser has loaded, then {name, device, note}."""
    return {"ok": True, "app": "editp", "version": __version__, "engine": inpaint.status()}


@app.get("/api/fonts")
def fonts() -> dict:
    """Which font families this computer has (a missing one is drawn with the sans font)."""
    return {"fonts": {family: family in render.FONTS for family in render.FONT_FILES}}


@app.get("/api/config")
def get_config() -> dict:
    cfg = config.load()
    return {"base_url": cfg["base_url"], "model": cfg["model"], "api_key": config.masked_key(cfg["api_key"])}


@app.post("/api/config")
def set_config(body: ConfigIn) -> dict:
    updates = {"base_url": body.base_url.strip(), "model": body.model.strip()}
    if body.api_key and body.api_key.strip():
        updates["api_key"] = body.api_key.strip()
    config.save(updates)
    return get_config()


@app.post("/api/config/test")
def test_config(body: ConfigIn) -> dict:
    cfg = {**config.load(), "base_url": body.base_url.strip(), "model": body.model.strip()}
    if body.api_key and body.api_key.strip():
        cfg["api_key"] = body.api_key.strip()
    models = llm.list_models(cfg)
    return {"ok": True, "models": models, "model_found": cfg["model"] in models}


@app.get("/api/current")
def current() -> dict:
    """The image open in the UI (lets a reloaded page pick up where it was); {} when none."""
    return _summary(_state) if "rgb" in _state else {}


@app.post("/api/open")
def open_image(body: OpenIn) -> dict:
    path = body.path or _dialog("open", title="開啟圖片", filetypes=IMAGE_TYPES)
    if not path:
        return {"cancelled": True}
    return _open(*_load(path))


@app.post("/api/close")
def close_image() -> dict:
    """Unload the open image, so a reloaded page starts empty instead of picking it up again.
    A new dict instead of clear(): a slow LLM read still holding the lock finishes on the old one."""
    global _state
    _state = {}
    return {"ok": True}


def _open(p: Path, rgb: np.ndarray, alpha: np.ndarray | None) -> dict:
    key = _key(rgb)
    with _lock:
        lines = _detect(rgb, key)
        cfg = config.load()
        read = _cache_get(_read_cache, (key, cfg["base_url"], cfg["model"]))
        _state.clear()
        _state.update(path=str(p), rgb=rgb, alpha=alpha, key=key, lines=lines if read is None else read,
                      read=read is not None,
                      erase_cache={}, result=None, version=0, results={}, saved=[])
        return _summary(_state)


def _upload_target(name: str, fmt: str, data: bytes) -> Path:
    """A free file name in UPLOAD_DIR (the same file dropped twice is reused, not copied again)."""
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(name).stem).strip(" .")[:80]
    stem = stem or time.strftime("貼上的圖片_%Y%m%d_%H%M%S")
    ext = images.EXTENSION.get(fmt, ".png")
    out, n = UPLOAD_DIR / f"{stem}{ext}", 2
    while out.exists() and out.read_bytes() != data:
        out, n = UPLOAD_DIR / f"{stem}_{n}{ext}", n + 1
    return out


def _save_upload(data: bytes, name: str) -> dict:
    rgb, alpha, fmt = images.load(data)
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = _upload_target(name, fmt, data)
        target.write_bytes(data)
    except OSError as e:
        raise HTTPException(500, f"無法把圖片存到 {UPLOAD_DIR}：{e}") from e
    return _open(target, rgb, alpha)


@app.post("/api/upload")
async def upload(request: Request, name: str = "") -> dict:
    """An image dropped on or pasted into the page (raw bytes): saved to UPLOAD_DIR, then opened from there."""
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD:
        raise HTTPException(413, f"圖片超過 {MAX_UPLOAD // 2**20} MB")
    data = await request.body()
    if not data:
        raise HTTPException(400, "沒有收到圖片內容")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"圖片超過 {MAX_UPLOAD // 2**20} MB")
    return await run_in_threadpool(_save_upload, data, name)


@app.post("/api/read")
def read_lines() -> dict:
    """LLM proofreading of the OCR lines (slow, cached per image)."""
    with _lock:
        s = _image()
        _ensure_read(s)
        return {"path": s["path"], "lines": s["lines"]}


@app.post("/api/region_text")
def region_text(body: BoxIn) -> dict:
    with _lock:
        s = _image()
        h, w = s["rgb"].shape[:2]
        box = clamp_box(body.box, w, h)
        return {"box": box, "text": ocr.read_region(s["rgb"], box)}


@app.post("/api/plan")
def plan(body: PlanIn) -> dict:
    with _lock:
        s = _image()
        _ensure_read(s)
        out = llm.plan(config.load(), s["rgb"], s["lines"], body.instruction.strip())
        return {**out, "lines": s["lines"]}


@app.post("/api/render")
def render_edits(body: RenderIn) -> dict:
    with _lock:
        s = _image()
        h, w = s["rgb"].shape[:2]
        edits = _boxes(body.edits, w, h)
        styles = []
        if edits:
            s["result"], styles = pipeline.apply(s["rgb"], edits, s["erase_cache"])
        else:
            s["result"] = None
        s["version"] += 1
        _remember(s, s["version"], s["result"])
        return {"styles": _jsonable(styles), "version": s["version"]}


@app.post("/api/restore")
def restore(body: RestoreIn) -> dict:
    """Undo/redo: show an earlier render again (version 0 = the original)."""
    with _lock:
        s = _image()
        if body.version and body.version not in s["results"]:
            raise HTTPException(404, "這一步的結果已不在記憶體中，需要重新套用")
        s["result"] = s["results"].get(body.version)
        return {"version": body.version}


@app.get("/api/image")
def image(which: str = "original") -> Response:
    rgb = _state.get("result") if which == "result" else _state.get("rgb")
    if rgb is None:
        raise HTTPException(404, "沒有這張圖")
    return Response(images.encode(rgb, "png", _state.get("alpha"), fast=True), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/save")
def save(body: SaveIn) -> dict:
    s = _image()
    if s["result"] is None:
        raise HTTPException(400, "還沒有套用任何修改")
    src = Path(s["path"])
    target = body.path
    if not target and body.dialog:
        out = default_output(src)
        target = _dialog("save", title="另存新檔", initialdir=str(out.parent), initialfile=out.name,
                         defaultextension=".png",
                         filetypes=[("PNG 圖片", "*.png"), ("JPG 圖片", "*.jpg"), ("WebP 圖片", "*.webp")])
        if not target:
            return {"cancelled": True}
    with _lock:
        out = _write_image(s["result"], Path(target) if target else default_output(src), src, s.get("alpha"))
        s["saved"].append(str(out))
    return {"saved": str(out), "name": out.name, "folder": str(out.parent)}


@app.get("/api/download")
def download(fmt: str = "png") -> Response:
    """The current result as a browser download (the browser shows its own save dialog or download bar)."""
    s = _image()
    if s["result"] is None:
        raise HTTPException(400, "還沒有套用任何修改")
    if fmt not in FORMATS:
        raise HTTPException(400, f"不支援的格式：{fmt}（可用 {'、'.join(FORMATS)}）")
    name = f"{Path(s['path']).stem}_edited.{fmt}"
    return Response(images.encode(s["result"], fmt, s.get("alpha")), media_type=FORMATS[fmt][1], headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}", "Cache-Control": "no-store"})


@app.post("/api/reveal")
def reveal(body: RevealIn) -> dict:
    """Open Explorer on a file this session saved, or on the open image."""
    allowed = {*_state.get("saved", ()), _state.get("path")}
    if body.path not in allowed or not Path(body.path).is_file():
        raise HTTPException(404, "只能開啟這次存過的檔案所在資料夾")
    explorer.show(Path(body.path), select=True)
    return {"ok": True}


@app.post("/api/shutdown")
def shutdown() -> dict:
    if on_exit is None:
        raise HTTPException(501, "這個執行方式不支援從網頁結束")
    threading.Timer(0.3, on_exit).start()  # after this response is sent
    return {"ok": True}


@app.post("/api/edit")
def edit(body: EditRequest) -> dict:
    """One call for scripts and Claude: read the image, plan from an instruction and/or take explicit
    edits, render, and save a PNG (unless dry_run). The UI state is not touched."""
    src, rgb, alpha = _load(body.path)
    h, w = rgb.shape[:2]
    explicit = _boxes(body.edits or [], w, h)
    key = _key(rgb)
    with _lock:
        lines = _detect(rgb, key)
        if body.instruction or not explicit:
            lines = _proofread(rgb, lines, key)
        result = {"edits": [], "reason": "", "source": "explicit"}
        if body.instruction:
            result = llm.plan(config.load(), rgb, lines, body.instruction.strip())
        by_id = {ln["id"]: ln for ln in lines}
        explicit = [{**e, "old_text": e["old_text"] or _text_in(lines, e["box"])} for e in explicit]
        edits = [{"box": by_id[e["id"]]["box"], "old_text": by_id[e["id"]]["text"], "new_text": e["new_text"]}
                 for e in result["edits"]] + explicit
        out = {"lines": lines, "edits": edits, "reason": result["reason"], "source": result["source"], "saved": None}
        if body.dry_run or not edits:
            return _jsonable(out)
        rendered, styles = pipeline.apply(rgb, edits)
    target = Path(body.output) if body.output else default_output(src)
    out["saved"] = str(_write_image(rendered, target, src, alpha))
    out["edits"] = [{**e, "style": st} for e, st in zip(edits, styles)]
    return _jsonable(out)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
