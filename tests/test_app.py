import io
from collections import OrderedDict

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from editp import __version__, config, render, server

KEY = "sk-test-0123456789abcdef"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")  # never touch the real config.json
    config.save({"api_key": KEY})
    return TestClient(server.app, base_url="http://127.0.0.1")


@pytest.fixture
def opened(client, tmp_path, monkeypatch):
    """A UI session with an image open and one render (version 1) applied."""
    src = tmp_path / "海報.png"
    rgb = np.zeros((8, 8, 3), np.uint8)
    result = np.full((8, 8, 3), 200, np.uint8)
    monkeypatch.setattr(server, "_state", {
        "path": str(src), "rgb": rgb, "alpha": None, "key": "k", "lines": [], "read": True, "erase_cache": {},
        "result": result,
        "version": 1, "results": {1: result}, "saved": []})
    return client


def test_masked_key():
    assert config.masked_key("") == ""
    assert config.masked_key("short") == "…"
    assert config.masked_key(KEY) == "sk-tes…cdef"


def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True and body["app"] == "editp" and body["version"] == __version__
    assert isinstance(body["engine"], dict)


def test_get_config_never_returns_the_key(client):
    body = client.get("/api/config").json()
    assert body["api_key"] == "sk-tes…cdef"
    assert KEY not in str(body)


def test_blank_key_keeps_the_saved_one(client):
    client.post("/api/config", json={"base_url": "http://x/v1", "model": "m", "api_key": ""})
    assert config.load()["api_key"] == KEY


def test_default_output_counts_up(tmp_path):
    src = tmp_path / "a.jpg"
    assert server.default_output(src).name == "a_edited.png"
    (tmp_path / "a_edited.png").touch()
    assert server.default_output(src).name == "a_edited_2.png"


def test_text_in_reads_the_line_under_the_box():
    lines = [{"id": 0, "text": "如何破解", "box": [1096, 102, 1299, 158]}]
    assert server._text_in(lines, [1090, 100, 1300, 160]) == "如何破解"
    assert server._text_in(lines, [0, 0, 50, 50]) == ""


def test_write_image_refuses_to_overwrite_original(tmp_path):
    src = tmp_path / "a.png"
    with pytest.raises(HTTPException):
        server._write_image(np.zeros((4, 4, 3), np.uint8), src, src)
    with pytest.raises(HTTPException):  # .bmp is written as .png: still the original
        server._write_image(np.zeros((4, 4, 3), np.uint8), tmp_path / "a.bmp", src)


def test_write_image_picks_the_format_from_the_suffix(tmp_path):
    out = server._write_image(np.zeros((4, 4, 3), np.uint8), tmp_path / "b.jpeg", tmp_path / "a.png")
    assert out.name == "b.jpeg"
    assert Image.open(out).format == "JPEG"
    assert not list(tmp_path.glob("*.part"))


def test_kai_falls_back_to_sans(monkeypatch):
    if "sans" not in render.FONTS:
        pytest.skip("Noto Sans TC 未安裝")
    monkeypatch.setattr(render, "FONTS", {"sans": render.FONTS["sans"]})
    render.get_font.cache_clear()
    try:
        assert render.get_font("kai", 40, 400).path == render.FONTS["sans"][0]
    finally:
        render.get_font.cache_clear()


def test_other_hosts_and_origins_are_refused(client):
    assert TestClient(server.app).get("/api/health").status_code == 403  # Host: testserver
    assert client.post("/api/config/test", json={"base_url": "x", "model": "m"},
                       headers={"Origin": "http://evil.example"}).status_code == 403


def test_restore_switches_between_renders(opened):
    assert opened.post("/api/restore", json={"version": 0}).status_code == 200
    assert server._state["result"] is None
    opened.post("/api/restore", json={"version": 1})
    assert server._state["result"][0, 0, 0] == 200
    assert opened.post("/api/restore", json={"version": 7}).status_code == 404


def test_close_unloads_the_image(opened):
    assert opened.get("/api/current").json()["name"] == "海報.png"
    assert opened.post("/api/close").status_code == 200
    assert opened.get("/api/current").json() == {}
    assert opened.get("/api/image").status_code == 404


def test_remember_drops_the_oldest_render(monkeypatch):
    monkeypatch.setattr(server, "RESULT_CACHE_BYTES", 500)
    s = {"results": {}}
    for v in (1, 2, 3):
        server._remember(s, v, np.zeros((8, 8, 3), np.uint8))  # 192 bytes each
    assert list(s["results"]) == [2, 3]


def test_download_formats(opened):
    res = opened.get("/api/download?fmt=jpg")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert "filename*=UTF-8''%E6%B5%B7%E5%A0%B1_edited.jpg" in res.headers["content-disposition"]
    assert opened.get("/api/download?fmt=gif").status_code == 400


def test_save_writes_beside_the_original(opened, tmp_path):
    body = opened.post("/api/save", json={}).json()
    assert body["name"] == "海報_edited.png"
    assert (tmp_path / "海報_edited.png").is_file()
    assert server._state["saved"] == [body["saved"]]


def test_reveal_only_opens_known_files(opened, tmp_path):
    other = tmp_path / "other.png"
    other.touch()
    assert opened.post("/api/reveal", json={"path": str(other)}).status_code == 404


def test_reveal_selects_the_saved_file(opened, monkeypatch):
    shown = []
    monkeypatch.setattr(server.explorer, "show", lambda path, select=False: shown.append((path, select)))
    saved = opened.post("/api/save", json={}).json()["saved"]
    assert opened.post("/api/reveal", json={"path": saved}).status_code == 200
    assert [(str(p), sel) for p, sel in shown] == [(saved, True)]


def _png(color=(10, 20, 30), size=(12, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def uploads(client, tmp_path, monkeypatch):
    """Uploads land in tmp_path/up; OCR is stubbed and counts its calls."""
    calls = []
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "up")
    monkeypatch.setattr(server, "_state", {})
    monkeypatch.setattr(server, "_ocr_cache", OrderedDict())
    monkeypatch.setattr(server, "_read_cache", OrderedDict())
    monkeypatch.setattr(server.ocr, "detect", lambda rgb: calls.append(1) or [])
    client.calls = calls
    return client


def test_upload_saves_then_opens(uploads, tmp_path):
    body = uploads.post("/api/upload?name=截圖 1.png", content=_png()).json()
    assert body["name"] == "截圖 1.png" and body["width"] == 12 and body["height"] == 8
    assert (tmp_path / "up" / "截圖 1.png").read_bytes() == _png()


def test_upload_reuses_the_same_file_and_numbers_a_different_one(uploads, tmp_path):
    uploads.post("/api/upload?name=a.png", content=_png())
    uploads.post("/api/upload?name=a.png", content=_png())
    assert uploads.post("/api/upload?name=a.png", content=_png((1, 2, 3))).json()["name"] == "a_2.png"
    assert sorted(p.name for p in (tmp_path / "up").iterdir()) == ["a.png", "a_2.png"]


def test_upload_cleans_the_name_and_uses_the_real_format(uploads):
    assert uploads.post("/api/upload?name=..%2F<x>.jpg", content=_png()).json()["name"] == "_x_.png"
    assert uploads.post("/api/upload", content=_png((5, 5, 5))).json()["name"].startswith("貼上的圖片_")


def test_upload_refuses_non_images(uploads, tmp_path):
    res = uploads.post("/api/upload?name=a.png", content=b"not an image")
    assert res.status_code == 400 and "無法讀取圖片" in res.json()["detail"]
    assert uploads.post("/api/upload", content=b"").status_code == 400
    assert not (tmp_path / "up").exists() or not list((tmp_path / "up").iterdir())


def test_reopening_the_same_image_skips_ocr(uploads, tmp_path):
    uploads.post("/api/upload?name=a.png", content=_png())
    uploads.post("/api/open", json={"path": str(tmp_path / "up" / "a.png")})
    assert len(uploads.calls) == 1


def test_fonts_lists_every_family(client):
    assert set(client.get("/api/fonts").json()["fonts"]) == set(render.FONT_FILES)


def test_bad_input_gets_one_readable_message(opened):
    res = opened.post("/api/render", json={"edits": [{"box": [1, 2, 3], "new_text": "x"}]})
    assert res.status_code == 422
    assert res.json()["detail"].startswith("輸入格式不正確（edits.0.box）")


def test_tiny_or_outside_box_is_refused(opened):
    res = opened.post("/api/render", json={"edits": [{"box": [50, 50, 60, 60], "new_text": "x"}]})
    assert res.status_code == 400 and "太小或在圖片外" in res.json()["detail"]
