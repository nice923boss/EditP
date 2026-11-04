"""EditP entry point: serve the HTTP API and web UI on 127.0.0.1, show a tray icon, then open the browser."""
import argparse
import importlib.util
import socket
import threading
import time
import webbrowser

import httpx

from editp import config

# packages start.bat checks before launching; a missing one sends it through install.bat first
REQUIRED = ("fastapi", "uvicorn", "httpx", "pydantic", "numpy", "PIL", "cv2", "rapidocr", "onnxruntime", "pystray")
OPTIONAL = ("torch", "simple_lama_inpainting")  # LaMa eraser; without it erasing falls back to OpenCV
PORT_TRIES = 10  # the configured port and the next ones, when another program already uses it


def running(url: str) -> bool:
    try:
        return httpx.get(url + "/api/health", timeout=1).json().get("app") == "editp"
    except (httpx.HTTPError, ValueError):
        return False


def port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def check() -> int:
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    print(f"[check] 缺少套件：{'、'.join(missing)}" if missing else "[check] 套件齊全", flush=True)
    if not missing and any(importlib.util.find_spec(name) is None for name in OPTIONAL):
        print("[check] 未安裝 LaMa，擦除會改用 OpenCV；跑 install.bat 可補裝", flush=True)
    return 1 if missing else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="EditP 圖片錯字修正")
    parser.add_argument("--no-browser", action="store_true", help="不要自動開啟瀏覽器")
    parser.add_argument("--no-tray", action="store_true", help="不要顯示系統托盤圖示")
    parser.add_argument("--check", action="store_true", help="只檢查套件是否齊全（缺少時結束代碼為 1）")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(check())
    wanted = config.load()["port"]
    for port in range(wanted, wanted + PORT_TRIES):
        url = f"http://127.0.0.1:{port}"
        if running(url):  # second start: just bring up the page
            if not args.no_browser:
                webbrowser.open(url)
            return
        if port_free(port):
            break
    else:
        raise SystemExit(f"[app] 埠 {wanted} 到 {wanted + PORT_TRIES - 1} 都被其他程式占用，請在 config.json 改 port")
    if port != wanted:
        print(f"[app] 埠 {wanted} 被其他程式占用，改用 {port}", flush=True)

    import uvicorn

    from editp import server, tray

    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="info"))
    icon = None

    def quit_app() -> None:
        srv.should_exit = True
        if icon is not None:
            icon.stop()

    def open_when_ready() -> None:
        for _ in range(120):
            if running(url):
                webbrowser.open(url)
                return
            time.sleep(0.5)

    server.on_exit = quit_app
    if not args.no_tray:
        icon = tray.start(url, quit_app)
    if not args.no_browser:
        threading.Thread(target=open_when_ready, daemon=True).start()
    threading.Thread(target=server.warm_up, daemon=True).start()
    try:
        srv.run()
    finally:
        if icon is not None:
            icon.stop()


if __name__ == "__main__":
    main()
