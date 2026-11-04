"""Notification-area (system tray) icon: shows EditP is running, reopens the page, and quits it."""
import threading
import webbrowser
from pathlib import Path

from PIL import Image

from . import config, explorer

LOG_DIR = Path.home() / ".editp"


def start(url: str, on_quit) -> object | None:
    """Show the tray icon on its own thread; returns it (call .stop() on exit), or None when pystray is missing."""
    try:
        import pystray
    except ImportError as e:  # an older install without pystray still runs, just without the icon
        print(f"[tray] 系統托盤圖示停用，缺少 pystray：{e}（重跑 install.bat 可補裝）", flush=True)
        return None

    def open_page(*_):
        webbrowser.open(url)

    def open_logs(*_):
        LOG_DIR.mkdir(exist_ok=True)
        explorer.show(LOG_DIR)

    def ready(icon):
        icon.visible = True
        try:
            icon.notify("EditP 已在背景執行。按右下角的圖示可再開啟頁面或結束程式。", "EditP 圖片錯字修正")
        except Exception as e:  # a balloon is a nicety; Focus Assist or policy may refuse it
            print(f"[tray] 通知無法顯示：{e}", flush=True)

    icon = pystray.Icon("EditP", Image.open(config.ROOT / "static" / "icon.png"), f"EditP 圖片錯字修正（{url}）",
                        pystray.Menu(
                            pystray.MenuItem("開啟 EditP", open_page, default=True),
                            pystray.MenuItem("開啟記錄檔資料夾", open_logs),
                            pystray.Menu.SEPARATOR,
                            pystray.MenuItem("結束 EditP", lambda *_: on_quit()),
                        ))
    threading.Thread(target=icon.run, kwargs={"setup": ready}, name="tray", daemon=True).start()
    return icon
