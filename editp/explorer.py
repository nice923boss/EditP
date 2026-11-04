"""Open File Explorer and bring its window to the front (Windows only, plain ctypes).

EditP runs under pythonw with no window of its own, and Windows does not let such a process hand the foreground
to the window it opens: Explorer shows up behind the browser and users think nothing happened."""
import ctypes
import os
import subprocess
import threading
import time
from ctypes import wintypes
from pathlib import Path

EXPLORER_CLASS = "CabinetWClass"
SW_RESTORE = 9
WAIT = 3.0  # how long to look for the window
REUSED_AFTER = 1.0  # no new window by then: Explorer showed the folder in a window that was already open
HOLD = 2.0  # how long to keep the window in front after bringing it there

ENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _api() -> tuple[ctypes.WinDLL, ctypes.WinDLL]:
    """Private DLL handles, so these argtypes never clash with other ctypes users of user32 (pystray is one)."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.EnumWindows.argtypes = [ENUMPROC, wintypes.LPARAM]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD
    return user32, kernel32


def shows(title: str, folder: Path) -> bool:
    """Whether an Explorer window title names this folder: "<name> - 檔案總管", or "<full path> - 檔案總管"
    when Explorer is set to show the full path in the title bar."""
    return any(title == name or title.startswith(f"{name} - ") for name in (folder.name, str(folder)) if name)


def show(path: Path, select: bool = False) -> None:
    """Open a folder in File Explorer (or, with select, the file's folder with the file selected) in front of
    every other window. The window is brought forward on a thread, so this returns at once."""
    user32, kernel32 = _api()
    before = set(_windows(user32))
    if select:
        subprocess.Popen(f'explorer /select,"{path}"')
    else:
        os.startfile(path)
    folder = path.parent if select else path
    threading.Thread(target=_bring_forward, args=(user32, kernel32, before, folder), name="explorer-front",
                     daemon=True).start()


def _windows(user32: ctypes.WinDLL) -> dict[int, str]:
    """The visible File Explorer windows and their titles."""
    found: dict[int, str] = {}
    buf = ctypes.create_unicode_buffer(512)

    def collect(hwnd: int, _: int) -> bool:
        user32.GetClassNameW(hwnd, buf, 512)
        if buf.value == EXPLORER_CLASS and user32.IsWindowVisible(hwnd):
            user32.GetWindowTextW(hwnd, buf, 512)
            found[hwnd] = buf.value
        return True

    user32.EnumWindows(ENUMPROC(collect), 0)
    return found


def _bring_forward(user32: ctypes.WinDLL, kernel32: ctypes.WinDLL, before: set[int], folder: Path) -> None:
    start = time.monotonic()
    while (waited := time.monotonic() - start) < WAIT:
        windows = _windows(user32)
        target = next((hwnd for hwnd in windows if hwnd not in before), None)
        if target is None and waited >= REUSED_AFTER:
            target = next((hwnd for hwnd, title in windows.items() if shows(title, folder)), None)
        if target is not None:
            _hold_front(user32, kernel32, target, folder)
            return
        time.sleep(0.1)
    print(f"[explorer] {WAIT:.0f} 秒內沒看到顯示 {folder} 的檔案總管視窗", flush=True)


def _hold_front(user32: ctypes.WinDLL, kernel32: ctypes.WinDLL, hwnd: int, folder: Path) -> None:
    """Activate the window, then for HOLD seconds activate it again whenever the foreground falls back to the
    window that had it before (the browser): about one time in a dozen, about a second after a new Explorer
    window got the foreground, Windows hands it back. A switch to any other window is left alone."""
    previous = user32.GetForegroundWindow()
    if not _activate(user32, kernel32, hwnd):
        print(f"[explorer] 檔案總管視窗無法移到最上層：{folder}", flush=True)
        return
    end = time.monotonic() + HOLD
    while time.monotonic() < end:
        time.sleep(0.1)
        if previous != hwnd and user32.GetForegroundWindow() == previous:
            print(f"[explorer] 前景被還給原視窗，再移一次：{folder}", flush=True)
            _activate(user32, kernel32, hwnd)


def _activate(user32: ctypes.WinDLL, kernel32: ctypes.WinDLL, hwnd: int) -> bool:
    """SetForegroundWindow is refused to a process that does not own the foreground, so share the foreground
    thread's input state while calling it."""
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    foreground = user32.GetForegroundWindow()
    fg_thread = user32.GetWindowThreadProcessId(foreground, None) if foreground else 0
    me = kernel32.GetCurrentThreadId()
    attached = bool(fg_thread and fg_thread != me and user32.AttachThreadInput(me, fg_thread, True))
    try:
        user32.BringWindowToTop(hwnd)
        return bool(user32.SetForegroundWindow(hwnd))
    finally:
        if attached:
            user32.AttachThreadInput(me, fg_thread, False)
