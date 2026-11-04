"""Wrapper for pythonw: redirect stdout/stderr BEFORE any import, log crashes to ~/.editp/app.log."""
import os
import sys
import traceback

_LOG_DIR = os.path.join(os.path.expanduser("~"), ".editp")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_PATH = os.path.join(_LOG_DIR, "app.log")
try:  # keep one old log (app.log.1) so the file never grows without limit
    if os.path.getsize(_LOG_PATH) > 5 * 2**20:
        os.replace(_LOG_PATH, _LOG_PATH + ".1")
except OSError:
    pass  # no log yet, or another running copy holds it open
_log = open(_LOG_PATH, "a", buffering=1, encoding="utf-8")
if sys.stdout is None:
    sys.stdout = _log
if sys.stderr is None:
    sys.stderr = _log
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import app

    app.main()
except SystemExit as e:
    _log.write(f"[launcher] SystemExit: {e}\n")
    raise  # keep the exit code: start.bat reads it from --check
except Exception:
    traceback.print_exc(file=_log)
    sys.exit(1)
finally:
    _log.flush()
