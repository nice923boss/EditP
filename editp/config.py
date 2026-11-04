"""User settings stored in config.json next to the app (kept out of source)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

DEFAULTS = {
    "base_url": "http://127.0.0.1:8080/v1",
    "api_key": "",
    "model": "qwen3.6-35b-a3b",
    "port": 8793,
    "llm_timeout": 180,  # seconds one LLM call may take
}


def _valid(cfg: dict) -> dict:
    """Wrong-typed values (hand-edited file) fall back to the default instead of failing later."""
    out = dict(cfg)
    for k, default in DEFAULTS.items():
        if not isinstance(out.get(k), type(default)) or isinstance(out.get(k), bool):
            out[k] = default
    if not 1024 <= out["port"] <= 65535:
        out["port"] = DEFAULTS["port"]
    if out["llm_timeout"] <= 0:
        out["llm_timeout"] = DEFAULTS["llm_timeout"]
    return out


def load() -> dict:
    if not CONFIG_PATH.exists():
        return dict(DEFAULTS)
    try:
        stored = json.loads(CONFIG_PATH.read_text("utf-8-sig"))
        if not isinstance(stored, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as e:
        bad = CONFIG_PATH.with_suffix(".json.bad")
        print(f"[config] config.json 讀不出來（{e}），已改名為 {bad.name}，改用預設設定", flush=True)
        try:
            CONFIG_PATH.replace(bad)
        except OSError:
            pass
        return dict(DEFAULTS)
    return _valid({**DEFAULTS, **stored})


def save(updates: dict) -> dict:
    """Written to a temp file first, so a crash mid-write never leaves a half config.json."""
    cfg = _valid({**load(), **{k: v for k, v in updates.items() if k in DEFAULTS}})
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")
    os.replace(tmp, CONFIG_PATH)
    return cfg


def masked_key(key: str) -> str:
    if not key:
        return ""
    return key[:6] + "…" + key[-4:] if len(key) > 12 else "…"
