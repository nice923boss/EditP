import json

import httpx
import pytest

from editp import config, llm

CFG = {"base_url": "http://llm.local/v1", "api_key": "", "model": "m", "llm_timeout": 180}
REPLY = {"choices": [{"message": {"content": '{"ok": true}'}}]}


@pytest.fixture
def server(monkeypatch):
    """Replies are taken in order from server.replies (a Response, a dict for 200 JSON, or an exception)."""
    calls, replies = [], []

    def fake(method, url, headers=None, timeout=None, **kw):
        calls.append({"method": method, "url": url, "headers": headers})
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        if isinstance(r, dict):
            r = httpx.Response(200, json=r)
        r.request = httpx.Request(method, url)
        return r

    monkeypatch.setattr(llm.httpx, "request", fake)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    server.calls, server.replies = calls, replies
    return server


def test_no_key_sends_no_authorization(server):
    server.replies.append(REPLY)
    assert llm.chat(CFG, [], {}, "t") == {"ok": True}
    assert server.calls[0]["headers"] == {}
    server.replies.append(REPLY)
    llm.chat({**CFG, "api_key": "k"}, [], {}, "t")
    assert server.calls[1]["headers"] == {"Authorization": "Bearer k"}


def test_wrong_key_and_wrong_path_get_plain_messages(server):
    server.replies.append(httpx.Response(401, text="no"))
    with pytest.raises(llm.LLMError, match="API Key"):
        llm.chat(CFG, [], {}, "t")
    server.replies.append(httpx.Response(404, text="no"))
    with pytest.raises(llm.LLMError, match="/v1 結尾"):
        llm.list_models(CFG)


def test_retries_once_while_the_server_starts(server):
    server.replies.extend([httpx.Response(503, text="loading"), REPLY])
    assert llm.chat(CFG, [], {}, "t") == {"ok": True}
    server.replies.extend([httpx.ConnectError("refused"), httpx.ConnectError("refused")])
    with pytest.raises(llm.LLMError, match="無法連線"):
        llm.chat(CFG, [], {}, "t")
    assert len(server.calls) == 4


def test_timeout_says_how_long_it_waited(server):
    server.replies.append(httpx.ReadTimeout("slow"))
    with pytest.raises(llm.LLMError, match="超過 30 秒"):
        llm.chat({**CFG, "llm_timeout": 30}, [], {}, "t")


def test_unusable_replies_raise_llm_errors(server):
    server.replies.append(httpx.Response(200, text="<html>"))
    with pytest.raises(llm.LLMError, match="不是有效 JSON"):
        llm.chat(CFG, [], {}, "t")
    server.replies.append({"error": "x"})
    with pytest.raises(llm.LLMError, match="格式不符"):
        llm.chat(CFG, [], {}, "t")
    with pytest.raises(llm.LLMError, match="http://"):
        llm.list_models({**CFG, "base_url": "llm.local/v1"})


def test_clean_reason_drops_rule_numbers():
    assert llm.clean_reason("根據規則8，亂碼行不算錯字。") == "亂碼行不算錯字。"
    assert llm.clean_reason("標題要改（規則 2）並保留其餘字") == "標題要改並保留其餘字"
    assert llm.clean_reason("沒有規則可言") == "沒有規則可言"


def test_corrupt_config_falls_back_and_is_kept_aside(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    (tmp_path / "config.json").write_text("{broken", "utf-8")
    assert config.load() == config.DEFAULTS
    assert (tmp_path / "config.json.bad").read_text("utf-8") == "{broken"


def test_config_fixes_wrong_values_and_saves_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    (tmp_path / "config.json").write_text(json.dumps({"port": "80", "llm_timeout": -1, "model": "x"}), "utf-8")
    cfg = config.load()
    assert cfg["port"] == 8793 and cfg["llm_timeout"] == 180 and cfg["model"] == "x"
    config.save({"port": 9000, "unknown": 1})
    assert json.loads((tmp_path / "config.json").read_text("utf-8"))["port"] == 9000
    assert "unknown" not in config.load()
    assert not list(tmp_path.glob("*.tmp"))
