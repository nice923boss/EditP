"""OpenAI-compatible vision LLM: correct the OCR reading of each line, then turn an instruction into edits.

Tuned for qwen3.6 on llama.cpp: no system prompt and the image first, so the read and plan calls share a
cached prefix; json_schema output is grammar-enforced; thinking is off (temperature 0).
"""
import base64
import io
import json
import re
import time

import httpx
import numpy as np
from PIL import Image

from .geometry import from_norm, region_name, to_norm

MAX_SIDE = 1600
TIMEOUT = 180  # seconds; config.json's llm_timeout overrides it
RETRY_STATUS = (502, 503)  # llama.cpp answers 503 while it is still loading the model


class LLMError(RuntimeError):
    """LLM unreachable or its reply unusable; the message is shown to the user."""


def _headers(cfg: dict) -> dict:
    """No Authorization header without a key: some local servers reject an empty Bearer."""
    return {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}


def _status_error(r: httpx.Response, cfg: dict) -> LLMError:
    if r.status_code in (401, 403):
        return LLMError(f"API Key 錯誤或沒有權限（{r.status_code}），請到「LLM 設定」確認")
    if r.status_code == 404:
        return LLMError(f"找不到 API 路徑（404）：{r.request.url}；Base URL 通常以 /v1 結尾，目前是 {cfg['base_url']}")
    return LLMError(f"LLM 回應錯誤 {r.status_code}：{r.text[:300]}")


def _send(method: str, url: str, cfg: dict, timeout: float, **kw) -> httpx.Response:
    """One request, retried once after 2 s when the connection drops or the server is still starting."""
    for attempt in (1, 2):
        try:
            r = httpx.request(method, url, headers=_headers(cfg), timeout=timeout, **kw)
        except httpx.TimeoutException as e:
            raise LLMError(f"LLM 超過 {timeout:g} 秒沒有回應（{cfg['base_url']}），可能還在載入模型或圖片太大") from e
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            if attempt == 2:
                raise LLMError(f"無法連線到 LLM（{cfg['base_url']}）：{e}；請確認伺服器已開啟、網址正確") from e
        except httpx.HTTPError as e:
            raise LLMError(f"無法連線到 LLM（{cfg['base_url']}）：{e}") from e
        else:
            if r.status_code not in RETRY_STATUS or attempt == 2:
                return r
        time.sleep(2)
    raise AssertionError("unreachable")


def _json(r: httpx.Response):
    try:
        return r.json()
    except ValueError as e:
        raise LLMError(f"LLM 回覆不是有效 JSON：{r.text[:300]}") from e


def encode_image(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb)
    scale = MAX_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_json(text: str) -> dict | list:
    """JSON from a model reply, tolerating code fences and chatter around it."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found")
    return json.JSONDecoder().raw_decode(text[min(starts):])[0]


def chat(cfg: dict, content: list[dict], schema: dict, name: str, max_tokens: int = 1500) -> dict:
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        # llama.cpp emits no reasoning while a json_schema grammar is active; the 'analysis' field replaces it
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
    }
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    timeout = float(cfg.get("llm_timeout") or TIMEOUT)
    r = _send("POST", url, cfg, timeout, json=body)
    if r.status_code == 400:  # servers without chat_template_kwargs / json_schema: plain request
        plain = {k: v for k, v in body.items() if k not in ("chat_template_kwargs", "response_format")}
        r = _send("POST", url, cfg, timeout, json=plain)
    if r.status_code != 200:
        raise _status_error(r, cfg)
    try:
        reply = _json(r)["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise LLMError(f"LLM 回覆格式不符（不是 OpenAI chat 格式）：{r.text[:300]}") from e
    try:
        out = parse_json(reply)
    except ValueError as e:
        raise LLMError(f"LLM 回覆不是有效 JSON：{reply[:300]}") from e
    if not isinstance(out, dict):
        raise LLMError(f"LLM 回覆格式不符：{reply[:300]}")
    return out


def list_models(cfg: dict) -> list[str]:
    """Model ids the server offers; used by the settings dialog to test a connection."""
    if not cfg["base_url"].startswith(("http://", "https://")):
        raise LLMError(f"Base URL 要以 http:// 或 https:// 開頭：{cfg['base_url']}")
    r = _send("GET", cfg["base_url"].rstrip("/") + "/models", cfg, 10)
    if r.status_code != 200:
        raise _status_error(r, cfg)
    data = _json(r)
    return [str(m.get("id")) for m in (data.get("data") if isinstance(data, dict) else None) or []
            if isinstance(m, dict)]


def describe_lines(lines: list[dict], w: int, h: int) -> str:
    return "\n".join(
        f"[{ln['id']}] 位置：{region_name(ln['box'], w, h)}，框：{to_norm(ln['box'], w, h)}，文字：「{ln['text']}」"
        for ln in lines
    )


def layout_map(lines: list[dict], w: int, cols: int = 80) -> str:
    """Lines laid out as text rows (same visual row -> same text row, indented by x position).

    Coordinates alone leave qwen confusing '第二個' with id [2]; a picture-like layout fixes most of that.
    """
    rows: list[list[dict]] = []
    for ln in sorted(lines, key=lambda l: l["box"][1] + l["box"][3]):
        cy, lh = (ln["box"][1] + ln["box"][3]) / 2, ln["box"][3] - ln["box"][1]
        for row in rows:
            first = row[0]["box"]
            row_h = min(r["box"][3] - r["box"][1] for r in row)
            if abs(cy - (first[1] + first[3]) / 2) <= 0.5 * min(lh, row_h):  # a tall title must not swallow rows
                row.append(ln)
                break
        else:
            rows.append([ln])
    out = []
    for items in rows:
        text = ""
        for ln in sorted(items, key=lambda l: l["box"][0]):
            col = round(ln["box"][0] / w * cols)
            text += " " * max(col - len(text), 2 if text else 0) + f"[{ln['id']}]{ln['text']}"
        out.append(text)
    return "\n".join(out)


READ_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"}, "text": {"type": "string"}, "garbled": {"type": "boolean"},
        }, "required": ["id", "text", "garbled"]}},
        "missing": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "garbled": {"type": "boolean"},
            "bbox_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
        }, "required": ["text", "garbled", "bbox_2d"]}},
    },
    "required": ["lines", "missing"],
}

READ_PROMPT = """這張圖片裡的文字由 OCR 偵測，結果如下（框座標 x1,y1,x2,y2 是 0 到 1000 的相對位置）：
{lines}

OCR 常把 AI 生成圖片裡變形的字讀錯。請對照圖片，逐行寫出每個框裡圖上實際寫的字：
- 照圖上原樣抄寫，圖上本來就寫錯的字也照抄，不要修正、不要潤飾。
- 不要合併或拆分行，每個編號回一筆。
- 框裡是看不懂的亂碼、假字或根本不是文字時，garbled 設 true，text 寫看起來最接近的字。

OCR 也可能漏掉文字，尤其是小字、淡色字、亂碼。圖上看得到、但不在上面清單裡的文字，每一行放進 missing：
text 照圖抄寫，bbox_2d 是那一行文字的框 [x1,y1,x2,y2]（0 到 1000 相對座標）。沒有漏掉就回空陣列。"""


def read(cfg: dict, rgb: np.ndarray, lines: list[dict]) -> list[dict]:
    """OCR lines with 'text' replaced by what the LLM sees in the image; adds 'ocr_text' and 'garbled'."""
    if not lines:
        return []
    h, w = rgb.shape[:2]
    content = [
        {"type": "image_url", "image_url": {"url": encode_image(rgb)}},
        {"type": "text", "text": READ_PROMPT.format(lines=describe_lines(lines, w, h))},
    ]
    reply = chat(cfg, content, READ_SCHEMA, "read")
    got = {e.get("id"): e for e in reply.get("lines") or [] if isinstance(e, dict)}
    out = []
    for ln in lines:
        e = got.get(ln["id"], {})
        text = str(e.get("text") or "").strip() or ln["text"]
        out.append({**ln, "ocr_text": ln.get("ocr_text", ln["text"]), "text": text, "garbled": bool(e.get("garbled"))})
    for e in reply.get("missing") or []:
        text, bbox = str(e.get("text") or "").strip(), e.get("bbox_2d")
        if not text or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        box = from_norm(bbox, w, h)
        if box[2] - box[0] < 4 or box[3] - box[1] < 4 or any(_overlaps(box, ln["box"]) for ln in out):
            continue
        out.append({"id": len(out), "text": text, "score": 0.0, "box": box, "ocr_text": "", "garbled": bool(e.get("garbled"))})
    return out


def _overlaps(a: list[int], b: list[int]) -> bool:
    """True when the boxes share over a third of the smaller one (the model re-listed a known line)."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return False
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih > smaller / 3


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        # scratchpad written before the answer; maxLength caps the repetition loops qwen can fall into
        "analysis": {"type": "string", "maxLength": 300},
        "found": {"type": "boolean"},
        "edits": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"}, "new_text": {"type": "string"},
        }, "required": ["id", "new_text"]}},
    },
    "required": ["analysis", "found", "edits"],
}

PLAN_PROMPT = """這張圖片上的文字行如下（編號、位置、框座標 x1,y1,x2,y2 為 0 到 1000、目前文字）：
{lines}

版面示意（照圖上位置排列，同一高度的放同一列，左右距離照比例）：
{layout}

使用者的修改指令：「{instruction}」{hint}

請找出指令要修改的行，回傳每一行修改後的完整新內容。規則：
1. 只列出需要修改的行，沒被指令影響的行不要列。
2. new_text 是整行改完後的完整文字，不是只寫被改的字；該行其餘的字一字不差保留。
   每個編號是圖上一個獨立的文字框，不要把別行的文字搬進來或合併成一行。
3. 要刪除整行時，new_text 給空字串。
4. 只做指令要求的修改。指令沒要求修正錯字時，就算看到錯字也不要改。
5. 指令用位置描述（右上角、第二行、左欄、最下面）時，看版面示意與框座標判斷：y 越小越上面，x 越小越左邊。
   「第幾個、第幾項、第幾行」指圖上由上往下（或由左往右）的排列順序，不是編號 [ ] 裡的數字。
   標題、欄位名稱通常不算在項目裡。
6. 指令說的文字或目標在清單裡不存在時，edits 回空陣列，不要改別的行來代替。
7. 指令寫「電話改成…」「營業時間改成…」這類「欄位名稱改成新內容」時，改的是欄位後面的內容，欄位名稱保留。
8. 看不出意思的亂碼行不算錯字；除非指令明說要處理亂碼，否則不要動它。
作答順序：
- analysis：先依圖上順序（由上到下、由左到右）寫出指令提到的區域裡每一行的 [編號]文字，再說要改哪幾行。
- found：圖上有沒有指令要改的目標，沒有就設 false。
- edits：要改的行。"""

_SEP = re.compile(r"[，,；;。\n]|還有|以及|並且|然後")
_REPLACE = re.compile(
    r"(?:請)?(?:把|將)?\s*[「『\"']?(?P<old>[^「」『』\"']+?)[」』\"']?\s*"
    r"(?:改成|改為|換成|替換成|替換為|更正為|更正成|→|->|=>)\s*"
    r"[「『\"']?(?P<new>[^「」『』\"']+?)[」』\"']?"
)


def explicit_replacements(instruction: str) -> list[tuple[str, str]] | None:
    """[(old, new)] when the whole instruction is literal 'A改成B' clauses, else None."""
    pairs = []
    for part in _SEP.split(instruction):
        part = part.strip()
        if not part:
            continue
        m = _REPLACE.fullmatch(part)
        if not m:
            return None
        pairs.append((m["old"].strip(), m["new"].strip()))
    return pairs or None


def replace_in_lines(lines: list[dict], pairs: list[tuple[str, str]]) -> list[dict] | None:
    """Edits from literal replacements; None when the LLM must decide.

    That is when some 'old' text is on no line, or is a field label ('電話：…'), where '電話改成X' means the value.
    """
    texts = {ln["id"]: ln["text"] for ln in lines}
    new = dict(texts)
    for old, repl in pairs:
        hits = [i for i, t in new.items() if old in t]
        if not hits or any(re.search(re.escape(old) + r"\s*[：:]", new[i]) for i in hits):
            return None
        for i in hits:
            new[i] = new[i].replace(old, repl)
    return [{"id": i, "new_text": t} for i, t in new.items() if t != texts[i]]


def normalize(edits: list, lines: list[dict]) -> list[dict]:
    """Keep edits that name a known line and actually change it; the last edit of a line wins."""
    texts = {ln["id"]: ln["text"] for ln in lines}
    out: dict[int, str] = {}
    for e in edits:
        if not isinstance(e, dict) or e.get("id") not in texts or not isinstance(e.get("new_text"), str):
            continue
        new_text = e["new_text"].strip()
        if new_text != texts[e["id"]]:
            out[e["id"]] = new_text
    return [{"id": i, "new_text": t} for i, t in out.items()]


def plan(cfg: dict, rgb: np.ndarray, lines: list[dict], instruction: str) -> dict:
    """{edits: [{id, new_text}], reason, source}. lines should already be corrected by read()."""
    pairs = explicit_replacements(instruction) or []
    if pairs:
        edits = replace_in_lines(lines, pairs)
        if edits is not None:
            return {"edits": edits, "reason": "指令是明確的文字替換，直接套用", "source": "rule"}
    # the model tends to invent a target for text that is not there; tell it which quoted words are absent
    absent = [old for old, _ in pairs if not any(old in ln["text"] for ln in lines)]
    hint = "".join(
        f"\n注意：「{old}」這幾個字不在任何一行裡。如果指令是要替換這幾個字，found 設 false；"
        "只有它是在描述某一行（例如「標題」「店名」「第三行」）時才修改。"
        for old in absent
    )
    h, w = rgb.shape[:2]
    prompt = PLAN_PROMPT.format(
        lines=describe_lines(lines, w, h), layout=layout_map(lines, w), instruction=instruction, hint=hint
    )
    content = [{"type": "image_url", "image_url": {"url": encode_image(rgb)}}, {"type": "text", "text": prompt}]
    out = chat(cfg, content, PLAN_SCHEMA, "plan")
    edits = normalize(out.get("edits") or [], lines) if out.get("found", True) else []
    return {"edits": edits, "reason": clean_reason(str(out.get("analysis", ""))), "source": "llm"}


_RULE_REF = re.compile(r"[（(]?(?:根據|依照|依據|按照|符合|違反)?\s*規則\s*[0-9０-９一二三四五六七八九十]+\s*[)）]?\s*[，,、：:]?\s*")


def clean_reason(text: str) -> str:
    """The model cites the prompt's rule numbers ('根據規則8'), which mean nothing to the user: drop them."""
    return _RULE_REF.sub("", text).strip()
