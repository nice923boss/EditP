// EditP web UI: open an image, point arrows at text lines, edit each line's new text, apply them all in one pass.
import { createHistory } from "./history.js";
import { createOverlay } from "./overlay.js";
import { loadList, saveList } from "./storage.js";
import { setupUpload, uploadFile } from "./upload.js";

const $ = (sel) => document.querySelector(sel);
const el = {
  viewport: $("#viewport"), stage: $("#stage"), img: $("#img"), svg: $("#overlay"), labels: $("#labels"),
  empty: $("#empty"), status: $("#status"), path: $("#path"), items: $("#items"), count: $("#count"),
  listEmpty: $("#list-empty"), instruction: $("#instruction"), reason: $("#plan-reason"),
  apply: $("#btn-apply"), hint: $("#tool-hint"), undo: $("#btn-undo"), redo: $("#btn-redo"), engine: $("#engine"),
  close: $("#btn-close"),
};
const history = createHistory();

const S = {
  gen: 0, // bumps on every open; responses for an older image are dropped
  image: null, // {path, name, width, height}
  lines: [], // OCR lines, text corrected by the LLM once read is true
  read: false,
  items: [], // {key, lineId, box, oldText, newText, touched, garbled, tail, tip, style, styleEdited}
  nextKey: 1,
  focusKey: null,
  editSeq: 0, // bumps on every change to the list; appliedSeq is the edit the shown result was rendered from
  appliedSeq: 0,
  hasResult: false,
  version: 0,
  savedVersion: 0, // the render last written to a file; leaving the page with a newer one asks first
  applying: null,
  applyQueued: false,
  restoring: false, // an undo/redo is swapping the shown render
  tool: "arrow",
  view: "result",
  zoom: "fit",
};

const HINTS = {
  arrow: "在文字上點一下，或從標籤位置拖曳到文字上；每支箭頭的標籤就是該行的新內容",
  box: "拖曳框出要改的文字範圍（OCR 沒抓到的字用這個）",
};

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? `HTTP ${res.status}`);
    throw new Error(detail);
  }
  return data;
}

function setStatus(text, kind = "") {
  el.status.textContent = text;
  el.status.className = `status ${kind}`;
  el.status.title = text;
}

let busyTimer = null;

// a busy status counts the seconds, so a slow LLM or CPU erase does not look frozen; returns the stopper
function busy(label) {
  const start = Date.now();
  clearInterval(busyTimer);
  setStatus(label, "busy");
  const timer = setInterval(() => setStatus(`${label}（${Math.round((Date.now() - start) / 1000)} 秒）`, "busy"), 1000);
  busyTimer = timer;
  return () => { clearInterval(timer); };
}

async function task(label, fn) {
  const stop = busy(label);
  try {
    return await fn();
  } catch (e) {
    setStatus(e.message, "error");
    throw e;
  } finally {
    stop();
  }
}

const dirty = () => S.editSeq !== S.appliedSeq;
const itemByKey = (key) => S.items.find((it) => it.key === key);

function touch() {
  S.editSeq += 1;
  el.apply.classList.toggle("dirty", dirty());
}

// ---- undo / redo -------------------------------------------------------------------------------------------

const snapshot = (label, tag = "") => ({
  label, tag, items: structuredClone(S.items), version: S.version, hasResult: S.hasResult, dirty: dirty(),
});

function syncUndo() {
  const u = history.nextUndo();
  const r = history.nextRedo();
  el.undo.disabled = !S.image || !u;
  el.redo.disabled = !S.image || !r;
  el.undo.title = u ? `復原：${u.label}（Ctrl+Z）` : "沒有可以復原的步驟（Ctrl+Z）";
  el.redo.title = r ? `重做：${r.label}（Ctrl+Y）` : "沒有可以重做的步驟（Ctrl+Y）";
}

function record(label, tag = "") {
  history.push(snapshot(label, tag));
  syncUndo();
}

async function restoreStep(step) {
  clearTimeout(styleTimer);
  const gen = S.gen;
  S.items = structuredClone(step.items);
  if (!itemByKey(S.focusKey)) S.focusKey = null;
  S.editSeq += 1;
  if (!step.dirty) S.appliedSeq = S.editSeq;
  if (step.version !== S.version) {
    S.restoring = true;
    try {
      await api("/api/restore", { version: step.version });
      S.version = step.version;
      S.hasResult = step.hasResult;
    } catch {
      // that render was dropped from memory: show the original and mark the list as not applied
      await api("/api/restore", { version: 0 }).catch(() => {});
      S.version = 0;
      S.hasResult = false;
      S.appliedSeq = -1;
    } finally {
      S.restoring = false;
    }
  }
  if (gen !== S.gen) return;
  overlay.draw();
  syncPanel();
  refreshImage();
  syncUndo();
}

async function travel(redo) {
  if (!S.image) return;
  if (S.applying || S.restoring) return setStatus("還在處理上一個動作，請稍候再試", "error");
  const moved = redo ? history.redo() : history.undo();
  if (!moved) return setStatus(redo ? "沒有可以重做的步驟" : "沒有可以復原的步驟");
  await restoreStep(moved.step);
  setStatus(redo ? `已重做：${moved.step.label}` : `已復原：${moved.undone.label}`);
}

// ---- image -------------------------------------------------------------------------------------------------

function loadImage(data) {
  keepNow(); // the previous image's last edits
  S.gen += 1;
  S.image = { path: data.path, name: data.name, width: data.width, height: data.height, key: data.key };
  S.lines = data.lines;
  S.read = data.read;
  S.items = [];
  S.focusKey = null;
  S.editSeq = S.appliedSeq = 0;
  S.hasResult = false;
  S.version = S.savedVersion = 0;
  el.path.value = data.path;
  el.close.disabled = false;
  el.reason.hidden = true;
  el.empty.hidden = true;
  el.stage.hidden = false;
  overlay.clear();
  refreshImage();
  overlay.layout();
  syncPanel();
  history.reset(snapshot("開啟圖片"));
  const kept = restoreList(data.key);
  syncUndo();
  document.title = `${data.name} - EditP`;
  if (!S.read) readLines();
  else setStatus(`已開啟 ${data.name}，共 ${S.lines.length} 行文字${kept ? `；已帶回上次的清單 ${kept} 項（Ctrl+Z 可撤回）` : ""}`);
}

// 清空: back to the empty page; the server forgets the image too, so a reload does not bring it back
async function closeImage() {
  if (!S.image) return;
  if (S.applying || S.restoring) return setStatus("還在處理上一個動作，請稍候再試", "error");
  if (S.hasResult && S.version !== S.savedVersion && !confirm("套用後的圖還沒存檔，確定要清空嗎？")) return;
  try {
    await api("/api/close", {});
  } catch (e) {
    return setStatus(`清空失敗：${e.message}`, "error");
  }
  keepNow(); // this image's list stays remembered for when it is opened again
  clearTimeout(styleTimer);
  S.gen += 1;
  S.image = null;
  S.lines = [];
  S.items = [];
  S.focusKey = null;
  S.editSeq = S.appliedSeq = 0;
  S.hasResult = false;
  S.version = S.savedVersion = 0;
  el.path.value = "";
  el.close.disabled = true;
  el.reason.hidden = true;
  el.stage.hidden = true;
  el.empty.hidden = false;
  el.img.removeAttribute("src");
  overlay.clear();
  syncPanel();
  syncUndo();
  document.title = "EditP 圖片錯字修正";
  clearInterval(busyTimer); // an AI read still running would keep counting over this message
  setStatus("已清空，尚未開啟圖片");
}

// the list this image had when the page was last closed comes back as one undoable step
function restoreList(key) {
  const items = loadList(key);
  if (!items) return 0;
  S.items = items;
  S.nextKey = Math.max(S.nextKey, ...items.map((it) => it.key + 1));
  touch();
  record(`帶回上次的清單 ${items.length} 項`);
  overlay.draw();
  syncPanel();
  return items.length;
}

function refreshImage() {
  const showResult = S.view === "result" && S.hasResult;
  el.img.src = showResult ? `/api/image?which=result&v=${S.version}` : `/api/image?which=original&g=${S.gen}`;
}

async function openImage(path) {
  const data = await task("開啟並辨識文字中…", () => api("/api/open", { path: path || null })).catch(() => null);
  if (!data) return;
  if (data.cancelled) {
    setStatus(S.image ? `已開啟 ${S.image.name}` : "尚未開啟圖片");
    return;
  }
  loadImage(data);
}

async function openUpload(file, name) {
  const data = await task("上傳並辨識文字中…", () => uploadFile(file, name)).catch(() => null);
  if (data) loadImage(data);
}

// the LLM read replaces OCR text with what is really drawn; untouched items follow it
function readItem(it, lines) {
  const ln = lines.find((l) => l.id === it.lineId);
  if (!ln) return it;
  return { ...it, oldText: ln.text, garbled: ln.garbled, newText: it.touched ? it.newText : ln.text };
}

function mergeLines(lines) {
  S.lines = lines;
  S.read = true;
  S.items = S.items.map((it) => readItem(it, lines));
  history.patch((step) => ({ ...step, items: step.items.map((it) => readItem(it, lines)) }));
  overlay.draw();
  syncPanel();
}

async function readLines() {
  const gen = S.gen;
  const stop = busy(`OCR 找到 ${S.lines.length} 行，AI 校對文字中（通常 10 到 20 秒）…`);
  try {
    const data = await api("/api/read", {});
    if (gen !== S.gen) return;
    mergeLines(data.lines);
    const garbled = data.lines.filter((l) => l.garbled).length;
    setStatus(`AI 校對完成：${data.lines.length} 行文字${garbled ? `，疑似亂碼 ${garbled} 行（橘框）` : ""}`);
  } catch (e) {
    if (gen === S.gen) setStatus(`AI 校對失敗（仍可用 OCR 結果操作）：${e.message}`, "error");
  } finally {
    stop();
  }
}

// ---- items -------------------------------------------------------------------------------------------------

// share of box a ([x1, y1, x2, y2]) that b covers
function overlap(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  const area = (a[2] - a[0]) * (a[3] - a[1]);
  return w > 0 && h > 0 && area > 0 ? (w * h) / area : 0;
}

function addItem({ lineId = null, box, text, garbled = false, tail = null, tip = null }) {
  const at = tail || overlay.placeTail(box);
  const it = {
    key: S.nextKey++, lineId, box, oldText: text, newText: text, touched: false, garbled,
    tail: at, tip: tip || overlay.tipToward(box, at), style: null, styleEdited: false,
  };
  S.items.push(it);
  touch();
  return it;
}

function lineItem(line, opts = {}) {
  return S.items.find((it) => it.lineId === line.id)
    || addItem({ lineId: line.id, box: line.box, text: line.text, garbled: line.garbled, ...opts });
}

function removeItem(key) {
  const n = S.items.findIndex((it) => it.key === key) + 1;
  S.items = S.items.filter((it) => it.key !== key);
  if (S.focusKey === key) S.focusKey = null;
  touch();
  record(`移除第 ${n} 項`);
  overlay.draw();
  syncPanel();
}

function focusItem(key, { label = false } = {}) {
  S.focusKey = key;
  overlay.draw();
  syncPanel();
  el.items.querySelector(`[data-key="${key}"]`)?.scrollIntoView({ block: "nearest" });
  if (label) overlay.focusLabel(key);
}

function setText(key, value) {
  const it = itemByKey(key);
  if (!it) return;
  it.newText = value;
  it.touched = true;
  touch();
  record(`修改第 ${S.items.indexOf(it) + 1} 項文字`, `text:${key}`);
  overlay.draw();
  syncPanel();
}

const overlay = createOverlay(el, S, {
  onArrow({ tip, tail, line }) {
    if (!line) {
      setStatus("箭頭尖端沒有指到偵測到的文字行；OCR 沒抓到的字請改用「框選」", "error");
      return;
    }
    const existing = S.items.find((it) => it.lineId === line.id);
    if (existing) {
      focusItem(existing.key, { label: true });
      setStatus(`這一行已經是第 ${S.items.indexOf(existing) + 1} 項`);
      return;
    }
    const it = lineItem(line, { tail, tip });
    record(`加入第 ${S.items.length} 項`);
    focusItem(it.key, { label: true });
    setStatus(`已加入第 ${S.items.length} 項：「${line.text}」，在標籤裡改成正確內容`);
  },
  async onBox(box) {
    const gen = S.gen;
    const data = await task("辨識框選範圍的文字中…", () => api("/api/region_text", { box })).catch(() => null);
    if (!data || gen !== S.gen) return;
    const twin = S.items.find((other) => overlap(data.box, other.box) > 0.5);
    const it = addItem({ box: data.box, text: data.text });
    record(`框選第 ${S.items.length} 項`);
    focusItem(it.key, { label: true });
    if (twin) {
      setStatus(`這個框和第 ${S.items.indexOf(twin) + 1} 項大半重疊，兩項都會擦除重畫；不需要的話按 ✕ 移除一項`, "error");
    } else {
      setStatus(data.text ? `已加入第 ${S.items.length} 項：「${data.text}」` : "框選範圍讀不到字，請直接在標籤輸入新內容");
    }
  },
  onText: setText,
  onFocus(key) {
    if (S.focusKey === key) return;
    S.focusKey = key;
    overlay.draw();
    syncPanel();
  },
});

// ---- batch list panel --------------------------------------------------------------------------------------

const getPath = (obj, path) => path.split(".").reduce((o, k) => o?.[k], obj);

function setPath(obj, path, value) {
  const [head, ...rest] = path.split(".");
  return rest.length ? { ...obj, [head]: setPath(obj[head] || {}, rest.join("."), value) } : { ...obj, [head]: value };
}

let styleTimer = null;

function onStyleInput(key, field) {
  const it = itemByKey(key);
  if (!it?.style) return;
  const raw = field.value;
  const value = field.type === "number" ? Number(raw) : raw;
  if (field.type === "number" && (raw === "" || Number.isNaN(value))) return;
  it.style = setPath(it.style, field.dataset.k, value);
  it.styleEdited = true;
  touch();
  record(`調整第 ${S.items.indexOf(it) + 1} 項樣式`, `style:${key}`);
  clearTimeout(styleTimer);
  if (S.hasResult) styleTimer = setTimeout(() => applyAll({ amend: true }), 500);
}

function makeRow(key) {
  const li = $("#item-tpl").content.firstElementChild.cloneNode(true);
  li.dataset.key = key;
  li.addEventListener("click", (ev) => { if (!ev.target.closest(".remove")) focusItem(key); });
  li.querySelector(".remove").addEventListener("click", () => removeItem(key));
  const input = li.querySelector(".new-text");
  input.addEventListener("input", () => setText(key, input.value));
  for (const field of li.querySelectorAll("[data-k]")) {
    field.addEventListener(field.tagName === "SELECT" || field.type === "color" ? "change" : "input",
      () => onStyleInput(key, field));
  }
  return li;
}

let keepTimer = null;
let keepNow = () => {}; // writes the list save still waiting out the typing

function keepList() {
  clearTimeout(keepTimer);
  const key = S.image?.key;
  const items = S.items;
  keepNow = () => {
    clearTimeout(keepTimer);
    keepNow = () => {};
    saveList(key, items);
  };
  keepTimer = setTimeout(keepNow, 400);
}

function syncPanel() {
  keepList();
  const rows = new Map([...el.items.children].map((li) => [Number(li.dataset.key), li]));
  for (const [key, li] of rows) if (!itemByKey(key)) li.remove();
  S.items.forEach((it, i) => {
    let li = rows.get(it.key);
    if (!li) li = makeRow(it.key);
    if (el.items.children[i] !== li) el.items.insertBefore(li, el.items.children[i] || null);
    li.classList.toggle("focused", it.key === S.focusKey);
    li.querySelector(".badge").textContent = i + 1;
    const old = li.querySelector(".old");
    old.textContent = it.lineId == null ? `框選：${it.oldText || "（讀不到字）"}` : `原文：${it.oldText}`;
    old.title = it.oldText;
    li.querySelector(".garbled").hidden = !it.garbled;
    const input = li.querySelector(".new-text");
    if (document.activeElement !== input && input.value !== it.newText) input.value = it.newText;
    input.classList.toggle("deleting", it.newText.trim() === "");
    for (const field of li.querySelectorAll("[data-k]")) {
      field.disabled = !it.style;
      const v = it.style ? getPath(it.style, field.dataset.k) : undefined;
      if (v !== undefined && document.activeElement !== field) field.value = v;
    }
    li.querySelector(".style-note").hidden = Boolean(it.style);
  });
  el.count.textContent = S.items.length;
  el.listEmpty.hidden = S.items.length > 0;
  el.apply.classList.toggle("dirty", dirty());
}

// ---- apply / save ------------------------------------------------------------------------------------------

function applyAll({ amend = false } = {}) {
  if (!S.image) return Promise.resolve();
  if (S.applying) {
    S.applyQueued = true;
    return S.applying;
  }
  const gen = S.gen;
  const seq = S.editSeq;
  const sent = S.items.map((it) => it.key);
  const edits = S.items.map((it) => ({
    box: it.box, old_text: it.oldText, new_text: it.newText, ...(it.styleEdited ? { style: it.style } : {}),
  }));
  const label = edits.length ? `擦除並重繪 ${edits.length} 行中…` : "清單是空的，還原原圖…";
  S.applying = (async () => {
    try {
      const data = await task(label, () => api("/api/render", { edits }));
      if (gen !== S.gen) return;
      data.styles.forEach((st, i) => {
        const it = itemByKey(sent[i]);
        if (it && !(it.styleEdited && dirty())) it.style = st;
      });
      S.appliedSeq = seq;
      S.hasResult = edits.length > 0;
      S.version = data.version;
      if (S.hasResult) setView("result");
      refreshImage();
      syncPanel();
      // a style tweak re-render belongs to the tweak's own step; a click on 全部套用 is a step of its own
      if (amend) history.amend(snapshot("")); else record(edits.length ? `全部套用 ${edits.length} 項` : "還原原圖");
      syncUndo();
      setStatus(edits.length ? `已套用 ${edits.length} 項修改（Ctrl+Z 可回到上一步）` : "已還原原圖");
    } catch {
      // task() already shows the error
    } finally {
      S.applying = null;
      if (S.applyQueued) {
        S.applyQueued = false;
        applyAll();
      }
    }
  })();
  return S.applying;
}

const PICKER_TYPES = [
  { description: "PNG 圖片", accept: { "image/png": [".png"] } },
  { description: "JPEG 圖片", accept: { "image/jpeg": [".jpg", ".jpeg"] } },
  { description: "WebP 圖片", accept: { "image/webp": [".webp"] } },
];
const stem = (name) => name.replace(/\.[^.]+$/, "");

function canSave() {
  if (!S.image) return setStatus("尚未開啟圖片", "error"), false;
  if (!S.items.length) return setStatus("清單是空的，沒有要存的修改", "error"), false;
  return true;
}

async function ensureApplied() {
  for (let i = 0; i < 3 && (S.applying || dirty() || !S.hasResult); i++) await (S.applying || applyAll());
  return !dirty() && S.hasResult;
}

// 另存新檔: write <name>_edited.png next to the original, then show where it went
async function saveBeside() {
  if (!canSave() || !(await ensureApplied())) return;
  const data = await task("存檔中…", () => api("/api/save", {})).catch(() => null);
  if (!data) return;
  S.savedVersion = S.version;
  setStatus(`已另存：${data.saved}`);
  showSaved({ name: data.name, path: data.saved, where: data.folder });
}

// 另存到…: the browser's own save dialog (asked for first, while the click still counts as a user action)
async function saveTo() {
  if (!canSave()) return;
  let handle = null;
  if (window.showSaveFilePicker) {
    try {
      handle = await window.showSaveFilePicker({
        suggestedName: `${stem(S.image.name)}_edited.png`, id: "editp-save", startIn: "pictures", types: PICKER_TYPES,
      });
    } catch (e) {
      if (e.name === "AbortError") return setStatus("已取消存檔");
      handle = null; // picker refused (policy or old browser): fall back to a normal download
    }
  }
  if (!(await ensureApplied())) return;
  const ext = (handle?.name.match(/\.(png|jpe?g|webp)$/i)?.[1] || "png").toLowerCase().replace("jpeg", "jpg");
  if (!handle) {
    downloadResult(ext);
    S.savedVersion = S.version;
    setStatus("已交給瀏覽器下載，檔案在瀏覽器的下載資料夾");
    return showSaved({ name: `${stem(S.image.name)}_edited.${ext}`, path: null, where: "瀏覽器的下載資料夾" });
  }
  try {
    await task("存檔中…", async () => {
      const res = await fetch(`/api/download?fmt=${ext}`);
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      const out = await handle.createWritable();
      await out.write(await res.blob());
      await out.close();
    });
  } catch {
    return; // task() already shows the error
  }
  S.savedVersion = S.version;
  setStatus(`已另存：${handle.name}`);
  showSaved({ name: handle.name, path: null, where: "你在存檔視窗選的資料夾" });
}

function downloadResult(fmt) {
  const a = document.createElement("a");
  a.href = `/api/download?fmt=${fmt}`;
  a.download = `${stem(S.image.name)}_edited.${fmt}`;
  document.body.append(a);
  a.click();
  a.remove();
}

const savedDlg = $("#saved-dlg");
let savedPath = null;

function showSaved({ name, path, where }) {
  savedPath = path;
  $("#saved-img").src = `/api/image?which=result&v=${S.version}`;
  $("#saved-name").textContent = name;
  $("#saved-where").textContent = path || where;
  $("#saved-reveal").hidden = !path;
  $("#saved-copy").hidden = !path;
  $("#saved-msg").textContent = "";
  savedDlg.showModal();
}

$("#saved-download").addEventListener("click", () => downloadResult($("#saved-fmt").value));
$("#saved-reveal").addEventListener("click", () => api("/api/reveal", { path: savedPath })
  .catch((e) => { $("#saved-msg").textContent = e.message; }));
$("#saved-copy").addEventListener("click", () => navigator.clipboard.writeText(savedPath)
  .then(() => { $("#saved-msg").textContent = "已複製路徑"; })
  .catch(() => { $("#saved-msg").textContent = "瀏覽器不允許複製，請手動選取上面的路徑"; }));
$("#saved-close").addEventListener("click", () => savedDlg.close());

// ---- AI instruction ----------------------------------------------------------------------------------------

async function runPlan() {
  const instruction = el.instruction.value.trim();
  if (!S.image) return setStatus("尚未開啟圖片", "error");
  if (!instruction) return setStatus("請先輸入修改指令", "error");
  const gen = S.gen;
  const btn = $("#btn-plan");
  btn.disabled = true;
  try {
    const data = await task("AI 解析指令中…", () => api("/api/plan", { instruction }));
    if (gen !== S.gen) return;
    mergeLines(data.lines);
    for (const e of data.edits) {
      const line = S.lines.find((l) => l.id === e.id);
      if (!line) continue;
      const it = lineItem(line);
      it.newText = e.new_text;
      it.touched = true;
    }
    touch();
    if (data.edits.length) record(`AI 解析加入 ${data.edits.length} 項`);
    overlay.draw();
    syncPanel();
    el.reason.hidden = !data.reason;
    el.reason.textContent = data.source === "rule" ? data.reason : `AI 判讀：${data.reason}`;
    setStatus(data.edits.length
      ? `AI 找到 ${data.edits.length} 處要改，已加入清單；確認後按「全部套用」`
      : "AI 沒有找到要改的地方，可改用箭頭指定");
  } catch {
    // task() already shows the error
  } finally {
    btn.disabled = false;
  }
}

// ---- settings dialog ---------------------------------------------------------------------------------------

const dlg = $("#settings");
const cfgBody = () => ({ base_url: $("#cfg-base").value, model: $("#cfg-model").value, api_key: $("#cfg-key").value });

async function openSettings() {
  const c = await api("/api/config");
  $("#cfg-base").value = c.base_url;
  $("#cfg-model").value = c.model;
  $("#cfg-key").value = "";
  $("#cfg-key").placeholder = c.api_key ? `已設定 ${c.api_key}，留空表示不變更` : "尚未設定";
  $("#cfg-msg").textContent = "";
  dlg.showModal();
}

$("#cfg-show").addEventListener("change", (ev) => { $("#cfg-key").type = ev.target.checked ? "text" : "password"; });
$("#cfg-cancel").addEventListener("click", () => dlg.close());
$("#cfg-test").addEventListener("click", async () => {
  const msg = $("#cfg-msg");
  msg.textContent = "連線測試中…";
  try {
    const r = await api("/api/config/test", cfgBody());
    const model = $("#cfg-model").value;
    msg.textContent = `連線成功，伺服器提供 ${r.models.length} 個模型；`
      + (r.model_found ? `找到 ${model}` : `沒有 ${model}，請確認模型名稱（可用：${r.models.join("、") || "無"}）`);
  } catch (e) {
    msg.textContent = `連線失敗：${e.message}`;
  }
});
$("#cfg-save").addEventListener("click", async () => {
  try {
    await api("/api/config", cfgBody());
    dlg.close();
    setStatus("LLM 設定已儲存");
  } catch (e) {
    $("#cfg-msg").textContent = `儲存失敗：${e.message}`;
  }
});

// ---- toolbar -----------------------------------------------------------------------------------------------

function setView(view) {
  S.view = view;
  document.querySelector(`input[name="view"][value="${view}"]`).checked = true;
}

function setTool(tool) {
  S.tool = tool;
  el.stage.classList.toggle("tool-arrow", tool === "arrow");
  el.stage.classList.toggle("tool-box", tool === "box");
  el.hint.textContent = HINTS[tool];
}

for (const r of document.querySelectorAll('input[name="tool"]')) r.addEventListener("change", () => setTool(r.value));
for (const r of document.querySelectorAll('input[name="view"]')) {
  r.addEventListener("change", () => {
    S.view = r.value;
    if (S.image) refreshImage();
  });
}
for (const r of document.querySelectorAll('input[name="zoom"]')) {
  r.addEventListener("change", () => {
    S.zoom = r.value;
    overlay.layout();
  });
}
$("#show-marks").addEventListener("change", (ev) => el.stage.classList.toggle("hide-marks", !ev.target.checked));
$("#btn-open").addEventListener("click", () => openImage(null));
el.close.addEventListener("click", closeImage);
el.path.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && el.path.value.trim()) openImage(el.path.value.trim()); });
$("#btn-settings").addEventListener("click", () => openSettings().catch((e) => setStatus(e.message, "error")));
$("#btn-plan").addEventListener("click", runPlan);
for (const chip of document.querySelectorAll(".chip")) {
  chip.addEventListener("click", () => {
    el.instruction.value = chip.dataset.instruction;
    runPlan();
  });
}
el.apply.addEventListener("click", () => applyAll());
$("#btn-save").addEventListener("click", saveBeside);
$("#btn-save-as").addEventListener("click", saveTo);
el.undo.addEventListener("click", () => travel(false));
el.redo.addEventListener("click", () => travel(true));

const editable = (t) => t instanceof HTMLElement && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));

document.addEventListener("keydown", (ev) => {
  if (!(ev.ctrlKey || ev.metaKey) || ev.altKey || document.querySelector("dialog[open]")) return;
  const k = ev.key.toLowerCase();
  // inside a text field Ctrl+Z belongs to the field (undo typing); outside it steps the edit list and the render
  if ((k === "z" || k === "y") && !editable(ev.target)) {
    ev.preventDefault();
    travel(k === "y" || ev.shiftKey);
  } else if (k === "s") {
    ev.preventDefault();
    if (ev.shiftKey) saveTo(); else saveBeside();
  } else if (k === "o") {
    ev.preventDefault();
    openImage(null);
  } else if (k === "enter") {
    ev.preventDefault();
    if (ev.target === el.instruction) runPlan(); else applyAll();
  }
});

// ---- help, drop/paste, startup notices ---------------------------------------------------------------------

const help = $("#help-dlg");
$("#btn-help").addEventListener("click", () => help.showModal());
$("#help-close").addEventListener("click", () => help.close());
document.addEventListener("keydown", (ev) => {
  if (ev.key !== "F1" || document.querySelector("dialog[open]")) return;
  ev.preventDefault(); // otherwise the browser opens its own help page
  help.showModal();
});

setupUpload({ target: el.viewport, onFile: openUpload, onRejected: (m) => setStatus(m, "error") });

// leaving with a render newer than the last save asks first
window.addEventListener("beforeunload", (ev) => {
  keepNow();
  if (!S.hasResult || S.version === S.savedVersion) return;
  ev.preventDefault();
  ev.returnValue = "";
});

// the eraser loads in the background after start; name it only when it is not LaMa on the GPU
async function watchEngine(tries) {
  const { engine } = await api("/api/health").catch(() => ({}));
  if (!engine?.name) {
    if (tries > 1) setTimeout(() => watchEngine(tries - 1), 3000);
    return;
  }
  const text = engine.name !== "lama" ? "OpenCV 擦除（品質較差）" : engine.device !== "cuda" ? "LaMa（CPU，較慢）" : "";
  el.engine.hidden = !text;
  el.engine.textContent = text;
  el.engine.title = engine.note || "";
}

// a font family this computer lacks is drawn with the sans font; every item's font list says so
api("/api/fonts").then(({ fonts }) => {
  for (const opt of $("#item-tpl").content.querySelectorAll('select[data-k="font"] option')) {
    if (fonts[opt.value] === false) opt.textContent += "（未安裝，改用黑體）";
  }
}).catch(() => {});

setTool("arrow");
syncUndo();
watchEngine(40);
api("/api/current").then((data) => { if (data.path) loadImage(data); }).catch(() => {});
