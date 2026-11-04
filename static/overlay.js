// Stage overlay: OCR line boxes, arrows and their text labels over the image, all in image-pixel coordinates.
const SVGNS = "http://www.w3.org/2000/svg";
const LABEL_W = 230; // screen px, roughly the size of a .tag-label
const LABEL_H = 34;
const CLICK_PX = 6; // a press that moves less than this is a click
const HIT_PX = 30; // a click this close to a line box still picks it

const area = (b) => (b[2] - b[0]) * (b[3] - b[1]);
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

function boxDistance(b, p) {
  const dx = Math.max(b[0] - p[0], 0, p[0] - b[2]);
  const dy = Math.max(b[1] - p[1], 0, p[1] - b[3]);
  return Math.hypot(dx, dy);
}

function overlap(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  return w > 0 && h > 0 ? w * h : 0;
}

function svgEl(tag, attrs) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

export function createOverlay(el, S, hooks) {
  let scale = 1; // screen px per image px
  let gesture = null; // {kind: "arrow"|"box", start, cur} or {kind: "tail", key}
  const labelEls = new Map();

  function layout() {
    if (!S.image) return;
    const { width: W, height: H } = S.image;
    if (S.zoom === "fit") {
      const cw = el.viewport.clientWidth - 32;
      const ch = el.viewport.clientHeight - 32;
      scale = Math.max(0.05, Math.min(cw / W, ch / H));
    } else {
      scale = Number(S.zoom);
    }
    el.stage.style.width = `${W * scale}px`;
    el.stage.style.height = `${H * scale}px`;
    el.svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    draw();
  }

  function toImage(ev) {
    const r = el.stage.getBoundingClientRect();
    const { width: W, height: H } = S.image;
    return [clamp((ev.clientX - r.left) / scale, 0, W), clamp((ev.clientY - r.top) / scale, 0, H)];
  }

  // smallest line box under the point, else the nearest one within HIT_PX screen pixels
  function hitLine(p) {
    const inside = S.lines.filter((l) => boxDistance(l.box, p) === 0);
    if (inside.length) return inside.reduce((a, b) => (area(a.box) <= area(b.box) ? a : b));
    let best = null;
    let bestD = HIT_PX / scale;
    for (const l of S.lines) {
      const d = boxDistance(l.box, p);
      if (d < bestD) [best, bestD] = [l, d];
    }
    return best;
  }

  function labelRect(p) {
    const lw = LABEL_W / scale;
    const lh = LABEL_H / scale;
    return [p[0] - lw / 2, p[1] - lh / 2, p[0] + lw / 2, p[1] + lh / 2];
  }

  // label position beside box that covers the least text and the fewest other labels
  function placeTail(box) {
    const { width: W, height: H } = S.image;
    const lw = LABEL_W / scale;
    const lh = LABEL_H / scale;
    const gap = 28 / scale;
    const cx = (box[0] + box[2]) / 2;
    const cy = (box[1] + box[3]) / 2;
    const left = box[0] - gap - lw / 2;
    const right = box[2] + gap + lw / 2;
    const above = box[1] - gap - lh / 2;
    const below = box[3] + gap + lh / 2;
    const candidates = [[left, cy], [right, cy], [cx, above], [cx, below],
      [left, above], [right, above], [left, below], [right, below]];
    const texts = S.lines.map((l) => l.box);
    const labels = S.items.map((it) => labelRect(it.tail));
    let best = candidates[0];
    let bestCost = Infinity;
    candidates.forEach((c, i) => {
      const r = labelRect(c);
      const outside = Math.max(0, -r[0]) + Math.max(0, r[2] - W) + Math.max(0, -r[1]) + Math.max(0, r[3] - H);
      const cost = outside * lh * 4
        + texts.reduce((s, t) => s + overlap(r, t), 0)
        + labels.reduce((s, t) => s + 3 * overlap(r, t), 0)
        + i; // ties go to the earlier candidate
      if (cost < bestCost) [best, bestCost] = [c, cost];
    });
    return [clamp(best[0], Math.min(lw / 2, W / 2), Math.max(W - lw / 2, W / 2)),
      clamp(best[1], Math.min(lh / 2, H / 2), Math.max(H - lh / 2, H / 2))];
  }

  // point on box nearest to p, nudged inside so the arrow head touches the text
  function tipToward(box, p) {
    const inset = Math.min(4 / scale, (box[2] - box[0]) / 2, (box[3] - box[1]) / 2);
    return [clamp(p[0], box[0] + inset, box[2] - inset), clamp(p[1], box[1] + inset, box[3] - inset)];
  }

  function drawArrow(tail, tip, focused) {
    const [tx, ty] = tail;
    const [px, py] = tip;
    const len = Math.hypot(px - tx, py - ty) || 1;
    const ux = (px - tx) / len;
    const uy = (py - ty) / len;
    const hl = Math.min(16 / scale, len * 0.6);
    const hw = 7 / scale;
    const bx = px - ux * hl;
    const by = py - uy * hl;
    const g = svgEl("g", { class: focused ? "focused" : "" });
    g.append(
      svgEl("line", { x1: tx, y1: ty, x2: bx, y2: by, class: "arrow-halo" }),
      svgEl("line", { x1: tx, y1: ty, x2: bx, y2: by, class: "arrow-shaft" }),
      svgEl("polygon", { points: `${px},${py} ${bx - uy * hw},${by + ux * hw} ${bx + uy * hw},${by - ux * hw}`, class: "arrow-head" }),
    );
    el.svg.append(g);
  }

  function rect(b, cls, tip) {
    const node = svgEl("rect", { x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1], class: cls });
    if (tip) node.append(Object.assign(svgEl("title", {}), { textContent: tip }));
    el.svg.append(node);
  }

  function draw() {
    if (!S.image) return;
    el.svg.replaceChildren();
    const picked = new Set(S.items.map((it) => it.lineId));
    for (const l of S.lines) {
      rect(l.box, `line-box${l.garbled ? " garbled" : ""}${picked.has(l.id) ? " picked" : ""}`,
        l.garbled ? `${l.text}（疑似亂碼）` : l.text);
    }
    for (const it of S.items) if (it.lineId == null) rect(it.box, "line-box region");
    for (const it of S.items) drawArrow(it.tail, it.tip, it.key === S.focusKey);
    if (gesture?.kind === "arrow" && gesture.moved) drawArrow(gesture.start, gesture.cur, true);
    if (gesture?.kind === "box" && gesture.moved) {
      const [a, b] = [gesture.start, gesture.cur];
      rect([Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])], "drag-rect");
    }
    syncLabels();
  }

  function makeLabel(key) {
    const node = document.createElement("div");
    node.className = "tag-label";
    node.innerHTML = '<span class="badge" title="拖曳可移動標籤"></span><input type="text" placeholder="留空＝刪除">';
    const input = node.querySelector("input");
    input.addEventListener("input", () => hooks.onText(key, input.value));
    input.addEventListener("focus", () => hooks.onFocus(key));
    const badge = node.querySelector(".badge");
    badge.addEventListener("pointerdown", (ev) => {
      ev.stopPropagation();
      ev.preventDefault();
      badge.setPointerCapture(ev.pointerId);
      gesture = { kind: "tail", key };
      hooks.onFocus(key);
    });
    badge.addEventListener("pointermove", (ev) => {
      if (gesture?.kind !== "tail") return;
      const it = S.items.find((x) => x.key === key);
      if (it) it.tail = toImage(ev);
      draw();
    });
    badge.addEventListener("pointerup", () => { gesture = null; });
    return node;
  }

  function syncLabels() {
    const keys = new Set(S.items.map((it) => it.key));
    for (const [k, node] of labelEls) {
      if (!keys.has(k)) {
        node.remove();
        labelEls.delete(k);
      }
    }
    S.items.forEach((it, i) => {
      let node = labelEls.get(it.key);
      if (!node) {
        node = makeLabel(it.key);
        labelEls.set(it.key, node);
        el.labels.append(node);
      }
      node.style.left = `${(it.tail[0] / S.image.width) * 100}%`;
      node.style.top = `${(it.tail[1] / S.image.height) * 100}%`;
      node.querySelector(".badge").textContent = i + 1;
      const input = node.querySelector("input");
      if (document.activeElement !== input && input.value !== it.newText) input.value = it.newText;
      node.classList.toggle("focused", it.key === S.focusKey);
    });
  }

  function focusLabel(key) {
    const input = labelEls.get(key)?.querySelector("input");
    if (!input) return;
    input.focus({ preventScroll: true });
    input.select();
  }

  function clear() {
    for (const node of labelEls.values()) node.remove();
    labelEls.clear();
    el.svg.replaceChildren();
  }

  el.stage.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0 || !S.image || ev.target.closest(".tag-label")) return;
    ev.preventDefault();
    el.stage.setPointerCapture(ev.pointerId);
    const p = toImage(ev);
    gesture = { kind: S.tool, start: p, cur: p, moved: false };
  });
  el.stage.addEventListener("pointermove", (ev) => {
    if (!gesture || gesture.kind === "tail") return;
    gesture.cur = toImage(ev);
    const d = Math.hypot(gesture.cur[0] - gesture.start[0], gesture.cur[1] - gesture.start[1]) * scale;
    gesture.moved = gesture.moved || d >= CLICK_PX;
    if (gesture.moved) draw();
  });
  el.stage.addEventListener("pointerup", () => {
    const g = gesture;
    if (!g || g.kind === "tail") return;
    gesture = null;
    draw();
    if (g.kind === "arrow") {
      const tip = g.moved ? g.cur : g.start;
      hooks.onArrow({ tip, tail: g.moved ? g.start : null, line: hitLine(tip) });
    } else if (g.moved) {
      const [a, b] = [g.start, g.cur];
      hooks.onBox([Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])].map(Math.round));
    }
  });
  window.addEventListener("resize", () => { if (S.zoom === "fit") layout(); });

  return { layout, draw, placeTail, tipToward, focusLabel, clear };
}
