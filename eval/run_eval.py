"""Instruction-following eval for the LLM planner on synthetic images with known text.

Usage: python eval/run_eval.py [case-substring ...]
Each case runs OCR -> llm.read -> llm.plan; a case passes when every expected line ends with the expected
text and no other line was edited. Images are written to eval/images/ for inspection.
"""
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from editp import config, llm, ocr, render  # noqa: E402

SANS = render.find_font("NotoSansTC-VF.ttf")  # system or per-user Fonts folder
REAL = ROOT / "143884.jpg"  # optional real AI poster; its cases are skipped without it
OUT = ROOT / "eval" / "images"


def font(size: int, weight: int = 500) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(SANS, size)
    f.set_variation_by_axes([weight])
    return f


def garbled(d: ImageDraw.ImageDraw, img: Image.Image, xy: tuple[int, int], size: int, fill: tuple) -> list[int]:
    """Pseudo-characters made of the left half of one glyph and the right half of another."""
    f = font(size)
    x, y = xy
    for a, b in ("鼠活", "龍語", "照灣", "影餐"):
        tile = Image.new("RGBA", (size + 4, int(size * 1.5)), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        td.text((2, 0), a, font=f, fill=fill)
        right = Image.new("RGBA", tile.size, (0, 0, 0, 0))
        ImageDraw.Draw(right).text((2, 0), b, font=f, fill=fill)
        tile.paste(right.crop((size // 2, 0, tile.width, tile.height)), (size // 2, 0))
        img.alpha_composite(tile, (x, y))
        x += size + 2
    return [xy[0], xy[1], x, xy[1] + int(size * 1.4)]


def text(d: ImageDraw.ImageDraw, xy: tuple[int, int], s: str, size: int, fill: tuple, weight: int = 500) -> list[int]:
    f = font(size, weight)
    b = d.textbbox(xy, s, font=f)
    d.text(xy, s, font=f, fill=fill)
    return [b[0], b[1], b[2], b[3]]


def make_poster() -> tuple[np.ndarray, dict[str, tuple[str, list[int]]]]:
    img = Image.new("RGBA", (1280, 720), (20, 24, 48, 255))
    d = ImageDraw.Draw(img)
    for y in range(720):
        d.line([(0, y), (1280, y)], fill=(20 + y // 12, 24, 48 + y // 10, 255))
    d.rounded_rectangle([880, 40, 1240, 300], 18, fill=(40, 60, 100, 255), outline=(120, 200, 255, 255), width=3)
    white, gold, cyan = (250, 250, 250, 255), (255, 210, 90, 255), (150, 230, 255, 255)
    g = {
        "title": ("夏日音樂祭", text(d, (80, 90), "夏日音樂祭", 96, gold, 800)),
        "date": ("2026年7月18日 晚上7點", text(d, (84, 240), "2026年7月18日 晚上7點", 40, white)),
        "place": ("地點：台北市信義區松高路11號", text(d, (84, 310), "地點：台北市信義區松高路11號", 34, white)),
        "phone": ("電話：02-2345-6789", text(d, (84, 370), "電話：02-2345-6789", 34, white)),
        "quota": ("入場免費，名額有線", text(d, (84, 450), "入場免費，名額有線", 34, gold)),
        "h0": ("活動亮點", text(d, (920, 60), "活動亮點", 36, cyan, 700)),
        "h1": ("樂團現場演出", text(d, (920, 130), "樂團現場演出", 30, white)),
        "h2": ("美食市集", text(d, (920, 185), "美食市集", 30, white)),
        "h3": ("煙火秀", text(d, (920, 240), "煙火秀", 30, white)),
        "host": ("主辦單位：星光文化基金會", text(d, (420, 640), "主辦單位：星光文化基金會", 28, (200, 200, 220, 255))),
        "junk": ("", garbled(d, img, (80, 30), 24, (200, 200, 220, 255))),
    }
    return np.array(img.convert("RGB")), g


def make_menu() -> tuple[np.ndarray, dict[str, tuple[str, list[int]]]]:
    img = Image.new("RGBA", (1280, 900), (246, 238, 222, 255))
    d = ImageDraw.Draw(img)
    brown, dark, red = (110, 70, 40, 255), (50, 40, 30, 255), (190, 50, 40, 255)
    g = {"title": ("晨光咖啡館", text(d, (450, 50), "晨光咖啡館", 80, brown, 800))}
    d.line([(640, 220), (640, 640)], fill=(180, 160, 130, 255), width=2)
    left = [("美式咖啡", "60"), ("拿鐵", "80"), ("卡布奇諾", "85")]
    right = [("紅茶拿鐵", "75"), ("抹茶拿鐵", "90"), ("熱可可", "70")]
    for col, items, x0, px in (("L", left, 120, 500), ("R", right, 720, 1100)):
        for i, (name, price) in enumerate(items):
            y = 250 + i * 120
            g[f"{col}{i}"] = (name, text(d, (x0, y), name, 44, dark, 600))
            g[f"{col}{i}p"] = (price, text(d, (px, y), price, 44, red, 700))
    g["hours"] = ("營業時間：週一至週五 8:00-18:00", text(d, (300, 720), "營業時間：週一至週五 8:00-18:00", 36, dark))
    g["promo"] = ("自備杯子折五元", text(d, (480, 800), "自備杯子折五元", 32, red))
    g["junk"] = ("", garbled(d, img, (1100, 40), 26, brown))
    return np.array(img.convert("RGB")), g


def make_notice() -> tuple[np.ndarray, dict[str, tuple[str, list[int]]]]:
    img = Image.new("RGBA", (900, 1200), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 900, 170], fill=(200, 40, 40, 255))
    white, dark, grey = (255, 255, 255, 255), (30, 30, 30, 255), (110, 110, 110, 255)
    g = {"title": ("社區防災講座", text(d, (210, 45), "社區防災講座", 76, white, 800))}
    rows = [
        ("time", "時間：11月3日（週日）下午2點"), ("place", "地點：里民活動中心二樓"),
        ("speaker", "講師：王小明 消防隊長"), ("tel", "報名專線：0912-345-678"),
    ]
    for i, (key, s) in enumerate(rows):
        g[key] = (s, text(d, (80, 230 + i * 80), s, 40, dark))
    for i, s in enumerate(("1. 地震應變", "2. 火災逃生", "3. 急救示範")):
        g[f"a{i}"] = (s, text(d, (120, 620 + i * 75), s, 38, dark, 600))
    g["footer"] = ("主辦：幸福里辦公處", text(d, (300, 1080), "主辦：幸福里辦公處", 32, grey))
    return np.array(img.convert("RGB")), g


def load_real() -> tuple[np.ndarray, dict[str, tuple[str, list[int]]]]:
    rgb = np.array(Image.open(REAL).convert("RGB"))
    return rgb, {
        "t1": ("亂世迷局", [1099, 43, 1296, 94]), "t2": ("如何破解", [1096, 102, 1299, 158]),
        "junk": ("", [1079, 4, 1166, 36]),
    }


# Cases tuned against while writing the prompts.
# (image, instruction, {gt line: expected final text}); every other line must stay untouched
CASES = [
    ("poster", "修正圖上的錯字", {"quota": "入場免費，名額有限"}),
    ("poster", "把有線改成有限", {"quota": "入場免費，名額有限"}),
    ("poster", "電話改成 02-8765-4321", {"phone": "電話：02-8765-4321"}),
    ("poster", "地點換成台中市西屯區台灣大道三段99號", {"place": "地點：台中市西屯區台灣大道三段99號"}),
    ("poster", "日期改成7月25日", {"date": "2026年7月25日 晚上7點"}),
    ("poster", "刪除最下面那一行", {"host": ""}),
    ("poster", "右上角面板裡的第二個活動改成手作市集", {"h2": "手作市集"}),
    ("poster", "標題改成秋日音樂祭", {"title": "秋日音樂祭"}),
    ("poster", "把咖啡改成茶", {}),
    ("poster", "活動亮點第一項改成搖滾樂團演出，第三項改成無人機秀", {"h1": "搖滾樂團演出", "h3": "無人機秀"}),
    ("poster", "刪掉左上角的亂碼", {"junk": ""}),
    ("poster", "把『夏日』改成『冬季』", {"title": "冬季音樂祭"}),
    ("poster", "把主辦單位改成協辦單位", {"host": "協辦單位：星光文化基金會"}),
    ("poster", "電話後面加上「分機123」", {"phone": "電話：02-2345-6789 分機123"}),
    ("menu", "右欄第二項的價格改成95", {"R1p": "95"}),
    ("menu", "左欄每個價格都加5元", {"L0p": "65", "L1p": "85", "L2p": "90"}),
    ("menu", "紅茶拿鐵改成鮮奶紅茶", {"R0": "鮮奶紅茶"}),
    ("menu", "營業時間改成每天 7:00-20:00", {"hours": "營業時間：每天 7:00-20:00"}),
    ("menu", "店名改成午後咖啡館", {"title": "午後咖啡館"}),
    ("menu", "刪除右上角那串亂碼", {"junk": ""}),
    ("menu", "熱可可改成熱巧克力，價格改成80", {"R2": "熱巧克力", "R2p": "80"}),
    ("menu", "把漢堡改成三明治", {}),
    ("menu", "刪除最下面的優惠文字", {"promo": ""}),
    ("menu", "拿鐵的價格改成100", {"L1p": "100"}),
]

# Held-out cases: written after the prompts were settled, never tuned against.
HOLDOUT = [
    ("notice", "講師名字改成李大華", {"speaker": "講師：李大華 消防隊長"}),
    ("notice", "第二個議程改成颱風防範", {"a1": "2. 颱風防範"}),
    ("notice", "時間改成11月10日（週日）下午2點", {"time": "時間：11月10日（週日）下午2點"}),
    ("notice", "把二樓改成三樓", {"place": "地點：里民活動中心三樓"}),
    ("notice", "刪除報名專線", {"tel": ""}),
    ("notice", "主辦單位後面加上「、消防局」", {"footer": "主辦：幸福里辦公處、消防局"}),
    ("notice", "把停車資訊刪掉", {}),
    ("notice", "標題的防災改成防颱", {"title": "社區防颱講座"}),
    ("poster", "最上面那行亂碼改成「限時活動」", {"junk": "限時活動"}),
    ("poster", "活動亮點的第一項和第三項對調", {"h1": "煙火秀", "h3": "樂團現場演出"}),
    ("menu", "所有拿鐵類飲品各漲10元", {"L1p": "90", "R0p": "85", "R1p": "100"}),
    ("menu", "把卡布奇諾刪掉，價格也一起刪", {"L2": "", "L2p": ""}),
    ("real", "把亂世迷局改成亂世謎局", {"t1": "亂世謎局"}),
    ("real", "刪除面板上方的小字亂碼", {"junk": ""}),
    ("real", "第二行改成如何化解", {"t2": "如何化解"}),
]


def norm(s: str) -> str:
    return "".join(unicodedata.normalize("NFKC", s).split())


def owner(box: list[int], gt: dict) -> str | None:
    """GT line whose (padded) box contains the OCR line's centre."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for key, (_, b) in gt.items():
        if b[0] - 8 <= cx <= b[2] + 8 and b[1] - 8 <= cy <= b[3] + 8:
            return key
    return None


def judge(gt: dict, lines: list[dict], edits: list[dict], expected: dict) -> list[str]:
    new = {e["id"]: e["new_text"] for e in edits}
    final: dict[str, list[tuple[int, str]]] = {}
    touched = set()
    for ln in lines:
        key = owner(ln["box"], gt)
        if ln["id"] in new:
            touched.add(key)
        if key is not None:
            final.setdefault(key, []).append((ln["box"][0], new.get(ln["id"], ln["text"])))
    problems = []
    for key, want in expected.items():
        got = "".join(t for _, t in sorted(final.get(key, [])))
        if key not in final:
            problems.append(f"{key}: OCR 沒偵測到")
        elif norm(got) != norm(want):
            problems.append(f"{key}: 得到「{got}」，應為「{want}」")
    for key in touched - set(expected):
        problems.append(f"{key}: 不該修改卻被改了")
    return problems


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if SANS is None:
        sys.exit("找不到 Noto Sans TC（NotoSansTC-VF.ttf），請先跑 install.bat 安裝字型")
    cfg = config.load()
    OUT.mkdir(parents=True, exist_ok=True)
    images = {"poster": make_poster(), "menu": make_menu(), "notice": make_notice()}
    if REAL.is_file():
        images["real"] = load_real()
    else:
        print(f"沒有 {REAL.name}，略過真實圖的案例")
    prepared = {}
    for name, (rgb, gt) in images.items():
        Image.fromarray(rgb).save(OUT / f"{name}.png")
        t = time.time()
        lines = ocr.detect(rgb)
        t_ocr = time.time() - t
        t = time.time()
        read = llm.read(cfg, rgb, lines)
        t_read = time.time() - t
        print(f"[{name}] OCR {t_ocr:.1f}s, read {t_read:.1f}s")
        for ln in read:
            key = owner(ln["box"], gt)
            want = gt[key][0] if key else "?"
            flag = "" if key == "junk" or norm(ln["text"]) == norm(want) else "  <-- 與原文不同"
            print(f"   [{ln['id']}] {key}: OCR「{ln['ocr_text']}」→ 讀「{ln['text']}」 garbled={ln['garbled']}{flag}")
        prepared[name] = (rgb, gt, read)

    only = sys.argv[1:]
    passed, total, results = 0, 0, []
    tagged = [(c, False) for c in CASES] + [(c, True) for c in HOLDOUT]
    for (image, instruction, expected), holdout in tagged:
        if (only and not any(s in instruction for s in only)) or image not in prepared:
            continue
        rgb, gt, lines = prepared[image]
        t = time.time()
        try:
            out = llm.plan(cfg, rgb, lines, instruction)
            problems = judge(gt, lines, out["edits"], expected)
        except llm.LLMError as e:
            out, problems = {"edits": [], "reason": "", "source": "error"}, [str(e)]
        secs = time.time() - t
        total += 1
        passed += not problems
        mark = "PASS" if not problems else "FAIL"
        print(f"{mark} {secs:5.1f}s [{out['source']}]{' (驗證集)' if holdout else ''} {instruction}")
        if problems:
            print(f"      reason: {out['reason']}")
            print(f"      edits: {json.dumps(out['edits'], ensure_ascii=False)}")
            for p in problems:
                print(f"      - {p}")
        results.append({"instruction": instruction, "pass": not problems, "secs": round(secs, 1),
                        "holdout": holdout, **out})
    hold = [r for r in results if r["holdout"]]
    print(f"\n通過 {passed}/{total}（其中驗證集 {sum(r['pass'] for r in hold)}/{len(hold)}）")
    (OUT / "last_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")


if __name__ == "__main__":
    main()
