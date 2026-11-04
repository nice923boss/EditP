"""One-click setup, run by install.bat: packages, fonts, the pythonw_editp.exe launcher, model downloads."""
import filecmp
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"  # no NVIDIA card: LaMa runs on the CPU (slower)
MIN_FREE_GB = 5
LOG = Path.home() / ".editp" / "install.log"
FONT_URL = "https://github.com/notofonts/noto-cjk/raw/main/{}/Variable/TTF/Subset/{}"
FONT_DOWNLOADS = {"NotoSansTC-VF.ttf": "Sans", "NotoSerifTC-VF.ttf": "Serif"}
VERIFY = """
import numpy as np
from editp import inpaint, ocr, render
ocr.detect(np.zeros((32, 32, 3), np.uint8))  # RapidOCR fetches its models on first use
inpaint.warm_up()  # big-lama.pt (~200 MB) goes to ~/.cache/torch, then loads on CUDA or the CPU
render.get_font("sans", 40, 400), render.get_font("serif", 40, 400)
s = inpaint.status()
print("擦除引擎：", "LaMa" if s["name"] == "lama" else "OpenCV（備援）", "/", "顯示卡" if s["device"] == "cuda" else "CPU")
"""


class SetupError(Exception):
    pass


def pip(*args: str) -> None:
    if subprocess.run([sys.executable, "-m", "pip", "install", *args]).returncode != 0:
        raise SetupError("pip 安裝失敗，請看上方錯誤訊息（常見原因：網路中斷、磁碟空間不足）")


def check_python() -> None:
    if sys.version_info < (3, 10):
        raise SetupError(f"需要 Python 3.10 以上，目前是 {sys.version.split()[0]}")
    if "WindowsApps" in sys.base_prefix:
        raise SetupError("這是 Microsoft Store 版 Python，無法建立啟動程式。請從 python.org 安裝 Python 3.12")
    free = shutil.disk_usage(ROOT).free / 2**30
    if free < MIN_FREE_GB:
        print(f"提醒：磁碟剩 {free:.1f} GB，第一次安裝約需 {MIN_FREE_GB} GB，空間不足時 pip 會失敗")


def has_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        print("找不到 NVIDIA 顯示卡驅動（nvidia-smi），擦除文字會用 CPU 跑，每次套用慢幾秒到幾十秒。\n"
              "有 NVIDIA 顯示卡的話，先裝驅動再重跑 install.bat：https://www.nvidia.com/Download/index.aspx")
        return False
    return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).returncode == 0


def check_gpu() -> None:
    global GPU
    GPU = has_gpu()


GPU = False


def torch_cuda() -> str | None:
    """CUDA version of the installed torch: '' for a CPU build, None when torch is not installed."""
    r = subprocess.run([sys.executable, "-c", "import torch; print(torch.version.cuda or '')"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def install_packages() -> None:
    pip("-r", str(ROOT / "requirements.txt"))
    cuda = torch_cuda()
    if cuda:
        print(f"已有 CUDA 版 torch（CUDA {cuda}），略過")
    elif cuda == "" and not GPU:
        print("已有 CPU 版 torch，沒有 NVIDIA 顯示卡，略過")
    elif cuda == "":
        print("目前的 torch 是 CPU 版，無法用顯示卡，改裝 CUDA 版")
        pip("--upgrade", "--force-reinstall", "--no-deps", "torch", "torchvision", "--index-url", TORCH_INDEX)
    else:
        pip("torch", "torchvision", "--index-url", TORCH_INDEX if GPU else TORCH_CPU_INDEX)
    # simple-lama-inpainting pins numpy<2 and pillow<10; honoring the pins would downgrade the versions this
    # app is tested with, and it runs fine on them, so install it without its dependency list.
    pip("--no-deps", "simple-lama-inpainting==0.1.2")


def install_fonts() -> None:
    from editp import render

    for name, family in FONT_DOWNLOADS.items():
        if found := render.find_font(name):
            print(f"{name}：已有（{found}）")
            continue
        target = ROOT / "fonts" / name
        target.parent.mkdir(exist_ok=True)
        print(f"{name}：下載中…")
        part = target.with_suffix(".part")
        urllib.request.urlretrieve(FONT_URL.format(family, name), part)
        part.replace(target)
    if not render.find_font("kaiu.ttf"):
        print("提醒：這台電腦沒有標楷體 kaiu.ttf，選楷體時會改用黑體")


def install_launcher() -> None:
    """A renamed pythonw (plus its DLLs and pyvenv.cfg) so the app runs, and stop.bat stops it, by its own name."""
    home = Path(sys.base_prefix)
    dll = f"python{sys.version_info.major}{sys.version_info.minor}.dll"
    for src_name, dst_name in {"pythonw.exe": "pythonw_editp.exe", dll: dll, "python3.dll": "python3.dll"}.items():
        src, dst = home / src_name, ROOT / dst_name
        if dst.exists() and filecmp.cmp(src, dst, shallow=False):
            continue
        try:
            shutil.copy2(src, dst)
        except PermissionError as e:
            raise SetupError(f"{dst.name} 正在使用中，請先執行 stop.bat 再重跑 install.bat") from e
    (ROOT / "pyvenv.cfg").write_text(f"home = {home}\ninclude-system-site-packages = true\n", encoding="utf-8")
    print(f"pythonw_editp.exe ← {home}")


def verify() -> None:
    if subprocess.run([sys.executable, "-c", VERIFY], cwd=ROOT).returncode != 0:
        raise SetupError("模型下載或載入失敗，請看上方錯誤訊息。訊息提到 CUDA driver 時，請更新 NVIDIA 驅動")


STEPS = [
    ("檢查 Python 與磁碟空間", check_python),
    ("檢查 NVIDIA 顯示卡", check_gpu),
    ("安裝 Python 套件（第一次約需下載 3 GB）", install_packages),
    ("準備字型", install_fonts),
    ("建立啟動程式", install_launcher),
    ("下載模型並驗證", verify),
]


def log(line: str) -> None:
    """One line per install run in ~/.editp/install.log, so a failed install can be looked at later."""
    try:
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Python {sys.version.split()[0]} {line}\n")
    except OSError:
        pass


def main() -> int:
    print(f"Python {sys.version.split()[0]}：{sys.executable}")
    step = ""
    try:
        for i, (title, run) in enumerate(STEPS, 1):
            step = title
            print(f"\n[{i}/{len(STEPS)}] {title}", flush=True)
            run()
    except SetupError as e:
        print(f"\n安裝中止：{e}")
        log(f"失敗於「{step}」：{e}")
        return 1
    log("安裝完成" + ("（NVIDIA 顯示卡）" if GPU else "（CPU）"))
    print("\n安裝完成。雙擊 start.bat 啟動 EditP，第一次使用請到「LLM 設定」填入 API Key。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
