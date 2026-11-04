# EditP 圖片錯字修正

修正 AI 生成圖片上的錯字：擦掉指定的文字行，用相近的字型、大小、顏色、描邊重畫正確文字，另存成 PNG、JPG 或 WebP（原圖不動）。

流程：RapidOCR 找出文字行 → LLM（OpenAI 相容 API）校正讀法、解析指令 → LaMa 擦除 → 依原字樣式重繪。

這一版的改動清單見 [IMPROVEMENTS.md](IMPROVEMENTS.md)。

## 需求

- Windows 10／11
- Python 3.10 以上，建議 3.12（沒有的話 install.bat 會用 winget 安裝）。不支援 Microsoft Store 版 Python。
- 一個 OpenAI 相容的 LLM 服務（開發時用 llama.cpp 跑 qwen3.6-35b-a3b）
- NVIDIA 顯示卡為選配：
  - 有顯示卡、驅動支援 CUDA 12.8：LaMa 用顯示卡擦除。
  - 沒有顯示卡：LaMa 用 CPU 擦除，每次套用慢幾秒到幾十秒。
  - LaMa 沒裝成或載入失敗：改用 OpenCV 擦除，品質較差，介面右上角會標示目前的擦除引擎。

## 安裝

下載專案（`git clone https://github.com/nice923boss/EditP.git`，或 GitHub 頁面的 Download ZIP），雙擊 `install.bat`。依序做六件事：檢查 Python 與磁碟空間、檢查 NVIDIA 顯示卡（沒有就改裝 CPU 版 torch）、安裝 Python 套件（第一次約下載 3 GB）、準備字型、建立啟動程式 `pythonw_editp.exe`、下載模型並驗證。任一步失敗會停下並說明原因，每次安裝的結果記在 `%USERPROFILE%\.editp\install.log`。

## 啟動與停止

- `start.bat`：先檢查套件，缺少時自動跑 install.bat，再啟動伺服器（http://127.0.0.1:8793）並開啟瀏覽器。
  - 8793 被其他程式占用時，會自動改用後面的埠。
  - EditP 已在執行時，只會再開一次瀏覽器。
- 啟動後系統托盤會出現 EditP 圖示：點一下開啟頁面，右鍵選單有「開啟記錄檔資料夾」「結束 EditP」。
- `stop.bat` 或托盤「結束 EditP」：停止伺服器。
- 執行紀錄與錯誤寫在 `%USERPROFILE%\.editp\app.log`，超過 5 MB 會輪替成 `app.log.1`。

第一次使用，按「LLM 設定」填入 Base URL、Model、API Key，按「測試連線」確認。設定存在 `config.json`（已列入 .gitignore，不會進版控）。
- 本機 LLM 回應慢時，可在 `config.json` 加大 `llm_timeout`（秒，預設 180）。
- `config.json` 損壞時會改名成 `config.json.bad`，用預設值啟動。

## 使用方式

1. 開啟圖片，四種方式：
   - 按「開啟圖片…」。
   - 在路徑欄貼上圖片路徑按 Enter。
   - 把圖片拖進視窗。
   - 複製圖片後按 Ctrl+V。

   拖進或貼上的圖片會先存到 `圖片\EditP`。
2. 指定要改的文字行，三種方式可混用，全部進入「批次修改清單」：
   - **箭頭**：從空白處拖到要改的那行（或直接點那行）。標籤會預填 AI 校正後的原文，直接改成整行的新內容；留空表示刪除這行。標籤可以拖曳移位。
   - **框選**：框住一段文字，標籤預填框內的 OCR 讀法。框要貼近文字：框內（再往外 6 px）的東西都會被擦掉重補，框到的邊框、圖案不一定補得回原樣。
   - **AI 指令**：輸入自然語言（例如「把『限時特賣』改成『限時優惠』、刪除底部的亂碼」）按「AI 解析」。「修正錯字」「刪除亂碼」是常用指令的捷徑。
3. 「全部套用」產生結果。清單中每一項的樣式（字型、粗細、字級、字距、顏色、對齊、光暈、描邊、位移）可微調，套用後再改會自動重繪。套用後不滿意，按 Ctrl+Z 回到套用前（Ctrl+Y 重做）。
4. 存檔，不能覆蓋原圖：
   - 「另存新檔」存到 `原檔名_edited.png`，已存在時改用 `_edited_2`、`_edited_3`…。存好會跳出預覽視窗，可再下載 PNG、JPG 或 WebP、開啟資料夾、複製路徑。
   - 「另存到…」自選路徑與格式。
   - 有套用但沒存的結果時，關閉頁面會先提醒。

批次修改清單會記在瀏覽器，重開同一張圖會帶回上次的清單（最多記 20 張）。

快捷鍵（F1 或右上角「?」可看說明）：

| 按鍵 | 動作 |
|---|---|
| Ctrl+O | 開啟圖片 |
| Ctrl+V | 貼上剪貼簿的圖片 |
| Ctrl+Enter | 全部套用（在 AI 指令框內是 AI 解析） |
| Ctrl+Z／Ctrl+Y | 復原／重做（含套用後的圖） |
| Ctrl+S | 另存新檔 |
| Ctrl+Shift+S | 另存到 |

介面跟隨 Windows 的深色或淺色模式；視窗寬度 900 px 以下改成上下排列。

## HTTP API

伺服器只聽本機 127.0.0.1，並拒絕 Host 或 Origin 不是本機的請求。一次完成「讀圖、改字、存檔」用 `POST /api/edit`：

| 欄位 | 說明 |
|---|---|
| `path` | 伺服器這台電腦上的圖片路徑 |
| `instruction` | 自然語言指令（可省略） |
| `edits` | 明確的修改：`[{"box": [x1, y1, x2, y2], "new_text": "...", "old_text": "..."}]`，像素座標，`new_text` 空字串表示刪除。`old_text`（框內現在的字）可省略，省略時用框中心那行的 OCR 讀法，作為比對字型的參考 |
| `output` | 輸出路徑（可省略，預設 `原檔名_edited.png`） |
| `dry_run` | `true` 只回傳解析結果，不畫也不存 |

`instruction` 與 `edits` 可同時給。只給 `edits` 時完全不呼叫 LLM，實測 3 到 9 秒；有 `instruction` 時 LLM 讀圖，實測約 25 秒。框座標建議先用 `dry_run` 取得 `lines` 裡各行的框，比手量的框貼字，擦除範圍較小。

中文 JSON 建議先寫成 UTF-8 檔再送出，避免終端機編碼問題。`req.json`：

```json
{"path": "D:/圖片/poster.jpg", "instruction": "把「迷局」改成「謎局」", "dry_run": true}
```

```bash
curl -s http://127.0.0.1:8793/api/edit -H "Content-Type: application/json" --data-binary @req.json
```

回傳 `lines`（偵測到的文字行與框）、`edits`（實際套用的修改與樣式）、`reason`（LLM 的判斷說明）、`source`（`llm`；`rule` 表示指令是單純的「A 改成 B」，沒經過 LLM 規劃；`explicit` 表示只有 `edits`）、`saved`（輸出檔路徑，dry_run 時為 null）。

用框座標直接改字並指定輸出：

```json
{"path": "D:/圖片/poster.jpg", "edits": [{"box": [1099, 43, 1296, 94], "new_text": "亂世謎局"}], "output": "D:/圖片/poster_fixed.png"}
```

UI 使用的其他端點（`/api/open`、`/api/close`、`/api/upload`、`/api/read`、`/api/plan`、`/api/render`、`/api/restore`、`/api/save`、`/api/download` 等）以目前在 UI 開啟的那張圖為對象，定義在 `editp/server.py`。`GET /api/health` 回傳版本與目前的擦除引擎。

## 測試

```bash
py -3.12 -m pip install pytest
```

```bash
py -3.12 -m pytest tests -q
```

單元測試不需要 GPU 與 LLM（LaMa 以假函式代替），也不會動到真的 `config.json`。

`eval/run_eval.py` 測 LLM 照指令修改的正確率：3 張合成圖加 1 張真實圖，共 39 個案例。需要 LLM 服務與已安裝的中文字型；專案根目錄沒有 `143884.jpg` 時跳過真實圖案例。
- 1.0 版：39/39 通過（其中保留驗證集 15/15）。
- 1.1.0（rapidocr 3.10、PP-OCRv6）：37/39，兩題失敗原因見 GAPS.md。

## 已知限制

見 [GAPS.md](GAPS.md)。

## 授權

MIT，見 [LICENSE](LICENSE)。
