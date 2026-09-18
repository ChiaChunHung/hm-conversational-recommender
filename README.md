# DSO585 — H&M Conversational Recommender System

以大型語言模型作為對話介面、背後接兩階段推薦引擎（候選召回 + LightGBM 排序）的對話式推薦系統。
資料來源為 Kaggle「H&M Personalized Fashion Recommendations」。

## 環境需求

- Python 3.10 以上
- 套件：`duckdb`

```bash
pip install duckdb
```

## 資料準備

**資料不在這個 repo 裡。** 原始檔有 3.2 GB，超過 GitHub 單檔 100 MB 的上限，
而且 Kaggle 競賽資料僅授權學術與競賽用途，不可公開散布。請自行下載。

到 <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data>
下載這三個檔案：`articles.csv`、`customers.csv`、`transactions_train.csv`（共約 3.7 GB）。

程式啟動時會依序找這三個地方，找到就用：

| 順序 | 位置 | 什麼時候會用到 |
|---|---|---|
| 1 | 環境變數 `DSO585_DATA` 指定的路徑 | 資料放在外接硬碟或其他專案共用時 |
| 2 | 這個 repo 裡的 `data/` | 最單純，clone 完直接在裡面建一個 `data/` 丟進去 |
| 3 | 這個 repo 旁邊的 `data/` | repo 只是某個大資料夾底下的一部分時 |

所以最省事的做法是在 clone 下來的資料夾裡建 `data/`，把三個 CSV 丟進去，什麼都不用設定：

```
dso585-hm-crs/
├── data/                  ← 在這裡建，丟三個 CSV 進去（不進版控）
├── main.py
└── README.md
```

如果資料放在別的地方，執行前設一次環境變數：

```bash
export DSO585_DATA=/Volumes/外接硬碟/hm_data
python main.py
```

想每次開終端機都自動生效，就把那行 `export` 加到 `~/.zshrc`（macOS 預設）或 `~/.bashrc`。

程式每次啟動都會印出它實際使用的資料目錄，所以不會搞錯用到哪一份。

## 執行

```bash
python main.py
```

第一次執行會把三份 CSV 轉成 Parquet（存在 `data/parquet/`，約 10 秒），
之後每次執行都會直接用 Parquet，速度約快三倍。轉出來的檔案不進版控，刪掉重跑即可。

| 指令 | 作用 |
|---|---|
| `python main.py` | 需要時轉檔，然後輸出探索統計 |
| `python main.py --rebuild` | 強制重新轉檔 |
| `python main.py --from-csv` | 略過 Parquet 直接查 CSV（用來比較速度） |

輸出是一張 Markdown 表格，19 項資料探索統計，可直接貼進報告。

## 為什麼用 DuckDB 而不是 SQLite

我們做的都是「掃過三千萬筆算統計」這類分析型查詢，那是欄式資料庫的強項；
SQLite 是列式儲存，為「抓出或更新單筆」設計。而且 DuckDB 可以直接對 CSV / Parquet
下 SQL，不需要先把資料匯入，省掉一個在 3.2 GB 規模下會爆記憶體的步驟。

## 注意事項

- `article_id` 長得像 `0108775015`，有前導零。讀檔時必須指定為 `VARCHAR`，
  否則會被當成整數、前導零消失，之後 JOIN 會全部對不上而且不會報錯。
  `customer_id` 同理。程式裡的 `TABLES` 常數就是在處理這件事。
- API 金鑰一律放在 `.env`（已在 `.gitignore` 中排除），絕對不要寫進程式碼。

## 協作方式

1. 開新分支再改：`git switch -c feature/你要做的事`
2. 推上去之前先確認清單：`git status`
3. 推上去後在 GitHub 開 Pull Request，請對方看過再合併
4. 拿最新版：`git switch main && git pull`

## 目前進度

| 週次 | 內容 | 狀態 |
|---|---|---|
| W3 | 執行提案 | 完成 |
| W4 | 資料管線、探索分析、評估框架、B0 / B1 基準線 | 進行中 |
| W5 | 召回模組 | 未開始 |
| W6–W7 | LightGBM 排序模型 | 未開始 |
