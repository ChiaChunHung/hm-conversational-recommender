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

**資料不放在這個 repo 裡，要放在 repo 的外面。** 原始檔 3.7 GB，超過 GitHub 單檔
100 MB 的上限，而且 Kaggle 競賽資料僅授權學術與競賽用途，不可公開散布。

到 <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data>
下載 `articles.csv`、`customers.csv`、`transactions_train.csv`，然後照這個結構擺：

```
DSO585/                    ← 隨便你取什麼名字，用來裝下面兩個資料夾
├── data/                  ← 資料放這裡，與 repo 並排
│   ├── articles.csv
│   ├── customers.csv
│   └── transactions_train.csv
└── dso585-hm-crs/         ← clone 下來的 repo
    ├── main.py
    └── README.md
```

也就是先建一個外層資料夾，在裡面 clone，再於旁邊建 `data/`：

```bash
mkdir DSO585 && cd DSO585
git clone https://github.com/<帳號>/dso585-hm-crs.git
mkdir data          # 三個 CSV 放這裡
```

### 為什麼一定要放在外面

放在 repo 外面，git 物理上看不到它，**不可能**誤傳。放在裡面就只能靠 `.gitignore`
擋，而且 `git clean -xdf`（很常見的「清乾淨重來」指令）會刪掉所有被 ignore 的檔案，
一個指令就讓 3.7 GB 資料蒸發，得重載一次。

程式偵測到資料被放在 repo 裡面時會直接停下來，並告訴你搬移指令。

### 資料放在別的地方

例如外接硬碟。執行前設一次環境變數即可：

```bash
export DSO585_DATA=/Volumes/外接硬碟/hm_data
python main.py
```

想每次開終端機都生效，把那行 `export` 加進 `~/.zshrc`（macOS 預設）或 `~/.bashrc`。
程式每次啟動都會印出實際使用的資料目錄，不會搞錯用到哪一份。

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
