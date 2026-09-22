# DSO585 — H&M Conversational Recommender System

以大型語言模型作為對話介面、背後接兩階段推薦引擎（候選召回 + LightGBM 排序）的對話式推薦系統。
資料來源為 Kaggle「H&M Personalized Fashion Recommendations」。

## 環境需求

- Python 3.10 以上
- 套件：`duckdb`

裝在虛擬環境裡，不要裝到全域 Python：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

之後每次開新的終端機，都要先 `source .venv/bin/activate` 才能跑 `main.py`。

## 資料準備

**資料不放在這個 repo 裡，要放在 repo 的外面。** 原始檔 3.7 GB，超過 GitHub 單檔
100 MB 的上限，而且 Kaggle 競賽資料僅授權學術與競賽用途，不可公開散布。

到 <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data>
下載 `articles.csv`、`customers.csv`、`transactions_train.csv`，然後照這個結構擺：

```
DSO585/                 ← 外層資料夾，名字隨意
├── data/               ← 三個 CSV 放這裡，與 repo 並排
└── <clone 下來的 repo>/ ← main.py 在這裡
```

指令大致如下：

```bash
mkdir DSO585 && cd DSO585
git clone git@github.com:ChiaChunHung/hm-conversational-recommender.git
mkdir data          # 三個 CSV 放進這裡
cd hm-conversational-recommender
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

這是 private repo，要先被加為 collaborator 才能 clone。網址是 SSH 格式，
需要先把你的公鑰加到 GitHub（`ssh -T git@github.com` 回覆 Hi 你的帳號就是設好了）。

### 為什麼一定要放在外面

放在 repo 外面，git 物理上看不到它，**不可能**誤傳。放在裡面就只能靠 `.gitignore`
擋，而且 `git clean -xdf`（很常見的「清乾淨重來」指令）會刪掉所有被 ignore 的檔案，
一個指令就讓 3.7 GB 資料蒸發，得重載一次。

程式會在啟動時檢查，資料缺檔或放錯位置都會直接停下來並告訴你怎麼修。

## 執行

先確認虛擬環境已啟用（提示字元開頭有 `(.venv)`），否則先執行 `source .venv/bin/activate`。

```bash
python main.py
```

第一次執行會把三份 CSV 轉成 Parquet，存在 `data/parquet/`，轉檔本身約 2 秒，
首次執行連同統計約 6 秒。之後每次執行都直接用 Parquet，統計從 12.5 秒降到 4 秒。
轉出來的檔案不進版控，刪掉重跑即可。

| 指令 | 作用 |
|---|---|
| `python main.py` | 需要時轉檔，然後輸出探索統計 |
| `python main.py --rebuild` | 強制重新轉檔 |
| `python main.py --from-csv` | 略過 Parquet 直接查 CSV（用來比較速度） |

輸出是一張 Markdown 表格，19 項資料探索統計，可直接貼進報告。

## 為什麼用 DuckDB 而不是 SQLite

我們做的都是「掃過三千萬筆算統計」這類分析型查詢，那是欄式資料庫的強項；
SQLite 是列式儲存，為「抓出或更新單筆」設計。而且 DuckDB 可以直接對 CSV / Parquet
下 SQL，不需要先把資料匯入，省掉一個在 3.5 GB 規模下會爆記憶體的步驟。
> **簡單來說：** DuckDB 讓我們能直接查詢 CSV/Parquet 檔案，
> 它只會在運算時分批讀取必要的資料進記憶體，不必像使用 Pandas / Polars 時那樣先把 3.5 GB 資料全數塞入 RAM；
> 也不必像 SQLite 需要先建立資料表並將 3.5 GB 資料完整匯入。這樣做可以節省時間也避免記憶體爆掉。

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
