# DSO585 — H&M Conversational Recommender System

以大型語言模型作為對話介面、背後接兩階段推薦引擎（候選召回 + LightGBM 排序）的對話式推薦系統。
資料來源為 Kaggle「H&M Personalized Fashion Recommendations」。

## 環境需求

- Python 3.10 以上
- 套件：`duckdb`、`pandas`、`numpy`、`scipy`、`implicit`、`matplotlib`、`ipykernel`（見 `requirements.txt`）
- `implicit` 是召回階段的 ALS 用的，只有 `recall.als()` 會 import 它

裝在虛擬環境裡，不要裝到全域 Python：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

之後每次開新的終端機，都要先 `source .venv/bin/activate` 才能跑 `main.py`。
在 VS Code 裡開終端機會自動啟用，不用手動打。

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
python main.py build
```

這是 private repo，要先被加為 collaborator 才能 clone。網址是 SSH 格式，
需要先把你的公鑰加到 GitHub（`ssh -T git@github.com` 回覆 Hi 你的帳號就是設好了）。

### 為什麼一定要放在外面

放在 repo 外面，git 物理上看不到它，**不可能**誤傳。放在裡面就只能靠 `.gitignore`
擋，而且 `git clean -xdf`（很常見的「清乾淨重來」指令）會刪掉所有被 ignore 的檔案，
一個指令就讓 3.7 GB 資料蒸發，得重載一次。

程式會在啟動時檢查，資料缺檔或放錯位置都會直接停下來並告訴你怎麼修。

## 程式結構

| 檔案 | 作用 |
|---|---|
| `main.py` | 入口。只做命令列分派，實際工作在模組裡 |
| `dataset.py` | 共用資料層：路徑、型別、時間邊界、DuckDB 連線 |
| `prepare.py` | CSV → Parquet 轉檔 |
| `evaluate.py` | 評估層：MAP@12 打分、答案卷、召回撈中率、B0 / B1 基準線 |
| `recall.py` | 第一階段候選召回：回購、分群熱門、共購、ALS 四源 + 候選池組裝 |
| `eda.ipynb` | 資料探索：推導建模共識（附錄 A 統計、缺失值規格、基準線） |

`evaluate.py` 與 `recall.py` 目前只供 import，沒有在 `main.py` 的 `COMMANDS`
註冊指令。之後的排序模組平行放在同一層。

**`recall.py` 的模組 docstring 是這個專案最長的一份說明**，裡面有【名詞對照】
（撈中率、MAP@12、2.5% 目標分別是什麼、屬於哪一階段）和【待解決】（目前的瓶頸
與決議）。要動召回或排序之前先讀那兩節。

## 執行

先確認虛擬環境已啟用（提示字元開頭有 `(.venv)`），否則先執行 `source .venv/bin/activate`。

### 轉檔

```bash
python main.py build
```

第一次執行會把三份 CSV 轉成 Parquet，存在 `data/parquet/`，約 2 秒。
之後重跑會自動跳過已存在的檔案。轉出來的檔案不進版控，刪掉重跑即可。

| 指令 | 作用 |
|---|---|
| `python main.py build` | 需要時轉檔 |
| `python main.py build --rebuild` | 強制重新轉檔 |

### 資料探索

轉檔完成後開 `eda.ipynb`，Run All 即可。輸出包含可直接貼回提案附錄 A 的
Markdown 表格，以及年齡分布、價格分布、月交易量、長尾集中度四張圖。

首次開啟時 VS Code 右上角要選 `.venv` 的 Python 直譯器。

所有聚合都在 SQL 裡完成，只把小結果交給 Python 畫圖 ——
3178 萬筆明細若整包拉進 pandas 會吃掉約 6.2 GB。

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

## 第一次 clone 後要做的事

除了建 venv、放資料之外，還要啟用 repo 內建的 git hook：

```bash
git config core.hooksPath .githooks
```

**只需做一次，但每次重新 clone 都要再做。** git 不會自動套用 repo 裡的 hook —— 那是刻意的
安全設計，否則光是 clone 別人的專案就會執行到裡面的腳本。

目前只有一個 hook：`pre-commit` 會擋下「提案兩份複本不一致」的 commit。
提案的正本在 vault 的 `proposals/`（在 repo 外面，進不了版控），repo 裡是複本；
兩份不會自己同步，最容易發生的是在 Obsidian 改完忘了同步過來。
沒有 vault 的人（例如只 clone repo 的組員）不受影響，hook 會自動略過。

要單次略過檢查用 `git commit --no-verify`。

## 協作方式

1. 開新分支再改：`git switch -c feature/你要做的事`
2. 推上去之前先確認清單：`git status`
3. 推上去後在 GitHub 開 Pull Request，請對方看過再合併
4. 拿最新版：`git switch main && git pull`

## 目前進度

| 週次 | 內容 | 狀態 |
|---|---|---|
| W3 | 執行提案 | 完成 |
| W4 | 資料管線、探索分析、評估框架、B0 / B1 基準線 | 完成（B0 0.00710、B1 0.00927，保留週）|
| W5 | 召回模組：四源 + 候選池組裝 + 撈中率 | 四源完成；驗收標準兩條中達成一條 |
| W6–W7 | LightGBM 排序模型 | 未開始 |
| W8 | 商品文字向量 + ChromaDB（語意檢索那一源） | 未開始 |

W5 的驗收標準是「保留週被召回的比例 ≥ 30%」與「ALS 兩種用法完成比較」。
後者已完成（原始 vs BM25 加權，結論記在 `recall.py` 的 `ALS_FACTORS` 註解）；
前者目前在驗證週是 17.0%，沒達到。決議是先做 W6 的排序模型，看它實際發揮到上限
的幾成，再決定要不要放大候選池 —— 完整理由見 `recall.py` 的【待解決】。

保留週的數字封存中，只在 W5 收尾時量一次，所以這裡引的召回數字一律是驗證週。
