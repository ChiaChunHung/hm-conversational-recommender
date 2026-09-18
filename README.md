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

1. 到 <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data> 下載這三個檔案：
   - `articles.csv`（約 36 MB）
   - `customers.csv`（約 207 MB）
   - `transactions_train.csv`（約 3.5 GB）
2. 放到 repo 外層的 `data/` 資料夾，也就是這個結構：

   ```
   DSO585/
   ├── data/                  ← 資料放這裡（不進版控）
   │   ├── articles.csv
   │   ├── customers.csv
   │   └── transactions_train.csv
   └── code/                  ← 這個 repo
       ├── main.py
       └── README.md
   ```

3. 如果你想把資料放在別的地方，設環境變數即可：

   ```bash
   export DSO585_DATA=/your/path/to/data
   ```

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
