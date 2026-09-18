"""
H&M 對話式推薦系統 — 資料管線與探索統計

做兩件事：
  1. 把 data/ 底下三份 CSV 轉成 Parquet（只做一次，之後重跑會自動跳過）
  2. 對 Parquet 下 SQL，產出提案附錄 A 的那張統計表

為什麼用 DuckDB 而不是 SQLite：
  SQLite 是列式儲存，為「抓出或更新某一筆」設計；我們做的全是「掃過三千萬筆算統計」，
  那是欄式儲存的強項。而且 DuckDB 可以直接對 CSV / Parquet 下 SQL，沒有匯入這一步。

為什麼轉 Parquet：
  CSV 是純文字，每個數字每次查詢都要重新從字元解析一遍，而且想讀任何一欄都得掃過整列。
  Parquet 是二進位欄式格式，只讀你點名的欄位，還會壓縮。轉一次，之後每次查詢都受益。

執行：
    python main.py              # 需要時轉檔，然後印出統計
    python main.py --rebuild    # 強制重新轉檔
    python main.py --from-csv   # 不用 Parquet，直接查 CSV（拿來比較速度）
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

try:
    import duckdb
except ModuleNotFoundError:
    sys.exit("缺少 duckdb，請先執行：pip install duckdb")


# 資料放在這個 repo 外面（見 README）。預設往上一層找 data/，
# 也就是 DSO585/data/。如果你的資料放在別的地方，設環境變數覆寫：
#     export DSO585_DATA=/your/path/to/data
DATA_DIR = Path(os.environ.get("DSO585_DATA", Path(__file__).parent.parent / "data"))
PARQUET_DIR = DATA_DIR / "parquet"

# 三份來源檔，以及必須強制指定型別的欄位。
#
# 這裡是整支程式最容易踩的坑：article_id 長得像 "0108775015"，開頭有零。
# 讓 DuckDB 自動推斷型別的話會被當成整數，前導零就消失了，之後跟 articles 表
# JOIN 會全部對不上。customer_id 是 64 字元雜湊，同樣必須是字串。
TABLES = {
    "articles": {"article_id": "VARCHAR", "product_code": "VARCHAR"},
    "customers": {"customer_id": "VARCHAR", "postal_code": "VARCHAR"},
    "transactions_train": {"article_id": "VARCHAR", "customer_id": "VARCHAR"},
}

# 資料集的時間邊界，直接寫死避免每次重算。最後一天是 2020-09-22。
LAST_DAY = "2020-09-22"
HOLDOUT_START = "2020-09-16"   # 保留週：最後 7 天
LAST_4W_START = "2020-08-26"   # 最近四週
PRICE_SCALE = 590              # 還原近似價的倍率，見提案 2.1.2


# --------------------------------------------------------------------------
# 第一步：CSV → Parquet
# --------------------------------------------------------------------------

def build_parquet(con: duckdb.DuckDBPyConnection, rebuild: bool = False) -> None:
    """把每份 CSV 轉成同名的 Parquet 檔。已存在就跳過，除非 rebuild=True。"""
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    for name, types in TABLES.items():
        csv_path = DATA_DIR / f"{name}.csv"
        pq_path = PARQUET_DIR / f"{name}.parquet"

        if not csv_path.exists():
            sys.exit(f"找不到 {csv_path}，請先把 Kaggle 資料放進 data/")

        if pq_path.exists() and not rebuild:
            print(f"  跳過 {name}.parquet（已存在，{pq_path.stat().st_size / 1e6:.0f} MB）")
            continue

        t0 = time.perf_counter()
        # read_csv 的 types 參數就是在鎖死型別，不讓它自動推斷。
        con.execute(f"""
            COPY (SELECT * FROM read_csv('{csv_path}', types={types!r}))
            TO '{pq_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        csv_mb = csv_path.stat().st_size / 1e6
        pq_mb = pq_path.stat().st_size / 1e6
        print(f"  {name}: {csv_mb:>7.0f} MB CSV → {pq_mb:>6.0f} MB Parquet"
              f"  （壓縮到 {pq_mb / csv_mb:.0%}，耗時 {time.perf_counter() - t0:.1f} 秒）")


# --------------------------------------------------------------------------
# 第二步：建立檢視表
# --------------------------------------------------------------------------

def register_views(con: duckdb.DuckDBPyConnection, from_csv: bool = False) -> None:
    """把三份檔案註冊成 SQL 檢視表 tx / art / cus。

    註冊成檢視表而不是實體表，是因為 DuckDB 查 Parquet 已經夠快，
    不需要再複製一份到記憶體或自己的資料庫檔裡。
    """
    for view, name in (("art", "articles"), ("cus", "customers"), ("tx", "transactions_train")):
        if from_csv:
            src = f"read_csv('{DATA_DIR / (name + '.csv')}', types={TABLES[name]!r})"
        else:
            src = f"read_parquet('{PARQUET_DIR / (name + '.parquet')}')"
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM {src}")


# --------------------------------------------------------------------------
# 第三步：附錄 A 的統計
# --------------------------------------------------------------------------

def appendix_a(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """重算提案附錄 A 的每一列，回傳 (項目, 數值) 的清單。"""
    q = lambda sql: con.execute(sql).fetchone()
    rows: list[tuple[str, str]] = []

    # 交易期間與總量
    mn, mx, n_tx, n_cust_tx, n_art_sold = q("""
        SELECT min(t_dat), max(t_dat), count(*),
               count(DISTINCT customer_id), count(DISTINCT article_id) FROM tx
    """)
    n_cust_all = q("SELECT count(*) FROM cus")[0]
    n_art_all = q("SELECT count(*) FROM art")[0]
    rows.append(("交易期間", f"{mn} 至 {mx}"))
    rows.append(("交易筆數", f"{n_tx:,}"))
    rows.append(("有交易的顧客數 / 顧客總數", f"{n_cust_tx:,} / {n_cust_all:,}"))
    rows.append(("有成交的品項數 / 品項總數", f"{n_art_sold:,} / {n_art_all:,}"))

    # 最近四週有成交的品項
    n_4w = q(f"SELECT count(DISTINCT article_id) FROM tx WHERE t_dat >= '{LAST_4W_START}'")[0]
    rows.append(("最近四週有成交的品項", f"{n_4w:,}（{n_4w / n_art_all:.1%}）"))

    # 通路占比。通路 2 在 COVID 封城期間佔 100%，這是判定 2 = 線上的依據。
    c1, c2 = q("""
        SELECT round(100.0 * count(*) FILTER (WHERE sales_channel_id = 1) / count(*), 1),
               round(100.0 * count(*) FILTER (WHERE sales_channel_id = 2) / count(*), 1) FROM tx
    """)
    rows.append(("通路 1（實體店）/ 通路 2（線上）交易占比", f"{c1}% / {c2}%"))

    a1, a2 = q("""
        SELECT round(100.0 * count(*) FILTER (WHERE sales_channel_id = 1) / count(*), 1),
               round(100.0 * count(*) FILTER (WHERE sales_channel_id = 2) / count(*), 1)
        FROM tx WHERE t_dat >= '2020-04-01' AND t_dat < '2020-05-01'
    """)
    rows.append(("2020 年 4 月通路 1（實體店）/ 通路 2（線上）占比（COVID 封城，門市關閉）",
                 f"{a1}% / {a2}%"))

    # 單日購物籃平均件數：線上訂單通常一次買比較多件，是通路判定的輔助證據。
    b1, b2 = q("""
        SELECT round(avg(n) FILTER (WHERE sales_channel_id = 1), 1),
               round(avg(n) FILTER (WHERE sales_channel_id = 2), 1)
        FROM (SELECT customer_id, t_dat, sales_channel_id, count(*) AS n
              FROM tx GROUP BY 1, 2, 3)
    """)
    rows.append(("單日購物籃平均件數：通路 1（實體店）/ 通路 2（線上）", f"{b1} / {b2}"))

    # 價格 ×590 還原後最常見的價位，用來佐證那個倍率是對的
    top_prices = con.execute(f"""
        SELECT round(price * {PRICE_SCALE}, 2) AS p FROM tx
        GROUP BY 1 ORDER BY count(*) DESC LIMIT 3
    """).fetchall()
    rows.append((f"價格 ×{PRICE_SCALE} 後最常見的三個價位",
                 "、".join(f"{p[0]:g}" for p in top_prices)))

    p10, p50, p90, pmax = q("""
        SELECT round(quantile_cont(price, 0.1), 4), round(quantile_cont(price, 0.5), 4),
               round(quantile_cont(price, 0.9), 4), round(max(price), 4) FROM tx
    """)
    rows.append(("價格 P10 / P50 / P90 / 最大", f"{p10} / {p50} / {p90} / {pmax}"))

    # 每位顧客的交易筆數分布
    med, mean, cp90, once = q("""
        SELECT quantile_cont(n, 0.5), round(avg(n), 1), quantile_cont(n, 0.9),
               count(*) FILTER (WHERE n = 1)
        FROM (SELECT customer_id, count(*) AS n FROM tx GROUP BY 1)
    """)
    rows.append(("每位顧客交易筆數中位數 / 平均 / P90", f"{med:g} / {mean} / {cp90:g}"))
    rows.append(("只買過一次的顧客", f"{once:,}（{once / n_cust_tx:.1%}）"))

    # 同品項回購：rn > 1 代表這位顧客先前買過同一件
    repeat = q("""
        SELECT round(100.0 * count(*) FILTER (WHERE rn > 1) / count(*), 1)
        FROM (SELECT row_number() OVER (PARTITION BY customer_id, article_id ORDER BY t_dat) AS rn
              FROM tx)
    """)[0]
    rows.append(("同品項回購占交易比例", f"{repeat}%"))

    # 長尾集中度：前 1% 與前 10% 的品項各佔多少交易
    t1, t10 = q("""
        WITH s AS (SELECT article_id, count(*) AS n FROM tx GROUP BY 1),
             r AS (SELECT n, row_number() OVER (ORDER BY n DESC) AS rk, count(*) OVER () AS tot FROM s)
        SELECT round(100.0 * sum(n) FILTER (WHERE rk <= tot * 0.01) / sum(n), 1),
               round(100.0 * sum(n) FILTER (WHERE rk <= tot * 0.10) / sum(n), 1) FROM r
    """)
    rows.append(("前 1% / 前 10% 品項的交易占比", f"{t1}% / {t10}%"))

    # 保留週（最後 7 天），也就是離線評估要預測的那一週
    h_tx, h_cust, h_art = q(f"""
        SELECT count(*), count(DISTINCT customer_id), count(DISTINCT article_id)
        FROM tx WHERE t_dat >= '{HOLDOUT_START}'
    """)
    rows.append((f"保留週（{HOLDOUT_START[5:]} 至 {LAST_DAY[5:]}）交易 / 顧客 / 品項",
                 f"{h_tx:,} / {h_cust:,} / {h_art:,}"))

    # 保留週買家裡有多少人在那之前就買過東西。剩下的就是冷啟動新客。
    with_hist = q(f"""
        SELECT round(100.0 * count(*) FILTER (WHERE n_prior > 0) / count(*), 1)
        FROM (SELECT customer_id,
                     count(*) FILTER (WHERE t_dat < '{HOLDOUT_START}') AS n_prior,
                     count(*) FILTER (WHERE t_dat >= '{HOLDOUT_START}') AS n_last
              FROM tx GROUP BY 1) WHERE n_last > 0
    """)[0]
    rows.append(("保留週顧客有歷史交易的比例", f"{with_hist}%"))

    # 品牌線與商品群組的組成
    idx = con.execute("""
        SELECT a.index_group_name, round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
        FROM tx t JOIN art a USING (article_id) GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()
    rows.append(("品牌線占比", "、".join(f"{n} {p}%" for n, p in idx)))

    grp = con.execute("""
        SELECT a.product_group_name, round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
        FROM tx t JOIN art a USING (article_id) GROUP BY 1 ORDER BY count(*) DESC LIMIT 4
    """).fetchall()
    rows.append(("商品群組占比（前四）", "、".join(f"{n} {p}%" for n, p in grp)))

    # 2020 年的季節性
    monthly = con.execute("""
        SELECT strftime(t_dat, '%m') AS m, count(*) AS n
        FROM tx WHERE t_dat >= '2020-01-01' GROUP BY 1 ORDER BY 1
    """).fetchall()
    rows.append(("2020 年月交易量",
                 "、".join(f"{int(m)} 月{'（至 22 日）' if m == '09' else ''} {n / 10000:.0f} 萬"
                            for m, n in monthly)))

    return rows


# --------------------------------------------------------------------------

def print_markdown(rows: list[tuple[str, str]]) -> None:
    """印成 Markdown 表格，可直接貼回提案附錄 A。"""
    print("\n| 項目 | 數值 |")
    print("|---|---|")
    for item, value in rows:
        print(f"| {item} | {value} |")


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    from_csv = "--from-csv" in sys.argv

    con = duckdb.connect()  # 記憶體內連線，不落地資料庫檔

    if not from_csv:
        print("[1/2] 準備 Parquet")
        build_parquet(con, rebuild=rebuild)
    else:
        print("[1/2] 略過 Parquet，直接查 CSV")

    register_views(con, from_csv=from_csv)

    print(f"\n[2/2] 計算附錄 A 統計（來源：{'CSV' if from_csv else 'Parquet'}）")
    t0 = time.perf_counter()
    rows = appendix_a(con)
    print(f"  {len(rows)} 項統計，耗時 {time.perf_counter() - t0:.1f} 秒")

    print_markdown(rows)


if __name__ == "__main__":
    main()
