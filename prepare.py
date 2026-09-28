"""
把 data/ 底下三份 CSV 轉成 Parquet。

為什麼轉 Parquet：
  CSV 是純文字，每個數字每次查詢都要重新從字元解析一遍，而且想讀任何一欄都得掃過整列。
  Parquet 是二進位欄式格式，只讀你點名的欄位，還會壓縮。轉一次，之後每次查詢都受益。

只做一次，之後重跑會自動跳過已存在的檔案。
"""

from __future__ import annotations

import sys
import time

import dataset
from dataset import DATA_DIR, PARQUET_DIR, TABLES


def build_parquet(rebuild: bool = False) -> None:
    """把每份 CSV 轉成同名的 Parquet 檔。已存在就跳過，除非 rebuild=True。

    自己檢查資料、自己開連線，呼叫端不必先準備任何東西 ——
    命令列與 notebook 都可以直接呼叫。
    """
    dataset.check_data_dir()
    con = dataset.raw_connect()

    print(f"資料目錄：{DATA_DIR}")
    print("準備 Parquet")
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    for name, types in TABLES.items():
        csv_path = DATA_DIR / f"{name}.csv"
        pq_path = PARQUET_DIR / f"{name}.parquet"

        if not csv_path.exists():
            sys.exit(f"資料目錄是 {DATA_DIR}，但裡面缺少 {name}.csv")

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
