"""
召回階段的兩項自動檢查。每次改召回源之後都要重跑。

    python recall_checks.py [日期]        預設 2020-09-09（驗證週）

一、資料洩漏測試（point-in-time correctness）

    同一個時鐘 t，用兩份資料各跑一次 recall.build()：
      A. 完整的交易表
      B. 只留 t_dat < t 的交易表
    兩次的 cand 必須逐列、逐欄完全相同。只要有差異，就代表某一步讀到了 t 以後的資料。

    唯一的例外是「要幫誰做推薦」（eval_cus）：它本來就由 [t, t+7) 有沒有買東西決定，
    所以 B 直接沿用 A 的名單 —— 那一步只決定對象，不碰任何候選內容。

二、ALS 可重現性

    同一份資料、同一個 seed 訓練兩次，候選 (顧客, 商品, 分數, 名次) 必須完全相同。
    不同的話，分數的變動分不出是改善還是雜訊，所有 CV 的比較都失去意義。

兩項都通過才印 PASS，否則以非零狀態結束。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import dataset
import recall


def _diff(a: Path, b: Path) -> tuple[int, int, int]:
    """回傳 (A 列數, 只在 A 的列數, 只在 B 的列數)。所有欄位都參與比較。"""
    con = dataset.raw_connect()
    return con.execute(f"""
        SELECT (SELECT count(*) FROM '{a}'),
               (SELECT count(*) FROM (SELECT * FROM '{a}' EXCEPT ALL SELECT * FROM '{b}')),
               (SELECT count(*) FROM (SELECT * FROM '{b}' EXCEPT ALL SELECT * FROM '{a}'))
    """).fetchone()


def leakage_test(t: str, test_end: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "full.parquet", Path(tmp) / "cut.parquet"

        full = dataset.connect()
        recall.build(full, t, test_end)
        audience = full.execute("SELECT customer_id FROM eval_cus").df()
        full.execute(f"COPY cand TO '{a}' (FORMAT PARQUET)")
        del full

        cut = dataset.connect()
        tx = dataset.PARQUET_DIR / "transactions_train.parquet"
        cut.execute(f"""
            CREATE OR REPLACE VIEW tx AS
            SELECT * FROM read_parquet('{tx}') WHERE t_dat < '{t}'
        """)
        cut.register("_audience", audience)

        def same_audience(con, *_):
            con.execute("CREATE OR REPLACE TABLE eval_cus AS SELECT * FROM _audience")

        original, recall.eval_customers = recall.eval_customers, same_audience
        try:
            recall.build(cut, t, test_end)
        finally:
            recall.eval_customers = original
        cut.execute(f"COPY cand TO '{b}' (FORMAT PARQUET)")

        n, only_a, only_b = _diff(a, b)
        same = only_a == 0 and only_b == 0
        print(f"  洩漏測試 cand {n:>11,} 列   只在完整版 {only_a:,}   只在截斷版 {only_b:,}   "
              f"{'相同' if same else '不同 ← 有洩漏'}")
        return same


def als_reproducibility(t: str, test_end: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for run in (1, 2):
            con = dataset.connect()
            recall.eval_customers(con, t, test_end)
            recall.als(con, t)
            f = Path(tmp) / f"als_{run}.parquet"
            con.execute(f"COPY cand_als TO '{f}' (FORMAT PARQUET)")
            files.append(f)
        n, only_a, only_b = _diff(*files)
        same = only_a == 0 and only_b == 0
        print(f"  ALS 重現性 {n:>11,} 列   只在第一次 {only_a:,}   只在第二次 {only_b:,}   "
              f"{'相同' if same else '不同'}")
        return same


def main() -> None:
    t = sys.argv[1] if len(sys.argv) > 1 else dataset.VAL_START
    test_end = dataset.pop_start_for(t, -7)
    print(f"召回檢查，時鐘 t = {t}")
    ok = leakage_test(t, test_end) & als_reproducibility(t, test_end)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
