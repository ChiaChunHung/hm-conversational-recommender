"""
共用的資料層。回答兩個問題：

  1. 「資料怎麼拿？」
     → connect() 給你一個能直接下 SQL 的連線，tx / art / cus 都註冊好了。

  2. 「資料的規格是什麼？」
     → 路徑在哪（DATA_DIR / PARQUET_DIR）、欄位什麼型別（TABLES）、
       關鍵日期是哪幾天（LAST_DAY / HOLDOUT_START / LAST_4W_START）、
       價格倍率多少（PRICE_SCALE）。

第一個是動作，第二個是事實。兩者都只寫在這裡一份，所有模組共用。
若未來，日期或型別要改時，改這一個地方，用到的地方全部跟著變。

用法範例（下面的 SQL 只是示意，換成你實際要查的就好）：

    # 拿連線
    import dataset
    con = dataset.connect()
    con.execute("SELECT count(*) FROM tx").fetchone()

    # 拿規格
    from dataset import HOLDOUT_START
    con.execute(f"SELECT * FROM tx WHERE t_dat >= '{HOLDOUT_START}'")
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import duckdb
except ModuleNotFoundError:
    sys.exit("缺少 duckdb，請先執行：pip install -r requirements.txt")


# 資料一律放在這個 repo 的「外面」，也就是與 repo 並排的 data/。
#
# 為什麼不放 repo 裡面：放外面的話，git 物理上看不到它，不可能誤 commit。
# 放裡面就只能靠 .gitignore 擋，而且 git clean -xdf 會把被 ignore 的檔案
# 全部刪掉，一個指令就讓 3.7 GB 資料蒸發。
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
PARQUET_DIR = DATA_DIR / "parquet"

# 三份來源檔，以及必須強制指定型別的欄位。
#
# article_id 長得像 "0108775015"，開頭有零；customer_id 是 64 字元雜湊。
# 兩者都必須是字串，被當成整數的話前導零會消失，跟 articles 表 JOIN 就全對不上。
#
# 為什麼不信任自動推斷：DuckDB 的型別推斷是「抽樣」決定的（預設只看前 20480 列）。
# 這份資料剛好每筆 article_id 都有前導零，所以它推得出 VARCHAR；但只要前面幾萬列
# 剛好都沒有前導零，它就會推成 BIGINT，然後把後面的 "0000555" 靜靜變成 555，
# 不報任何錯。寫死型別就不必賭這件事。
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


def check_data_dir() -> None:
    """確認三份資料都在該在的位置，不在就說清楚該怎麼辦。"""
    missing = [n for n in TABLES if not (DATA_DIR / f"{n}.csv").exists()]
    if not missing:
        return

    # 目錄在、只是少幾個檔案，就直接點名少哪些
    if len(missing) < len(TABLES):
        listed = "\n".join(f"    {n}.csv" for n in missing)
        sys.exit(f"資料目錄 {DATA_DIR} 缺少這些檔案：\n{listed}")

    # 放在 repo 裡面是常見的誤會，特別提醒而不是讓他一頭霧水
    inside = HERE / "data"
    if (inside / "articles.csv").exists():
        sys.exit(
            f"資料放在 repo 裡面了：{inside}\n\n"
            "請把它移到 repo 的外面，與 repo 並排：\n"
            f"    mv {inside} {DATA_DIR}\n\n"
            "原因：放外面 git 就看不到它，不可能誤傳；放裡面只能靠 .gitignore 擋，\n"
            "而且 git clean -xdf 會把 .gitignore 忽略的檔案全部刪掉，資料會整包消失。"
        )

    sys.exit(
        f"找不到資料。應該放在：{DATA_DIR}\n\n"
        "請到 Kaggle 下載這三個檔案放進去：\n"
        "    articles.csv / customers.csv / transactions_train.csv\n"
        "    https://www.kaggle.com/competitions/"
        "h-and-m-personalized-fashion-recommendations/data"
    )


def check_parquet() -> None:
    """確認 Parquet 都轉好了，沒有就叫他先去轉。"""
    missing = [n for n in TABLES if not (PARQUET_DIR / f"{n}.parquet").exists()]
    if missing:
        listed = "\n".join(f"    {n}.parquet" for n in missing)
        sys.exit(
            f"Parquet 目錄 {PARQUET_DIR} 缺少這些檔案：\n{listed}\n\n"
            "請先執行轉檔：\n"
            "    python main.py build"
        )


def register_views(con: duckdb.DuckDBPyConnection) -> None:
    """把三份 Parquet 註冊成 SQL 檢視表 tx / art / cus。

    註冊成檢視表而不是實體表，是因為 DuckDB 查 Parquet 已經夠快，
    不需要再複製一份到記憶體或自己的資料庫檔裡。
    """
    for view, name in (("art", "articles"), ("cus", "customers"), ("tx", "transactions_train")):
        src = f"read_parquet('{PARQUET_DIR / (name + '.parquet')}')"
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM {src}")


def raw_connect() -> duckdb.DuckDBPyConnection:
    """開一個什麼都還沒註冊的記憶體連線。轉檔時用，那時 Parquet 還不存在。"""
    return duckdb.connect()


def connect() -> duckdb.DuckDBPyConnection:
    """開一個已經註冊好 tx / art / cus 的連線。分析與 EDA 的起手式。"""
    check_data_dir()
    check_parquet()
    con = duckdb.connect()
    register_views(con)
    return con
