"""
共用的資料層。回答兩個問題：

  1. 「資料怎麼拿？」
     → connect() 給你一個能直接下 SQL 的連線，六張檢視表都註冊好了：

           tx              交易明細（3,178 萬筆）
           cus / art       顧客、商品 —— 原始，NULL 原封不動
           cus_c / art_c   同上，但套好建模共識 §3 的缺失值規格   ← 建模一律用這兩張

       為什麼原始版要留著：eda.ipynb §3 得數缺失值才推得出那張規格表，
       原始表一旦被補好，那一節會印出「全部無缺失」，共識的推導會把自己抹掉。

     → build_segments(con, train_end) 再建出第七張 seg：每位顧客的
       (年齡層, 主要品牌線)。B1 基準線與召回的「分群熱門」共用這一組標籤。

  2. 「資料的規格是什麼？」
     → 路徑在哪（DATA_DIR / PARQUET_DIR）、欄位什麼型別（TABLES）、
       時間怎麼切（FIRST_DAY / VAL_START / VAL_END / HOLDOUT_START / HOLD_END / LAST_DAY）、
       熱銷榜往回幾天（POP_DAYS / POP_START）、價格倍率多少（PRICE_SCALE）。

第一個是動作，第二個是事實。兩者都只寫在這裡一份，所有模組共用。
若未來，日期或型別要改時，改這一個地方，用到的地方全部跟著變。

打分是另一件事，在 evaluate.py：MAP@12、答案卷、B0 / B1。

用法範例（下面的 SQL 只是示意，換成你實際要查的就好）：

    import dataset

    con = dataset.connect()
    con.execute("SELECT count(*) FROM tx").fetchone()

    # 建模讀乾淨版，日期不要自己打字
    con.execute(f"SELECT * FROM cus_c WHERE ageband = '20-29'")
    con.execute(f"SELECT * FROM tx WHERE t_dat < '{dataset.HOLDOUT_START}'")

    # 分群
    dataset.build_segments(con, dataset.HOLDOUT_START)
    con.execute("SELECT * FROM seg LIMIT 5").df()
"""
from __future__ import annotations

import datetime as dt
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
#
# 三段切分（見 eda.ipynb 的建模共識 §1）：
#
#     訓練   t_dat <  2020-09-09          驗證週以前
#     驗證週 2020-09-09 ~ 09-15           調參數只能看這裡
#     保留週 2020-09-16 ~ 09-22           只在最後跑一次
#
# 結束日一律寫成「不含」的那一天，SQL 用 t_dat >= start AND t_dat < end，
# 這樣不會有「邊界那天算哪邊」的爭議 —— 09-15 放錯邊，保留週就會多 26,053 筆。
FIRST_DAY = "2018-09-20"       # 資料的第一天
LAST_DAY = "2020-09-22"        # 資料的最後一天
VAL_START = "2020-09-09"       # 驗證週起（含）
VAL_END = "2020-09-16"         # 驗證週迄（不含）
HOLDOUT_START = "2020-09-16"   # 保留週起（含），最後 7 天
HOLD_END = "2020-09-23"        # 保留週迄（不含）
LAST_4W_START = "2020-08-26"   # 最近四週

# 熱銷榜往回取幾天。14 天是在驗證週上掃描 7/14/21/30/60/全部 之後選出來的，
# 完全沒有用到保留週的分數 —— 推導過程見 eda.ipynb §4.2。
POP_DAYS = 14
POP_START = (dt.date.fromisoformat(HOLDOUT_START) - dt.timedelta(days=POP_DAYS)).isoformat()

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


def register_clean_views(con: duckdb.DuckDBPyConnection) -> None:
    """再加兩張檢視表 cus_c / art_c，套用建模共識 §3 的缺失值規格。

    為什麼要兩套而不是直接把 cus / art 補好：eda.ipynb §3 要數缺失值才推得出那張規格表，
    原始表一旦被補好，那一節會印出「全部無缺失」，共識的推導過程會把自己抹掉。
    檢視表存的是 SQL 不是資料，多一套不佔記憶體。

        cus / art       原始，NULL 原封不動     → eda.ipynb 用
        cus_c / art_c   套好共識 §3            → 召回、排序、對話層用
    """
    # age 不補值：補中位數會讓「真的 32 歲」和「不知道幾歲」分不出來。
    # 缺的年齡改以 ageband 的「未填」承接 —— 分群鍵必須讓每位顧客都落在某一群，
    # 否則那 15,861 人（含 292 位保留週買家）會分不到群、拿不到推薦。
    con.execute("""
        CREATE OR REPLACE VIEW cus_c AS
        SELECT customer_id,
               -- FN / Active 這兩欄只出現過 1.0，資料裡沒有任何 0。
               -- NULL 的語意是「沒有這個標記」，補 0 是還原它本來的意思。
               coalesce(FN, 0) AS FN,
               coalesce(Active, 0) AS Active,
               -- NULL 在 SQL 裡問 = 'ACTIVE' 和 != 'ACTIVE' 都不會被選到，
               -- 兩邊都漏且不報錯。給它一個看得見的值，過濾時才一定落在某一邊。
               coalesce(club_member_status, 'UNKNOWN') AS club_member_status,
               -- 'NONE'（877,711）與 'None'（2）是同一個意思的兩種寫法，先合併再補。
               CASE WHEN fashion_news_frequency IS NULL
                         OR upper(fashion_news_frequency) = 'NONE'
                    THEN 'NONE' ELSE fashion_news_frequency END AS fashion_news_frequency,
               age,
               CASE WHEN age IS NULL THEN '未填'
                    WHEN age < 20 THEN '<20'
                    WHEN age < 30 THEN '20-29'
                    WHEN age < 40 THEN '30-39'
                    WHEN age < 50 THEN '40-49'
                    WHEN age < 60 THEN '50-59'
                    ELSE '60+' END AS ageband
        FROM cus
        -- postal_code 整欄不取：352,899 個相異值裡最大的一區有 120,303 人，
        -- 第二名只有 261 人，差 461 倍 —— 那是佔位值，不是真郵遞區號。
    """)

    # detail_desc 缺失的 416 件商品裡有 17 件是保留週正解，不能刪。
    # 補空字串會讓它們的文字向量全部退化成同一個點，等於在語意檢索裡永遠找不到；
    # 用名稱 + 類型 + 色系拼出來的替代文字雖然粗，至少保有可比對的語意。
    # 用 concat_ws 而不是 ||，因為 || 只要有一個運算元是 NULL，整串就變成 NULL。
    con.execute("""
        CREATE OR REPLACE VIEW art_c AS
        SELECT * EXCLUDE (detail_desc),
               coalesce(detail_desc,
                        concat_ws('、', prod_name, product_type_name, colour_group_name))
                 AS detail_desc
        FROM art
    """)


def build_segments(con: duckdb.DuckDBPyConnection, train_end: str) -> None:
    """建立檢視表 seg：每位顧客的 (年齡層, 主要品牌線)。

    這是 B1 基準線的分群依據，也是召回「分群熱門」那一源的分群依據 ——
    兩邊必須是同一組標籤，所以定義只放這裡一份。
    7 個年齡層 × 6 條品牌線（5 種 + 沒有購買紀錄的「未知」）＝ 42 組，顧客身上全部出現得到。
    但其中 7 組整組都是「未知」品牌線，共 15,271 位顧客 —— 他們在 train_end 以前沒買過東西，
    自然也不會出現在熱銷區間的交易裡，所以排不出分群熱銷榜，會整份落回 B0。
    B1 實際用到的是剩下的 35 群，共識 §4 的「35 個分群」指的就是這個。

    train_end 一定要傳，不能寫死：主要品牌線是從交易歷史算出來的，若把要評估的那一週
    也算進去，就變成用「他那週買了什麼」決定他屬於哪一群，再拿那一群的熱門去預測
    他那週會買什麼 —— 先看答案再作答。評估保留週傳 HOLDOUT_START，驗證週傳 VAL_START。
    """
    con.execute(f"""
        CREATE OR REPLACE VIEW seg AS
        SELECT c.customer_id, c.ageband, coalesce(d.ig, '未知') AS ig
        FROM cus_c c
        LEFT JOIN (
            -- 每位顧客買最多的那條品牌線。平手時以名稱字母序決勝：
            -- arg_max() 平手不具決定性，同一份資料連跑五次分群人數會跳動。
            SELECT customer_id, (list(index_group_name ORDER BY n DESC, index_group_name))[1] AS ig
            FROM (SELECT t.customer_id, a.index_group_name, count(*) AS n
                  FROM tx t JOIN art_c a USING (article_id)
                  WHERE t.t_dat < '{train_end}' GROUP BY 1, 2)
            GROUP BY 1
        ) d USING (customer_id)
        -- LEFT JOIN 而非 JOIN：有 9,699 位顧客從未買過任何東西，
        -- 用 JOIN 他們會整個消失，用 LEFT JOIN 則歸進「未知」品牌線。
    """)


def raw_connect() -> duckdb.DuckDBPyConnection:
    """開一個什麼都還沒註冊的記憶體連線。轉檔時用，那時 Parquet 還不存在。"""
    return duckdb.connect()


def connect() -> duckdb.DuckDBPyConnection:
    """開一個已經註冊好 tx / art / cus 的連線。分析與 EDA 的起手式。"""
    check_data_dir()
    check_parquet()
    con = duckdb.connect()
    register_views(con)
    register_clean_views(con)
    return con
