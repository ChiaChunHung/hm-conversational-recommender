"""
第一階段：候選召回。

從十萬件商品裡，先幫每位顧客縮到幾十件「有機會」的，再交給第二階段的排序模型。
為什麼要兩階段：對每位顧客的每件商品都算一次分數是 1,371,980 × 105,542 次，跑不完；
先粗篩再精排，是 Kaggle 獲獎方案與業界的共同做法（提案 §2.2.2）。

召回的唯一職責是**不要漏掉正解**。排序模型再強，也救不回第一關就被丟掉的商品 ——
所以這一層的指標是覆蓋率，不是準確率。W5 的驗收標準是保留週實際購買的品項有
30% 以上被召回。

覆蓋率本身在 evaluate.coverage()，不在這裡：那是評估指標，而且它要拿答案卷去對，
答案卷的定義只能有一份（evaluate.truth_view）。這個檔案只負責產候選。

**每一支函式都要一個日期，那個日期的意思是「把時鐘撥到哪一天」。**

候選的定義本身就綁著時間：「他近 12 週買過什麼」在 9/09 問和 9/16 問答案不一樣，
中間他又買了東西；「他那一群最近 14 天賣最好的 50 件」更是每週都在換。
實測同一位顧客：

    時鐘撥到 9/09  →  73 個候選
    時鐘撥到 9/16  →  82 個候選，其中 34 個是 9/09 那次沒有的

所以**沒有「這個人的候選池」這種東西，只有「某個時間點上，這個人的候選池」**。

為什麼需要好幾個時間點：W6 的排序模型要學「什麼樣的候選會被買」，而要學就得有
題目和答案 —— 題目是候選池，答案是他到底買了沒。答案只能從歷史裡挖，所以得假裝
現在是 9/16：用 9/16 以前的資料撈候選（題目），再翻開 9/16 那週的實際購買紀錄
（答案）。**一個日期 ＝ 一組完整的題目 + 答案。** 多撥幾次鐘，就有多組訓練資料，
模型才不會把某一週的促銷或天氣當成通則。

上線之後不用撥鐘 —— 那時「現在」就是今天，只有一份候選池。
歷史時間點是離線訓練階段才需要的東西。

    候選來源            函式                狀態
    回購                repurchase()        ✓
    共購相似度          尚未實作
    ALS 隱因子          尚未實作（要先 pip install implicit）
    分群熱門            segment_popular()   ✓
    語意檢索            尚未實作（W8）

每一源各自建一張 cand_<名字> 表，欄位一律是 (customer_id, article_id)。
分開建是為了看得出每一源的邊際貢獻 —— 加了一源覆蓋率沒動，那一源就不值得留。

**記憶體**：召回是展開，不是聚合。輸出遠大於輸入，一不小心就是上億列。
所有候選一律留在 DuckDB 裡（CREATE TABLE），不要 .df() 拉進 pandas ——
DuckDB 算不下會溢寫到 .tmp，pandas 不會，只會讓整台機器沒記憶體可用。
只有最後那幾個數字才拉出來。
"""

from __future__ import annotations

import duckdb

import dataset

# 回購往回看幾週。提案 §2.2.2 寫的是 12 週。
REPURCHASE_WEEKS = 12

# 分群熱門每群取前幾名。
#
# 在驗證週上掃過 12 / 30 / 50 / 100 / 200（不看保留週，看了就是拿測試集調參）：
#
#   top_n   這一源單獨   加上回購   每人候選數   每多一個候選換到的覆蓋率
#      12       2.0%       6.9%         60
#      30       4.1%       8.8%         78            0.107%
#      50       5.8%      10.4%         98            0.081%     ← 目前採用
#     100       9.3%      13.7%        148            0.067%
#     200      15.0%      19.0%        247            0.053%
#
# 為什麼選 50，而不是覆蓋率最高的 200：
#
# （這個 top_n 是「每群撈幾件當候選」，跟 evaluate.py 的 k=12「最終推薦幾件」無關）
#
# 1. 只看覆蓋率的話，答案永遠是「k 越大越好」，這個問題問不出停損點。
#    我們真正要最大化的是 MAP@12。top_n 拉大等於要排序模型從 247 件裡挑 12 件，
#    而不是從 98 件裡挑 —— 召回的上限提高了，但排序的準度可能反而掉。
#    哪一邊的效應大，要等排序模型做出來、在驗證週上掃 top_n 才知道。現在沒有那個證據。
#
# 2. 曲線上沒有拐點。報酬遞減但非常平緩，沒有哪裡突然掉一半。
#    也就是說，光看覆蓋率曲線本身「看不出」正確答案在哪。
#
# 3. 剩下還有共購、ALS、語意三源沒做。那三源會改變 top_n 的邊際價值 ——
#    如果它們各補 7-8%，top_n=50 就到 30% 了，根本不用動這個數字。
#
# top_n 只是一個數字，改完重跑幾秒。反過來，先把候選池撐到每人 247 件、
# 等 W6 才發現排序模型的訓練資料大 2.5 倍跑不動，代價高得多。
# 提案 §2.2.2 的預算是每人 50-100 件，50 剛好貼齊上限。
#
# 結論：五源做完先看總覆蓋率，不到 30% 才考慮提 top_n；
#       最終的 top_n 等排序模型出來後用驗證週的 MAP 決定，不是用覆蓋率。
SEGMENT_TOP_N = 50


def eval_customers(con: duckdb.DuckDBPyConnection, test_start: str, test_end: str) -> int:
    """決定「要幫誰做推薦」，答案存成檢視表 eval_cus。

    挑的是 [test_start, test_end) 這段期間有買東西的人。測保留週就是那七天的
    68,984 位買家，不是全部 1,371,980 位顧客。

    為什麼不用全部人：那 130 萬個沒買東西的人，你推薦什麼都不可能中。把他們算進去，
    每個模型的分數都會一起被稀釋成 1/19.9，好模型跟爛模型的差距也跟著縮小，
    反而看不出誰比較好。共識 §2 定的評估母體就是買家，提案拿來對照的 Kaggle 分數
    也是買家分母，換一種算法就沒得比。順帶好處是候選池小 19.9 倍，跑得動。

    「用『那週有買』來挑人，不算偷看答案嗎？」不算，差別在這裡：

        偷看答案  =  用「他那週買了什麼」去挑候選
        這裡做的  =  用「他那週有沒有買」去決定要不要幫他做推薦

    買了什麼完全沒用到。挑候選的函式一律只讀 test_start 以前的交易。
    """
    con.execute(f"""
        CREATE OR REPLACE VIEW eval_cus AS
        SELECT DISTINCT customer_id FROM tx
        WHERE t_dat >= '{test_start}' AND t_dat < '{test_end}'
    """)
    return con.execute("SELECT count(*) FROM eval_cus").fetchone()[0]


def repurchase(con: duckdb.DuckDBPyConnection, train_end: str,
               weeks: int = REPURCHASE_WEEKS) -> int:
    """建 cand_repurchase：顧客近 weeks 週買過的品項，以及那些款的其他顏色。

    依據是提案 §2.2.2：14.1% 的交易是同品項回購（排除同一籃買多件後仍有 4.4%）。
    「同款其他顏色」靠 product_code 展開 —— article_id 是「款 + 色」，
    product_code 只到「款」，所以同一個 product_code 底下就是同款的各種顏色。

    只讀 train_end 以前的交易。往後看一天都是拿答案當特徵。
    """
    start = dataset.pop_start_for(train_end, weeks * 7)
    con.execute(f"""
        CREATE OR REPLACE TABLE cand_repurchase AS
        SELECT DISTINCT t.customer_id, a2.article_id
        FROM tx t
        JOIN eval_cus USING (customer_id)
        JOIN art_c a1 ON a1.article_id = t.article_id
        JOIN art_c a2 ON a2.product_code = a1.product_code   -- 同款的所有顏色
        WHERE t.t_dat >= '{start}' AND t.t_dat < '{train_end}'
    """)
    return con.execute("SELECT count(*) FROM cand_repurchase").fetchone()[0]


def segment_popular(con: duckdb.DuckDBPyConnection, train_end: str,
                    top_n: int = SEGMENT_TOP_N,
                    days: int = dataset.POP_DAYS) -> int:
    """建 cand_segment：每位顧客拿自己那一群最近 days 天賣最好的前 top_n 名。

    分群是 (年齡層 × 主要品牌線)，共 35 群，定義在 dataset.build_segments。
    B1 基準線用的是同一組標籤 —— 兩邊若各分一次群，分數就不可比。

    這一源專門補回購補不到的洞：回購只給得出「他買過的東西」，
    新客與換口味的人身上是空的。分群熱門對誰都給得出東西。

    top_n 要在驗證週上掃描決定，不要看保留週的覆蓋率挑 —— 那是拿測試集調參數（共識 §1）。
    掃描結果與選 50 的理由見上面 SEGMENT_TOP_N 的註解。

    注意 top_n 跟 evaluate.py 裡那些 k=12 不是同一回事：
    那邊的 k 是「最終推薦幾件」（Kaggle 規定 12，不能改），
    這裡的 top_n 是「每群撈幾件當候選」（我們自己選的）。

    排序加了 article_id 決勝，否則同票時每次跑出來的前 top_n 名會不一樣（共識 §4）。
    """
    dataset.build_segments(con, train_end)
    start = dataset.pop_start_for(train_end, days)
    con.execute(f"""
        CREATE OR REPLACE TABLE cand_segment AS
        WITH pop AS (
            SELECT s.ageband, s.ig, t.article_id, count(*) AS n
            FROM tx t JOIN seg s USING (customer_id)
            WHERE t.t_dat >= '{start}' AND t.t_dat < '{train_end}'
            GROUP BY 1, 2, 3
        ),
        ranked AS (
            SELECT ageband, ig, article_id,
                   row_number() OVER (PARTITION BY ageband, ig
                                      ORDER BY n DESC, article_id) AS rk
            FROM pop
        )
        SELECT e.customer_id, r.article_id
        FROM eval_cus e
        JOIN seg s USING (customer_id)
        JOIN ranked r ON r.ageband = s.ageband AND r.ig = s.ig
        WHERE r.rk <= {top_n}
    """)
    return con.execute("SELECT count(*) FROM cand_segment").fetchone()[0]


# 每一源在最終候選池裡對應的欄位名。加新的一源就在這裡多一行，build() 不用改。
SOURCES = {
    "cand_repurchase": "from_repurchase",
    "cand_segment": "from_segment",
}


def build(con: duckdb.DuckDBPyConnection, test_start: str, test_end: str,
          save: bool = False) -> int:
    """跑完所有召回源，合成最終候選池 cand。這是這個檔案的成品。

    cand 一列一個 (顧客, 商品)，每一源各一個 0/1 欄位標記它有沒有撈到這件：

        customer_id   article_id   from_repurchase   from_segment
          顧客A       0751471001         1                0
          顧客A       0918890002         0                1
          顧客A       0448509014         1                1     ← 兩源都撈到

    同一件被兩源撈到不是重複，是訊號 —— 兩個不同的理由都指向它。
    所以這裡不能只做 UNION 去重，要把來源留下來：排序模型會拿「候選來自哪一源」
    當特徵用（提案 §2.2.2 的交互特徵）。

    test_start 就是「時鐘撥到哪一天」（見本檔最上面的說明）。所有召回源都只讀它
    以前的交易 —— 往後看一天就是拿答案當特徵。test_end 只用來決定要幫誰做推薦，
    不影響任何候選的內容。

        recall.build(con, "2020-09-16", "2020-09-23")   # 撥到 9/16，考 9/16 那一週
        recall.build(con, "2020-09-09", "2020-09-16")   # 撥到 9/09，考 9/09 那一週

    save=True 會順便把 cand 寫成 data/candidates/cand_<test_start>.parquet。

    為什麼要存檔：W6 的排序模型要用好幾個週次的候選池訓練（提案 §2.2.2 的多 fold），
    每次訓練都重算一遍太浪費；更重要的是**記憶體裡的東西沒有身分** ——
    今天用 top_n=50 算一份、明天用 top_n=100 算一份，事後你分不出模型 A 是拿哪一份訓練的。
    存成檔案才追得回來。

    預設不存，因為量覆蓋率一天要跑好幾次，每次都寫 176 MB 沒意義。
    """
    eval_customers(con, test_start, test_end)
    repurchase(con, test_start)
    segment_popular(con, test_start)

    # 每一源先貼上自己的旗標，再用 GROUP BY 把同一個 (顧客, 商品) 併成一列。
    # 內層用 UNION ALL 而不是 UNION：反正外層要 GROUP BY，UNION 會多做一次無用的去重。
    cols = list(SOURCES.values())
    parts = [
        "SELECT customer_id, article_id, "
        + ", ".join(f"{1 if c == flag else 0} AS {c}" for c in cols)
        + f" FROM {table}"
        for table, flag in SOURCES.items()
    ]
    con.execute(f"""
        CREATE OR REPLACE TABLE cand AS
        SELECT customer_id, article_id, {", ".join(f"max({c}) AS {c}" for c in cols)}
        FROM ({" UNION ALL ".join(parts)})
        GROUP BY 1, 2
    """)
    if save:
        dataset.CAND_DIR.mkdir(parents=True, exist_ok=True)
        f = dataset.CAND_DIR / f"cand_{test_start}.parquet"
        con.execute(f"COPY cand TO '{f}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    return con.execute("SELECT count(*) FROM cand").fetchone()[0]
