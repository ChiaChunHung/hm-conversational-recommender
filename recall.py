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

# 分群熱門每群取前幾名。在驗證週上掃過 12 / 30 / 50 / 100 / 200：
#
#     k     這一源單獨   加上回購   每人候選數
#      12       2.0%       6.9%        60
#      30       4.1%       8.8%        78
#      50       5.8%      10.4%        98     ← 採用
#     100       9.3%      13.7%       148
#     200      15.0%      19.0%       247
#
# 覆蓋率隨 k 幾乎線性成長，這一源還沒飽和 —— 但候選池也跟著線性成長，
# 排序模型要篩的東西變多、推論變慢。提案 §2.2.2 的預算是每人 50-100 件，
# 50 剛好貼齊上限。要換取更高覆蓋率就得先放寬那個預算。
SEGMENT_TOP_K = 50


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
                    k: int = SEGMENT_TOP_K,
                    days: int = dataset.POP_DAYS) -> int:
    """建 cand_segment：每位顧客拿自己那一群最近 days 天賣最好的前 k 名。

    分群是 (年齡層 × 主要品牌線)，共 35 群，定義在 dataset.build_segments。
    B1 基準線用的是同一組標籤 —— 兩邊若各分一次群，分數就不可比。

    這一源專門補回購補不到的洞：回購只給得出「他買過的東西」，
    新客與換口味的人身上是空的。分群熱門對誰都給得出東西。

    k 要在驗證週上掃描決定，不要看保留週的覆蓋率挑 ——
    那是拿測試集調參數（共識 §1）。

    排序加了 article_id 決勝，否則同票時每次跑出來的前 k 名會不一樣（共識 §4）。
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
        WHERE r.rk <= {k}
    """)
    return con.execute("SELECT count(*) FROM cand_segment").fetchone()[0]
