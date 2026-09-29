"""
第一階段：候選召回。

從十萬件商品裡，先幫每位顧客縮到幾十件「有機會」的，再交給第二階段的排序模型。
為什麼要兩階段：對每位顧客的每件商品都算一次分數是 1,371,980 × 105,542 次，跑不完；
先粗篩再精排，是 Kaggle 獲獎方案與業界的共同做法（提案 §2.2.2）。

召回的唯一職責是**不要漏掉正解**。排序模型再強，也救不回第一關就被丟掉的商品 ——
所以這一層的指標是覆蓋率（coverage），不是準確率。W5 的驗收標準是保留週實際購買的
品項有 30% 以上被召回。

    候選來源            函式                狀態
    回購                repurchase()        ✓
    共購相似度          尚未實作
    ALS 隱因子          尚未實作（要先 pip install implicit）
    分群熱門            尚未實作（分群標籤用 dataset.build_segments）
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


def eval_customers(con: duckdb.DuckDBPyConnection, test_start: str, test_end: str) -> int:
    """建檢視表 eval_cus：要被評估的顧客，也就是 [test_start, test_end) 有買東西的人。

    這不是洩漏。共識 §2 把評估母體定義成「保留週有購買的 68,984 位顧客」，
    這裡用到的只是「這個人那週有買」，不是「他那週買了什麼」。
    候選本身一律只看 test_start 以前的資料。

    只為這些人建候選，而不是全部 1,371,980 位顧客，同時也是記憶體的考量：
    母體小 19.9 倍，候選池就小 19.9 倍。
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


def coverage(con: duckdb.DuckDBPyConnection, sources: list[str],
             test_start: str, test_end: str) -> dict:
    """召回覆蓋率：[test_start, test_end) 真正買的東西，有多少被這些候選源撈到了。

    sources 是表名清單，例如 ['cand_repurchase']；多個會先 UNION 去重。

    回傳兩種覆蓋率，兩個都要看：

      micro  命中的 (顧客, 品項) 組數 / 總正解組數。W5 的 30% 說的是這個。
             買很多件的顧客在這個數字裡權重較大。
      macro  每位顧客各自的覆蓋率再平均。每個人一票，看的是「一般顧客」的體驗。

    兩者差很多，代表召回對重度買家和輕度買家的效果不一樣，值得往下查。

    全部在 SQL 裡算完，只把幾個數字拉出來。
    """
    union = " UNION ".join(f"SELECT customer_id, article_id FROM {s}" for s in sources)
    row = con.execute(f"""
        WITH cand AS ({union}),
             truth AS (
                 SELECT DISTINCT customer_id, article_id FROM tx
                 WHERE t_dat >= '{test_start}' AND t_dat < '{test_end}'
             ),
             hit AS (
                 SELECT t.customer_id, t.article_id, c.article_id IS NOT NULL AS hit
                 FROM truth t
                 LEFT JOIN cand c
                        ON c.customer_id = t.customer_id AND c.article_id = t.article_id
             ),
             per_cus AS (
                 SELECT customer_id, avg(CASE WHEN hit THEN 1.0 ELSE 0.0 END) AS c
                 FROM hit GROUP BY 1
             )
        -- 全部寫成純量子查詢。把 hit 和 per_cus 擺在同一個 FROM 會變成交叉連接，
        -- 213,728 × 68,984 ＝ 147 億列。
        SELECT (SELECT count(*) FROM hit)                      AS 正解組數,
               (SELECT count(*) FROM hit WHERE hit)            AS 命中組數,
               (SELECT avg(c) FROM per_cus)                    AS macro,
               (SELECT count(*) FROM cand)                     AS 候選組數,
               (SELECT count(DISTINCT customer_id) FROM cand)  AS 有候選的顧客數
    """).fetchone()
    n_truth, n_hit, macro, n_cand, n_cus = row
    return {"正解組數": n_truth, "命中組數": n_hit,
            "micro": n_hit / n_truth, "macro": macro,
            "候選組數": n_cand, "有候選的顧客數": n_cus,
            "平均每人候選數": n_cand / n_cus if n_cus else 0.0}
