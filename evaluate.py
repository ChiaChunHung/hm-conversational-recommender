"""
評估層。回答一個問題：「這份推薦清單有多好？」

規格全部來自 eda.ipynb 的建模共識 §2，寫在這裡一份，所有模型共用同一把尺 ——
B0、B1、排序模型、加了對話的版本，都得用同一支函式打分才比得出高下。

    答案卷    truth_view()           (顧客, 品項) 相異組合 ← **唯一定義**，下面三支都讀它
              truth_of()             同上，一位買家一列，不帶名字
              truth_by_customer()    同上但鍵是 customer_id，逐顧客打 MAP 時用
    答案紙    top_k()                全站熱銷前 12（B0 的輸出，也是補位用的清單）
              segment_top()          各分群的熱銷前 12（B1 的輸出）
              pad_to_k()             不足 12 件時拿上面那份清單補滿，跳過重複
    打分      ap_at_k()              單一顧客，分母 min(正解數, 12)，與 Kaggle 一致
              evaluate()             跑完整個保留週，回傳 B0 / B1 的 MAP 與命中率
              coverage()             第一階段的指標：召回漏掉了多少正解

所有需要查資料的函式，第一個參數都是 con。不從外面偷抓全域變數：
函式簽名看得出它要什麼，放到哪個檔案都能跑。
"""

from __future__ import annotations

import duckdb
import numpy as np

import dataset


def ap_at_k(truth: set, pred: list, k: int = 12) -> float:
    """單一顧客的 Average Precision@K，分母 min(正解數, K)，與 Kaggle 一致。

    先去重：同一件商品在推薦清單裡出現兩次，會被算成兩次命中，
    分數可能超過 1（實測 [A,A,B] 對正解 {A,B} 會得到 1.5）。
    共識規定清單不得重複，這裡再擋一次，讓錯誤的清單也算出對的分數。
    """
    pred = list(dict.fromkeys(pred))   # 去重且保留順序
    hits = 0
    score = 0.0
    for i, a in enumerate(pred[:k], 1):
        if a in truth:
            hits += 1
            score += hits / i
    return score / min(len(truth), k)


def pad_to_k(pred: list, filler: list, k: int = 12) -> list:
    """把不足 k 件的推薦清單用 filler（通常是 B0 熱銷榜）補滿，補的時候跳過重複。"""
    out = list(pred)
    for a in filler:
        if len(out) >= k:
            break
        if a not in out:
            out.append(a)
    return out[:k]


def top_k(con: duckdb.DuckDBPyConnection, start: str, end: str, k: int = 12) -> list[str]:
    """[start, end) 內成交最多的 k 個品項。同票以 article_id 決勝，確保可重現。"""
    return [r[0] for r in con.execute(f"""
        SELECT article_id FROM tx WHERE t_dat >= '{start}' AND t_dat < '{end}'
        GROUP BY 1 ORDER BY count(*) DESC, article_id LIMIT {k}
    """).fetchall()]


def truth_view(con: duckdb.DuckDBPyConnection, start: str, end: str) -> None:
    """建檢視表 truth：[start, end) 內相異的 (customer_id, article_id) 組合。

    **答案卷的唯一定義。** 底下 truth_of / truth_by_customer / coverage 都讀這一張，
    誰都不准自己再寫一段撈正解的 SQL。

    為什麼要這樣管：覆蓋率在召回層算、MAP 在這裡算，兩個分數都得拿同一張答案卷去對。
    定義各寫一份的話，哪天日期邊界改了而只改到一份，兩邊就會在批改不同的考卷 ——
    保留週從 09-16 改成 09-15 起算，母體會從 68,984 位買家變成 75,481 位，
    然後你會看到覆蓋率達標但 MAP 低得莫名其妙，而且全程不報錯。

    這是檢視表，換日期重新呼叫即可，不佔記憶體。
    """
    con.execute(f"""
        CREATE OR REPLACE VIEW truth AS
        SELECT DISTINCT customer_id, article_id FROM tx
        WHERE t_dat >= '{start}' AND t_dat < '{end}'
    """)


def truth_of(con: duckdb.DuckDBPyConnection, start: str, end: str) -> list[list[str]]:
    """答案卷，一位買家一列，但不帶 customer_id。

    保留週跑出來是 68,984 位買家、213,728 組（顧客 × 品項）；
    撈出來不是這個數字就是哪裡寫錯了，最常見是忘了去重或日期邊界差一天。
    """
    truth_view(con, start, end)
    return list(con.execute(
        "SELECT list(article_id) FROM truth GROUP BY customer_id").df().iloc[:, 0])


def truth_by_customer(con: duckdb.DuckDBPyConnection, start: str,
                      end: str) -> dict[str, set[str]]:
    """同樣是答案卷，但鍵是 customer_id：{顧客: {他買過的品項}}。

    用在**逐顧客打 MAP@12** —— ap_at_k() 要的是一個 Python set，
    而那時候每人只剩 12 件，拉進記憶體很便宜。

    覆蓋率不要用這支：那是 285 萬列候選對 21 萬組正解的比對，
    留在 SQL 裡 join 只要零點幾秒，拉進 Python 做集合運算又慢又吃記憶體。
    覆蓋率請用 coverage()。
    """
    truth_view(con, start, end)
    df = con.execute(
        "SELECT customer_id, list(article_id) AS truth FROM truth GROUP BY 1").df()
    return {r.customer_id: set(r.truth) for r in df.itertuples()}


def coverage(con: duckdb.DuckDBPyConnection, sources: list[str],
             test_start: str, test_end: str) -> dict:
    """召回覆蓋率：[test_start, test_end) 真正買的東西，有多少被這些候選源撈到了。

    sources 是 recall.py 建出來的表名清單，例如 ['cand_repurchase']；多個會先 UNION 去重。

    這是第一階段的指標。召回的職責是不要漏掉正解 —— 排序模型再強，也救不回第一關
    就被丟掉的商品，所以覆蓋率是整個系統的上限。W5 的驗收標準是 30%。

    回傳兩種，兩個都要看：

      micro  命中的 (顧客, 品項) 組數 / 總正解組數。W5 的 30% 說的是這個。
             買很多件的顧客在這裡權重較大。
      macro  每位顧客各自的覆蓋率再平均。每個人一票，看的是「一般顧客」的體驗。

    兩者差很多，代表召回對重度與輕度買家的效果不一樣，值得往下查。

    全部在 SQL 裡算完，只把幾個數字拉出來。
    """
    truth_view(con, test_start, test_end)
    union = " UNION ".join(f"SELECT customer_id, article_id FROM {s}" for s in sources)
    row = con.execute(f"""
        WITH cand AS ({union}),
             hit AS (
                 SELECT t.customer_id, c.article_id IS NOT NULL AS hit
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


def segment_top(con: duckdb.DuckDBPyConnection, pop_start: str, pop_end: str,
                k: int = 12) -> dict[tuple[str, str], list[str]]:
    """每個分群在 [pop_start, pop_end) 的熱銷前 k 名，鍵是 (年齡層, 品牌線)。

    呼叫前 seg 檢視表要先存在（dataset.build_segments）。
    """
    return {(r.ageband, r.ig): list(r.top) for r in con.execute(f"""
        SELECT ageband, ig, list(article_id ORDER BY n DESC, article_id)[1:{k}] AS top
        FROM (SELECT s.ageband, s.ig, t.article_id, count(*) AS n
              FROM tx t JOIN seg s USING (customer_id)
              WHERE t.t_dat >= '{pop_start}' AND t.t_dat < '{pop_end}'
              GROUP BY 1, 2, 3)
        GROUP BY 1, 2
    """).df().itertuples()}


def evaluate(con: duckdb.DuckDBPyConnection, test_start: str, test_end: str,
             pop_start: str) -> dict:
    """在 [test_start, test_end) 上評估 B0 與 B1。

    分群用 test_start 當訓練上界重建一次 —— 分群依據是交易歷史，
    把要評估的那一週算進去就等於先看答案。

    pop_start 請一律用 dataset.pop_start_for(test_start) 算，不要填 dataset.POP_START：
    那個常數是拿保留週算出來的，換成驗證週會讓熱銷區間從 14 天悄悄縮成 7 天，
    而且不會報錯。

        evaluate(con, HOLDOUT_START, HOLD_END, dataset.pop_start_for(HOLDOUT_START))
        evaluate(con, VAL_START,     VAL_END,  dataset.pop_start_for(VAL_START))
    """
    dataset.build_segments(con, test_start)

    b0 = top_k(con, pop_start, test_start)
    segs = segment_top(con, pop_start, test_start)

    gt = con.execute(f"""
        SELECT s.ageband, s.ig, list(DISTINCT t.article_id) AS truth
        FROM tx t JOIN seg s USING (customer_id)
        WHERE t.t_dat >= '{test_start}' AND t.t_dat < '{test_end}'
        GROUP BY t.customer_id, 1, 2
    """).df()

    a0 = np.array([ap_at_k(set(t), b0) for t in gt["truth"]])
    a1 = np.array([ap_at_k(set(r.truth), pad_to_k(segs.get((r.ageband, r.ig), []), b0))
                   for r in gt.itertuples()])
    return {"買家數": len(gt), "分群數": len(segs),
            "B0": a0.mean(), "B1": a1.mean(),
            "B0命中率": (a0 > 0).mean(), "B1命中率": (a1 > 0).mean()}
