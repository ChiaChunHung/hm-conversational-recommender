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
    共購相似度          cooccurrence()      ✓
    ALS 隱因子          als()               ✓（需要 implicit 套件）
    分群熱門            segment_popular()   ✓
    語意檢索            尚未實作（W8）

每一源各自建一張 cand_<名字> 表，欄位一律是 (customer_id, article_id)。
分開建是為了看得出每一源的邊際貢獻 —— 加了一源覆蓋率沒動，那一源就不值得留。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【待解決】覆蓋率只有 14%，把 MAP@12 的上限壓在 15.8%

  決議（2026-09-29）：先做 W6 的排序模型，看它實際發揮到上限的幾成。
  離目標 2.5% 還很遠的話，再回頭放大候選池。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

保留週實測，邏輯一條一條接下去：

    四個源做完，覆蓋率 14.0%（29,997 / 213,728 組正解）
       ↓
    68,984 位買家裡，有 48,486 人（70.3%）
    我們給他的 163 件候選中，一件他那週真的會買的都沒有
       ↓
    那 70% 的人，排序模型再厲害也是 0 分 —— 東西不在名單上，它連看都看不到
       ↓
    所以 MAP@12 最高只能到 15.84%

**「最高 15.84%」是怎麼算出來的**

排序模型唯一能做的事，是把候選池裡的正確答案往前擠。全部擠到最前面時，
AP 的每一項都是 1/1、2/2、3/3……**全部等於 1**，加起來就是命中件數，
再除以他買的件數 —— 那剛好就是他自己的覆蓋率。

    完美排序的 AP ＝ 命中件數 / 買的件數 ＝ 他的覆蓋率

每個人都這樣，所以「完美排序的 MAP」＝「每人覆蓋率的平均」。
實測 15.84%，與 macro 覆蓋率 15.76% 只差 0.08 個百分點
（差額來自那 1.6% 買超過 12 件的人，他們的 AP 分母是 min(件數, 12)）。

順帶：**「只能推 12 件」不是壓低分數的原因**。AP 的分母 min(正解數, 12)
就是在處理這件事 —— 買 3 件的人推對 3 件是滿分，買 50 件的人推對 12 件也是滿分。
而且 98.4% 的顧客買不到 12 件，12 個名額根本用不完。壓低分數的是上面那 70%。

**目標還在上限底下，所以不是做不到**

     0.93%   B1 基準線（現在的對手）
     2.50%   提案 §3.2 的目標
    15.84%   上限

排序模型要做的事，是在那 29.7% 有機會的顧客身上，把答案擠進前 12 名。

**放大候選池可以把上限推高，但不是免費的**（保留週實測）

    每人候選    覆蓋率     上限      排序要從幾件裡挑 12 件
       150      14.0%    15.8%            150
       301         —     21.6%            301
       694      29.0%    33.5%            694

要過 W5 原訂的「覆蓋率 ≥ 30%」，大概要每人 700 個候選，是提案 §2.2.2
寫的 50-100 預算的七倍。而且上限變高的同時，排序模型要排除的干擾項也多 4.6 倍。
**兩個效應方向相反，淨效果是正是負，只有真的做出排序模型才知道** —— 這就是
先走 W6 的理由。放大池子隨時能做（改三個數字重跑），W6 的時間花掉就沒了。

**語意檢索（ChromaDB）補不了這個洞**

它要有一句查詢才動得起來（「我要找婚禮穿的」），而離線評估裡沒有人說話，
所以它對 MAP@12 的貢獻是零 —— 不是效果不好，是結構上參與不了這個測量。
它的價值在對話路徑，提案 §3.2 用「模擬對話命中率（相對無對話版本提升 ≥ 20%）」
另外量。對話時使用者會說出意圖，那比離線盲猜多得到太多資訊，
實際體感準度可能遠高於 MAP@12 —— 但那要用對話那組指標才量得出來。

**記憶體**：召回是展開，不是聚合。輸出遠大於輸入，一不小心就是上億列。
所有候選一律留在 DuckDB 裡（CREATE TABLE），不要 .df() 拉進 pandas ——
DuckDB 算不下會溢寫到 .tmp，pandas 不會，只會讓整台機器沒記憶體可用。
只有最後那幾個數字才拉出來。
"""

from __future__ import annotations

import duckdb
import pandas as pd

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

# 共購統計看多久，以及每件商品保留幾個共購夥伴。weeks=12 與回購對齊。
#
# 在驗證週掃 top_n（合計 = 回購 12 週 + 分群 50 + 共購）：
#
#   top_n   這一源單獨   合計    每人候選數
#      —         —      10.4%      98      （沒有共購）
#      5       3.9%     11.5%     121
#     10       5.6%     12.5%     149      ← 採用
#     20       7.6%     13.9%     206
#     50      11.4%     16.7%     369
#
# 注意單獨 5.6% 但只讓合計 +2.1 個百分點 —— 其中 3.5 個百分點跟前兩源重疊。
# 合理：常被一起買的東西，本來就多半是熱銷品，分群熱門已經撈走了。
#
# 每多一個候選換到的覆蓋率，三源對比（驗證週）：
#
#   分群熱門 top_n=50     0.1160%   ← 效率最高
#   回購 4 週             0.1261%   （但只到 3.8%，窗口一拉長效率就掉）
#   回購 12 週            0.0842%
#   共購 top_n=10         0.0410%   ← 效率最低
#
# 那為什麼還留著共購：它是唯一能推出「他沒買過、也不在他那群熱銷榜上」的商品的源。
# 覆蓋率上重疊不代表沒價值 —— 排序模型會拿 from_cooc 這個旗標當特徵，
# 「三源都指向它」和「只有分群熱門撈到」是不同的訊號。
COOC_WEEKS = 12
COOC_TOP_N = 10

# ALS：看多久的互動、隱因子幾維、每人取幾個推薦。
# 近 12 週是 350 萬筆互動、52 萬位顧客、41,585 件商品，稀疏矩陣約 28 MB。
#
# 驗證週掃過的設定（三源基準 12.5%、每人 149）：
#
#   設定                  單獨    合計    邊際
#   factors=64 原始       2.6%   13.1%   +0.6pp / +25 候選
#   factors=64 BM25       3.8%   13.2%   +0.8pp / +21 候選   ← 採用
#   factors=128 BM25      4.2%   13.1%   +0.7pp / +19 候選
#   factors=64 BM25 30輪  3.7%   13.2%   +0.7pp / +21 候選
#
# 加維度、加輪數都沒有讓邊際貢獻動 —— ALS 撈到的東西，前三源多半也撈得到。
# 它是四源裡效率最差的（0.038%/候選，分群熱門是 0.116%）。
ALS_WEEKS = 12
ALS_FACTORS = 64
ALS_TOP_N = 50


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


def cooccurrence(con: duckdb.DuckDBPyConnection, train_end: str,
                 top_n: int = COOC_TOP_N, weeks: int = COOC_WEEKS) -> int:
    """建 cand_cooc：顧客買過的每件商品，帶出最常跟它一起被買的 top_n 件。

    「一起被買」的定義是**同一位顧客、同一天**算一籃。訓練期 900 萬籃，平均 3.1 件，
    最大一籃 159 件。用同籃而不是「這個人買過的所有東西兩兩配對」，是因為後者會把
    半年前買的襪子跟今天買的外套配成一對 —— 那不是「一起買」，只是「都買過」。

    這一源補的是另外兩源都補不到的：回購只給得出他買過的東西，分群熱門給的是全群
    一樣的榜單，兩者都無法從「他的個人品味」推到「他沒買過但可能喜歡」的商品。

    weeks 同時決定兩件事：共購統計看多久，以及拿顧客哪一段的購買紀錄去查表。
    top_n 是每件商品保留幾個共購夥伴，要在驗證週上掃描決定。

    排序加了 j 決勝，同票時才不會每次跑出不同的夥伴清單（共識 §4）。
    """
    start = dataset.pop_start_for(train_end, weeks * 7)

    # 先算商品對商品的共購次數。雙向存（i→j 和 j→i 各一列），查表時才不用管方向。
    # 近 12 週是 1,031 萬列、0.3 秒；先用 sum(n*(n-1)/2) 估過才建，沒有盲目展開。
    con.execute(f"""
        CREATE OR REPLACE TABLE cooc AS
        WITH basket AS (
            SELECT DISTINCT customer_id, t_dat, article_id FROM tx
            WHERE t_dat >= '{start}' AND t_dat < '{train_end}'
        )
        SELECT a.article_id AS i, b.article_id AS j, count(*) AS n
        FROM basket a JOIN basket b USING (customer_id, t_dat)
        WHERE a.article_id <> b.article_id
        GROUP BY 1, 2
    """)

    # 每件商品只留最強的 top_n 個夥伴。不留的話平均每件有 255 個夥伴，候選池會炸。
    con.execute(f"""
        CREATE OR REPLACE TABLE cand_cooc AS
        WITH top AS (
            SELECT i, j FROM (
                SELECT i, j, row_number() OVER (PARTITION BY i ORDER BY n DESC, j) AS rk
                FROM cooc
            ) WHERE rk <= {top_n}
        )
        SELECT DISTINCT t.customer_id, top.j AS article_id
        FROM tx t
        JOIN eval_cus USING (customer_id)
        JOIN top ON top.i = t.article_id
        WHERE t.t_dat >= '{start}' AND t.t_dat < '{train_end}'
    """)
    return con.execute("SELECT count(*) FROM cand_cooc").fetchone()[0]


def als(con: duckdb.DuckDBPyConnection, train_end: str, top_n: int = ALS_TOP_N,
        weeks: int = ALS_WEEKS, factors: int = ALS_FACTORS, seed: int = 42) -> int:
    """建 cand_als：用矩陣分解找出「口味相近的人在買什麼」，每人取 top_n 件。

    前三源都是規則：回購看他買過什麼、分群熱門看他那群在買什麼、共購看什麼跟什麼
    一起賣。ALS 不是規則，它從「誰買過什麼」這張稀疏表裡自己學出隱藏的口味維度，
    把每位顧客和每件商品都壓成一個 64 維向量，兩個向量相乘就是預測分數。
    所以它撈得到的東西，前三源不見得撈得到。

    只有隱性回饋：資料裡只有「買了」，沒有「不喜歡」。沒買過不等於不喜歡，
    可能只是沒看過 —— 這就是 implicit 這個套件名字的由來，它的損失函數是為這種
    資料設計的，不能拿評分預測那一套來套。

    共識 §2 允許推薦顧客已買過的商品，所以 filter_already_liked_items=False。

    seed 一定要固定。ALS 的因子是隨機初始化的，不固定的話同一份資料每次跑出不同的
    候選池，你會分不出分數變動是模型改善還是雜訊（共識 §4）。

    記憶體：只把整數索引拉進 Python，字串 ID 留在 DuckDB 最後再 join 回來。
    近 12 週的稀疏矩陣約 28 MB。
    """
    try:
        import numpy as np
        import scipy.sparse as sp
        from implicit.als import AlternatingLeastSquares
        from implicit.nearest_neighbours import bm25_weight
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"ALS 需要 implicit 套件（缺少 {e.name}）。請執行：pip install -r requirements.txt"
        ) from e

    start = dataset.pop_start_for(train_end, weeks * 7)

    # 先在 SQL 裡把字串 ID 換成連號整數 —— 稀疏矩陣只吃得下整數，
    # 而且 64 字元的 customer_id 拉 350 萬筆進 Python 太浪費。
    con.execute(f"""
        CREATE OR REPLACE TABLE als_ix AS
        WITH pair AS (
            SELECT DISTINCT customer_id, article_id FROM tx
            WHERE t_dat >= '{start}' AND t_dat < '{train_end}'
        ),
        u AS (SELECT customer_id, row_number() OVER (ORDER BY customer_id) - 1 AS ui
              FROM (SELECT DISTINCT customer_id FROM pair)),
        a AS (SELECT article_id, row_number() OVER (ORDER BY article_id) - 1 AS ai
              FROM (SELECT DISTINCT article_id FROM pair))
        SELECT p.customer_id, p.article_id, u.ui, a.ai
        FROM pair p JOIN u USING (customer_id) JOIN a USING (article_id)
    """)
    ix = con.execute("SELECT ui, ai FROM als_ix").df()
    n_u = int(ix["ui"].max()) + 1
    n_a = int(ix["ai"].max()) + 1
    mat = sp.csr_matrix(
        (np.ones(len(ix), dtype="float32"), (ix["ui"].to_numpy(), ix["ai"].to_numpy())),
        shape=(n_u, n_a),
    )
    del ix

    # BM25 加權。不加權的話，買很多的人和熱門商品會主導整個矩陣，
    # 學出來的因子等於在記熱銷榜 —— 而那件事分群熱門已經做了，重疊會更嚴重。
    # 驗證週實測：不加權單獨 2.6%，加了 3.8%。
    mat = bm25_weight(mat, K1=100, B=0.8).tocsr()

    model = AlternatingLeastSquares(factors=factors, random_state=seed)
    model.fit(mat, show_progress=False)

    # 只對要評估的顧客做推薦。他們之中有人在這段窗口沒有互動（新客），
    # ALS 給不出東西，那些人靠其他源補。
    users = con.execute("""
        SELECT DISTINCT i.ui FROM als_ix i JOIN eval_cus e USING (customer_id) ORDER BY 1
    """).df()["ui"].to_numpy()
    ids, _ = model.recommend(users, mat[users], N=top_n, filter_already_liked_items=False)

    rec = pd.DataFrame({
        "ui": np.repeat(users, ids.shape[1]).astype("int32"),
        "ai": ids.reshape(-1).astype("int32"),
    })
    con.register("als_rec", rec)
    con.execute("""
        CREATE OR REPLACE TABLE cand_als AS
        SELECT DISTINCT u.customer_id, a.article_id
        FROM als_rec r
        JOIN (SELECT DISTINCT customer_id, ui FROM als_ix) u USING (ui)
        JOIN (SELECT DISTINCT article_id, ai FROM als_ix) a USING (ai)
    """)
    con.unregister("als_rec")
    return con.execute("SELECT count(*) FROM cand_als").fetchone()[0]


# 每一源在最終候選池裡對應的欄位名。加新的一源就在這裡多一行，build() 不用改。
SOURCES = {
    "cand_repurchase": "from_repurchase",
    "cand_segment": "from_segment",
    "cand_cooc": "from_cooc",
    "cand_als": "from_als",
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
    cooccurrence(con, test_start)
    als(con, test_start)

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
