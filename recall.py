"""
第一階段：候選召回。

【這份檔案在做啥】

從 105,542 件商品裡，幫每位顧客挑出約 190 件「有機會」的，交給第二階段的排序模型
（ranking.py，W6）決定最後推哪 12 件。四個召回源各建一張 cand_<名字> 表，
欄位一律是 (customer_id, article_id)，最後由 build() 合成一張 cand。

    回購         他近 12 週買過什麼            repurchase()
    分群熱門     跟他同群的人最近在買什麼      segment_popular()   ← 主力，佔池子九成
    共購         跟他買過的東西常被一起買的    cooccurrence()
    ALS          口味相近的人在買什麼          als()
    語意檢索     尚未實作（W8）

分開建是為了看得出每一源的邊際貢獻 —— 加了一源撈中率沒動，那一源就不值得留。

每支函式都要給一個日期，意思是**把時鐘撥到哪一天**：那天以前的交易可以讀，那天
（含）以後一律不讀。候選的定義本身綁著時間 —— 同一位顧客撥到 9/02 是 363 個候選、
撥到 9/09 是 307 個，其中 94 個是上次沒有的。所以沒有「這個人的候選池」這種東西，
只有「某個時間點上，這個人的候選池」。上線之後不用撥鐘，那時「現在」就是今天。

【目的是啥】

這一層的唯一職責是**不要漏掉正解**。排序模型只能把池子裡已經有的東西往前排，不能
無中生有 —— 第一關就被丟掉的商品，整個系統再也拿不回來。

所以這一層量的是**撈中率**：那一週真的被買的品項，有幾成被我們撈進了池子。
（同一個數字有四個名字：這裡叫撈中率、evaluate.py 的函式叫 coverage、W5 驗收標準寫
「被召回的比例」、Kaggle 討論串寫 recall rate。）

**不量準確率** —— 池子裡夾帶他不會買的東西是排序模型要處理的事，這一層寧可多撈。
**也不能用 MAP@12 調參** —— 池子放大撈中率一定升，但 MAP 可能升也可能降（多撈進來的
同時也是干擾項），而現在還沒有排序模型，量不出淨效果。撈中率和 MAP@12 的唯一正當
關係是**撈中率決定 MAP@12 的上限**；兩者不能比大小，看到「撈中率 19.3%、目標 MAP
2.5%」不要讀成已經超標六倍，那是兩把不同的尺上的兩個刻度。

【結果是啥】

驗證週（2020-09-09 ~ 09-15）實測。**保留週封存中**，只在 W5 收尾時量一次 —— 反覆看
它，後續的判斷就會被它影響（共識 §1）。

    每人候選數    193.0 件
    撈中率        19.28%
    上限          21.18%   排序模型再完美，整個系統的 MAP@12 最高就是這個數
    目標           2.50%   提案 §3.2 —— 在上限底下，所以召回沒有把路堵死

72,019 位買家全部拿得到候選（新客缺口已經關掉），但其中 43,220 人（60.0%）的池子裡
沒有一件他那週真的會買的，那些人排序模型再厲害也是 0 分 —— 沒撈中的正解有 65.7% 落在
近 14 天熱銷 1000 名之外、7.2% 近 14 天根本沒賣過，那裡沒有訊號可以學。

而這個任務本身就難：同一份資料同一個指標，Kaggle 三千隊的銀牌方案是 MAP@12 0.030、
論文的雙塔模型約 0.034，我們的目標 0.025 落在那個量級裡。「百分比小」是題目的性質。

【啥等待定】

**一、W5 原訂「30% 以上被召回」沒達到。** 放大池子就幾乎搆得到：2026-10-18 兩折實測，
四個源全部放大到每人 462 件時是 29.5% / 29.9%，代價是候選和干擾項都變 2.4 倍。**決議（2026-09-29）：先做 W6 的排序模型，
看它發揮到上限的幾成，再決定要不要放大池子。**

**二、十個常數裡只有三個做過兩折驗證。** 小組決議（2026-10-17）：先用這組數字往下做
W6，之後有時間再回來補測。哪三個、為什麼需要兩折，見下面常數區。

**三、ALS 兩種用法都做完了，結論是第一種不值得用。** 第二種（model.recommend）接在
池子裡；第一種（商品隱因子算相似度，als_similar）實作完成但沒接進 SOURCES —— 它的
邊際比四源裡最低的還差。比較表見 ALS_SIM_TOP_N 的註解。

**四、語意檢索（W8）補不了撈中率這個洞** —— 它要有一句查詢才動得起來，而離線評估裡
沒有人說話。它的價值在對話路徑，提案 §3.2 另外用「模擬對話命中率」量。

排序階段的設計在 ranking.py。

**記憶體**：召回是展開不是聚合，輸出遠大於輸入。所有候選一律留在 DuckDB 裡
（CREATE TABLE），不要 .df() 拉進 pandas —— DuckDB 算不下會溢寫到磁碟，pandas 不會，
只會讓整台機器沒記憶體可用。只有最後那幾個數字才拉出來。
"""

from __future__ import annotations

import duckdb
import pandas as pd

import dataset

# ═══ 下面那十個常數是怎麼定出來的 ═══
#
# 旋鈕分兩類，分法就一句話：**把它調大，每人候選數會不會跟著變多？**
#
# **A 不會變多（3 個）—— 只換「是哪幾件」。沒有取捨，直接選撈中率最高的那個值。**
#
#   SEGMENT_DAYS     看 5 天還是 14 天，每群還是取 170 名
#   ALS_FACTORS      64 維還是 128 維，每人還是拿 25 件
#   SEGMENT_LADDER   換成哪一份榜，每人還是取 170 名（格子太小的人拿不滿，見它的註解）
#
# **B 會變多（7 個）—— 它多吃的名額就是別人少拿的。有取捨，只能比邊際。**
#
#   SEGMENT_TOP_N、COOC_TOP_N、ALS_TOP_N      直接就是「每人／每件拿幾個」
#   REPURCHASE_WEEKS、COOC_WEEKS、ALS_WEEKS   窗口拉長 → 撈得到東西的人和商品都變多
#   EXPAND_COLOURS                            打開就把同款所有顏色加進來（10.8 → 61.5 件）
#
# **B 類問「這一源最佳是幾週」問不出答案** —— 撈中率對它們單調遞增，答案永遠是
# 「拉到最大」。所以 B 類不能單獨問，只能比較「同樣多的候選，給誰換到更多撈中率」，
# 而這種比較要有個基準。我們用的基準是**維持在目前這個量級（每人 190~200 件）**，
# 來自 2026-09-29 的決議：先做 W6 的排序模型，再決定要不要放大池子。
#
# **這不是硬上限，我們沒訂過「每人最多幾件」。** 提案 §2.2.4 本來就把「每人候選數
# 上限」列為要在驗證週上選的參數；風險表那個「每人 50」是記憶體撐不住時的退路。
# 三輪走下來實際是 191.1 → 198.9 → 194.8 → 193.0，上下都動過幾件 —— 所以約束是
# 「不准大幅放大」這個判斷，不是某個數字。決議一改（例如 W6 做完決定衝 30%），
# 下面整套取捨就要重算。
#
# 在這個前提下，剩下能做的動作就是「把名額從一個源挪到另一個源」—— 所以 B 類的調參
# 和分名額是同一件事，沒辦法分開問。做法三步：
#
#   1. 標準是**效率 ＝ 撈中率 ÷ 每人候選數**（一個候選換到多少撈中率），不是撈中率
#   2. 一次只動一個參數掃一輪，相鄰兩列相減 ＝ **邊際效率**（在現在這套設定下，
#      多給這一源一個名額，四源合計多幾個百分點）。分名額看邊際，不看單獨效率
#   3. 從邊際低的源收名額給高的，**挪到四個打平就是最佳**（還挪得動就還沒最佳）
#
# 三條守則，不照著做會調出假的改善：
#
# **一、邊際只在你量的那一點成立，不能外插。** 它擋的是這種算法：「現在 19.28%，要
# 30% 還差 10.7pp，一個候選換 0.064%，那多加 167 個就到了」。實際量過（兩折）：把四個
# 源全部放大（回購 52 週 / 分群 300 名 / 共購 50 個夥伴 / ALS 200 件）要每人 462 件、
# 多 269 個候選，才到 29.5%，比預估的 167 個多了六成。因為榜是按熱銷排好的，往後拿的是
# 越來越不會被買的東西，一個候選的平均價錢從 0.064% 掉到 0.038%。
#
# **二、邊際不能相加。** 四源單獨是 2.77 + 2.72 + 2.94 + 15.42 ＝ 23.85%，四個一起跑
# 只有 19.28% —— 差的 4.57pp 是重複撈到的。所以組合的結果只能實測。
#
# **三、窗口類的參數要兩折**（驗證週 + 緊貼它前面的 09-02 ~ 09-08，看平均）。只在一週
# 上挑最大值會挑到手氣，實例見 SEGMENT_DAYS。第二折必須相鄰：隔太遠連目錄都換過了，
# 分不出是雜訊還是市場真的變了。第二折不是保留週，保留週繼續封存。
#
# 三條合起來的意思是：**名額一挪，整套設定就變，手上的表全部作廢**，所以流程是迴圈
# （掃 → 比 → 挪 → 重掃），下面那三輪就是跑了三次。實例見 COOC_TOP_N 的註解：共購的
# 窗口改了之後，它 top_n 的邊際翻了一倍。
#
# ── 底下那些掃描表怎麼讀 ──
#
# 掃一個旋鈕的時候，其他旋鈕必須釘死 —— 不然四個一起動，撈中率變了也分不出是誰造成的。
# 每張表的撈中率都是**四源合計**（不是那一源自己的成績），而且都是 2026-10-18 在現行
# 設定下兩折量的，所以可以直接跟下面的 19.28% 比。
#
# **但表隨時會再過期。** 名額一挪別的源就變了，而「這個值勝出」往往是靠跟別源的邊際比
# 出來的。最好的實例：ALS 窗口原本的結論是「12 週被 6 週**嚴格支配**，多拉六週是負貢
# 獻」，2026-10-18 重掃之後 12 週反而兩折最高 —— 結論沒變（還是 6 週），但理由整個換掉
# 了。所以動過任何參數之後，別的源的表都要當作可疑。
#
# ── 結果：現在這套設定（驗證週）──
#
#   來源        每人候選   單獨撈中率   單獨效率   邊際效率（兩折平均）
#   回購            10.8       2.77%    0.2565%    0.0414%   ← 邊際最低
#   共購            23.0       2.72%    0.1183%    0.0602%   ← 邊際最高
#   ALS             25.0       2.94%    0.1176%    0.0450%
#   分群熱門       169.8      15.42%    0.0908%    0.0549%
#   四源合計       193.0      19.28%    0.0999%
#
# 邊際那一欄是「把名額加給這一源」換到多少，2026-10-18 在現在這套設定下兩折重量的。
# 每個源的邊際都是用它自己最划算的旋鈕量的：共購是 top_n 5→10、分群是 170→220 名、
# ALS 是 top_n 25→50、回購是 12→26 週。共購還有第二個旋鈕（窗口 4→8 週）邊際 0.0496%，
# 排在分群之後、ALS 之前。
#
# **還沒打平**，最高與最低差 1.45 倍（第一輪之前是 2.5 倍）。要加候選的話第一個該加
# 的是共購的 top_n（5 → 10），不是分群；而那是放大池子，W6 之後才該碰。
#
#   第一輪 2026-10-06   重新分名額：共購 10→5、ALS 50→25、分群 100→150
#                       191.1 → 198.9 件，15.52% → 16.96%
#   第二輪 2026-10-06   調窗口：分群 14→5 天、ALS 12→6 週、共購 12→4 週，名額給分群 170
#                       198.9 → 194.8 件，16.96% → 18.11%（候選變少而撈中率變高）
#   第三輪 2026-10-15   換分群維度：42 組 → 四階降級鏈
#                       194.8 → 193.0 件，18.11% → 19.28%，新客掛零 5,395 人消成 0
#
# ── 守則二實際做到哪裡：全部都量了兩折 ──
#
# 2026-10-18 把所有過期的掃描表在現在這套設定下重跑了一遍，兩折都量（驗證週 +
# 09-02 ~ 09-08）。所以下面每一張表都是現行設定下的數字，沒有「舊設定」那種註腳了。
#
#   參數                     重量日期      兩折結果
#   REPURCHASE_WEEKS（12 週） 2026-10-18   12 週，但 8 週只差一點（見它的註解）
#   EXPAND_COLOURS（關閉）    2026-10-05   單獨效率差 3 倍以上，差距大到不必重量
#   SEGMENT_TOP_N（170）      2026-10-18   曲線沒拐點，170 是停損點不是最佳點
#   SEGMENT_DAYS（5 天）      2026-10-18   5 天在兩折都最高（之前是 3 天贏驗證週）
#   SEGMENT_LADDER（降級鏈）  2026-10-15   四源合計沒差，留著是原則
#   COOC_WEEKS（4 週）        2026-10-18   4 週，但理由跟舊註解寫的不一樣
#   COOC_TOP_N（5）           2026-10-18   邊際四源最高，下一輪要加就加這裡
#   ALS_WEEKS（6 週）         2026-10-18   6 週，但**「12 週被嚴格支配」那句話是錯的**
#   ALS_FACTORS（64）         2026-10-18   **128 維在兩折都更好而且不花候選**，待決定
#   ALS_SIM_TOP_N（未採用）   2026-10-17   邊際比四源最低的還差，實作完成但不接進池子
#
# **小組決議（2026-10-17）**：回購用 12 週、分群熱門用 5 天定案，先往下做 W6，之後有
# 時間再回來補兩折。**2026-10-18 已經補完**（就是上面那張表），兩個定案都沒被推翻。


##########################################################################
##########################################################################
# 這區討論回購參數：

# 回購往回看幾週。REPURCHASE_WEEKS 目前 12 週。
# 怎麼掃的：另外三個源釘死不動，只改回購的週數，每改一次就把四個源全部跑一遍、量整個
# 池子。所以**下表的撈中率都是四源合計**，不是回購自己的成績。2026-10-18 兩折重量：
#
#   回購看幾週   池子每人   驗證週      第二折      兩折平均   邊際
#        4        188.0   18.9316%   19.6641%   19.2978%
#        8        189.9   19.1132%   19.8161%   19.4647%   0.0878%
#       12        193.0   19.2826%   19.9631%   19.6229%   0.0510%   ← 採用
#       26        199.8   19.5498%   20.2597%   19.9047%   0.0414%
#       52        211.0   19.8657%   20.6608%   20.2633%   0.0320%
#
# 窗口拉長，「他買過的東西」就變多，所以池子變大、撈中率也變高。52 週確實比 12 週多撈
# 0.64pp，但要多吃 18 個候選 —— 一個只值 0.036%，那些名額給分群（0.0549%）換得更多。
#
# **12 週不是最佳點，是個「差不多」的點。** 8→12 這一步的邊際 0.0510% 其實略低於分群的
# 0.0549%，嚴格按打平原則該縮到 8 週。差距只有 7%，比兩折之間的落差還小，所以沒改；
# 但別說「12 週是量出來的最佳解」。往下縮到 4 週就明顯不行 —— 4→8 的邊際 0.0878% 是整
# 個設計裡最高的一段，砍掉很貴。
#
# 這個旋鈕影響本來就小：回購只佔每人 10.8 個候選，而且窗口再長也長不出他沒買過的東西。
# 提案 §2.2.2 也寫 12 週，但那不是留著 12 的理由 —— 掃過才留的。
REPURCHASE_WEEKS = 12

# 回購要不要把「同款其他顏色」也納入候選。EXPAND_COLOURS 目前 False（實測結論是不要）。
#
# （article_id 是「款 + 色」、product_code 只到「款」，所以展開就是同一個
#   product_code 底下的所有顏色。）
#
# 它讓回購這一源的候選從每人 10.8 個
# 漲到 61.5 個，撈中率只從 2.77% 到 5.18%，
# 效率從 0.257% 掉到 0.084% —— 等於拿全場最好的訊號去餵一堆雜訊。
#
# 對照組更直接（同樣分群取 100，只差有沒有換色）：多 0.64 個百分點、多吃 26 個
# 候選 ＝ 0.024%／候選，是整個設計裡最不划算的一項。把那些名額挪給分群熱門
# （50 → 100）換到的是撈中率 13.21% → 15.52%，而且**整體效率也跟著變好** ——
# 放大池子通常效率會掉，這裡反而升，因為名額是從爛的地方挪到好的地方。
#
# 換色不是完全沒用：實測有 5,480 組命中是「他沒買過那個顏色」才撈到的，別源補不上。
# 留著這個開關是因為候選預算之後若放寬（見【啥等待定】），值得重新評估。
EXPAND_COLOURS = False

##########################################################################
##########################################################################
# 這裡討論分群熱門：

# 分群熱門每群取前幾名。SEGMENT_TOP_N 目前 170。（這個 SEGMENT_TOP_N 是「每群撈幾件當候選」）
# 驗證週掃過，2026-10-18 兩折重量：
#
#   top_n   每人候選   驗證週      第二折      兩折平均   邊際
#     100     124.4   14.9778%   15.4662%   15.2220%
#     170     193.0   19.2826%   19.9631%   19.6229%   0.0641%   ← 採用
#     250     271.5   23.3605%   24.0602%   23.7104%   0.0521%
#     400     418.2   29.1830%   29.9516%   29.5673%   0.0399%
#
# **曲線沒有拐點**，報酬平緩遞減，所以這條曲線本身問不出「最佳值在哪」。170 不是最佳
# 值，是**停損點**：再往上加就是放大池子。要加的話這裡也不是第一順位，共購的邊際更高。
#
# 這個數字讓每人候選數（193）超出提案原訂的 50–100，是刻意的 —— §2.2.4 本來就把它
# 列為待選參數。代價落在 W6：目前總共 13,898,930 組訓練資料。真的跑不動，照上面那張
# 邊際表從最低的那一源先砍。
SEGMENT_TOP_N = 170

# 分群熱銷榜取最近幾天。SEGMENT_DAYS 目前 5 天。
# 這個旋鈕不花候選 —— 每群都是取 top_n 名，改天數只換「是哪 170 件」，所以沒有取捨，
# 純比高低。
#
#   天數   驗證週      第二折      兩折平均
#     3   19.2813%   19.8350%   19.5582%
#     5   19.2826%   19.9631%   19.6229%   ← 採用，兩折都最高
#     7   19.0264%   19.8627%   19.4446%
#    10   18.5393%   19.5317%   19.0355%
#    14   18.3625%   18.9597%   18.6611%
#
# 2026-10-18 在降級鏈下重掃的結果。**5 天在兩折都最高**，14 天在兩折都最低，差 0.96pp
# —— 零候選成本換到的，是整個檔案裡最便宜的一次改善。
#
# 值得記一筆：2026-10-06 第一次掃（當時是 42 組分群）量到「3 天在驗證週最高、在第二折
# 輸掉」，那是當初堅持要兩折的理由。重掃之後 3 天在兩折都輸，那個現象不見了 —— 但兩折
# 這條紀律該留著，它當初確實擋下了一個會選錯的決定。
#
# 為什麼短的贏：快時尚的熱銷榜換得快，14 天前賣得好的東西下週不見得還在賣。
#
# 這個常數**刻意不等於 dataset.POP_DAYS（14 天）** —— 那是 W4 用 B1 的 MAP 選的，目標
# 不同，而且動它會改掉已經交付的 B0 / B1。
SEGMENT_DAYS = 5

# 分群用哪些欄位切。SEGMENT_LADDER 目前走四階降級鏈：每位顧客用他的標籤湊得出來的第一階。
# 兩個欄位（定義見 dataset.build_segments）的來源不同，這是關鍵：ageband 年齡層 7 段
# 和 eda 同個切法、來自註冊資料，新客也有；ig 主要品牌線 5 條要從交易歷史算「他買哪線
# 最多」，新客是 'Unknown'。
#
#    階  條件                用哪份榜          驗證週人數   那份榜最少幾件
#    1  年齡有、品牌線有    (年齡 × 品牌線)     66,343            6   ← 10 格湊不滿 170
#    2  只有年齡（純新客）  (年齡)               5,355          612
#    3  只有品牌線          (品牌線)               281          792   ← 真訊號優先於 'Missing'
#    4  兩個都沒有          全站熱銷                40        湊得滿 ← 人人有候選的保證
#
# 為什麼不統一用一種切法 —— 拆開量，兩群人的結果相反（也因此第 2、3 階的榜都超過
# 170 件，降級之後不必補位）：
#
#   格子湊得滿 170 的 66,384 人   細切 15.38%   只用年齡 15.23%   細的略好
#   格子湊不滿的 5,635 人         細切  0.27%   只用年齡 15.68%   細的幾乎掛零（空榜）
#
# **四源合計沒有變好**：驗證週 +0.04pp、第二折 −0.04pp，一升一降，按兩折紀律就是雜訊。
# 留著它是因為這個做法可以確保兩件事：每人用手上最好的訊號；而最後一階的全站榜加上
# 函式裡那條 assert，讓「人人有候選」不靠剛好、靠程式強制（只用年齡時人人有候選，是
# 因為剛好每個年齡層都有 612 件以上可挑）。要簡化成兩階（年齡 → 全站）也站得住腳。
#
# 還沒修的一個缺口：降級鏈判斷要不要降級，看的是「你有沒有這個標籤」，不是「你的
# 格子裡商品夠不夠」。所以一位兩個標籤都有的顧客會走第 1 階，即使他那一格五天內只賣過
# 27 件 —— 他就只拿到 27 件候選，不會被降到年齡榜（那裡有 612 件以上可挑）。驗證週有
# 187 人是這樣，他們沒掛零，只是候選比別人少。
# 修法：多一條規則「格子不足 top_n 就降到下一階」。為什麼沒做 —— 那 187 人會從第 1 階
# 移到第 2 階，第 1 階人數一變整個池子就變了，前面七張掃描表全部得重跑一輪。
#
# eda 測出「加品牌線比較好」而這裡相反，是因為量法不同：eda 量 B1 的 MAP@12、每群取
# 12 件、不足用全站熱銷補滿；這裡量撈中率、取 170 件、不補位。
SEGMENT_LADDER = (("ageband", "ig"), ("ageband",), ("ig",), ())

# 每個分群欄位的「缺值」長什麼樣 —— 降級鏈靠這個判斷某一階的標籤湊不湊得出來。
# 這兩個字串**不是資料裡的值**，是 dataset.build_segments 自己填的記號：ageband 的
# CASE 用 'Missing' 承接缺年齡的人，ig 用 coalesce(..., 'Unknown') 承接沒有購買紀錄的
# 人。**兩邊要一字不差**，不然降級鏈會把缺值當成有效標籤，所有人都被判到第 1 階，
# 而且不會報錯。
SEGMENT_UNKNOWN = {"ageband": "Missing", "ig": "Unknown"}

##########################################################################
##########################################################################
# 這裡討論共購：

# 共購：統計看多久、每件商品留幾個夥伴。COOC_WEEKS 目前 4 週、COOC_TOP_N 目前每件留 5 個。
#
# 先說為什麼留著這一源：它跟別源重疊得很兇（單獨只有 2.72%，而且大部分別源也撈得到），
# 曾經被考慮砍掉。留著的理由有兩個，而**「它是唯一給得出他沒買過的東西的源」不是其中
# 之一** —— ALS 第二種用法也給得出，別把這句話寫回來：
#
#   一、它的推薦掛在「某一件他買過的商品」上。ALS 給的是「跟他整體口味相近」，說不出
#       是哪一件引起的；共購說得出「因為你買了 X」。提案 §2.2.3 要求每件推薦附一句
#       理由、且理由必須來自工具回傳的欄位，這一源是少數給得出來的。
#   二、排序模型會拿 from_cooc 當特徵，「三源都指向它」和「只有分群撈到」是不同的訊號。
#
# ── weeks = 4（兩折）──  原本寫 12，理由是「與回購對齊」—— 那不是證據，是為了好記。
# （提案沒有指定共購的窗口，12 是我們自己初版寫死的。）2026-10-18 兩折重量：
#
#   週數   這源每人   單獨    四源每人   驗證週      第二折      兩折平均   邊際
#     2      18.1   2.07%     190.0   19.0351%   19.7182%   19.3767%
#     4      23.0   2.72%     193.0   19.2826%   19.9631%   19.6229%   0.0821%  ← 採用
#     8      33.5   3.44%     201.2   19.6964%   20.3521%   20.0243%   0.0489%
#    12      47.1   3.92%     213.7   20.0474%   20.6864%   20.3669%   0.0274%
#    26      76.2   4.73%     241.9   20.6472%   21.3581%   21.0027%   0.0225%
#
# 往下縮不行：2→4 的邊際 0.0821% 是整個設計裡最高的一段。
# 往上加不划算：4→8 的 0.0489% 低於分群的 0.0549%，那 8 個候選給分群換得更多。
#
# **舊註解說「8→12 和 12→26 的邊際是全場最低」—— 那部分仍然成立（0.027% / 0.023%），
# 但不要因此以為共購窗口整體不值錢：4→8 那一段其實比 ALS（0.0450%）和回購（0.0414%）
# 都好。** 為什麼短的贏：十二週前跟外套一起買的襪子，跟「這週會不會一起買」沒什麼關係。
#
# ── top_n = 5 ──  **這是四源裡邊際效率最高的旋鈕**（5 → 10 換到 0.0602%／候選，
# 2026-10-18 兩折平均）。停在 5 是因為加上去會讓每人候選漲到約 200 件，那是放大池子。
# **下一輪要加候選的話，這裡是第一個該加的地方。**
#
# 2026-10-06 那一輪實測過一個交互作用：weeks 從 12 縮到 4 之後，top_n 的邊際翻了一倍
# （0.0302% → 0.0568%）。那兩個數字是舊設定下量的，但那件事本身確實發生過 ——
# **改了 weeks 就不能沿用舊的 top_n 結論，要重掃。**
COOC_WEEKS = 4
COOC_TOP_N = 5

##########################################################################
##########################################################################
# 這裡討論 ALS：

# ALS：互動窗口幾週、隱因子幾維、每人取幾件。ALS_WEEKS 目前 6 週、ALS_FACTORS 目前 64 維
# + BM25 加權、ALS_TOP_N 目前每人 25 件。這一源在做什麼見 als() 的 docstring。
#
# ── weeks = 6（兩折）──  2026-10-18 重量：
#
#   週數   每人候選   驗證週      第二折      兩折平均   邊際
#     4      190.6   19.1462%   19.7779%   19.4620%
#     6      193.0   19.2826%   19.9631%   19.6229%   0.0670%   ← 採用
#     8      194.7   19.3818%   20.0194%   19.7006%   0.0457%
#    12      197.7   19.4353%   20.0383%   19.7368%   0.0121%
#
# **這張表推翻了 2026-10-06 的結論。** 當時寫「12 週被 6 週嚴格支配 —— 候選更少而撈中率
# 更高，多拉那六週的舊互動是負貢獻」。重掃之後，12 週其實是兩折最高的（比 6 週多 0.11pp），
# 只是很貴：8→12 這一步一個候選只值 0.0121%，整個設計裡最差的一段。
# 所以 6 週仍然是對的停點，但理由是**「再加不划算」，不是「加了有害」**。
# 往下縮也不行：4→6 的邊際 0.0670% 高過分群的 0.0549%，砍掉很貴。
#
# ── factors = 64 + BM25（兩折）──  2026-10-18 重量：
#
#   設定            ALS 單獨（驗證／第二折）   四源每人   四源合計（驗證／第二折）
#   64 原始          1.99% / 1.83%            193.0    19.0939% / 19.7338%
#   64 BM25          2.94% / 2.79%            193.0    19.2826% / 19.9631%   ← 現行
#   128 BM25         3.30% / 3.13%            192.8    19.3243% / 19.9967%
#   64 BM25 30 輪    2.93% / 2.75%            193.1    19.2984% / 19.9732%
#
# **BM25 加權是免費的改善**：候選數一樣，四源合計多 0.19~0.23pp。不加權的話，買很多的
# 人和熱門商品會主導整個矩陣，學出來等於在記熱銷榜 —— 而那件事分群熱門已經做了。
#
# **128 維在兩折都比 64 好（+0.042 / +0.034pp），而且候選數還少一點點 —— 這是一個待決定
# 的改動，代價是訓練時間變兩倍（21 秒 → 41 秒）。** 2026-10-06 的舊結論寫「加維度沒讓
# 邊際動」，那是在只有三源的基準下量的，現在四源都在、窗口也都調過，結果不一樣。
# 加迭代次數（15 → 30 輪）沒有用，兩折都只有 +0.01pp 等級。
#
# ── top_n = 25 ──  邊際 0.0450%／候選（兩折平均），四源倒數第二，不是加候選的第一
# 順位。砍到 25 之後它的**單獨**效率反而變高（0.118%）—— 前 25 名本來就比較準，
# 第 26~50 名多半是雜訊。
ALS_WEEKS = 6
ALS_FACTORS = 64
ALS_TOP_N = 25

# 第一種用法（als_similar）每件商品帶幾個最像的。**目前 5，但這一源沒有接進 SOURCES**
# —— 它實作完成、量完，結論是不值得給名額。
#
# ── ALS 兩種用法的比較（驗證週；提案 §2.2.2 承諾的那個比較）──
#
#   用法                      每人候選   單獨撈中率   單獨效率   加進池子的邊際
#   第二種 recommend()            25.0       2.94%   0.1176%        0.0482%
#   第一種 similar_items()        30.1       1.88%   0.0623%        0.0277%
#
# 第一種用法掃過鄰居數 1/2/3/5/10，邊際最好的是 2 個鄰居的 0.0308% —— 仍然低於四源裡
# 最低的回購（0.0414%），更不用說共購的 0.0602%。所以在「池子不放大」的前提下，
# 給它名額一定虧。
#
# **為什麼第二種贏**：它用的是顧客的「整體口味向量」（把他買過的東西平均成一個方向，
# 零星買的襪子會被稀釋），然後對全目錄打分、取最好的 25 件。第一種是每件買過的商品
# 各給 5 個名額，不管那件商品代不代表他 —— 買了襪子就拿 5 件類似襪子，而且沒有跨商品
# 的排名，好錨點和爛錨點分到一樣多的額度。
#
# 有一件事值得記著：第一種撈到的東西**八成以上別人都沒撈到**（只有 15.9% 跟共購重疊、
# 9.8% 跟第二種重疊）。它不是在重複別人的工作，是撈到了新東西而那些新東西沒人買。
# 之後若放寬候選預算、或 W6 要「這個候選有幾個不同理由指向它」這種特徵，可以重新評估。
ALS_SIM_TOP_N = 5


def eval_customers(con: duckdb.DuckDBPyConnection, test_start: str, test_end: str) -> int:
    """決定「要幫誰做推薦」，答案存成檢視表 eval_cus。

    挑的是 [test_start, test_end) 這段期間有買東西的人 —— 驗證週是 72,019 位買家，
    不是全部 1,371,980 位顧客。

    為什麼不用全部人：那 130 萬個沒買東西的人，你推薦什麼都不可能中，把他們算進去
    只會讓每個模型的分數一起被稀釋成 1/19.9，好模型跟爛模型的差距也跟著縮小。
    共識 §2 定的評估母體就是買家，提案拿來對照的 Kaggle 分數也是買家分母。

    「用『那週有買』來挑人，不算偷看答案嗎？」不算：

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
               weeks: int = REPURCHASE_WEEKS,
               expand_colours: bool = EXPAND_COLOURS) -> int:
    """建 cand_repurchase：顧客近 weeks 週買過的品項。

    依據是提案 §2.2.2：14.1% 的交易是同品項回購（排除同一籃買多件後仍有 4.4%）。
    這是所有召回源裡效率最高的一個：每人 10.8 個候選就換到 2.77% 撈中率。

    expand_colours=True 會把「同款的其他顏色」也納入（同一個 product_code 底下的
    各種顏色）。**預設關閉**，理由見上面 EXPAND_COLOURS 的註解。

    只讀 train_end 以前的交易。往後看一天都是拿答案當特徵。
    """
    start = dataset.pop_start_for(train_end, weeks * 7)
    source = (
        """SELECT DISTINCT t.customer_id, a2.article_id
           FROM tx t
           JOIN eval_cus USING (customer_id)
           JOIN art_c a1 ON a1.article_id = t.article_id
           JOIN art_c a2 ON a2.product_code = a1.product_code"""
        if expand_colours else
        """SELECT DISTINCT t.customer_id, t.article_id
           FROM tx t
           JOIN eval_cus USING (customer_id)"""
    )
    con.execute(f"""
        CREATE OR REPLACE TABLE cand_repurchase AS
        {source}
        WHERE t.t_dat >= '{start}' AND t.t_dat < '{train_end}'
    """)
    return con.execute("SELECT count(*) FROM cand_repurchase").fetchone()[0]


def _popular_by(con: duckdb.DuckDBPyConnection, train_end: str, keys: tuple[str, ...],
                top_n: int, days: int, table: str, audience: str = "eval_cus") -> int:
    """「按 keys 分群，每群取最近 days 天的熱銷前 top_n 名」的共用實作。

    **這支不是生產路徑** —— segment_popular() 走的是四階降級鏈。留著它是為了跑單鍵
    的對照實驗（「只用年齡」vs「只用品牌線」vs「兩個都用」），SEGMENT_LADDER 註解裡
    那幾張表就是用它量出來的。

    keys 是 seg 檢視表的欄位名；audience 是「要發給誰」的檢視表名。

    排序固定加 article_id 決勝，否則同票時每次跑出來的前 top_n 名會不一樣（共識 §4）。
    """
    allowed = {"ageband", "ig"}
    bad = set(keys) - allowed
    assert keys and not bad, f"分群鍵只能用 {allowed}，收到 {keys}"

    dataset.build_segments(con, train_end)
    start = dataset.pop_start_for(train_end, days)
    cols = ", ".join(keys)
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        WITH pop AS (
            SELECT {", ".join(f"s.{c}" for c in keys)}, t.article_id, count(*) AS n
            FROM tx t JOIN seg s USING (customer_id)
            WHERE t.t_dat >= '{start}' AND t.t_dat < '{train_end}'
            GROUP BY {", ".join(str(i + 1) for i in range(len(keys) + 1))}
        ),
        ranked AS (
            SELECT {cols}, article_id,
                   row_number() OVER (PARTITION BY {cols}
                                      ORDER BY n DESC, article_id) AS rk
            FROM pop
        )
        SELECT e.customer_id, r.article_id
        FROM {audience} e
        JOIN seg s USING (customer_id)
        JOIN ranked r ON {" AND ".join(f"r.{c} = s.{c}" for c in keys)}
        WHERE r.rk <= {top_n}
    """)
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def segment_popular(con: duckdb.DuckDBPyConnection, train_end: str,
                    top_n: int = SEGMENT_TOP_N,
                    days: int = SEGMENT_DAYS,
                    ladder: tuple[tuple[str, ...], ...] = SEGMENT_LADDER) -> int:
    """建 cand_segment：每位顧客拿「跟他同一群」的人最近 days 天賣最好的前 top_n 名。

    「同一群」怎麼定，走的是**四階降級鏈**：有年齡又有品牌線的人用細的榜（對他們
    更準），只有一個標籤的人用那個標籤的榜，兩個都沒有的人用全站榜。階層表與人數
    分布見上面 SEGMENT_LADDER 的註解。

    這一源是整個召回的主力，吃掉池子約九成的容量，而且它是**唯一對任何人都給得出
    東西**的源：回購要他買過、共購要他買過、ALS 要他有互動，只有這一源不看歷史 ——
    最後那一階連「什麼標籤都沒有」的人都給得出 top_n 件。**所以每一位買家都拿得到
    候選。** 這件事在 2026-10-15 之前不成立：當時分群寫死成 (年齡 × 品牌線)，新客的
    品牌線是「未知」、那一群訓練期內零交易、排不出榜，5,395 人因此四源全空。

    注意 B1 基準線用的還是固定的 (年齡層 × 品牌線)，沒有降級鏈 —— 兩邊**刻意分岔**，
    B1 的分群是用 MAP@12 選的，這一源是用撈中率選的。共用的只有
    dataset.build_segments 產出的那份標籤（所以 days 吃 SEGMENT_DAYS 而不是
    dataset.POP_DAYS，理由見 SEGMENT_DAYS 的註解）。

    排序固定加 article_id 決勝，否則同票時每次跑出來的前 top_n 名會不一樣（共識 §4）。
    """
    for keys in ladder:
        bad = set(keys) - set(SEGMENT_UNKNOWN)
        assert not bad, "分群鍵只能用 %s，收到 %s" % (set(SEGMENT_UNKNOWN), keys)
    assert ladder and ladder[-1] == (), \
        "最後一階必須是 ()（全站榜），否則不保證每位顧客都拿得到候選"

    dataset.build_segments(con, train_end)
    start = dataset.pop_start_for(train_end, days)

    # 每一階各排一份榜，四份都從同一個 days 天窗口算，只是分群粗細不同。
    # 窗口裡只有十幾萬筆交易，四份榜加起來也不大，所以全部先排好再路由 ——
    # 比「先判斷每個人要哪一階、再各自算一份」簡單得多，而且可以一條 SQL 做完。
    rungs = []
    for n, keys in enumerate(ladder):
        if keys:
            cols = ", ".join(keys)
            picked = ", ".join("s.%s" % c for c in keys)
            rungs.append(
                "r%d AS (SELECT %s, article_id, row_number() OVER ("
                "PARTITION BY %s ORDER BY n DESC, article_id) AS rk FROM ("
                "SELECT %s, t.article_id, count(*) AS n FROM tx t "
                "JOIN seg s USING (customer_id) "
                "WHERE t.t_dat >= '%s' AND t.t_dat < '%s' GROUP BY ALL))"
                % (n, cols, cols, picked, start, train_end)
            )
        else:
            rungs.append(
                "r%d AS (SELECT article_id, row_number() OVER ("
                "ORDER BY n DESC, article_id) AS rk FROM ("
                "SELECT article_id, count(*) AS n FROM tx "
                "WHERE t_dat >= '%s' AND t_dat < '%s' GROUP BY 1))"
                % (n, start, train_end)
            )

    # 每位顧客落在第幾階：第一個「所有標籤都不是缺值」的那一階；都不符合就是最後一階。
    whens = []
    for n, keys in enumerate(ladder):
        if not keys:
            continue
        tests = " AND ".join("s.%s <> '%s'" % (c, SEGMENT_UNKNOWN[c]) for c in keys)
        whens.append("WHEN %s THEN %d" % (tests, n))
    tier = "CASE %s ELSE %d END" % (" ".join(whens), len(ladder) - 1)

    picks = []
    for n, keys in enumerate(ladder):
        on = " AND ".join("r.%s = w.%s" % (c, c) for c in keys) if keys else "TRUE"
        picks.append(
            "SELECT w.customer_id, r.article_id FROM who w JOIN r%d r ON %s "
            "WHERE w.tier = %d AND r.rk <= %d" % (n, on, n, top_n)
        )

    con.execute("""
        CREATE OR REPLACE TABLE cand_segment AS
        WITH %s,
        who AS (
            SELECT e.customer_id, %s, %s AS tier
            FROM eval_cus e JOIN seg s USING (customer_id)
        )
        SELECT DISTINCT customer_id, article_id FROM (%s)
    """ % (", ".join(rungs),
           ", ".join("s.%s" % c for c in SEGMENT_UNKNOWN),
           tier,
           " UNION ALL ".join(picks)))
    return con.execute("SELECT count(*) FROM cand_segment").fetchone()[0]


def cooccurrence(con: duckdb.DuckDBPyConnection, train_end: str,
                 top_n: int = COOC_TOP_N, weeks: int = COOC_WEEKS) -> int:
    """建 cand_cooc：顧客買過的每件商品，帶出最常跟它一起被買的 top_n 件。

    「一起被買」的定義是**同一位顧客、同一天**算一籃 —— 資料裡沒有籃子編號，日期是
    代理指標。**這個定義跑過對照組**：同一天 / 同一天+同一通路 / 同一週 / 完全不分天，
    兩折都量，四源合計的差距 ≤0.014pp，在雜訊範圍內。而且不是因為名單一樣（同一天與
    不分天的 top-5 夥伴只有 70.7% 重疊，換掉了三成），是換掉的那三成一樣有效。所以在
    top_n=5 這個深度上，共購訊號對籃子怎麼定不敏感；留著「同一天」是因為它語意上最
    接近「一起買」。

    籃子的規模：整份訓練資料是 900 萬籃、平均 3.1 件；近 4 週只有 34 萬籃、平均
    2.9 件。**共購次數只從後者統計** —— weeks 把窗口關小了，不是拿 900 萬籃去算。

    這一源補的是另外兩源都補不到的：回購只給得出他買過的東西，分群熱門給的是全群
    一樣的榜單，兩者都無法從「他的個人品味」推到「他沒買過但可能喜歡」的商品。

    weeks 同時決定兩件事：共購統計看多久，以及拿顧客哪一段的購買紀錄去查表。

    排序加了 j 決勝，同票時才不會每次跑出不同的夥伴清單（共識 §4）。
    """
    start = dataset.pop_start_for(train_end, weeks * 7)

    # 先算商品對商品的共購次數。雙向存（i→j 和 j→i 各一列），查表時才不用管方向。
    # 時鐘撥到驗證週、近 4 週是 309 萬列（近 12 週時是 1,131 萬 —— 這個數字跟著
    # 時鐘和窗口走）；先用 sum(n*(n-1)/2) 估過才建，沒有盲目展開。
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

    # 每件商品只留最強的 top_n 個夥伴。不留的話平均每件有 109 個夥伴（驗證週、
    # 近 4 週），候選池會炸。
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


def _als_deps():
    """ALS 要用的套件。單獨一支，是為了把「缺套件」的錯誤講清楚。"""
    try:
        import numpy as np
        import scipy.sparse as sp
        from implicit.als import AlternatingLeastSquares
        from implicit.nearest_neighbours import bm25_weight
        return np, sp, AlternatingLeastSquares, bm25_weight
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"ALS 需要 implicit 套件（缺少 {e.name}）。請執行：pip install -r requirements.txt"
        ) from e


def _als_fit(con: duckdb.DuckDBPyConnection, train_end: str,
             weeks: int, factors: int, seed: int):
    """建 als_ix（字串 ID ↔ 整數索引）、訓練 ALS，回傳 (model, mat)。

    **兩種用法共用這一段**，因為它們只差在用哪一邊的隱因子：
    als() 用顧客隱因子（model.user_factors），als_similar() 用商品隱因子
    （model.item_factors）。訓練一次約 20 秒，要比較兩種用法時把回傳值傳給
    als_similar(model_mat=...) 就不用訓練兩次。

    seed 一定要固定。ALS 的因子是隨機初始化的，不固定的話同一份資料每次跑出不同的
    候選池，你會分不出分數變動是模型改善還是雜訊（共識 §4）。

    記憶體：只把整數索引拉進 Python，字串 ID 留在 DuckDB 最後再 join 回來。
    近 6 週的稀疏矩陣約 11 MB（149 萬個非零值）。
    """
    np, sp, AlternatingLeastSquares, bm25_weight = _als_deps()
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
    return model, mat


def als(con: duckdb.DuckDBPyConnection, train_end: str, top_n: int = ALS_TOP_N,
        weeks: int = ALS_WEEKS, factors: int = ALS_FACTORS, seed: int = 42,
        model_mat=None) -> int:
    """建 cand_als：**ALS 的第二種用法** —— 用顧客隱因子對全商品打分，每人取 top_n 件。

    問的是「這個人會喜歡什麼」：拿他的 64 維向量去跟所有商品向量相乘，取分數最高的
    top_n 件。所以它給得出「他沒買過、跟他買過的也不像」的東西 —— 那是前三源都給不出
    來的，理由是口味相近的別人買了。

    只有隱性回饋：資料裡只有「買了」，沒有「不喜歡」。沒買過不等於不喜歡，可能只是
    沒看過 —— 這就是 implicit 這個套件名字的由來，它的損失函數是為這種資料設計的，
    不能拿評分預測那一套來套。

    共識 §2 允許推薦顧客已買過的商品，所以 filter_already_liked_items=False。

    另一種用法見 als_similar()。兩種的比較見 ALS_SIM_TOP_N 的註解。
    """
    np, _, _, _ = _als_deps()
    model, mat = model_mat if model_mat else _als_fit(con, train_end, weeks, factors, seed)

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


def als_similar(con: duckdb.DuckDBPyConnection, train_end: str,
                top_n: int = ALS_SIM_TOP_N, weeks: int = ALS_WEEKS,
                factors: int = ALS_FACTORS, seed: int = 42, model_mat=None) -> int:
    """建 cand_als_sim：**ALS 的第一種用法** —— 用商品隱因子算商品相似度。

    問的是「跟這件最像的是哪幾件」：拿商品的 64 維向量互比，每件留 top_n 個最像的，
    再把顧客近 weeks 週買過的每一件各帶 top_n 件進候選池。

    **形狀跟 cooccurrence() 一模一樣**（都是「你買過 X，所以給你跟 X 有關的」），
    差別只在「有關」怎麼定義：

        共購          同一顧客同一天被一起買的次數      直接數出來的
        ALS 相似度    兩件商品的隱因子向量夾角          模型學出來的

    所以這兩源可以直接比，提案 §2.2.2 承諾的「ALS 兩種用法比較」就是比這個。

    model_mat 可以把 _als_fit() 的回傳值傳進來共用一次訓練（省 20 秒），
    平常不用給。
    """
    np, _, _, _ = _als_deps()
    model, mat = model_mat if model_mat else _als_fit(con, train_end, weeks, factors, seed)
    start = dataset.pop_start_for(train_end, weeks * 7)

    # 只對「評估對象真的買過的商品」找鄰居 —— 別人沒買過的商品，找了也沒人用得到。
    items = con.execute("""
        SELECT DISTINCT i.ai FROM als_ix i JOIN eval_cus e USING (customer_id) ORDER BY 1
    """).df()["ai"].to_numpy()

    # N 要多取一個：similar_items 的第一名一定是自己（跟自己的夾角為 0）。
    ids, _ = model.similar_items(items, N=top_n + 1)
    sim = pd.DataFrame({
        "i": np.repeat(items, ids.shape[1]).astype("int32"),
        "j": ids.reshape(-1).astype("int32"),
    })
    sim = sim[sim["i"] != sim["j"]]

    con.register("als_sim_raw", sim)
    con.execute("""
        CREATE OR REPLACE TABLE als_sim AS
        SELECT ai.article_id AS i, aj.article_id AS j
        FROM als_sim_raw s
        JOIN (SELECT DISTINCT article_id, ai FROM als_ix) ai ON ai.ai = s.i
        JOIN (SELECT DISTINCT article_id, ai FROM als_ix) aj ON aj.ai = s.j
    """)
    con.unregister("als_sim_raw")

    # 這一段跟 cooccurrence() 的第二段只差 JOIN 的是 als_sim 還是 cooc 的 top 表。
    con.execute(f"""
        CREATE OR REPLACE TABLE cand_als_sim AS
        SELECT DISTINCT t.customer_id, s.j AS article_id
        FROM tx t
        JOIN eval_cus USING (customer_id)
        JOIN als_sim s ON s.i = t.article_id
        WHERE t.t_dat >= '{start}' AND t.t_dat < '{train_end}'
    """)
    return con.execute("SELECT count(*) FROM cand_als_sim").fetchone()[0]


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

        customer_id  article_id  from_repurchase from_segment from_cooc from_als
          顧客A      0751471001         1             0          0        0
          顧客A      0918890002         0             1          0        1

    同一件被多源撈到不是重複，是訊號 —— 好幾個不同的理由都指向它。所以這裡不能只做
    UNION 去重，要把來源留下來：排序模型會拿它當特徵（提案 §2.2.2 的交互特徵）。

    每人的候選數不固定（驗證週：最少 27、中位數 191、P99 285、最多 627）。變動組長的
    LambdaRank 本來就吃得下，而「池子大小」跟「會不會買」相關，那是 W6 該放進去的
    特徵，不是瑕疵。**每一位買家都拿得到候選**（驗證週 72,019 人全部有，193.0 ＝
    13,898,930 ÷ 72,019），所以共識 §2 那條「不足 12 件以 B0 補滿」在這一層一次都不會
    觸發。那個「分母剛好等於買家數」不是保證，陷阱見 evaluate.coverage() 的 docstring。

    test_start 就是「時鐘撥到哪一天」。所有召回源都只讀它以前的交易 —— 往後看一天就是
    拿答案當特徵。test_end 只用來決定要幫誰做推薦，不影響任何候選的內容。

        recall.build(con, "2020-09-09", "2020-09-16")   # 撥到 9/09，考 9/09 那一週

    save=True 會把 cand 寫成 data/candidates/cand_<test_start>.parquet。為什麼要存檔：
    W6 要用好幾個週次的候選池訓練，而**記憶體裡的東西沒有身分** —— 今天用 top_n=50 算
    一份、明天用 100 算一份，事後分不出模型 A 是拿哪一份訓練的。預設不存，因為量撈中率
    一天要跑好幾次，每次都寫 176 MB 沒意義。
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
