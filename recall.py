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

這一層的唯一職責是**不要漏掉正解**。排序模型只能把池子裡已經有的東西往前排，
不能無中生有 —— 第一關就被丟掉的商品，整個系統再也拿不回來。

所以這一層量的是**撈中率**：那一週真的被買的品項，有幾成被我們撈進了池子。
（同一個數字有四個名字：這裡叫撈中率、evaluate.py 的函式叫 coverage、W5 驗收標準
寫「被召回的比例」、Kaggle 討論串寫 recall rate。）

**不量準確率** —— 池子裡夾帶他不會買的東西是排序模型要處理的事，這一層寧可多撈、
只怕漏撈。**也不能用 MAP@12 調參** —— 池子放大撈中率一定升，但 MAP 可能升也可能
降（多撈進來的東西同時也是干擾項），而現在還沒有排序模型，量不出淨效果。

撈中率和 MAP@12 的唯一正當關係是**撈中率決定 MAP@12 的上限**。兩者不能比大小
（一個是比例、一個是排名分數），看到「撈中率 19.2%、目標 MAP 2.5%」不要讀成
已經超標六倍，那是兩把不同的尺上的兩個刻度。

【結果是啥】

驗證週（2020-09-09 ~ 09-15）實測。**保留週封存中**，只在 W5 收尾時量一次 ——
反覆看它，後續的判斷就會被它影響（共識 §1）。

    每人候選數    193.0 件
    撈中率        19.28%
    上限          21.18%   排序模型再完美，整個系統的 MAP@12 最高就是這個數
    目標           2.50%   提案 §3.2 —— 在上限底下，所以召回沒有把路堵死

72,019 位買家全部拿得到候選，新客缺口已經關掉。但其中 43,220 人（60.0%）的池子裡
沒有一件他那週真的會買的，那些人排序模型再厲害也是 0 分。

撈不到的那些，資料裡沒有訊號指向它們：沒撈中的正解有 66.4% 落在近 14 天熱銷
1000 名之外、7.3% 近 14 天根本沒賣過。任何靠歷史統計的方法都撈不到那些。
而這個任務本身就難 —— 同一份資料同一個指標，Kaggle 三千隊的銀牌方案是
MAP@12 0.030、學術論文的雙塔模型約 0.034，我們的目標 0.025 落在那個量級裡。
「百分比小」是題目的性質，不是我們的分數難看。

【啥等待定】

**一、W5 原訂「30% 以上被召回」沒達到。** 放大池子就大致搆得到（實測每人 664 件
時撈中率 28.3%），代價是候選變 3.4 倍、排序模型要排除的干擾項也變 3.4 倍。
**決議（2026-09-29）：先做 W6 的排序模型，看它實際發揮到上限的幾成，
離 2.5% 還很遠的話再回頭放大候選池。** 後續三輪調參都守住這條，每人候選數一路
沒放大（191.1 → 198.9 → 194.8 → 193.0）。

**二、下面十個常數裡只有三個做過兩折驗證**，其餘只在驗證週上量過。
小組決議（2026-10-17）：先用現在這組數字往下做 W6，之後有時間再回來補測。
哪三個、為什麼窗口類的參數需要兩折，見下面常數區開頭那張表。

**三、ALS 的第一種用法還沒實作。** 提案 §2.2.2 承諾「兩種用法都實作並比較」，
目前只有第二種（model.recommend，以顧客隱因子對全商品打分）；第一種（拿商品隱因子
算商品相似度）沒寫。這是 W5 的第二條欠帳。

**四、語意檢索（W8）補不了撈中率這個洞。** 它要有一句查詢才動得起來，而離線評估裡
沒有人說話。它的價值在對話路徑，提案 §3.2 另外用「模擬對話命中率」量 ——
所以不要期待 W8 把撈中率推上去。

召回階段的完整工作紀錄（含哪些設計從來沒驗證過）在 vault 的《候選池施工紀錄》；
排序階段的設計在 ranking.py；商業價值估算在提案 §3.3。這個檔案只講候選池怎麼撈
出來、為什麼是這些參數。

**記憶體**：召回是展開不是聚合，輸出遠大於輸入，一不小心就是上億列。所有候選一律
留在 DuckDB 裡（CREATE TABLE），不要 .df() 拉進 pandas —— DuckDB 算不下會溢寫到
磁碟，pandas 不會，只會讓整台機器沒記憶體可用。只有最後那幾個數字才拉出來。
"""

from __future__ import annotations

import duckdb
import pandas as pd

import dataset

# ═══ 四個源的名額怎麼分：方法，以及目前分到哪裡 ═══
#
# 下面十個常數每一個都是這套方法挑出來的。這一節只講方法和結論，個別參數的掃描表
# 在各自的常數旁邊。
#
# ── 方法：三步 ──
#
# **第一步，不能拿撈中率當標準**，不然答案永遠是「每個窗口都拉到最大」。用的是
# **效率 ＝ 撈中率 ÷ 每人候選數**，意思是「一個候選換到多少撈中率」。
#
# **第二步，一次只動一個參數掃一輪，相鄰兩列相減就是邊際效率** ——「在目前這個
# 操作點上，多給這一源一個名額，四源合計的撈中率會多幾個百分點」。分名額要看邊際，
# 不是單獨效率：回購的單獨效率是第二名的兩倍，但它只能撈「他自己買過的」，
# 一個人買過幾件就是幾件，給再多名額也長不出新東西。
#
# **第三步，從邊際低的源收名額、給邊際高的源，一直挪到四個打平。** 若 A 的邊際比 B
# 高，從 B 收一個名額給 A，候選總量不變而撈中率上升 —— 代表原來那個點不是最佳。
# 四個打平之後就再也挪不出好處了，**那就是最佳條件**。
#
# ── 兩條守則：不照著做會調出假的改善 ──
#
# **一、邊際綁在操作點上，不能外插也不能相加。** 名額一挪操作點就變，下面那張表就
# 過期；而且四源會撈到同一批東西（砍掉共購之後分群新撈到的，有一部分本來就是共購
# 在撈的），所以組合的結果只能實測。實例：共購窗口從 12 週縮到 4 週之後，它的
# top_n 邊際從 0.0302% 跳到 0.0612%，從四源最低變成最高 —— 沒人預測得到，是量出來的。
#
# **二、窗口類的參數要用兩折。** 只在一週上挑最大值會挑到手氣（實例見 SEGMENT_DAYS）。
# 兩折 ＝ 驗證週 + 緊貼它前面的 09-02 ~ 09-08，看兩折的平均。第二折**必須相鄰** ——
# 要隔離的是運氣，隔太遠的一週連目錄都換過了，答案不一樣也分不出是雜訊還是市場真的
# 變了；而且太早的週次連窗口都湊不滿（測 2018-10 的話回購 12 週會少 73 天，程式不
# 報錯，但量到的是殘廢版本）。第二折不是保留週，保留週繼續封存。
#
# ── 結果：目前的操作點 ──
#
# 驗證週、現在的設定（回購 12 週 / 分群取 170 名看 5 天 / 共購 5 取 4 週 / ALS 25 取 6 週）：
#
#   來源        每人候選   單獨撈中率   單獨效率   邊際效率
#   回購            10.8       2.77%    0.2565%    0.0395%   ← 邊際最低
#   共購            23.0       2.72%    0.1183%    0.0612%   ← 邊際最高
#   ALS             25.0       2.94%    0.1176%    0.0482%
#   分群熱門       169.8      15.42%    0.0908%    0.0551%
#   四源合計       193.0      19.28%    0.0999%
#
# **還沒打平**：最高（共購 0.0612%）與最低（回購 0.0395%）差 1.55 倍，第一輪之前是
# 2.5 倍。但已經沒有明顯被浪費的名額了 —— 池子有 88% 的容量在分群熱門，而它的邊際
# 排第二。要加候選的話第一個該加的是共購（5 → 10），不是分群；而那是放大池子，
# W6 之後才該碰（見上面【啥等待定】）。
#
# ── 走到這個操作點用了三輪 ──
#
#   2026-10-06  重新分名額：共購 top_n 10→5、ALS 50→25、分群 100→150
#               每人 191.1 → 198.9，撈中率 15.52% → 16.96%
#   2026-10-06  調窗口：分群熱銷 14→5 天、ALS 12→6 週、共購 12→4 週，省下的名額
#               給分群 150→170。每人 198.9 → 194.8，撈中率 16.96% → 18.11%
#               —— **候選變少而撈中率變高**，所以這一輪沒有成本
#   2026-10-15  換分群維度：42 組 →（年齡層 7 群）→ 四階降級鏈
#               每人 194.8 → 193.0，撈中率 18.11% → 19.28%，上限 19.82% → 21.18%
#               順便把新客缺口整個消掉：原本 5,395 人（7.5%）四源全空，現在 0 人
#
# ── 守則二實際做到哪裡：只有三個參數真的量了兩折 ──
#
#   參數                  兩折？   實際怎麼量的
#   SEGMENT_DAYS（分群天數） ✓     驗證週 + 09-02
#   SEGMENT_LADDER（分群維度）✓     驗證週 + 09-02
#   ALS_WEEKS（ALS 窗口）    ✓     驗證週 + 09-02
#   ─────────────────────────────────────────
#   REPURCHASE_WEEKS（12 週）✗     只有驗證週   ← 這是窗口類的
#   COOC_WEEKS（4 週）       ✗     只有驗證週   ← 這也是
#   SEGMENT_TOP_N（170）     ✗     只有驗證週
#   COOC_TOP_N（5）          ✗     只有驗證週
#   ALS_TOP_N（25）          ✗     只有驗證週
#   ALS_FACTORS（64+BM25）   ✗     只有驗證週
#   EXPAND_COLOURS（關閉）   ✗     只有驗證週
#
# **小組決議（2026-10-17）：回購用 12 週、分群熱門用 5 天定案；ALS 與共購的參數
# 今天才會討論完。先用這組數字往下做 W6，之後有時間再回來補兩折（或更多折）。**

# 回購往回看幾週。**目前 12 週。**
#
# 驗證週掃過（當時操作點是分群 150 取 14 天 / 共購 5 取 12 週 / ALS 25 取 12 週）：
#
#   週數   每人候選   撈中率    邊際
#     4      195.5   16.78%
#     8      197.0   16.89%   0.0748%
#    12      198.9   16.96%   0.0358%   ← 採用
#    26      206.0   17.20%   0.0344%
#    52      218.0   17.52%   0.0260%
#
# 52 週確實多撈 0.56 個百分點，但多吃 19 個候選 ＝ 0.0260%／候選，當時全場最低 ——
# 同樣 19 個名額給分群熱門換到的更多。所以 52 週是輸在機會成本，不是輸在沒用。
# 往下縮也不划算：縮到 8 週只省 1.9 個候選。
#
# 提案 §2.2.2 也寫 12 週，但那不是留著 12 的理由 —— 掃過才留的。
# 這個旋鈕本來影響就小：回購只佔每人 10.8 個候選，而且窗口再長也長不出他沒買過的東西。
REPURCHASE_WEEKS = 12

# 回購要不要把「同款其他顏色」也納入候選。實測結論是**不要**。
#
# （article_id 是「款 + 色」、product_code 只到「款」，所以展開就是同一個
#   product_code 底下的所有顏色。）
#
# 它讓回購這一源的候選從每人 10.8 個漲到 61.5 個，撈中率只從 2.77% 到 5.18%，
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

# 分群熱門每群取前幾名。**目前 170。**
#
# （這個 top_n 是「每群撈幾件當候選」，跟 evaluate.py 的 k=12「最終推薦幾件」無關。
#   k=12 是 Kaggle 規定的不能改；top_n 是我們自己選的。）
#
# 驗證週掃過（**在降級鏈之前的 7 群版本上掃的**，所以 170 那一列跟上面的
# 193.0 / 19.28% 差一點點，但各列之間的比較仍然有效）：
#
#   top_n   每人候選   撈中率    邊際
#     100     124.5   14.82%
#     170     193.2   19.24%   0.0643%   ← 採用
#     250     272.1   23.56%   0.0548%
#     400     420.3   29.57%   0.0406%
#
# **曲線沒有拐點**，報酬平緩遞減，所以這條曲線本身問不出「最佳值在哪」。
# 170 不是最佳值，是**停損點** —— 再往上加就是放大候選池，而決議是先做 W6
# （見【啥等待定】）。要加的話這裡也不是第一順位，共購（5 → 10）的邊際更高。
#
# **這個數字已經超出提案的預算。** §2.2.2 寫「每位顧客約 50–100 個候選」、風險表更
# 保守寫「上限 50」，現在是每人 193.0。超出是刻意的 —— §2.2.4 本來就把「每人候選數
# 上限」列為要在驗證週上選的參數。代價落在 W6：候選越多，排序模型的訓練資料越大
# （目前總共 13,898,930 組）。真的跑不動，照上面那張邊際表從最低的那一源先砍。
#
# 歷史：50（初版，當時共購與 ALS 還沒做）→ 100（關掉同款換色，省下的名額挪過來）
# → 150（第一輪分名額）→ 170（第二輪，共購縮窗口省下的名額只有這一源吃得下，
# 它佔池子 88% 容量）→ 第三輪換分群維度後重掃，170 不動。
SEGMENT_TOP_N = 170

# 分群熱銷榜取最近幾天。**目前 5 天。**
#
# **這個旋鈕不花候選** —— 每群都是取 top_n 名，改天數只換「是哪 170 件」，不換數量。
# 所以沒有取捨，純比高低。這也是有做兩折的三個參數之一：
#
#   天數   驗證週    第二折    兩折平均
#     3   17.88%   18.16%   18.02%   ← 只看驗證週會選這個
#     5   17.82%   18.28%   18.05%   ← 採用
#     7   17.51%   18.28%   17.90%
#    10   17.11%      —
#    14   16.96%   17.48%   17.22%   ← 原本
#
# 3 天在驗證週最高、在第二折輸給 5 天和 7 天：三天的熱銷榜只有三天交易量，排名本來
# 就抖，而驗證週那三天還是 9/06~9/08、週日到週二，週末效應會被當成通則。
# 5 天在兩折都接近最佳、都沒當過最差。
#
# **這是整個檔案裡最便宜的一次改善：零候選成本換 0.86 個百分點。**
# 為什麼短的贏：快時尚的熱銷榜換得快，14 天前賣得好的東西下週不見得還在賣。
#
# 這個常數**刻意不等於 dataset.POP_DAYS（14 天）**：那個 14 天是 W4 用 B1 基準線的
# MAP 選出來的，這裡要的是撈中率，兩個不同的目標；而且動 dataset.POP_DAYS 會連帶
# 改掉已經交付的 B0 / B1。（分群**標籤**兩邊還是共用 dataset.build_segments 那一份，
# 那個不能分岔。）
SEGMENT_DAYS = 5

# 分群用哪些欄位切。**目前走四階降級鏈：每位顧客用他的標籤湊得出來的第一階。**
#
# 可用的欄位只有 seg 檢視表那兩個（定義在 dataset.build_segments）：
#
#   ageband   年齡層 7 段。**來自 customers.csv**，跟購買紀錄無關，缺值是「未填」
#   ig        主要品牌線 5 條。**從交易歷史算出來的**，沒買過的人是「未知」
#
# 關鍵差別：新客一定有 ageband（註冊資料），但一定沒有 ig（要有購買才算得出來）。
#
#   階  條件                用哪份榜          驗證週人數   那份榜最少幾件
#    1  年齡有、品牌線有    (年齡 × 品牌線)     66,343            6   ← 10 格湊不滿 170
#    2  只有年齡（純新客）  (年齡)               5,355          612
#    3  只有品牌線          (品牌線)               281          792
#    4  兩個都沒有          全站熱銷                40        湊得滿
#
# **為什麼要分階，而不是全部用同一份榜** —— 2026-10-15 把顧客拆開量才看出來的：
#
#   族群                            (年齡 × 品牌線)   (年齡)     誰贏
#   格子湊得滿 170 的 66,384 人           15.38%     15.23%   前者贏 0.15pp
#   格子湊不滿的 5,635 人                  0.27%     15.68%   後者贏一整段
#
# 對九成的顧客來說，切細確實比較準；而原本 42 組版本輸掉的 1.13 個百分點幾乎全部
# 來自新客拿到零候選 —— 不是榜單品質問題，是他們根本沒有榜單。所以正確答案不是
# 二選一，是**各取所長**。第 2、3 階的榜永遠湊得滿 170（見上表），所以降級之後不需要
# 補位；最後一階的全站榜是最後的保證，函式裡那條斷言就是在守它。
#
# 第 3 階給品牌線榜而不給「未填」年齡榜（那一格也有 612 件，湊得滿），是因為
# **品牌線是他買出來的真實偏好訊號，「未填」只是資料缺失** —— 有真訊號就不該退回
# 缺失值那一格。實際差距：他們的 (未填 × 品牌線) 格子只有 6~499 件可挑，
# 改用品牌線榜是 792~16,316 件。
#
# **但降級鏈對四源合計沒有改善**：驗證週 +0.04pp、第二折 -0.04pp，一正一負，按兩折
# 的紀律就是雜訊。分群這一源單獨倒是升了（驗證週 15.27% → 15.42%），併進四源卻被
# 吃掉 —— 因為第 1 階接到的人就是「有購買歷史的人」，而回購、共購、ALS 也只對這些人
# 作用，細榜替他們多撈到的東西那三源本來就撈到了。
#
# 那為什麼還留著四階：既然**資料分不出四階和兩階的差別**，就用原則決定 ——
# 每位顧客用他手上最好的訊號，而不是退回全站榜；而且最後一階加上那條斷言，把
# 「每位顧客都拿得到候選」變成**結構保證**，不再是靠「剛好每個年齡層都有 612 件
# 以上」這個資料巧合。這是判斷不是數據結論，要退回兩階（年齡 → 全站）也站得住腳，
# 少 30 行 SQL 產生碼。
#
# **還沒修的一個缺口**：降級鏈是按「有哪些標籤」路由，不是按「格子夠不夠大」，所以
# 兩個標籤都有、但自己那格商品太少的人不會被降級 —— 驗證週有 187 人拿不到 170 件
# （最少的只有 27 件），第二折 413 人。他們沒掛零，只是候選少一點。要修就是加第五條
# 規則「格子不足 top_n 就降級」，但那會動到第 1 階的人數，整輪要重測。目前沒做。
#
# 為什麼 eda 當初測出「加品牌線比較好」而這裡相反：**操作點完全不同。** eda 量的是
# B1 的 MAP@12、每群只取 12 件、不足還用全站熱銷補滿；這裡量撈中率、每群取 170 件、
# 沒有補位。MAP 獎勵「精準」、撈中率獎勵「涵蓋廣」，兩把尺獎勵的東西相反，而取 12 件
# 和取 170 件對小格子的容忍度也完全不同。eda 沒測錯，是把它的結論套到不同的操作點
# 卻沒重測才錯。
SEGMENT_LADDER = (("ageband", "ig"), ("ageband",), ("ig",), ())

# 每個分群欄位的「缺值」長什麼樣。降級鏈靠這個判斷某一階的標籤湊不湊得出來。
SEGMENT_UNKNOWN = {"ageband": "未填", "ig": "未知"}

# 共購：統計看多久（weeks）、每件商品留幾個夥伴（top_n）。**目前 4 週 / 每件留 5 個。**
#
# 先說為什麼留著這一源：它的撈中率跟別源重疊得很兇（單獨只有 2.72%，而且大部分
# 別源也撈得到），曾經被考慮砍掉。留著的理由是**它是唯一推得出「他沒買過、也不在
# 他那群熱銷榜上」的商品的源**；而且排序模型會拿 from_cooc 這個旗標當特徵，
# 「三源都指向它」和「只有分群熱門撈到」是不同的訊號。
#
# ── weeks = 4：統計看多久 ──
#
# 原本寫 12，理由是「與回購對齊」—— 那不是證據。驗證週掃過（舊操作點）：
#
#   週數   每人候選   撈中率    效率
#     4      179.6   16.41%   0.0914%   ← 採用
#     8      187.8   16.75%   0.0892%
#    12      198.9   16.96%   0.0853%   ← 原本
#    26      229.4   17.56%   0.0765%
#
# 8→12 週和 12→26 週的邊際（0.0190% / 0.0196%）都是當時全場最低。反過來縮到 4 週
# 是省下每人 19 個候選、撈中率掉 0.55 個百分點，那些名額給分群熱門換到的更多，
# 實測挪完淨賺 0.24 個百分點。
# 為什麼短的贏：十二週前跟外套一起買的襪子，跟「這週會不會一起買」已經沒什麼關係。
#
# ── top_n = 5：每件留幾個夥伴 ──
#
# **這個旋鈕現在是四源裡邊際效率最高的**：5 → 10 能換到 0.0612%／候選。那為什麼還
# 停在 5 —— 加上去會讓每人候選從 193 件漲到約 200 件，那是放大池子，而決議是先做 W6。
# **下一輪要加候選的話，這裡是第一個該加的地方。**
#
# 順帶記一個交互作用：weeks 從 12 縮到 4 之後，top_n 的邊際從 0.0302% 一路跳到
# 0.0612%，從四源最低變成四源最高。四週內一起被買的東西比十二週內的準得多，所以多
# 留幾個夥伴才值得。**改了 weeks 就不能沿用舊的 top_n 結論，要重掃。**
COOC_WEEKS = 4
COOC_TOP_N = 5

# ALS：互動窗口幾週（weeks）、隱因子幾維（factors）、每人取幾件（top_n）。
# **目前 6 週 / 64 維 + BM25 加權 / 每人 25 件。**
#
# 這一源跟前三個不一樣：前三個是規則（他買過什麼、他那群在買什麼、什麼跟什麼一起
# 賣），ALS 是模型，自己從稀疏矩陣裡學出隱藏的口味維度，所以它撈得到的東西前三源
# 不見得撈得到。完整說明見下面 als() 的 docstring。
#
# ── weeks = 6：互動窗口（有做兩折）──
#
#   週數   驗證週每人   驗證週撈中率   第二折撈中率
#     4       195.1      16.97%        17.52%
#     6       196.7      17.01%        17.56%   ← 採用，兩折都最高
#     8       197.7      17.01%        17.54%
#    12       198.9      16.96%        17.48%   ← 原本，兩折都最差
#
# 12 週被 6 週**嚴格支配**：候選少 2.2 個、撈中率還高 0.05 個百分點。多拉那六週的
# 舊互動不是遞減報酬，是**負貢獻** —— 隱因子會把已經過季的東西學進口味向量裡。
#
# ── factors = 64 + BM25：矩陣怎麼分解 ──
#
# 驗證週掃過（當時三源基準是 12.5%、每人 149）：
#
#   設定                  單獨    合計    邊際
#   factors=64 原始       2.6%   13.1%   +0.6pp / +25 候選
#   factors=64 BM25       3.8%   13.2%   +0.8pp / +21 候選   ← 採用
#   factors=128 BM25      4.2%   13.1%   +0.7pp / +19 候選
#   factors=64 BM25 30輪  3.7%   13.2%   +0.7pp / +21 候選
#
# 加維度、加輪數都沒有讓邊際貢獻動 —— ALS 撈到的東西前三源多半也撈得到，所以把它
# 本身調得更準並不會讓四源合計變好。**BM25 加權是唯一有感的那一項**（單獨 2.6% →
# 3.8%），理由見 als() 裡那段註解。這兩個值從初版至今沒變過。
#
# ── top_n = 25：每人取幾件 ──
#
# 邊際是 25 → 50 換到 0.0439%／候選，四源裡倒數第二，所以不是加候選的第一順位。
# 砍到 25 之後它的**單獨**效率反而變高（0.118%，四源第二）—— 前 25 名本來就是比較
# 準的那幾個，第 26~50 名多半是雜訊。
ALS_WEEKS = 6
ALS_FACTORS = 64
ALS_TOP_N = 25


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

    「一起被買」的定義是**同一位顧客、同一天**算一籃。用同籃而不是「這個人買過的
    所有東西兩兩配對」，是因為後者會把半年前買的襪子跟今天買的外套配成一對 ——
    那不是「一起買」，只是「都買過」。

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


def als(con: duckdb.DuckDBPyConnection, train_end: str, top_n: int = ALS_TOP_N,
        weeks: int = ALS_WEEKS, factors: int = ALS_FACTORS, seed: int = 42) -> int:
    """建 cand_als：用矩陣分解找出「口味相近的人在買什麼」，每人取 top_n 件。

    它從「誰買過什麼」這張稀疏表裡自己學出隱藏的口味維度，把每位顧客和每件商品都
    壓成一個 64 維向量，兩個向量相乘就是預測分數。

    只有隱性回饋：資料裡只有「買了」，沒有「不喜歡」。沒買過不等於不喜歡，可能只是
    沒看過 —— 這就是 implicit 這個套件名字的由來，它的損失函數是為這種資料設計的，
    不能拿評分預測那一套來套。

    共識 §2 允許推薦顧客已買過的商品，所以 filter_already_liked_items=False。

    seed 一定要固定。ALS 的因子是隨機初始化的，不固定的話同一份資料每次跑出不同的
    候選池，你會分不出分數變動是模型改善還是雜訊（共識 §4）。

    記憶體：只把整數索引拉進 Python，字串 ID 留在 DuckDB 最後再 join 回來。
    近 6 週的稀疏矩陣約 11 MB（149 萬個非零值，每個 8 bytes）。
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

        customer_id  article_id  from_repurchase from_segment from_cooc from_als
          顧客A      0751471001         1             0          0        0
          顧客A      0918890002         0             1          0        1
          顧客A      0448509014         1             0          1        0

    同一件被多源撈到不是重複，是訊號 —— 好幾個不同的理由都指向它。

    每人的候選數不固定（驗證週：最少 27、中位數 191、P99 285、最多 627）。
    變動組長 LambdaRank 本來就吃得下。而「池子大小」跟「會不會買」相關
    （買多的人候選也多），那是 W6 該放進去的特徵，不是瑕疵。

    **每一位買家都拿得到候選**，驗證週 72,019 人全部有 —— 因為分群熱門的最後一階
    不看購買歷史。所以共識 §2 那條「不足 12 件以 B0 補滿」在這一層**一次都不會觸發**
    （不足 12 件的人是 0 位）。2026-10-15 之前不是這樣：當時分群寫死成
    (年齡層 × 品牌線)，5,395 人（7.5%）四個源全空。上線後新客比例只會更高，
    所以「主源不依賴購買歷史」這個性質要守住，它是冷啟動唯一的出路。

    （驗證週是 193.0 ＝ 13,898,930 ÷ 72,019，分母與買家數剛好相等。那個「剛好」不是
    保證 —— 分群鍵一改就可能又不等，陷阱的說明在 evaluate.coverage() 的 docstring。）

    這裡不能只做 UNION 去重，要把來源留下來：排序模型會拿「候選來自哪一源」當特徵
    （提案 §2.2.2 的交互特徵）。

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

    預設不存，因為量撈中率一天要跑好幾次，每次都寫 176 MB 沒意義。
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
