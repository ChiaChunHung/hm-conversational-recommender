"""
第一階段：候選召回。

這份檔案只做一件事：從 105,542 件商品裡，幫每位顧客挑出幾百件「有機會」的，
再交給第二階段的排序模型（W6）決定最後要推哪 12 件。

為什麼要切成兩段 —— 每位顧客 × 每件商品各算一次分數是
1,371,980 × 105,542 ≈ 1,448 億次，筆電跑不完。先用便宜的規則粗篩（這一層），
再用模型精排（W6），是 Kaggle 獲獎方案與業界的共同配方（提案 §2.2.2）。

這一層的唯一職責是**不要漏掉正解**。排序模型只能把池子裡已經有的東西往前排，
不能無中生有 —— 第一關就被丟掉的商品，整個系統再也拿不回來。所以這一層量的是
**撈中率**：那一週真的被買的品項，有幾成被我們撈進了池子。

不量準確率。池子裡夾帶一堆他不會買的東西，那是排序模型要處理的事，不是這一層的
罪 —— 這一層寧可多撈，只怕漏撈。

W5 原訂的驗收標準是「保留週實際購買的品項有 30% 以上被召回」—— 目前沒到，
原因與我們的決定見下面的【待解決】。


【名詞對照】撈中率、MAP@12、2.5% 目標，是三個不同的東西

這三個最容易混，先擺在一起看：

    名字              量的是什麼                      誰的分數     在哪裡算
    撈中率            正解有幾成被撈進候選池          第一階段     recall.py
                      進了池子就算，不管排第幾名      （這個檔案） evaluate.coverage()
    MAP@12            推薦的 12 格裡，正解排得多前面  整個系統     evaluate.ap_at_k()
                      排第 1 和排第 12 差很多         召回＋排序
    2.5%              驗收目標，量的是 MAP@12         整個系統     提案 §3.2

**同一件事有四個名字**：這個檔案叫「撈中率」，evaluate.py 的函式叫 coverage
（召回覆蓋率），W5 驗收標準寫「被召回的比例」，Kaggle 的討論串寫 recall 或
recall rate。四個講的是同一個數字，不是四件事。

**而 recall rate 跟 MAP@12 是兩件事**，名字都有「recall」只是巧合 ——
前者是這一層的比例，後者是整個系統的排名分數。

然後是三條規矩，記住這三條就不會再混：

**一、撈中率和 MAP@12 不能互相比大小。** 連單位都不一樣 —— 撈中率是「比例」，
MAP@12 是「排名品質分數」。看到「撈中率 17.0%、目標 MAP 2.5%」，不要讀成
「已經超標六倍」，那是兩把不同的尺上的兩個刻度。

**二、撈中率決定 MAP@12 的上限。** 這是兩者唯一的正當關係。假設排序模型完美
（把池子裡的正解全部排到最前面），它能拿到的 MAP@12 就是：

    每位顧客的撈中率取平均（macro）   18.80%
    完美排序實際算出來的 MAP@12       18.89%   ← 真正的上限

為什麼幾乎相等：正解全排在最前面時，第 1 格到第 h 格每一格的精確率都是
1/1、2/2、3/3 … ＝ 1，於是 AP ＝ 命中數 ÷ 正解數，剛好就是那個人的撈中率，
平均起來就是 macro。差的那 0.096 個百分點來自買超過 12 件的 1.8% 顧客 ——
他們的 AP 分母被 12 卡住，算出來比自己的撈中率高一點。

所以**這一層決定「最多能拿幾分」，W6 排序模型決定「拿到上限的幾成」**。
目標 2.5% 在 18.89% 底下，所以召回沒有把路堵死。

**三、這一層不能用 MAP@12 調參。** 池子放大，撈中率一定升，但 MAP@12 可能升也
可能降 —— 多撈進來的東西同時也是干擾項，排序模型要從更多雜訊裡挑 12 件。兩個
效應方向相反，而現在還沒有排序模型，量不出淨效果。所以召回階段只看撈中率和
效率（一個候選換到多少撈中率），候選預算的最終決定要等 W6（見【待解決】）。


【先搞懂尺度】為什麼每個百分比看起來都這麼小

這一層的撈中率只有十幾個百分點（確切數字見【待解決】），整個專案的 MAP@12 目標
是 0.025。第一次看到這兩個數字，正常反應是「這不是做壞了嗎」。不是。這一段把尺度
講清楚，省得每個人各自嚇一次。

**先看外面的人考幾分。** 同一份資料、同一個指標（參照點取自提案 §3.2）：

    我們的目標                                MAP@12  0.025
    Kaggle 銀牌方案（第 45 名 / 3,006 隊）    MAP@12  0.02996
    學術論文的雙塔模型                        MAP@12  約 0.034

全世界三千隊打同一份資料，銀牌 0.03、論文 0.034。**所以「小」不是我們做壞了，
是題目本身就這麼難。** 難在四件事疊在一起：

一、規模懸殊。我們推 12 件，目錄有 105,542 件 —— 一次只能指出 0.011% 的商品。
    亂猜的期望值趨近於零。

二、購買行為本身帶大量隨機性。一個人下週買什麼，看天氣、看心情、看那天剛好滑到
    什麼、看朋友說了什麼。**那些資訊我們一個都沒有**，手上只有交易紀錄。

三、時間窗很短。這裡說的是**答案卷那一邊**：要猜的是「他未來七天會買什麼」，
    不是「未來三個月」。窗口越短，運氣佔的比重越高。
    （召回往回看多久是另一回事，那是參數，見下面時鐘那節。）

四、長尾。驗證週實測，那些我們一件都沒撈中的顧客，他們買的 117,906 組正解分布是：

        近 14 天熱銷 1000 名之外    62.5%
        101 ~ 1000 名               27.8%
        近 14 天完全沒賣過           7.1%
        熱銷前 100                   2.5%

    將近七成落在 1000 名之外、或近 14 天根本沒賣過。那些商品在資料裡沒有任何
    訊號指向它們，任何靠歷史統計的方法都撈不到 —— 不是參數調得不好。

**所以絕對值沒有意義，要看相對值。** 用同一支 evaluate.evaluate() 量驗證週：

    B0 近 14 天全站熱銷前 12（最笨的做法）    MAP@12  0.00700
    目標（提案 §3.2）                         MAP@12  0.025    ← B0 的 3.6 倍

「2.5%」不是「只對了 2.5%」，是「在一個亂猜趨近於零的任務上，比最笨的做法好
3.6 倍」。這才是這個數字的意思。

（B0 屬於排序階段的對照組，量的是 MAP；這一層量的是撈中率，兩把不同的尺，
不要互相比。放在這裡只是為了讓人知道 0.025 這個目標有多遠。）


【先搞懂尺度】2.5% 的 MAP 對 H&M 有沒有價值

有，但價值不是來自 MAP 這個數字本身。提案 §3.3 的估法是：

    增量營收 ≈ 線上銷售 × 採用率 × 轉換提升

MAP 只影響最後那一項，而且是間接的 —— 推得準，使用者才願意用；用了才會買。
H&M 線上銷售約 SEK 685 億，最保守的格子（採用率 5%、轉換提升 5%）就是增量
SEK 1.7 億。價值的量級來自那個乘數，不是來自小數點後第二位。

還有一件事要誠實寫下來：**離線 MAP 會系統性低估真實價值。**

MAP 只量一件事 —— 「他本來就要買的東西，你猜中沒有」。但推薦系統真正賺錢的地方
是「影響」：他本來沒打算買，看到推薦覺得不錯就買了。那一塊離線評估**結構上量不到**，
因為歷史資料裡不存在「如果當時推薦了會怎樣」這筆紀錄。對話式推薦更是如此 ——
真實使用者會說話，離線資料裡沒有人說話。提案 §3.2 另外用「模擬對話命中率」補這
一塊，就是這個原因。

所以這一層要追求的不是「數字好看」，而是**在固定的候選預算下，把正解盡量撈進
池子**。下面每一個常數的註解，都在回答同一個問題：這個預算花在這裡划不划算。

**每一支函式都要一個日期，那個日期的意思是「把時鐘撥到哪一天」。**

候選的定義本身就綁著時間：「他近 12 週買過什麼」在 9/02 問和 9/09 問答案不一樣，
中間他又買了東西；「他那一群最近 14 天賣最好的那幾件」更是每週都在換。
實測同一位顧客：

    時鐘撥到 9/02  →  580 個候選
    時鐘撥到 9/09  →  568 個候選，其中 88 個是 9/02 那次沒有的

所以**沒有「這個人的候選池」這種東西，只有「某個時間點上，這個人的候選池」**。

**那「撥到 9/09」到底讀了哪些資料？不是整個訓練集** —— 每一源有自己的回看窗口：

    回購        近 12 週（84 天）   2020-06-17 ~ 09-08
    共購        近 12 週            同上
    ALS         近 12 週            同上
    分群熱門    近 14 天            2020-08-26 ~ 09-08
    分群標籤    全部歷史            2018-09-20 ~ 09-08（主要品牌線要夠多資料才穩）

全部都是嚴格「< 9/09」，一天都沒有往後看。窗口長度本身是參數（回購掃過
4 / 8 / 12 / 26 / 52 週，見 REPURCHASE_WEEKS），不是「有多少資料就用多少」。

**答案卷那邊則是整整一週合起來算一張。** 驗證週 9/09 ~ 9/15 這七天內買過的
(顧客, 商品) 去重，不分哪一天買 —— 9/09 買的和 9/15 買的都算對。所以我們在做的
是「猜他這一週會買什麼」，不是「猜他星期三會買什麼」。

為什麼需要好幾個時間點：W6 的排序模型要學「什麼樣的候選會被買」，而要學就得有
題目和答案 —— 題目是候選池，答案是他到底買了沒。答案只能從歷史裡挖，所以得假裝
現在是 9/09：用 9/09 以前的資料撈候選（題目），再翻開 9/09 那週的實際購買紀錄
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
分開建是為了看得出每一源的邊際貢獻 —— 加了一源撈中率沒動，那一源就不值得留。

【待解決】撈中率 17.0%，讓 MAP@12 最高只能到 18.9%

**以下數字全部在驗證週（2020-09-09 ~ 09-15）量的。保留週封存中**，只在 W5 收尾時
量一次。反覆看保留週，後續的判斷就會被它影響 —— 那是共識 §1 訂這條紀律的理由。

驗證週實測：每人 198.9 件候選，但 72,019 位買家裡有 46,013 人（63.9%），
他的候選裡一件那週真的會買的都沒有。那些人排序模型再厲害也是 0 分。

排序模型只能把候選池裡的東西往前排，不能無中生有 —— 召回漏掉的，就是整個系統
永遠拿不到的分數。完美排序模型能拿到的 MAP@12 是 18.89%（推導見【名詞對照】）。

    上限   18.89%   召回決定的天花板
    目標    2.50%   提案 §3.2      ← 在上限底下，所以召回沒有把路堵死

放大候選池能把上限推高。實測每人 664 件時（回購 52 週 / 分群 300 / 共購 50 /
ALS 200）：撈中率 28.3%、macro 30.2%、撈不到任何東西的人從 63.9% 降到 48.9%。
**W5 原訂的 30% 大致搆得到**，代價是候選數變 3.3 倍，而排序模型要排除的干擾項
也跟著多 3.3 倍。兩個效應方向相反，要有排序模型才驗得出淨效果是正是負。

**決議（2026-09-29）：先做 W6 的排序模型，看它實際發揮到上限的幾成。
離目標 2.5% 還很遠的話，再回頭放大候選池。**

（2026-10-06 那次調參不違反這條決議：它是把名額在四源之間重新分配，池子只從
每人 191.1 變成 198.9，沒有放大。詳情見下面四源效率表的註解。）

兩個常見的誤會：

  「只推 12 件」不是壓低分數的原因。AP 的分母是 min(買的件數, 12)，
  買 3 件推對 3 件、買 50 件推對 12 件，兩個都是滿分。98.2% 的顧客買不到 12 件。

  語意檢索（ChromaDB）補不了這個洞。它要有一句查詢才動得起來，離線評估裡沒有
  人說話。它的價值在對話路徑，提案 §3.2 用「模擬對話命中率」另外量。

**記憶體**：召回是展開，不是聚合。輸出遠大於輸入，一不小心就是上億列。
所有候選一律留在 DuckDB 裡（CREATE TABLE），不要 .df() 拉進 pandas ——
DuckDB 算不下會溢寫到 .tmp，pandas 不會，只會讓整台機器沒記憶體可用。
只有最後那幾個數字才拉出來。
"""

from __future__ import annotations

import duckdb
import pandas as pd

import dataset

# ═══ 四個召回源的效率。下面每一個常數的取捨都看這張表 ═══
#
# 驗證週、目前的設定（回購 12 週 / 分群 150 / 共購 5 / ALS 25、不換色）：
#
#   來源        每人候選   單獨撈中率   單獨效率（一個候選換到多少撈中率）
#   回購            10.8       2.77%    0.2567%   ← 單獨最划算
#   ALS             25.0       2.60%    0.1041%
#   共購            47.1       3.92%    0.0834%
#   分群熱門       149.9      12.40%    0.0827%
#   四源合計       198.9      16.96%    0.0853%
#
# **單獨效率不能直接拿來分名額**，因為四源會撈到同一批東西。回購的單獨效率是
# 第二名的兩倍半，但它只能撈「他自己買過的」—— 一個人買過幾件就是幾件，
# 給再多名額也長不出新東西，所以 REPURCHASE_WEEKS 那個旋鈕的上限很低。
#
# 真正的依據是**邊際效率**：在目前這個操作點上，多給某一源一個名額，四源合計
# 的撈中率會多幾個百分點。在舊操作點（分群 100 / 共購 10 / ALS 50）掃出來是：
#
#   分群熱門  100 → 150     0.0546%   ← 最高
#   回購      12 → 26 週    0.0343%
#   ALS       50 → 100      0.0240%
#   共購      10 → 20       0.0216%   ← 最低
#
# 這張表就是 2026-10-06 那次調整的依據：**從最低的兩源收名額（共購 10→5、
# ALS 50→25），交給最高的那源（分群 100→150）**。實測池子幾乎沒變大
# （每人 191.1 → 198.9），撈中率 15.52% → 16.96%，整體效率 0.0812% → 0.0853%。
# 加量通常會讓效率掉，這次沒掉，因為是「挪」不是「加」。
#
# 邊際效率綁在操作點上。名額一挪、操作點就變了，上面那張邊際表對「下一步該給
# 誰」已經不準，要重掃才知道 —— 不要直接拿它外插。
#
# 也不要把兩張表相加來預測組合的結果。四源重疊，砍掉共購之後分群新撈到的東西，
# 有一部分本來就是共購在撈的。組合只能實測，這次就是實測才發現挪完比預期好。

# 回購往回看幾週。提案 §2.2.2 寫的是 12 週。
REPURCHASE_WEEKS = 12

# 回購要不要把「同款其他顏色」也一起納入候選。實測結論是**不要**。
#
# 驗證週四源合計，只改這一項與分群深度：
#
#   設定                    每人候選   撈中率    效率（每候選換到的撈中率）
#   換色 + 分群 50            170.2   13.21%   0.0776%    原本的做法
#   不換色 + 分群 100         191.1   15.52%   0.0812%    ← 當時採用
#   不換色 + 分群 150         238.9   18.13%   0.0759%
#   換色 + 分群 100           217.4   16.16%   0.0743%    對照組
#
# 為什麼關掉，三個理由：
#
# 1. 「他買過的那幾件」是所有源裡效率最高的訊號 —— 每人只要 10.8 個候選就換到
#    2.77% 撈中率（0.257%/候選，是第二名 ALS 0.104% 的兩倍半，見上面那張效率表）。
#    展開成同款所有顏色後，候選暴增到 61.5 個，撈中率只到 5.18%，
#    效率掉到 0.084%。等於拿最好的訊號去餵一堆雜訊。
#
# 2. 第四列那個對照組，證實問題出在換色本身，不是「分群取深剛好有用」：
#    同樣分群 100，加了換色只多 0.64 個百分點，卻多吃 26 個候選 ——
#    0.024%/候選，比四源裡單獨效率最差的分群熱門（0.083%）還差三倍多，
#    是整個設計裡最不划算的一項。
#
# 3. 把那些名額挪給分群熱門（50 → 100），撈中率從 13.21% 升到 15.52%，
#    而且**整體效率也跟著變好**。放大池子通常效率會掉，這裡反而升，
#    因為名額是從爛的地方挪到好的地方，不是單純加量。
#
# 換色不是完全沒用：實測有 5,480 組命中是「他沒買過那個顏色」才撈到的，
# 其他源補不上。但那些命中的單位成本太高，同樣的預算給分群熱門換到更多。
# 之後若放寬候選預算（見檔案最上面的【待解決】），這個開關值得重新評估。
EXPAND_COLOURS = False

# 分群熱門每群取前幾名。
#
# 在驗證週上掃過 12 / 30 / 50 / 100 / 200（不看保留週，看了就是拿測試集調參）：
#
#   top_n   這一源單獨   加上回購   每人候選數   每多一個候選換到的撈中率
#      12       2.0%       6.9%         60
#      30       4.1%       8.8%         78            0.107%
#      50       5.8%      10.4%         98            0.081%     ← 當時採用
#     100       9.3%      13.7%        148            0.067%
#     200      15.0%      19.0%        247            0.053%
#
# 為什麼當時選 50，而不是撈中率最高的 200（2026-09-29 的紀錄，
# 結論已被下面兩次「後續更新」取代，留著是為了留下推理過程）：
#
# （這個 top_n 是「每群撈幾件當候選」，跟 evaluate.py 的 k=12「最終推薦幾件」無關）
#
# 1. 只看撈中率的話，答案永遠是「top_n 越大越好」，這個問題問不出停損點。
#    我們真正要最大化的是 MAP@12。top_n 拉大等於要排序模型從 247 件裡挑 12 件，
#    而不是從 98 件裡挑 —— 召回的上限提高了，但排序的準度可能反而掉。
#    哪一邊的效應大，要等排序模型做出來、在驗證週上掃 top_n 才知道。現在沒有那個證據。
#
# 2. 曲線上沒有拐點。報酬遞減但非常平緩，沒有哪裡突然掉一半。
#    也就是說，光看撈中率曲線本身「看不出」正確答案在哪。
#
# 3. 當時共購與 ALS 還沒做，本來期待它們補上缺口。後來實測：共購只加 2.1 個百分點、
#    ALS 只加 0.8 個百分點，四源合計 13.2%，離 30% 還很遠。
#    **所以「靠新的召回源補足」這條路已經走完了**，剩下的槓桿只有放大候選池。
#
# top_n 只是一個數字，改完重跑幾秒。反過來，先把候選池撐大、等 W6 才發現排序模型的
# 訓練資料跑不動，代價高得多。提案 §2.2.2 的預算是每人 50-100 件，50 剛好貼齊上限。
#
# 後續更新（2026-10-05）：50 改成 100。
#
# 上面那張表是在「回購含同款換色」的設定下掃的。後來實測換色划不來（見
# EXPAND_COLOURS 的註解），關掉它省下每人約 50 個候選，把那些名額挪給這一源 ——
# 分群取 100 而不是 50，撈中率從 13.21% 升到 15.52%，整體效率也變好。
#
# 所以這個 100 不是「放大池子」的結果，是**把名額從效率最差的地方挪到第二好的
# 地方**。真正的放大池子（每人 700 件以上）還沒做，決議見檔案最上面的【待解決】。
#
# 後續更新（2026-10-06）：100 改成 150。
#
# 這次不是重新發現這一源有多好，而是量出了邊際效率（見檔案上方那張效率表）：
# 在 100 的位置上，多給分群一個名額換到 0.0546% 撈中率，是四源裡最高的；
# 共購和 ALS 分別只有 0.0216% 和 0.0240%，最低。名額從低的挪到高的，
# 池子幾乎沒變大而撈中率多了 1.44 個百分點。
#
# 要誠實記一筆：**每人 198.9 個候選已經超出提案的預算。** §2.2.2 寫「每位顧客約
# 50–100 個候選」，風險表更保守，寫「每人候選上限 50」。超出是刻意的 —— §2.2.4
# 本來就把「每人候選數上限」列為要在驗證週上選的參數，而驗證週的證據是加到 198.9
# 還在換到撈中率。代價落在 W6：候選越多，排序模型的訓練資料越大。真的跑不動，
# 往回縮的順序就照上面那張邊際表，從最低的那一源先砍。
#
# 最終的 top_n 等排序模型出來後用驗證週的 MAP 決定，不是用撈中率。
SEGMENT_TOP_N = 150

# 共購統計看多久，以及每件商品保留幾個共購夥伴。weeks=12 與回購對齊。
#
# 在驗證週掃 top_n（合計 = 回購 12 週 + 分群 50 + 共購）：
#
#   top_n   這一源單獨   合計    每人候選數
#      —         —      10.4%      98      （沒有共購）
#      5       3.9%     11.5%     121
#     10       5.6%     12.5%     149      ← 當時採用
#     20       7.6%     13.9%     206
#     50      11.4%     16.7%     369
#
# 注意單獨 5.6% 但只讓合計 +2.1 個百分點 —— 其中 3.5 個百分點跟前兩源重疊。
# 合理：常被一起買的東西，本來就多半是熱銷品，分群熱門已經撈走了。
#
# 跨源的效率對比見檔案上方那張四源效率表。（這裡原本抄了一份，設定一改就過期，
# 所以刪掉改成指過去。）
#
# 後續更新（2026-10-06）：top_n 從 10 改成 5。共購的邊際效率是四源裡最低的
# （0.0216%），名額挪給分群熱門。單獨砍這一刀（其餘不動）是每人少 26 個候選、
# 四源合計撈中率從 15.52% 掉到 14.73%，掉的 0.79 個百分點比省下的預算便宜。
#
# 那為什麼還留著共購：它是唯一能推出「他沒買過、也不在他那群熱銷榜上」的商品的源。
# 撈中率上重疊不代表沒價值 —— 排序模型會拿 from_cooc 這個旗標當特徵，
# 「三源都指向它」和「只有分群熱門撈到」是不同的訊號。
COOC_WEEKS = 12
COOC_TOP_N = 5

# ALS：看多久的互動、隱因子幾維、每人取幾個推薦。
# 時鐘撥到驗證週、近 12 週是 350 萬筆互動、52 萬位顧客、41,585 件商品，
# 稀疏矩陣約 28 MB。（撥到保留週是 324 萬 / 51 萬 / 41,376，數字跟著時鐘走。）
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
#
# 後續更新（2026-10-06）：top_n 從 50 改成 25，factors=64 + BM25 不變。
# ALS 的邊際效率 0.0240% 是四源裡倒數第二（見檔案上方那張效率表），名額挪給
# 分群熱門。單獨砍這一刀（其餘不動）是每人少 13 個候選、四源合計撈中率從
# 15.52% 掉到 15.13%。砍到 25 之後它的單獨效率反而是四源裡第二高（0.104%）——
# 前 25 名本來就是比較準的那幾個，第 26~50 名多半是雜訊。
ALS_WEEKS = 12
ALS_FACTORS = 64
ALS_TOP_N = 25


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
               weeks: int = REPURCHASE_WEEKS,
               expand_colours: bool = EXPAND_COLOURS) -> int:
    """建 cand_repurchase：顧客近 weeks 週買過的品項。

    依據是提案 §2.2.2：14.1% 的交易是同品項回購（排除同一籃買多件後仍有 4.4%）。
    這是所有召回源裡效率最高的一個：每人 10.8 個候選就換到 2.77% 撈中率。

    expand_colours=True 會把「同款的其他顏色」也納入。article_id 是「款 + 色」，
    product_code 只到「款」，所以同一個 product_code 底下就是同款的各種顏色。
    **預設關閉** —— 它讓候選暴增 5.7 倍但命中只多 1.9 倍，理由見上面 EXPAND_COLOURS
    的註解。留著開關是因為之後若放寬候選預算，值得重新評估。

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


def segment_popular(con: duckdb.DuckDBPyConnection, train_end: str,
                    top_n: int = SEGMENT_TOP_N,
                    days: int = dataset.POP_DAYS) -> int:
    """建 cand_segment：每位顧客拿自己那一群最近 days 天賣最好的前 top_n 名。

    分群是 (年齡層 × 主要品牌線)，共 35 群，定義在 dataset.build_segments。
    B1 基準線用的是同一組標籤 —— 兩邊若各分一次群，分數就不可比。

    這一源專門補回購補不到的洞：回購只給得出「他買過的東西」，換口味的人身上
    是空的。分群熱門不看他買過什麼，只看他屬於哪一群，所以換口味的人也有東西。

    **但純新客還是空的。** 他的品牌線標籤是「未知」，而「未知」那一群在訓練期內
    沒有任何交易，排不出熱銷榜。驗證週有 5,395 人（7.5% 的買家）因此四源全空，
    細節見 build() 的說明。

    top_n 要在驗證週上掃描決定，不要看保留週的撈中率挑 —— 那是拿測試集調參數（共識 §1）。
    掃描結果與每一次改動的理由見上面 SEGMENT_TOP_N 的註解（目前是 150）。

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

    「一起被買」的定義是**同一位顧客、同一天**算一籃。用同籃而不是「這個人買過的
    所有東西兩兩配對」，是因為後者會把半年前買的襪子跟今天買的外套配成一對 ——
    那不是「一起買」，只是「都買過」。

    籃子的規模分兩個層次，別混在一起看：

        整份訓練資料   900 萬籃   平均 3.1 件   最大一籃 159 件
        近 12 週       116 萬籃   平均 3.1 件   最大一籃  74 件   ← 這一源實際讀的

    共購次數只從下面那一列統計 —— weeks 把窗口關小了，不是拿 900 萬籃去算。

    這一源補的是另外兩源都補不到的：回購只給得出他買過的東西，分群熱門給的是全群
    一樣的榜單，兩者都無法從「他的個人品味」推到「他沒買過但可能喜歡」的商品。

    weeks 同時決定兩件事：共購統計看多久，以及拿顧客哪一段的購買紀錄去查表。
    top_n 是每件商品保留幾個共購夥伴，要在驗證週上掃描決定。

    排序加了 j 決勝，同票時才不會每次跑出不同的夥伴清單（共識 §4）。
    """
    start = dataset.pop_start_for(train_end, weeks * 7)

    # 先算商品對商品的共購次數。雙向存（i→j 和 j→i 各一列），查表時才不用管方向。
    # 時鐘撥到驗證週、近 12 週是 1,131 萬列、0.3 秒（撥到保留週是 1,032 萬 ——
    # 這個數字跟著時鐘走）；先用 sum(n*(n-1)/2) 估過才建，沒有盲目展開。
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

    # 每件商品只留最強的 top_n 個夥伴。不留的話平均每件有 278 個夥伴（驗證週），
    # 候選池會炸。
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
    近 12 週的稀疏矩陣約 28 MB（350 萬個非零值，每個 8 bytes）。
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

    每人的候選數不固定（驗證週：最少 23、中位數 188、P99 393、最多 1032）。
    變動組長 LambdaRank 本來就吃得下。而「池子大小」跟「會不會買」相關
    （買多的人候選也多），那是 W6 該放進去的特徵，不是瑕疵。

    **但有一群人的候選是零。** 驗證週 72,019 位買家裡，只有 66,624 人拿到候選，
    剩下 5,395 人（7.5%）四個源一個都撈不到。查過了：這 5,395 人全部是「驗證週
    以前從來沒買過東西」的新客。四個源都要歷史才動得起來 —— 回購要他買過、
    共購要他買過、ALS 要他有互動，分群熱門要他那一群有銷售紀錄，而新客的品牌線
    標籤是「未知」，那一群在訓練期內沒有任何交易，連熱銷榜都生不出來。

    所以共識 §2 那條「不足 12 件以 B0 補滿」**真的會觸發**，觸發在這 7.5% 的人
    身上 —— 他們拿到的 12 格全部來自 B0。（上面「最少 23 件」是在「有候選的人」
    裡取的最小值，看不到掛零的人；注意 coverage() 的「平均每人候選數」分母也是
    有候選的顧客數，不是買家數。）上線後新客比例只會更高，B0 補位不是形式，
    是冷啟動唯一的出路。
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

    預設不存，因為量撈中率一天要跑好幾次，每次都寫 209 MB 沒意義。
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
