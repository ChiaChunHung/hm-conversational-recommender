"""
H&M 對話式推薦系統 — 專案入口。

這個檔案只做一件事：看你下了哪個指令，然後呼叫對應模組。
所有實際工作都在模組裡，方便 notebook 與其他模組直接 import 重用。

    python main.py build              需要時轉檔 CSV → Parquet
    python main.py build --rebuild    強制重新轉檔

資料探索統計在 eda.ipynb。

為什麼用 DuckDB 而不是 SQLite：我們做的全是「掃過三千萬筆算統計」，那是欄式儲存
的強項，而 SQLite 是列式、為「抓出或更新某一筆」設計。完整理由與實測見 README，
以及 vault 的《為什麼用 DuckDB + Parquet 而不是 SQLite》—— 這裡不重複。
"""

from __future__ import annotations

import sys

import prepare

USAGE = """用法：
    python main.py build [--rebuild]    把 CSV 轉成 Parquet

資料探索統計請開 eda.ipynb。
"""


def cmd_build(argv: list[str]) -> None:
    """build 指令：請 prepare 把 data/ 的三份 CSV 轉成 Parquet。

    轉出來的檔案放在 data/parquet/，之後所有查詢都讀那裡。
    帶 --rebuild 就強制重轉，不帶的話已存在的檔案會被跳過。
    """
    prepare.build_parquet(rebuild="--rebuild" in argv)


# 指令名字 → 要呼叫的函式。
#
# 右邊的 cmd_build 沒有括號，存的是「函式本身」而不是「執行結果」；
# 加了括號的話程式一啟動就會直接轉檔，而不是等你下指令。
#
# 之後新增模組就在這裡多一行，下面的 main() 完全不用改。
#
# **不是每個模組都該註冊。** evaluate.py 與 recall.py 刻意沒有指令 —— 它們是函式庫，
# 每次呼叫的參數都不一樣（測哪一週、掃哪個 top_n），包成單一指令反而綁手綁腳，
# 用 import 的方式從 notebook 或實驗腳本呼叫才合用。
# 會變成指令的是「跑一次就結束」的動作，例如 build 轉檔，或之後排序模型的訓練。
COMMANDS = {
    "build": cmd_build,
}


def main() -> None:
    """讀命令列的第一個字當指令，查表找到對應函式，把剩下的參數交給它。

    以 python main.py build --rebuild 為例：
        sys.argv            = ['main.py', 'build', '--rebuild']
        sys.argv[1]         = 'build'         → 查 COMMANDS 得到 cmd_build
        sys.argv[2:]        = ['--rebuild']   → 當參數傳進去

    指令打錯或沒給指令，就印出 USAGE 並以失敗狀態結束。
    """
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(USAGE)
    COMMANDS[sys.argv[1]](sys.argv[2:])


# 只有被直接執行（python main.py ...）時才跑 main()。
# 別人 import main 時不會觸發，免得光是 import 就意外開始轉檔。
if __name__ == "__main__":
    main()
