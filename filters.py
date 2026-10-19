"""
硬性條件的共用篩選模組。semantic_search、search_catalog、recommend 三個工具都呼叫這一份。

為什麼獨立成一個檔案：組員版 catalog.py 把篩選寫在語意檢索裡，看起來像語意檢索專屬的，
另外兩個工具容易各寫一套。放在這裡，誰都不用依賴誰。（vault《設計決策紀錄》第 6 項）

**條件格式是一張固定表格**，LLM 從對話裡聽出硬性條件後只能照表填空：

    欄位              對應的商品欄位                  能填的值
    index_group       index_group_name               5 種品牌線，例如 Ladieswear
    product_group     product_group_name             18 種大類，例如 Garment Upper body
    product_type      product_type_name              127 種細類，例如 Dress
    colour            perceived_colour_master_name   18 種大色系，例如 Black
    colour_exclude    同上                           同上

    例：「女裝洋裝，不要黑色也不要灰色」
        {"index_group": "Ladieswear", "product_type": "Dress",
         "colour_exclude": ["Black", "Grey"]}

為什麼不讓 LLM 直接寫資料庫欄位（組員的寫法）：同一件事有好幾欄可以選，LLM 可能選錯。
例如顏色有細色系（50 種）和大色系（20 種）兩欄，選了細色系寫「不要 Grey」，
Dark Grey、Light Grey 還是會出現。這裡每個條件只對應一個欄位，由我們決定。
（vault《設計決策紀錄》第 7 項）

**規則**

  - 每欄可以填一個值或一串值；一串值代表「其中之一」。沒填、None、空清單都代表沒有限制
  - 大小寫不拘，"black" 會對到 "Black"
  - 值不在清單上就直接報錯，錯誤訊息會列出正確選項，讓 LLM 重填。
    不能默默放行：值寫錯卻不報錯的話，「不要黑色」就等於沒篩，而且沒人會發現
  - 大類和細類**是「或」的關係**：都填的話，符合其中一個就留下。
    兩欄是同一件事的粗細兩種說法，用「且」的話「上衣或洋裝」會什麼都找不到。
    原則是寧寬勿窄：拿不準時填大類，不相關的會被查詢句排到後面，不會被剔掉
  - 只有使用者明確說「要／不要」的才填。「最好是碎花」這類偏好寫進查詢句，不要填進表格
    （vault《LLM 提示詞》規則 4）

**價格條件暫緩。** 表格目前沒有價格欄位，原因與已查過的資料見 vault《設計決策紀錄》第 10 項。
LLM 若傳了 price_max 或 price_band 會直接報錯，不會默默忽略 —— 忽略的話使用者說
「30 歐元以內」卻拿到 80 歐元的大衣。要補的位置在 _PRICE_KEYS 與 to_sql() 裡標了 TODO。
"""
from __future__ import annotations

import duckdb

import dataset

# 表格欄位 → 商品表欄位。改這張表就改了 LLM 能用的條件。
FORM = {
    "index_group": "index_group_name",
    "product_group": "product_group_name",
    "product_type": "product_type_name",
    "colour": "perceived_colour_master_name",
    "colour_exclude": "perceived_colour_master_name",
}

# 資料裡代表「不知道」的值，不開放給 LLM 選。
# 它們不會被剔除：「不要黑色」時這些商品照樣留下，只是不能指定「要 Unknown」。
_HIDDEN = {"Unknown", "undefined"}

# 細類裡寫法不同、其實是同一樣東西的值。左邊併進右邊，選右邊時兩種都會篩到。
# 只有大小寫不同的（Dog Wear / Dog wear）不用列，大小寫本來就不拘。
_SAME_AS = {
    "Earrings": "Earring",
    "Flat shoes": "Flat shoe",
}

# TODO（價格，vault《設計決策紀錄》第 10 項）：補做時把 price_max、price_band 加進 FORM，
# 並在 to_sql() 加上對應的條件。在那之前，傳這兩欄會報錯。
_PRICE_KEYS = {"price_max", "price_band"}

_values: dict[str, dict[str, list[str]]] | None = None


def _load_values(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, list[str]]]:
    """從商品表讀出每欄的合法值。

    回傳 {表格欄位: {小寫鍵: [資料裡的原始寫法, ...]}}。一個鍵可能對到好幾種寫法
    （大小寫不同、或 _SAME_AS 併進來的），篩選時全部都要算進去。

    從資料讀而不是寫死：細類有 131 種，寫死的清單遲早會跟資料對不上。
    """
    out: dict[str, dict[str, list[str]]] = {}
    for key, col in FORM.items():
        raw = [r[0] for r in con.execute(
            f"SELECT DISTINCT {col} FROM art_c WHERE {col} IS NOT NULL ORDER BY 1").fetchall()]
        groups: dict[str, list[str]] = {}
        for v in raw:
            if v in _HIDDEN:
                continue
            canon = _SAME_AS.get(v, v)
            groups.setdefault(canon.lower(), []).append(v)
        out[key] = groups
    return out


def _vals(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, dict[str, list[str]]]:
    """合法值只讀一次，之後重用。"""
    global _values
    if _values is None:
        _values = _load_values(con or dataset.connect())
    return _values


def allowed_values(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, list[str]]:
    """每個表格欄位能填哪些值。W9 寫 LLM 工具說明時，可以直接把這份清單放進去。

    每個值以第一種寫法代表（例如 Earring 代表 Earring 與 Earrings）。
    """
    return {key: sorted(_SAME_AS.get(g[0], g[0]) for g in groups.values())
            for key, groups in _vals(con).items()}


def to_sql(constraints: dict | None, alias: str = "a",
           con: duckdb.DuckDBPyConnection | None = None) -> tuple[list[str], list]:
    """把表格翻成 SQL 條件與參數。條件之間是「且」，大類與細類之間是「或」。

    alias 是商品表在呼叫端 SQL 裡的別名。值一律以參數傳入，不拼進 SQL 字串，
    所以 LLM 的輸出塞不進任何指令。

    回傳 (條件清單, 參數清單)，呼叫端用 " AND ".join(條件) 接進 WHERE。
    """
    constraints = constraints or {}
    vals = _vals(con)

    price = _PRICE_KEYS & constraints.keys()
    if price:
        raise ValueError(
            f"價格條件 {sorted(price)} 目前還不支援（暫緩，見 vault《設計決策紀錄》第 10 項）。"
            "拿掉價格條件再查一次，並告訴使用者目前還不能依價格篩選。"
            "不要把價格寫進查詢句：商品文字裡沒有價格，寫了也比對不到。")
    unknown = constraints.keys() - FORM.keys()
    if unknown:
        raise ValueError(f"沒有 {sorted(unknown)} 這個條件。能用的條件：{sorted(FORM)}")

    # 先把每欄的值換成資料裡的原始寫法
    picked: dict[str, list[str]] = {}
    for key, v in constraints.items():
        if v is None:
            continue
        items = [v] if isinstance(v, str) else list(v)
        if not items:
            continue
        raw: list[str] = []
        for item in items:
            groups = vals[key].get(_SAME_AS.get(str(item), str(item)).lower())
            if groups is None:
                raise ValueError(
                    f"{key} 沒有 {item!r} 這個值。能填的值：{allowed_values(con)[key]}")
            raw.extend(groups)
        picked[key] = raw

    conds: list[str] = []
    params: list = []

    def isin(col: str, values: list[str], negate: bool = False) -> str:
        params.extend(values)
        return (f"{alias}.{col} {'NOT ' if negate else ''}IN "
                f"({', '.join('?' * len(values))})")

    if "index_group" in picked:
        conds.append(isin(FORM["index_group"], picked["index_group"]))

    kinds = [isin(FORM[k], picked[k]) for k in ("product_group", "product_type") if k in picked]
    if kinds:
        conds.append("(" + " OR ".join(kinds) + ")")

    if "colour" in picked:
        conds.append(isin(FORM["colour"], picked["colour"]))
    if "colour_exclude" in picked:
        # 顏色是 NULL 的商品要另外放行：NOT IN 碰到 NULL 會得到 NULL，商品就被剔掉了，
        # 但「不要黑色」不該剔掉顏色不明的商品
        col = f"{alias}.{FORM['colour_exclude']}"
        conds.append(f"({col} IS NULL OR {isin(FORM['colour_exclude'], picked['colour_exclude'], negate=True)})")

    sellable = sellable_condition(alias)
    if sellable:
        conds.append(sellable)

    return conds, params


def sellable_condition(alias: str = "a") -> str | None:
    """「只留可售商品」的位置，目前不啟用，永遠回傳 None。

    Kaggle 資料沒有庫存與上架狀態。用「最近有沒有人買」來推測是在解決一個
    和 H&M 合作後就不存在的問題，所以不做（vault《設計決策紀錄》第 11 項）。
    拿到 H&M 的可售資料後，在這裡回傳對應的 SQL 條件，三個工具就都會套用。
    """
    return None
