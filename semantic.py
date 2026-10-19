"""
語意檢索：使用者用一句話描述想要的東西，從 10.5 萬件商品裡找出意思最接近的 k 件。
提案附錄 B 的 semantic_search。

======================================================================
 寫 W9 對話 LLM 提示詞之前，先讀這兩點
======================================================================

 1. **情境詞要由對話 LLM 先改寫，這個工具本身不改寫。**
    H&M 的商品描述只寫材質和剪裁，幾乎不寫場合：10.5 萬件裡出現 wedding 的只有 1 件、
    camping 0 件、office 0 件。「朋友婚禮要穿的」直接拿來查幾乎找不到對的商品。
    對話 LLM 呼叫這個工具前，要先改寫成商品描述會用的英文字，例如
        「朋友婚禮要穿的」→ "elegant midi dress in chiffon or satin, flowy"
    改寫規則與提示詞草稿在 vault《LLM 提示詞》。（vault《設計決策紀錄》第 5 項）

 2. **價格條件暫緩。** 條件表格目前不能篩價格，LLM 傳了會報錯。
    原因、已查過的資料與初步提案在 vault《設計決策紀錄》第 10 項；要補的位置在 filters.py。

======================================================================

**這份程式和組員版 catalog.py 的關係**

骨架沿用組員的設計：事先把每件商品轉成向量、查詢時只轉使用者的話；每個顏色一組向量，
查詢時同款只留最像的顏色；商品文字不放名稱、缺描述就留空；用 DuckDB 跟每件商品都比一次。
這些為什麼合理，見 vault《語意檢索說明書》。

改動的地方（理由見 vault《設計決策紀錄》）：
  - 硬性條件改用 filters.py 的固定表格，三個工具共用（第 6、7、8 項）
  - 修好「中途換模型會拿錯向量」：每個模型各自載入、各自快取（第 9 項）
  - 模型的查詢前綴與商品前綴跟著模型走，之後比較 bge-base、e5-base 不用改程式（第 13 項）
  - 「只留還在賣的」改成 filters.sellable_condition() 的保留位置，目前不啟用（第 11 項）

**怎麼跑**

    pip install sentence-transformers         還沒寫進 requirements.txt，併回 main 時要補
    python main.py build                      先要有 Parquet
    python semantic.py build                  建向量（M4 Pro 幾分鐘）
    python semantic.py search "elegant midi dress in chiffon" --where '{"colour_exclude": "Black"}'

或在 Python 裡：

    import semantic
    semantic.semantic_search("elegant midi dress in chiffon",
                             constraints={"product_type": "Dress", "colour_exclude": ["Black"]})

向量放在 repo 外面的 data/semantic/<模型>.parquet，跟其他資料一樣不進版控，刪掉可以重建。
模型本身由 sentence-transformers 下載到 ~/.cache/huggingface。
"""
from __future__ import annotations

import argparse
import json
import time

import duckdb

import dataset
import filters

# 能用的模型，以及各自要求的前綴。
#
# 前綴是模型訓練時的規矩：bge 學的是「一句短問題」配「一段長文章」，問題前面要加一句
# 說明告訴它「這是要拿去找東西的問題」，文章不加。e5 則是兩邊都要加，各加各的。
# 前綴加錯，相似度會整體偏掉。換模型時前綴一定要跟著換，所以綁在一起寫。
#
# 第一個是預設。另外三個是之後比較模型的候選名單（vault《設計決策紀錄》第 13 項）。
MODELS = {
    "BAAI/bge-small-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages: ", "doc": ""},
    "BAAI/bge-base-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages: ", "doc": ""},
    "BAAI/bge-large-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages: ", "doc": ""},
    "intfloat/e5-base-v2": {"query": "query: ", "doc": "passage: "},
}
DEFAULT_MODEL = next(iter(MODELS))

VEC_DIR = dataset.DATA_DIR / "semantic"


def _vec_path(model: str):
    """一個模型一個檔，換模型不會蓋掉舊的，可以並排比較。"""
    return VEC_DIR / (model.replace("/", "__") + ".parquet")


def _check_model(model: str) -> dict:
    if model not in MODELS:
        raise ValueError(f"沒有登記 {model!r} 的前綴。能用的模型：{list(MODELS)}")
    return MODELS[model]


_encoders: dict = {}


def _encoder(model: str):
    """模型一個行程只載入一次：載入要幾秒，轉一句話只要幾毫秒。"""
    if model not in _encoders:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"語意檢索需要 sentence-transformers（缺少 {e.name}）。"
                "請執行：pip install sentence-transformers") from e
        _encoders[model] = SentenceTransformer(model)
    return _encoders[model]


def item_texts(con: duckdb.DuckDBPyConnection) -> "pd.DataFrame":
    """每件商品的「自我介紹」，也就是要轉成向量的那段文字。

    順序是「類型、色系、圖樣 → 分區、品牌線 → 文字描述」，最能區分商品的放最前面。

    **不放商品名稱。** H&M 的名稱常跟商品無關：一套睡衣叫「CAMP …」，搜露營會撈到它。
    使用者用名稱找商品，交給 search_catalog 的關鍵字比對。

    **描述讀原始的 art，不讀 art_c。** art_c 把缺描述的 416 件補成「名稱、類型、色系」，
    等於把名稱塞回來。這裡缺描述就留空（concat_ws 會跳過 NULL），靠類型和顏色一樣找得到。
    這 416 件所屬的款式，其他顏色也都沒有描述，沒得借。
    """
    return con.execute("""
        SELECT a.article_id,
               concat_ws('. ',
                   concat_ws(', ', a.product_type_name, a.colour_group_name,
                                   a.graphical_appearance_name),
                   concat_ws(', ', a.section_name, a.index_group_name),
                   raw.detail_desc) AS text
        FROM art_c a
        JOIN art raw USING (article_id)
        ORDER BY a.article_id
    """).df()


def build_index(model: str = DEFAULT_MODEL, rebuild: bool = False) -> None:
    """把每件商品轉成向量，存成 (article_id, v) 的 Parquet。已存在就跳過，除非 rebuild。

    只存編號和向量。名稱、顏色這些查詢時再從 art_c 拿，只有一份，不會兩份對不上。
    """
    prefix = _check_model(model)["doc"]
    path = _vec_path(model)
    if path.exists() and not rebuild:
        print(f"  跳過：{path.name} 已存在（{path.stat().st_size / 1e6:.0f} MB）")
        return

    con = dataset.connect()
    df = item_texts(con)
    print(f"建向量：{len(df):,} 件商品，模型 {model}")

    t0 = time.perf_counter()
    # 一字不差的自我介紹只轉一次：10.5 萬件商品只有約 9.5 萬段不同的文字。
    # 向量正規化成長度 1，兩個向量的內積就直接等於 cosine 相似度。
    uniq = df["text"].drop_duplicates().tolist()
    pos = {t: i for i, t in enumerate(uniq)}
    vec = _encoder(model).encode([prefix + t for t in uniq], batch_size=256,
                                 normalize_embeddings=True, show_progress_bar=True)
    df["v"] = list(vec[df["text"].map(pos).to_numpy()].astype("float32"))
    print(f"  轉換花了 {time.perf_counter() - t0:.0f} 秒")

    VEC_DIR.mkdir(parents=True, exist_ok=True)
    con.register("vec_df", df[["article_id", "v"]])
    con.execute(f"COPY vec_df TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.unregister("vec_df")
    print(f"  寫入 {path}（{path.stat().st_size / 1e6:.0f} MB）")


_cons: dict[str, tuple[duckdb.DuckDBPyConnection, int]] = {}


def _search_con(model: str) -> tuple[duckdb.DuckDBPyConnection, int]:
    """回傳 (載好這個模型向量的連線, 向量維度)。每個模型各自一條連線、各自快取。

    組員版只有一條全域連線，載入第一個模型的向量後就一直用那份；中途換模型時，
    查詢句用新模型轉、卻拿去跟舊模型的向量比，結果是錯的而且不會報錯。
    商品和查詢一定要用同一個模型轉才能比，所以這裡以模型名稱為鍵分開存。
    """
    if model not in _cons:
        path = _vec_path(model)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 不存在。先執行：\n    python semantic.py build --model {model}")
        con = dataset.connect()
        dim = con.execute(f"SELECT len(v) FROM read_parquet('{path}') LIMIT 1").fetchone()[0]
        # Parquet 沒有固定長度陣列，讀回來是不定長清單；轉回 FLOAT[dim]，
        # array_inner_product 才吃得下
        con.execute(f"CREATE TABLE vec AS SELECT article_id, v::FLOAT[{dim}] AS v "
                    f"FROM read_parquet('{path}')")
        _cons[model] = (con, dim)
    return _cons[model]


def semantic_search(query_text: str, constraints: dict | None = None, k: int = 12,
                    one_per_product: bool = True, model: str = DEFAULT_MODEL) -> list[dict]:
    """找出意思最接近 query_text 的 k 件商品。

    query_text   **要是改寫過的英文**，見檔案開頭第 1 點
    constraints  硬性條件，格式見 filters.py。先篩掉不符合的，剩下的才比相似度
    one_per_product
                 同款只留最像的那個顏色，預設開。不開的話前 12 名可能是同一件洋裝的
                 12 種顏色。使用者指定的顏色在 constraints 先篩，不受影響

    回傳每件商品一個 dict：編號、名稱、類型、大類、顏色、品牌線、相似度。
    多帶名稱和類型，LLM 寫回覆時可以直接引用，不用再查一次。

    條件太嚴時回傳會少於 k 件，甚至是空的。呼叫端要自己看長度；
    要不要放寬條件或追問使用者，是對話 LLM 的決定，不在這裡處理。
    """
    prefix = _check_model(model)["query"]
    con, dim = _search_con(model)
    q = _encoder(model).encode([prefix + query_text], normalize_embeddings=True)[0]

    conds, params = filters.to_sql(constraints, alias="a", con=con)

    # 相似度一樣時用編號排，同樣的查詢每次結果都一樣
    rows = con.execute(f"""
        SELECT a.article_id, a.prod_name, a.product_type_name, a.product_group_name,
               a.perceived_colour_master_name, a.index_group_name,
               array_inner_product(e.v, ?::FLOAT[{dim}]) AS similarity
        FROM vec e JOIN art_c a USING (article_id)
        {"WHERE " + " AND ".join(conds) if conds else ""}
        {"QUALIFY row_number() OVER (PARTITION BY a.product_code "
         "ORDER BY similarity DESC, a.article_id) = 1" if one_per_product else ""}
        ORDER BY similarity DESC, a.article_id
        LIMIT {int(k)}
    """, [q.tolist()] + params).fetchall()

    cols = ["article_id", "prod_name", "product_type", "product_group", "colour",
            "index_group", "similarity"]
    out = [dict(zip(cols, r)) for r in rows]
    for r in out:
        r["similarity"] = round(r["similarity"], 4)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="語意檢索：建向量或試查詢")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="把商品轉成向量")
    b.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    b.add_argument("--rebuild", action="store_true", help="已存在也重建")

    s = sub.add_parser("search", help="試一句查詢")
    s.add_argument("query")
    s.add_argument("--where", default=None, help='條件表格的 JSON，例如 \'{"colour_exclude": "Black"}\'')
    s.add_argument("-k", type=int, default=12)
    s.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))

    a = p.parse_args()
    if a.cmd == "build":
        build_index(a.model, a.rebuild)
    else:
        rows = semantic_search(a.query, json.loads(a.where) if a.where else None, a.k,
                               model=a.model)
        for i, r in enumerate(rows, 1):
            print(f"{i:>2}. {r['similarity']:.3f}  {r['article_id']}  {r['product_type']:<14} "
                  f"{r['colour']:<10} {r['index_group']:<14} {r['prod_name']}")


if __name__ == "__main__":
    main()
