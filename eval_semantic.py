"""
語意檢索的驗收：W8 標準是「情境詞查詢人工檢視相關率 ≥ 80%」。

做法（vault《設計決策紀錄》第 12 項）：

  1. 約 20 個情境，每個測兩個版本：
       原句     使用者會講的話，例如 "outfit for a friend's wedding"
       改寫後   照 vault《LLM 提示詞》規則改寫，例如 "elegant midi dress in chiffon or satin"
     改寫版的相關率明顯比原句高，就證明改寫有用（第 5 項的決定）
  2. 每個查詢取前 12 件，由**獨立的 agent** 標記相關／不相關
       - 只看得到情境和商品，看不到是哪個模型、哪個版本找到的
       - 用原本的情境判斷（「適不適合參加婚禮」），不是用改寫句判斷，
         否則改寫寫偏了也看不出來
  3. 使用者抽查一部分，跟 agent 的判斷差太多就修正標記規則重標

**同一件商品只標一次。** 標記以 (情境, 商品) 為鍵存在 labels.csv，之後換模型跑，
找到同一件商品就直接沿用，只標新出現的。

**流程**

    python eval_semantic.py run              跑查詢，產生待標記清單 to_label.csv
    （交給獨立 agent，把結果寫進 labels.csv）
    python eval_semantic.py score            算相關率
    python eval_semantic.py review -n 40     抽樣給使用者抽查，產生 review.csv
    （使用者在 review.csv 的 your_label 欄填 1 或 0）
    python eval_semantic.py agree            算使用者和 agent 的一致率

所有檔案放在 repo 外面的 data/semantic/eval/。

**labels.csv 的格式**：scenario_id, article_id, relevant（1 或 0）, reason（一句話）
"""
from __future__ import annotations

import argparse
import random

import pandas as pd

import dataset
import semantic

EVAL_DIR = dataset.DATA_DIR / "semantic" / "eval"
K = 12
PASS = 0.80

_WOMEN = {"index_group": ["Ladieswear", "Divided"]}
_MEN = {"index_group": "Menswear"}
_KIDS = {"index_group": "Baby/Children"}

# 情境清單。
#   situation    給標記者看的情境（中文），判斷相關與否只看這個
#   constraints  LLM 從對話裡聽得出來的硬性條件。只填品牌線：情境本身通常就隱含了
#                是誰要穿，不填的話男裝會混進女性情境，這不是語意檢索該負責的錯
#   raw          使用者原本會講的話（英文）
#   rewritten    照 vault《LLM 提示詞》規則改寫：寫材質、剪裁、風格，不寫場合本身
SCENARIOS = [
    dict(id="wedding_guest", situation="女性受邀參加朋友的婚禮，當賓客要穿的衣服", constraints=_WOMEN,
         raw="outfit for a friend's wedding",
         rewritten="elegant midi dress in chiffon or satin, flowy, feminine"),
    dict(id="office_women", situation="女性平日上班穿的衣服", constraints=_WOMEN,
         raw="clothes for the office",
         rewritten="tailored blazer, straight trousers or neat blouse in smooth woven fabric"),
    dict(id="party_women", situation="女性晚上去派對要穿的衣服", constraints=_WOMEN,
         raw="party outfit",
         rewritten="sequin or satin mini dress, glittery fitted top, shiny details"),
    dict(id="date_women", situation="女性去晚餐約會要穿的衣服", constraints=_WOMEN,
         raw="outfit for a dinner date",
         rewritten="fitted dress or blouse in soft fabric with lace or satin details"),
    dict(id="interview_women", situation="女性去面試要穿的衣服", constraints=_WOMEN,
         raw="what to wear to a job interview",
         rewritten="tailored blazer and suit trousers, neat button-up blouse"),
    dict(id="festival_women", situation="女性去夏天戶外音樂祭要穿的衣服", constraints=_WOMEN,
         raw="music festival outfit",
         rewritten="denim shorts, crochet or fringe top, boho dress, bucket hat"),
    dict(id="graduation_women", situation="女性參加畢業典禮要穿的衣服", constraints=_WOMEN,
         raw="graduation ceremony outfit",
         rewritten="elegant knee-length dress or smart blazer, light and polished"),
    dict(id="yoga", situation="女性上瑜伽課要穿的衣服", constraints=_WOMEN,
         raw="yoga class",
         rewritten="high-waisted sports leggings and sports bra in stretchy fabric"),
    dict(id="beach", situation="去海邊度假要穿或要帶的衣物", constraints=None,
         raw="beach vacation",
         rewritten="swimsuit, bikini, lightweight linen shirt, loose cotton shorts, sarong"),
    dict(id="camping", situation="去露營要穿的衣物", constraints=None,
         raw="going camping",
         rewritten="durable outdoor jacket or trousers, water-repellent, practical pockets"),
    dict(id="hiking", situation="去山上健行要穿的衣物", constraints=None,
         raw="hiking trip",
         rewritten="outdoor trousers, windproof jacket, functional sportswear, fleece"),
    dict(id="rain", situation="下雨天出門要穿的衣物", constraints=None,
         raw="rainy day",
         rewritten="water-repellent raincoat with hood, rain jacket, rubber boots"),
    dict(id="winter", situation="冬天很冷時保暖要穿的衣物", constraints=None,
         raw="keep warm in winter",
         rewritten="padded jacket, wool coat, thick knitted sweater, warm lining"),
    dict(id="home", situation="在家放鬆或睡覺時穿的衣服", constraints=None,
         raw="something comfy to wear at home",
         rewritten="soft cotton pyjama set, loungewear, relaxed fit"),
    dict(id="office_men", situation="男性平日上班穿的衣服", constraints=_MEN,
         raw="work clothes for the office",
         rewritten="slim fit cotton shirt with collar, chinos, suit trousers, blazer"),
    dict(id="wedding_men", situation="男性參加婚禮要穿的衣服", constraints=_MEN,
         raw="outfit for a wedding",
         rewritten="slim fit suit jacket and suit trousers, formal shirt, tie"),
    dict(id="gym_men", situation="男性去健身房運動要穿的衣服", constraints=None,
         raw="men's gym workout clothes",
         rewritten="men's sports t-shirt and training shorts in fast-drying functional fabric"),
    dict(id="school_kids", situation="小孩平常上學穿的衣服", constraints=_KIDS,
         raw="clothes for school",
         rewritten="comfortable cotton t-shirt, sweatshirt and joggers for kids, durable"),
    dict(id="newborn_gift", situation="送給剛出生嬰兒的禮物（衣物）", constraints=_KIDS,
         raw="gift for a newborn baby",
         rewritten="soft organic cotton baby bodysuit, romper, newborn set"),
    dict(id="christmas", situation="參加聖誕派對要穿的衣服", constraints=None,
         raw="christmas party outfit",
         rewritten="velvet or sequin dress, festive glitter details, knitted holiday sweater"),
]
VERSIONS = ("raw", "rewritten")


def _run_path(model: str):
    return EVAL_DIR / "runs" / (model.replace("/", "__") + ".csv")


def _labels() -> pd.DataFrame:
    path = EVAL_DIR / "labels.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"article_id": str})
    return pd.DataFrame(columns=["scenario_id", "article_id", "relevant", "reason"])


def run(model: str = semantic.DEFAULT_MODEL) -> None:
    """跑全部查詢，存下結果，並產生「還沒標過的 (情境, 商品)」待標記清單。

    待標記清單刻意不含模型與版本，順序也打亂，標記者看不出商品是哪個查詢找到的。
    """
    rows = []
    for s in SCENARIOS:
        for ver in VERSIONS:
            for rank, r in enumerate(semantic.semantic_search(
                    s[ver], s["constraints"], k=K, model=model), 1):
                rows.append(dict(scenario_id=s["id"], version=ver, rank=rank,
                                 article_id=r["article_id"], similarity=r["similarity"]))
    runs = pd.DataFrame(rows)
    _run_path(model).parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(_run_path(model), index=False)

    done = _labels()
    todo = (runs[["scenario_id", "article_id"]].drop_duplicates()
            .merge(done[["scenario_id", "article_id"]], how="left", indicator=True)
            .query("_merge == 'left_only'").drop(columns="_merge"))

    con = dataset.connect()
    con.register("todo", todo)
    sheet = con.execute("""
        SELECT t.scenario_id, t.article_id, a.prod_name, a.product_type_name AS product_type,
               a.colour_group_name AS colour, a.graphical_appearance_name AS pattern,
               a.section_name AS section, a.index_group_name AS index_group,
               raw.detail_desc AS description
        FROM todo t JOIN art_c a USING (article_id) JOIN art raw USING (article_id)
    """).df()
    situation = {s["id"]: s["situation"] for s in SCENARIOS}
    sheet.insert(1, "situation", sheet["scenario_id"].map(situation))
    sheet = sheet.sample(frac=1, random_state=0).reset_index(drop=True)
    sheet.to_csv(EVAL_DIR / "to_label.csv", index=False)

    print(f"{model}：{len(SCENARIOS)} 個情境 × {len(VERSIONS)} 版 × 前 {K} 件 → {len(runs)} 筆結果")
    print(f"  不重複的 (情境, 商品) 共 {len(runs[['scenario_id', 'article_id']].drop_duplicates())} 組，"
          f"已標過 {len(runs[['scenario_id', 'article_id']].drop_duplicates()) - len(todo)} 組，"
          f"待標記 {len(todo)} 組 → {EVAL_DIR / 'to_label.csv'}")


def score(model: str = semantic.DEFAULT_MODEL) -> pd.DataFrame:
    """每個情境、每個版本的相關率（前 12 件裡相關的比例），以及兩個版本的平均。"""
    runs = pd.read_csv(_run_path(model), dtype={"article_id": str})
    lab = _labels()
    m = runs.merge(lab[["scenario_id", "article_id", "relevant"]], how="left")
    missing = m["relevant"].isna().sum()
    if missing:
        print(f"注意：還有 {missing} 筆結果沒有標記，相關率只算已標記的部分")
    table = (m.dropna(subset=["relevant"])
             .groupby(["scenario_id", "version"])["relevant"].mean()
             .unstack("version").reindex([s["id"] for s in SCENARIOS]))
    print(table.round(2).to_string())
    for ver in VERSIONS:
        mean = table[ver].mean()
        print(f"{ver:>10} 平均相關率 {mean:.1%}  {'✅ 過' if mean >= PASS else '❌ 未過'} {PASS:.0%}")
    return table


def review(n: int = 40, seed: int = 0) -> None:
    """從已標記的結果抽 n 筆給使用者抽查。表上不顯示 agent 的判斷，避免先入為主。"""
    lab = _labels()
    if lab.empty:
        raise SystemExit("labels.csv 還沒有任何標記")
    pick = lab.sample(n=min(n, len(lab)), random_state=seed)[["scenario_id", "article_id"]]
    con = dataset.connect()
    con.register("pick", pick)
    sheet = con.execute("""
        SELECT p.scenario_id, p.article_id, a.prod_name, a.product_type_name AS product_type,
               a.colour_group_name AS colour, a.index_group_name AS index_group,
               raw.detail_desc AS description
        FROM pick p JOIN art_c a USING (article_id) JOIN art raw USING (article_id)
    """).df()
    situation = {s["id"]: s["situation"] for s in SCENARIOS}
    sheet.insert(1, "situation", sheet["scenario_id"].map(situation))
    sheet["your_label"] = ""
    sheet.to_csv(EVAL_DIR / "review.csv", index=False)
    print(f"抽了 {len(sheet)} 筆 → {EVAL_DIR / 'review.csv'}，請在 your_label 欄填 1（相關）或 0（不相關）")


def agree() -> None:
    """使用者抽查的判斷跟 agent 一致的比例，並列出不一致的。"""
    rev = pd.read_csv(EVAL_DIR / "review.csv", dtype={"article_id": str})
    rev = rev[rev["your_label"].notna() & (rev["your_label"].astype(str).str.strip() != "")]
    m = rev.merge(_labels(), on=["scenario_id", "article_id"])
    m["your_label"] = m["your_label"].astype(int)
    same = (m["your_label"] == m["relevant"]).mean()
    print(f"抽查 {len(m)} 筆，一致率 {same:.0%}")
    diff = m[m["your_label"] != m["relevant"]]
    if len(diff):
        print("\n不一致的：")
        print(diff[["situation", "prod_name", "product_type", "your_label", "relevant", "reason"]]
              .to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="語意檢索驗收")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "score"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", default=semantic.DEFAULT_MODEL, choices=list(semantic.MODELS))
    sp = sub.add_parser("review")
    sp.add_argument("-n", type=int, default=40)
    sub.add_parser("agree")

    a = p.parse_args()
    if a.cmd == "run":
        run(a.model)
    elif a.cmd == "score":
        score(a.model)
    elif a.cmd == "review":
        review(a.n)
    else:
        agree()


if __name__ == "__main__":
    main()
