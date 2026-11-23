"""
排序模型的滾動驗證比較：LightGBM / XGBoost / CatBoost，外加名次融合。

    python ranking_compare.py

每一輪用一週的候選池訓練、下一週評估 MAP@12（ROUNDS 三輪）。候選池要先用
recall.build(..., save=True) 產生 data/candidates/cand_<週>.parquet。

特徵：候選池的 13 欄來源訊號（from_ / score_ / rank_ 各源 + n_sources），加上
商品近 7 天銷量、顧客距上次購買天數、顧客年齡。全部只用該週開始以前的資料。

訓練資料保留全部正例與 NEG_FRAC 的負例，並去掉沒有任何正例的顧客（排序模型
在這種群組裡學不到東西）。評估用該週全部候選，不抽樣。
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import dataset
import evaluate

ROUNDS = [("2020-08-19", "2020-08-26"), ("2020-08-26", "2020-09-02"),
          ("2020-09-02", "2020-09-09")]
NEG_FRAC = 0.1
SEED = 0
LAST_MODEL: list = []


def frame(con, t: str) -> pd.DataFrame:
    """候選池 + 標籤 + 特徵。所有特徵只用 t 以前的交易。"""
    end = dataset.pop_start_for(t, -7)
    d = lambda days: dataset.pop_start_for(t, days)
    return con.execute(f"""
        WITH cand AS (SELECT * FROM '{dataset.CAND_DIR}/cand_{t}.parquet'),
        truth AS (SELECT DISTINCT customer_id, article_id, 1 AS y FROM tx
                  WHERE t_dat >= '{t}' AND t_dat < '{end}'),
        hist AS (SELECT t.* FROM tx t
                 WHERE t.t_dat < '{t}'
                   AND t.customer_id IN (SELECT DISTINCT customer_id FROM cand)),
        -- 商品特徵
        item AS (
            SELECT article_id,
                   count(*) FILTER (WHERE t_dat >= '{d(7)}')  AS item_sales_7d,
                   count(*) FILTER (WHERE t_dat >= '{d(14)}' AND t_dat < '{d(7)}') AS item_sales_prev7d,
                   count(*) FILTER (WHERE t_dat >= '{d(28)}') AS item_sales_28d,
                   count(DISTINCT customer_id) FILTER (WHERE t_dat >= '{d(7)}') AS item_buyers_7d,
                   avg(price) FILTER (WHERE t_dat >= '{d(28)}') AS item_price_28d,
                   date_diff('day', min(t_dat), DATE '{t}') AS item_age_days
            FROM tx WHERE t_dat < '{t}' GROUP BY 1),
        -- 顧客整體
        cu AS (SELECT customer_id, count(*) AS cust_n, avg(price) AS cust_avg_price,
                      date_diff('day', max(t_dat), DATE '{t}') AS days_since_last
               FROM hist GROUP BY 1),
        -- 顧客 × 商品屬性的購買次數
        h AS (SELECT h.customer_id, a.product_type_no, a.colour_group_code, a.department_no
              FROM hist h JOIN art a USING (article_id)),
        pt AS (SELECT customer_id, product_type_no, count(*) AS n FROM h GROUP BY 1, 2),
        co AS (SELECT customer_id, colour_group_code, count(*) AS n FROM h GROUP BY 1, 2),
        dp AS (SELECT customer_id, department_no, count(*) AS n FROM h GROUP BY 1, 2),
        same AS (SELECT customer_id, article_id,
                        date_diff('day', max(t_dat), DATE '{t}') AS days_since_bought_item
                 FROM hist GROUP BY 1, 2)
        SELECT c.*, coalesce(truth.y, 0) AS y,
               cu.age,
               coalesce(item.item_sales_7d, 0) AS item_sales_7d,
               coalesce(item.item_sales_28d, 0) AS item_sales_28d,
               (coalesce(item.item_sales_7d, 0) + 1.0) / (coalesce(item.item_sales_prev7d, 0) + 1.0)
                   AS item_trend,
               coalesce(item.item_buyers_7d, 0) AS item_buyers_7d,
               item.item_price_28d, item.item_age_days,
               a.product_group_name, a.index_group_no,
               k.days_since_last,
               coalesce(pt.n, 0) / k.cust_n AS pref_product_type,
               coalesce(co.n, 0) / k.cust_n AS pref_colour,
               coalesce(dp.n, 0) / k.cust_n AS pref_department,
               item.item_price_28d / k.cust_avg_price AS price_ratio,
               same.days_since_bought_item
        FROM cand c
        JOIN art a USING (article_id)
        LEFT JOIN truth USING (customer_id, article_id)
        LEFT JOIN item USING (article_id)
        LEFT JOIN cus cu USING (customer_id)
        LEFT JOIN cu k USING (customer_id)
        LEFT JOIN pt ON pt.customer_id = c.customer_id AND pt.product_type_no = a.product_type_no
        LEFT JOIN co ON co.customer_id = c.customer_id AND co.colour_group_code = a.colour_group_code
        LEFT JOIN dp ON dp.customer_id = c.customer_id AND dp.department_no = a.department_no
        LEFT JOIN same USING (customer_id, article_id)
        ORDER BY customer_id
    """).df()


PRODUCT_GROUPS: dict = {}


def models():
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRanker, Pool

    def lgb_fit(X, y, grp, qid, Xv):
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05,
                           num_leaves=63, random_state=SEED, verbose=-1)
        m.fit(X, y, group=grp)
        return m.predict(Xv)

    def xgb_fit(X, y, grp, qid, Xv):
        m = xgb.XGBRanker(objective="rank:ndcg", n_estimators=300, learning_rate=0.05,
                          max_depth=6, tree_method="hist", random_state=SEED)
        m.fit(X, y, qid=qid)
        LAST_MODEL[:] = [m]
        return m.predict(Xv)

    def cb_fit(X, y, grp, qid, Xv):
        m = CatBoostRanker(loss_function="YetiRank", iterations=300, learning_rate=0.1,
                           depth=6, random_seed=SEED, verbose=0)
        m.fit(Pool(X, y, group_id=qid))
        return m.predict(Xv)

    return {"LightGBM": lgb_fit, "XGBoost": xgb_fit, "CatBoost": cb_fit}


def run_round(con, train_t: str, test_t: str, only=None) -> dict:
    tr = frame(con, train_t)
    tr = tr[(tr.y == 1) | (np.random.default_rng(SEED).random(len(tr)) < NEG_FRAC)]
    tr = tr[tr.groupby("customer_id").y.transform("max") == 1]
    va = frame(con, test_t)
    if not PRODUCT_GROUPS:
        names = con.execute("SELECT DISTINCT product_group_name FROM art ORDER BY 1").fetchall()
        PRODUCT_GROUPS.update({n: i for i, (n,) in enumerate(names)})
    for df in (tr, va):
        df["product_group_name"] = df["product_group_name"].map(PRODUCT_GROUPS).fillna(-1)
    feats = [c for c in tr.columns if c not in ("customer_id", "article_id", "y")]
    X, Xv = tr[feats].astype("float32"), va[feats].astype("float32")
    grp = tr.groupby("customer_id", sort=False).size().to_numpy()
    qid = tr.customer_id.astype("category").cat.codes.to_numpy()

    end = dataset.pop_start_for(test_t, -7)
    truth = {c: set(a) for c, a in con.execute(f"""
        SELECT customer_id, list(DISTINCT article_id) FROM tx
        WHERE t_dat >= '{test_t}' AND t_dat < '{end}' GROUP BY 1""").fetchall()}
    cust = va.customer_id.to_numpy()

    def map12(score):
        d = pd.DataFrame({"c": cust, "a": va.article_id.to_numpy(), "s": score})
        d = d.sort_values(["c", "s"], ascending=[True, False]).groupby("c").head(12)
        preds = d.groupby("c").a.apply(list)
        return float(np.mean([evaluate.ap_at_k(truth[c], preds.get(c, [])) for c in truth]))

    scores, res = {}, {}
    for name, fit in models().items():
        if only and name not in only:
            continue
        t = time.time()
        scores[name] = fit(X, tr.y, grp, qid, Xv)
        res[name] = map12(scores[name])
        print(f"  {name:12s} {res[name]:.5f}  ({time.time() - t:.0f}s)", flush=True)
    if len(scores) > 1:
        fused = -sum(pd.Series(s).groupby(cust).rank(ascending=False).to_numpy()
                     for s in scores.values())
        res["RankAverage"] = map12(fused)
    res["SegmentOrder"] = map12(-va.rank_segment.fillna(999).to_numpy())
    print(f"  {'SegmentOrder':12s} {res['SegmentOrder']:.5f}", flush=True)
    if only:
        imp = pd.Series(LAST_MODEL[0].feature_importances_, index=feats).sort_values(ascending=False)
        print("  feature importance (gain share):\n" + (imp / imp.sum()).round(3).to_string())
    return res


def main() -> None:
    import sys
    con = dataset.connect()
    if sys.argv[1:] == ["quick"]:          # 一輪、只跑 XGBoost
        print("=== quick: XGBoost, train 2020-09-02 -> test 2020-09-09", flush=True)
        run_round(con, "2020-09-02", "2020-09-09", only={"XGBoost"})
        return
    allres = {}
    for train_t, test_t in ROUNDS:
        print(f"=== train {train_t} -> test {test_t}", flush=True)
        allres[test_t] = run_round(con, train_t, test_t)
    names = list(next(iter(allres.values())))
    print("\nMAP@12")
    print("test week   " + "".join(f"{n:>13s}" for n in names))
    for w, r in allres.items():
        print(f"{w}  " + "".join(f"{r[n]:13.5f}" for n in names))
    print("mean        " + "".join(f"{np.mean([r[n] for r in allres.values()]):13.5f}"
                                   for n in names))


if __name__ == "__main__":
    main()
