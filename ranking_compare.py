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


def frame(con, t: str) -> pd.DataFrame:
    end, d7 = dataset.pop_start_for(t, -7), dataset.pop_start_for(t, 7)
    return con.execute(f"""
        WITH truth AS (SELECT DISTINCT customer_id, article_id, 1 AS y FROM tx
                       WHERE t_dat >= '{t}' AND t_dat < '{end}'),
        item7 AS (SELECT article_id, count(*) AS item_sales_7d FROM tx
                  WHERE t_dat >= '{d7}' AND t_dat < '{t}' GROUP BY 1),
        last AS (SELECT customer_id, date_diff('day', max(t_dat), DATE '{t}') AS days_since_last
                 FROM tx WHERE t_dat < '{t}' GROUP BY 1)
        SELECT c.*, coalesce(truth.y, 0) AS y,
               coalesce(item7.item_sales_7d, 0) AS item_sales_7d,
               last.days_since_last, cu.age
        FROM '{dataset.CAND_DIR}/cand_{t}.parquet' c
        LEFT JOIN truth USING (customer_id, article_id)
        LEFT JOIN item7 USING (article_id)
        LEFT JOIN last USING (customer_id)
        LEFT JOIN cus cu USING (customer_id)
        ORDER BY customer_id
    """).df()


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
        return m.predict(Xv)

    def cb_fit(X, y, grp, qid, Xv):
        m = CatBoostRanker(loss_function="YetiRank", iterations=300, learning_rate=0.1,
                           depth=6, random_seed=SEED, verbose=0)
        m.fit(Pool(X, y, group_id=qid))
        return m.predict(Xv)

    return {"LightGBM": lgb_fit, "XGBoost": xgb_fit, "CatBoost": cb_fit}


def run_round(con, train_t: str, test_t: str) -> dict:
    tr = frame(con, train_t)
    tr = tr[(tr.y == 1) | (np.random.default_rng(SEED).random(len(tr)) < NEG_FRAC)]
    tr = tr[tr.groupby("customer_id").y.transform("max") == 1]
    va = frame(con, test_t)
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
        t = time.time()
        scores[name] = fit(X, tr.y, grp, qid, Xv)
        res[name] = map12(scores[name])
        print(f"  {name:12s} {res[name]:.5f}  ({time.time() - t:.0f}s)", flush=True)
    fused = -sum(pd.Series(s).groupby(cust).rank(ascending=False).to_numpy()
                 for s in scores.values())
    res["RankAverage"] = map12(fused)
    res["SegmentOrder"] = map12(-va.rank_segment.fillna(999).to_numpy())
    print(f"  {'RankAverage':12s} {res['RankAverage']:.5f}\n"
          f"  {'SegmentOrder':12s} {res['SegmentOrder']:.5f}", flush=True)
    return res


def main() -> None:
    con = dataset.connect()
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
