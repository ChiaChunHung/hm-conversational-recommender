"""
排序模型的滾動驗證比較：LightGBM / XGBoost / CatBoost，外加名次融合。

    python ranking_compare.py

每一輪用一週的候選池訓練、下一週評估 MAP@12（ROUNDS 四輪，測試週與 Stage 1 的四個驗證週相同）。候選池要先用
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

ROUNDS = [("2020-08-12", "2020-08-19"), ("2020-08-19", "2020-08-26"), ("2020-08-26", "2020-09-02"),
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
                        date_diff('day', max(t_dat), DATE '{t}') AS days_since_bought_item,
                        count(*) AS times_bought_item
                 FROM hist GROUP BY 1, 2),
        -- 顧客最近一次購買那天買的東西
        lastday AS (SELECT h.customer_id, a.department_no, a.product_type_no
                    FROM hist h JOIN art a USING (article_id)
                    JOIN cu ON cu.customer_id = h.customer_id
                         AND h.t_dat = DATE '{t}' - cu.days_since_last::INTEGER),
        ld_dp AS (SELECT DISTINCT customer_id, department_no FROM lastday),
        ld_pt AS (SELECT DISTINCT customer_id, product_type_no FROM lastday)
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
               same.days_since_bought_item,
               coalesce(same.times_bought_item, 0) AS times_bought_item,
               (ld_dp.customer_id IS NOT NULL)::INTEGER AS last_buy_same_department,
               (ld_pt.customer_id IS NOT NULL)::INTEGER AS last_buy_same_type
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
        LEFT JOIN ld_dp ON ld_dp.customer_id = c.customer_id AND ld_dp.department_no = a.department_no
        LEFT JOIN ld_pt ON ld_pt.customer_id = c.customer_id AND ld_pt.product_type_no = a.product_type_no
        ORDER BY customer_id
    """).df()


NEW_FEATURES = ["times_bought_item", "last_buy_same_department", "last_buy_same_type"]
CACHE_DIR = dataset.DATA_DIR / "ranking_cache"
MIN_EFFECT = 0.001          # MAP@12 平均提升至少 0.001，與 Stage 1 同一套規則
MIN_POSITIVE_ROUNDS = 3


def cached_frame(con, t: str) -> pd.DataFrame:
    """frame() 的結果存成 parquet，重複實驗不必重算特徵。改了 frame() 要刪掉 CACHE_DIR。"""
    CACHE_DIR.mkdir(exist_ok=True)
    f = CACHE_DIR / f"frame_{t}.parquet"
    if not f.exists():
        df = frame(con, t)
        names = sorted(con.execute("SELECT DISTINCT product_group_name FROM art").df()
                       ["product_group_name"].dropna())
        df["product_group_name"] = df["product_group_name"].map(
            {n: i for i, n in enumerate(names)}).fillna(-1)
        con.register("_frame", df)
        con.execute(f"COPY _frame TO '{f}' (FORMAT PARQUET)")
        con.unregister("_frame")
    return con.execute(f"SELECT * FROM '{f}'").df()


def fitters(params: dict | None = None):
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRanker, Pool
    xp = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, **(params or {})}

    def lgb_fit(X, y, grp, qid, Xv):
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05,
                           num_leaves=63, random_state=SEED, verbose=-1)
        m.fit(X, y, group=grp)
        return m.predict(Xv)

    def xgb_fit(X, y, grp, qid, Xv):
        m = xgb.XGBRanker(objective="rank:ndcg", tree_method="hist", random_state=SEED, **xp)
        m.fit(X, y, qid=qid)
        LAST_MODEL[:] = [m]
        return m.predict(Xv)

    def cb_fit(X, y, grp, qid, Xv):
        m = CatBoostRanker(loss_function="YetiRank", iterations=300, learning_rate=0.1,
                           depth=6, random_seed=SEED, verbose=0, train_dir="/tmp/catboost_info")
        m.fit(Pool(X, y, group_id=qid))
        return m.predict(Xv)

    return {"LightGBM": lgb_fit, "XGBoost": xgb_fit, "CatBoost": cb_fit}


def per_customer_ap(con, va: pd.DataFrame, test_t: str, score) -> np.ndarray:
    """每位買家的 AP@12（依 customer_id 排序），沒有候選的買家記 0。"""
    end = dataset.pop_start_for(test_t, -7)
    truth = con.execute(f"""
        SELECT customer_id, list(DISTINCT article_id) FROM tx
        WHERE t_dat >= '{test_t}' AND t_dat < '{end}' GROUP BY 1 ORDER BY 1""").fetchall()
    d = pd.DataFrame({"c": va.customer_id.to_numpy(), "a": va.article_id.to_numpy(), "s": score})
    d = d.sort_values(["c", "s"], ascending=[True, False]).groupby("c").head(12)
    preds = d.groupby("c").a.apply(list)
    return np.array([evaluate.ap_at_k(set(a), preds.get(c, [])) for c, a in truth])


def run(con, models=("XGBoost",), drop=(), params=None, rounds=None, verbose=True) -> dict:
    """每一輪訓練、評估，回傳 {模型: {測試週: 每位買家 AP 陣列}}。"""
    out = {}
    for train_t, test_t in rounds or ROUNDS:
        tr = cached_frame(con, train_t)
        tr = tr[(tr.y == 1) | (np.random.default_rng(SEED).random(len(tr)) < NEG_FRAC)]
        tr = tr[tr.groupby("customer_id").y.transform("max") == 1]
        va = cached_frame(con, test_t)
        feats = [c for c in tr.columns if c not in ("customer_id", "article_id", "y", *drop)]
        X, Xv = tr[feats].astype("float32"), va[feats].astype("float32")
        grp = tr.groupby("customer_id", sort=False).size().to_numpy()
        qid = tr.customer_id.astype("category").cat.codes.to_numpy()
        scores = {}
        for name, fit in fitters(params).items():
            if name in models:
                scores[name] = fit(X, tr.y, grp, qid, Xv)
        if "RankAverage" in models:
            scores["RankAverage"] = -sum(
                pd.Series(scores[m]).groupby(va.customer_id.values).rank(ascending=False).to_numpy()
                for m in ("LightGBM", "XGBoost", "CatBoost"))
        if "SegmentOrder" in models:
            scores["SegmentOrder"] = -va.rank_segment.fillna(999).to_numpy()
        for name, sc in scores.items():
            out.setdefault(name, {})[test_t] = per_customer_ap(con, va, test_t, sc)
        if verbose:
            print(f"  {test_t}  " + "  ".join(f"{n} {out[n][test_t].mean():.5f}" for n in scores),
                  flush=True)
    return out


def compare(base: dict, new: dict) -> dict:
    """paired per-customer 比較（新 − 基準），規則同 Stage 1：平均 Δ ≥ MIN_EFFECT、
    至少 MIN_POSITIVE_ROUNDS 輪為正、合併 95% CI 下界 > 0。"""
    per = {w: new[w].mean() - base[w].mean() for w in base}
    d = np.concatenate([new[w] - base[w] for w in base])
    se = d.std(ddof=1) / np.sqrt(len(d))
    mean = float(np.mean(list(per.values())))
    pos = sum(v > 0 for v in per.values())
    return {"per_round": per, "mean": mean, "ci": (d.mean() - 1.96 * se, d.mean() + 1.96 * se),
            "positive": pos,
            "adopt": mean >= MIN_EFFECT and pos >= MIN_POSITIVE_ROUNDS and d.mean() - 1.96 * se > 0}


def table(res: dict) -> None:
    names = list(res)
    weeks = list(next(iter(res.values())))
    print("test week   " + "".join(f"{n:>13s}" for n in names))
    for w in weeks:
        print(f"{w}  " + "".join(f"{res[n][w].mean():13.5f}" for n in names))
    print("mean        " + "".join(f"{np.mean([res[n][w].mean() for w in weeks]):13.5f}"
                                   for n in names))


def main() -> None:
    """
    python ranking_compare.py               三個模型 + 名次融合，四輪
    python ranking_compare.py features      XGBoost：新特徵加不加的比較
    python ranking_compare.py tune          XGBoost：幾組參數的比較
    """
    import sys
    con = dataset.connect()
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if mode == "features":
        base = run(con, drop=NEW_FEATURES)["XGBoost"]
        for f in NEW_FEATURES:
            t = time.time()
            new = run(con, drop=[x for x in NEW_FEATURES if x != f], verbose=False)["XGBoost"]
            c = compare(base, new)
            print(f"+{f:26s} Δ={c['mean']:+.5f} CI=[{c['ci'][0]:+.5f},{c['ci'][1]:+.5f}] "
                  f"+rounds={c['positive']}/4 adopt={c['adopt']}  ({time.time() - t:.0f}s)", flush=True)
        allnew = run(con, verbose=False)["XGBoost"]
        c = compare(base, allnew)
        print(f"+all three                  Δ={c['mean']:+.5f} +rounds={c['positive']}/4 adopt={c['adopt']}")
    elif mode == "tune":
        grid = [{}, {"n_estimators": 600, "learning_rate": 0.03},
                {"max_depth": 8}, {"max_depth": 4, "n_estimators": 600},
                {"subsample": 0.8, "colsample_bytree": 0.8},
                {"min_child_weight": 10}]
        drop = [x for x in sys.argv[2:]]
        base = run(con, drop=drop, params=grid[0], verbose=False)["XGBoost"]
        print(f"default                                   mean {np.mean([v.mean() for v in base.values()]):.5f}")
        for g in grid[1:]:
            t = time.time()
            new = run(con, drop=drop, params=g, verbose=False)["XGBoost"]
            c = compare(base, new)
            print(f"{str(g):42s} mean {np.mean([v.mean() for v in new.values()]):.5f} "
                  f"Δ={c['mean']:+.5f} +rounds={c['positive']}/4 adopt={c['adopt']}  "
                  f"({time.time() - t:.0f}s)", flush=True)
    else:
        res = run(con, models=("LightGBM", "XGBoost", "CatBoost", "RankAverage", "SegmentOrder"),
                  drop=[*NEW_FEATURES, *sys.argv[2:]])   # 三個新特徵未通過採用規則
        print("\nMAP@12")
        table(res)


if __name__ == "__main__":
    main()
