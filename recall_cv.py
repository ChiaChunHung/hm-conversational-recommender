"""
召回參數的 cross-validation。

    python recall_cv.py step1     quality 參數：coordinate descent（最多 2 輪）
    python recall_cv.py step2     budget 參數：greedy marginal allocation + Pareto frontier
    python recall_cv.py all       兩步依序執行，最後寫 report.md
    python recall_cv.py report    只從已有結果重寫 report.md

輸出全部在 data/recall_cv/（repo 外面）：
    <fold>/                各 fold 的快取（評估對象、ground truth、各源在最大 k 的候選）
    step1_log.json         每一次比較的完整數字
    selected_config.json   step1 選出的設定（step2 從這裡起跑）
    frontier.csv           step2 的 recall–budget 曲線
    report.md              兩步的摘要表（python recall_cv.py report 重新產生）


【評估設定】

Validation folds：FOLDS 四週，rolling-origin。每個 fold 只用該週起點 t 以前的資料產生
候選，以 [t, t+7) 的實際購買為 ground truth。Test set（2020-09-16 起）不在這裡出現。

主指標是 micro candidate recall：被候選池涵蓋的 (顧客, 商品) 正解對 ÷ 全部正解對。
每個結果都另外依 recency 分層：<=4w / 4-12w / lapsed（> 12 週沒買）/ cold_start。


【決策規則】

新設定 vs 現行設定，以顧客為單位做 paired comparison。每位顧客的命中數差 d_i，
fold f 的 Δ_f = Σd_i / N_f（N_f = 該 fold 正解對數），標準誤用常態近似
SE_f = √(n_f · Var(d_i)) / N_f；四個 fold 合併時假設彼此獨立。

採用新設定需同時滿足：
  1. 四個 fold 的平均 Δ ≥ MIN_EFFECT（0.1 個百分點，minimum practically significant effect）
  2. 至少 3/4 的 fold 方向為正
  3. 合併的 95% CI 下界 > 0
否則保留現行設定（parsimony）。同一參數有多個候選值時，取通過規則且平均 Δ 最大者。

（四個 fold 時間相鄰、歷史重疊，不是嚴格獨立；同一參數比較多個值也沒有做多重比較
校正。所以這套規則是篩選標準，不是嚴格的顯著性檢定。）


【為什麼 quality 和 budget 分開處理】

quality 參數（回看窗口、ALS 超參數、籃子定義）決定每一源「撈得準不準」；
budget 參數（各源的 cutoff k）決定每一源「分到多少名額」。各源的候選高度重疊，
名額只能一起分配；但「準不準」主要影響自己那一源，可以逐一比較。

**等預算比較**：quality 參數一改，候選數也可能跟著變（例如回購窗口拉長）。候選多
撈中率本來就高，直接比不公平。所以比較時調整 segment popularity 的 cutoff，讓新設定的
每人平均候選數貼齊現行設定。選 segment 當調節閥，是因為它佔池子約九成、各名次的
邊際效率接近平坦，是最不扭曲比較的那一源。


【Step 2 的理論依據】

固定 quality 參數之後，「在總預算 B 下怎麼分配各源的 k，使 recall 最大」是
budgeted maximum coverage 問題。recall（覆蓋的正解數）是 monotone submodular 函數，
每次把一個增量分給「單位候選換到最多新增覆蓋」的那一源，這個 greedy 解有
(1 − 1/e) 的近似保證。greedy 的路徑本身就是 recall–budget 曲線上的一串點。


【快取】

每一源只在「最大 k」產生一次（KMAX），存 (c, a, rk)：c 是該 fold 評估對象的整數編號，
a 是 article_id 轉成的整數，rk 是 cutoff 用的名次。任何 k 都只是 rk <= k 的篩選，
不必重跑召回。快取檔名含該源相關參數的 hash，參數改了就是另一個檔。
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time

import numpy as np

import dataset
import evaluate
import recall

FOLDS = ["2020-08-19", "2020-08-26", "2020-09-02", "2020-09-09"]
CV_DIR = dataset.DATA_DIR / "recall_cv"

MIN_EFFECT = 0.001      # 0.1 個百分點
MIN_POSITIVE_FOLDS = 3

# 各源產生候選時的最大 cutoff。cooc 的 k 是「每件商品留幾個共購夥伴」。
KMAX = {"segment": 300, "cooc": 20, "als": 100}

# 每一源受哪些 quality 參數影響（決定快取 key）
QUALITY_KEYS = {
    "repurchase": ["repurchase_weeks", "expand_colours"],
    "segment": ["segment_days"],
    "cooc": ["cooc_weeks", "cooc_basket_days"],
    "als": ["als_weeks", "als_factors", "als_regularization", "als_bm25_k1", "als_bm25_b"],
}

# Step 1 的搜尋空間。第一個值不一定是現行值；現行值一定要在清單裡。
SEARCH = [
    ("segment_days", [3, 5, 7]),
    ("repurchase_weeks", [4, 8, 12, 26]),
    ("expand_colours", [False, True]),
    ("cooc_weeks", [2, 4, 8, 12]),
    ("cooc_basket_days", [1, 7]),
    ("als_weeks", [4, 6, 8, 12]),
    ("als_factors", [32, 64, 128]),
    ("als_regularization", [0.01, 0.05, 0.1]),
    ("als_bm25_k1", [50, 100, 200]),
    ("als_bm25_b", [0.5, 0.8, 1.0]),
]

# Step 2 每一步的增量
DELTA = {"segment": 10, "cooc": 1, "als": 5}
FRONTIER_TARGETS = [50, 100, 150, 200, 250, 300, 400]

# 現行設定，取自 recall.py 的常數；三個沒有常數的值等於 recall.py 函式參數的預設值。
START_CONFIG = {
    "repurchase_weeks": recall.REPURCHASE_WEEKS,
    "expand_colours": recall.EXPAND_COLOURS,
    "segment_top_n": recall.SEGMENT_TOP_N,
    "segment_days": recall.SEGMENT_DAYS,
    "cooc_top_n": recall.COOC_TOP_N,
    "cooc_weeks": recall.COOC_WEEKS,
    "cooc_basket_days": 1,
    "als_top_n": recall.ALS_TOP_N,
    "als_weeks": recall.ALS_WEEKS,
    "als_factors": recall.ALS_FACTORS,
    "als_regularization": 0.01,
    "als_bm25_k1": 100,
    "als_bm25_b": 0.8,
}


# ─────────────────────────── 設定 → 各源的 cutoff ───────────────────────────

def cutoffs(cfg: dict) -> dict:
    """設定 → {源: k}。repurchase 沒有 cutoff（由窗口控制），用 None 表示全收。"""
    k = {"repurchase": None,
         "segment": cfg["segment_top_n"],
         "cooc": cfg["cooc_top_n"],
         "als": cfg["als_top_n"]}
    return k


def _tag(src: str, cfg: dict) -> str:
    key = {"src": src, "kmax": KMAX.get(src), **{k: cfg[k] for k in QUALITY_KEYS[src]}}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def _end(t: str) -> str:
    return dataset.pop_start_for(t, -7)


# ─────────────────────────── 每個 fold 的基本資料 ───────────────────────────

class Fold:
    """一個 fold 的評估對象、ground truth 與分層。全部是以 c（整數編號）排序的陣列。"""

    def __init__(self, t: str, gen_con):
        self.t = t
        self.dir = CV_DIR / t
        self.dir.mkdir(parents=True, exist_ok=True)
        self.base_f = self.dir / "base.parquet"
        self.truth_f = self.dir / "truth.parquet"
        if not self.base_f.exists():
            self._build(gen_con)
        con = dataset.raw_connect()
        b = con.execute(f"SELECT c, n_truth, stratum FROM '{self.base_f}' ORDER BY c").df()
        self.n_truth = b["n_truth"].to_numpy()
        self.stratum = b["stratum"].to_numpy()
        self.NB = len(b)
        self.NT = int(self.n_truth.sum())

    def _build(self, con) -> None:
        t, end = self.t, _end(self.t)
        recall.eval_customers(con, t, end)
        evaluate.truth_view(con, t, end)
        d = lambda days: dataset.pop_start_for(t, days)
        con.execute(f"""
            CREATE OR REPLACE TABLE cv_base AS
            WITH idx AS (
                SELECT customer_id, (row_number() OVER (ORDER BY customer_id) - 1)::INTEGER AS c
                FROM eval_cus
            ),
            h AS (SELECT customer_id, max(t_dat) AS last_d FROM tx
                  WHERE t_dat < '{t}' GROUP BY 1)
            SELECT idx.c, idx.customer_id,
                   (SELECT count(*) FROM truth WHERE truth.customer_id = idx.customer_id) AS n_truth,
                   CASE WHEN h.last_d IS NULL THEN 'cold_start'
                        WHEN h.last_d < DATE '{d(84)}' THEN 'lapsed'
                        WHEN h.last_d < DATE '{d(28)}' THEN '4-12w'
                        ELSE '<=4w' END AS stratum
            FROM idx LEFT JOIN h USING (customer_id)
        """)
        con.execute(f"COPY cv_base TO '{self.base_f}' (FORMAT PARQUET)")
        con.execute(f"""
            COPY (SELECT b.c, t.article_id::BIGINT AS a FROM truth t JOIN cv_base b USING (customer_id))
            TO '{self.truth_f}' (FORMAT PARQUET)
        """)

    def use(self, con) -> None:
        """讓 con 的 eval_cus 與 cv_base 指向這個 fold（產生候選前呼叫）。"""
        recall.eval_customers(con, self.t, _end(self.t))
        con.execute(f"CREATE OR REPLACE TABLE cv_base AS SELECT * FROM '{self.base_f}'")


# ─────────────────────────── 候選快取 ───────────────────────────

def source_file(gen_con, fold: Fold, src: str, cfg: dict):
    f = fold.dir / f"{src}_{_tag(src, cfg)}.parquet"
    if f.exists():
        return f
    t = fold.t
    fold.use(gen_con)

    def dump(table: str, rk_col: str, out):
        gen_con.execute(f"""
            COPY (SELECT b.c, x.article_id::BIGINT AS a, x.{rk_col}::INTEGER AS rk
                  FROM {table} x JOIN cv_base b USING (customer_id))
            TO '{out}' (FORMAT PARQUET)
        """)

    if src == "repurchase":
        recall.repurchase(gen_con, t, weeks=cfg["repurchase_weeks"],
                          expand_colours=cfg["expand_colours"])
        dump("cand_repurchase", "rk", f)
    elif src == "segment":
        recall.segment_popular(gen_con, t, top_n=KMAX["segment"], days=cfg["segment_days"])
        dump("cand_segment", "rk", f)
    elif src == "cooc":
        recall.cooccurrence(gen_con, t, top_n=KMAX["cooc"], weeks=cfg["cooc_weeks"],
                            basket_days=cfg["cooc_basket_days"])
        # cand_cooc 的 rk 是顧客內的名次；這裡要的是「每件商品第幾個夥伴」，
        # 因為 cooc 的 k 是每件商品留幾個夥伴。用 recall.cooccurrence 留下的 cooc 表重算。
        start = dataset.pop_start_for(t, cfg["cooc_weeks"] * 7)
        gen_con.execute(f"""
            CREATE OR REPLACE TABLE cv_cooc AS
            WITH top AS (
                SELECT i, j, row_number() OVER (PARTITION BY i ORDER BY n DESC, j) AS prk
                FROM cooc
            ),
            mine AS (
                SELECT DISTINCT t.customer_id, t.article_id
                FROM tx t JOIN eval_cus USING (customer_id)
                WHERE t.t_dat >= '{start}' AND t.t_dat < '{t}'
            )
            SELECT m.customer_id, top.j AS article_id, min(top.prk) AS rk
            FROM mine m JOIN top ON top.i = m.article_id
            WHERE top.prk <= {KMAX["cooc"]}
            GROUP BY 1, 2
        """)
        dump("cv_cooc", "rk", f)
    elif src == "als":
        recall.als(gen_con, t, top_n=KMAX["als"], weeks=cfg["als_weeks"],
                   factors=cfg["als_factors"], regularization=cfg["als_regularization"],
                   bm25_k1=cfg["als_bm25_k1"], bm25_b=cfg["als_bm25_b"])
        dump("cand_als", "rk", f)
    return f


def files_for(gen_con, fold: Fold, cfg: dict) -> dict:
    """{源: (檔案, k)}，只含 cfg 啟用的源。"""
    return {src: (source_file(gen_con, fold, src, cfg), k) for src, k in cutoffs(cfg).items()}


def _select(path, k) -> str:
    return (f"SELECT c, a FROM '{path}'" if k is None
            else f"SELECT c, a FROM '{path}' WHERE rk <= {int(k)}")


# ─────────────────────────── 評估 ───────────────────────────

def per_customer(fold: Fold, files: dict) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (每位顧客的候選數, 每位顧客的命中數)，以 c 排序。"""
    con = dataset.raw_connect()
    con.execute(f"""
        CREATE TABLE pool AS SELECT DISTINCT c, a FROM (
            {" UNION ALL ".join(_select(p, k) for p, k in files.values())})
    """)
    r = con.execute(f"""
        SELECT b.c, coalesce(p.n, 0) AS n, coalesce(h.h, 0) AS h
        FROM '{fold.base_f}' b
        LEFT JOIN (SELECT c, count(*) AS n FROM pool GROUP BY c) p USING (c)
        LEFT JOIN (SELECT c, count(*) AS h FROM pool JOIN '{fold.truth_f}' USING (c, a)
                   GROUP BY c) h USING (c)
        ORDER BY b.c
    """).df()
    return r["n"].to_numpy(), r["h"].to_numpy()


def segment_curve(fold: Fold, files: dict) -> tuple[np.ndarray, np.ndarray]:
    """segment cutoff = 1..KMAX 時，整個池子的 (候選總數, 命中總數)。一次查詢算完全部 k。"""
    seg_path = files["segment"][0]
    others = {s: v for s, v in files.items() if s != "segment"}
    con = dataset.raw_connect()
    con.execute(f"""
        CREATE TABLE others AS SELECT DISTINCT c, a FROM (
            {" UNION ALL ".join(_select(p, k) for p, k in others.values())})
    """)
    n0, h0 = con.execute(f"""
        SELECT count(*), (SELECT count(*) FROM others JOIN '{fold.truth_f}' USING (c, a))
        FROM others
    """).fetchone()
    g = con.execute(f"""
        SELECT s.rk, count(*) AS n, count(t.a) AS h
        FROM '{seg_path}' s
        ANTI JOIN others o USING (c, a)
        LEFT JOIN '{fold.truth_f}' t USING (c, a)
        GROUP BY 1 ORDER BY 1
    """).df()
    n = np.zeros(KMAX["segment"] + 1); h = np.zeros(KMAX["segment"] + 1)
    n[g["rk"].to_numpy()] = g["n"].to_numpy(); h[g["rk"].to_numpy()] = g["h"].to_numpy()
    return n0 + np.cumsum(n), h0 + np.cumsum(h)


def evaluate_config(gen_con, folds: list[Fold], cfg: dict, budget: float | None = None):
    """評估 cfg。budget 有給時，調整 segment cutoff 讓每人平均候選數貼齊 budget。

    回傳 (實際使用的 cfg, {fold.t: (n 陣列, h 陣列)}, 每人平均候選數)。
    """
    cfg = dict(cfg)
    files = {f.t: files_for(gen_con, f, cfg) for f in folds}
    if budget is not None:
        per_k = np.mean([segment_curve(f, files[f.t])[0] / f.NB for f in folds], axis=0)
        k = int(np.argmin(np.abs(per_k[1:] - budget)) + 1)
        cfg["segment_top_n"] = k
        files = {f.t: files_for(gen_con, f, cfg) for f in folds}
    res = {f.t: per_customer(f, files[f.t]) for f in folds}
    B = float(np.mean([res[f.t][0].sum() / f.NB for f in folds]))
    return cfg, res, B


def recall_of(folds, res) -> dict:
    """micro recall：各 fold 與平均，以及依 recency 分層（四個 fold 合併）。"""
    out = {"folds": {f.t: float(res[f.t][1].sum() / f.NT) for f in folds}}
    out["mean"] = float(np.mean(list(out["folds"].values())))
    strata = {}
    for s in ("<=4w", "4-12w", "lapsed", "cold_start"):
        hit = sum(res[f.t][1][f.stratum == s].sum() for f in folds)
        tot = sum(f.n_truth[f.stratum == s].sum() for f in folds)
        strata[s] = float(hit / tot) if tot else float("nan")
    out["strata"] = strata
    return out


def compare(folds, base_res, new_res) -> dict:
    """paired comparison（新 − 現行），回傳各 fold Δ、平均 Δ、合併 95% CI 與是否採用。"""
    per, num, var_sum, tot = {}, 0.0, 0.0, 0
    for f in folds:
        d = new_res[f.t][1].astype(float) - base_res[f.t][1].astype(float)
        per[f.t] = d.sum() / f.NT
        num += d.sum(); var_sum += len(d) * d.var(ddof=1); tot += f.NT
    pooled = num / tot
    se = math.sqrt(var_sum) / tot
    mean = float(np.mean(list(per.values())))
    positive = sum(v > 0 for v in per.values())
    adopt = (mean >= MIN_EFFECT and positive >= MIN_POSITIVE_FOLDS
             and pooled - 1.96 * se > 0)
    return {"delta_folds": {t: float(v) for t, v in per.items()}, "delta_mean": mean,
            "delta_pooled": float(pooled),
            "ci95": [float(pooled - 1.96 * se), float(pooled + 1.96 * se)],
            "positive_folds": int(positive), "adopt": bool(adopt)}


# ─────────────────────────── Step 1 ───────────────────────────

def step1(gen_con, folds: list[Fold]) -> dict:
    cfg = dict(START_CONFIG)
    log = []
    for rnd in (1, 2):
        changed = False
        for param, values in SEARCH:
            _, base_res, B0 = evaluate_config(gen_con, folds, cfg)
            base_rec = recall_of(folds, base_res)
            best = None
            for v in values:
                if v == cfg[param]:
                    continue
                alt_cfg, alt_res, B1 = evaluate_config(gen_con, folds, {**cfg, param: v},
                                                       budget=B0)
                cmp = compare(folds, base_res, alt_res)
                entry = {"round": rnd, "param": param, "current": cfg[param], "value": v,
                         "budget_current": B0, "budget_value": B1,
                         "segment_top_n_value": alt_cfg["segment_top_n"],
                         "recall_current": base_rec, "recall_value": recall_of(folds, alt_res),
                         **cmp}
                log.append(entry)
                print(f"  [r{rnd}] {param:20s} {str(cfg[param]):>9s} → {str(v):<9s} "
                      f"Δ={100 * cmp['delta_mean']:+.3f}pp  CI=[{100 * cmp['ci95'][0]:+.3f},"
                      f"{100 * cmp['ci95'][1]:+.3f}]  +folds={cmp['positive_folds']}/4  "
                      f"B={B1:.1f}/{B0:.1f}  {'ADOPT' if cmp['adopt'] else ''}", flush=True)
                if cmp["adopt"] and (best is None or cmp["delta_mean"] > best[1]["delta_mean"]):
                    best = (alt_cfg, cmp)
            if best:
                cfg = best[0]
                changed = True
        _dump("step1_log.json", log)
        if not changed:
            break
    _, res, B = evaluate_config(gen_con, folds, cfg)
    result = {"config": cfg, "budget": B, "recall": recall_of(folds, res),
              "start_config": START_CONFIG}
    _, res0, B0 = evaluate_config(gen_con, folds, START_CONFIG)
    result["start_recall"] = recall_of(folds, res0)
    result["start_budget"] = B0
    result["vs_start_same_budget"] = compare(
        folds, res0, evaluate_config(gen_con, folds, cfg, budget=B0)[1])
    _dump("selected_config.json", result)
    return result


# ─────────────────────────── Step 2 ───────────────────────────

def _pool_file(gen_con, fold: Fold, cfg: dict):
    """把各源（最大 k）併成一張「每個 (c, a) 在各源的名次」的表，greedy 只掃這張表。"""
    srcs = cutoffs(cfg)
    f = fold.dir / f"pool_{hashlib.sha1(json.dumps({s: _tag(s, cfg) for s in srcs}, sort_keys=True).encode()).hexdigest()[:10]}.parquet"
    if f.exists():
        return f, list(srcs)
    files = {s: source_file(gen_con, fold, s, cfg) for s in srcs}
    con = dataset.raw_connect()
    union = " UNION ALL ".join(f"SELECT c, a, rk, '{s}' AS src FROM '{p}'" for s, p in files.items())
    rank_cols = ", ".join(f"min(rk) FILTER (WHERE src = '{s}')::SMALLINT AS r_{s}"
                          for s in srcs if s != "repurchase")
    con.execute(f"""
        COPY (
            SELECT u.c, bool_or(src = 'repurchase') AS rep, {rank_cols},
                   bool_or(t.c IS NOT NULL) AS truth
            FROM ({union}) u
            LEFT JOIN '{fold.truth_f}' t USING (c, a)
            GROUP BY u.c, u.a
        ) TO '{f}' (FORMAT PARQUET)
    """)
    return f, list(srcs)


def _covered(k: dict, srcs: list[str]) -> str:
    parts = ["rep"] + [f"coalesce(r_{s} <= {k[s]}, FALSE)" for s in srcs if s != "repurchase"]
    return "(" + " OR ".join(parts) + ")"


def step2(gen_con, folds: list[Fold], cfg: dict) -> list[dict]:
    pools = {}
    for f in folds:
        pools[f.t], srcs = _pool_file(gen_con, f, cfg)
    budget_srcs = [s for s in srcs if s != "repurchase"]
    paths = "[" + ", ".join(f"'{pools[f.t]}'" for f in folds) + "]"
    NB = {str(pools[f.t]): f.NB for f in folds}
    NT = {str(pools[f.t]): f.NT for f in folds}
    con = dataset.raw_connect()

    def totals(k):
        rows = con.execute(f"""
            SELECT filename, count(*) FILTER (WHERE {_covered(k, srcs)}),
                   count(*) FILTER (WHERE {_covered(k, srcs)} AND truth)
            FROM read_parquet({paths}, filename = true) GROUP BY 1
        """).fetchall()
        B = np.mean([n / NB[fn] for fn, n, _ in rows])
        R = np.mean([h / NT[fn] for fn, _, h in rows])
        return float(B), float(R)

    k = {s: 0 for s in budget_srcs}
    path = []
    B, R = totals(k)
    path.append({"budget": B, "recall": R, **{f"k_{s}": 0 for s in budget_srcs}})
    while B < max(FRONTIER_TARGETS) * 1.05:
        moves = [s for s in budget_srcs if k[s] < KMAX[s]]
        if not moves:
            break
        cov = _covered(k, srcs)
        aggs = []
        for s in moves:
            band = f"r_{s} > {k[s]} AND r_{s} <= {k[s] + DELTA[s]} AND NOT {cov}"
            aggs += [f"count(*) FILTER (WHERE {band})", f"count(*) FILTER (WHERE {band} AND truth)"]
        rows = con.execute(f"""
            SELECT filename, {", ".join(aggs)}
            FROM read_parquet({paths}, filename = true) GROUP BY 1
        """).fetchall()
        best, best_eff = None, -1.0
        for i, s in enumerate(moves):
            dn = np.mean([r[1 + 2 * i] / NB[r[0]] for r in rows])
            dh = np.mean([r[2 + 2 * i] / NT[r[0]] for r in rows])
            eff = dh / dn if dn > 0 else 0.0
            if eff > best_eff:
                best, best_eff = s, eff
        k[best] = min(k[best] + DELTA[best], KMAX[best])
        B, R = totals(k)
        path.append({"budget": B, "recall": R, "move": best, "marginal_eff": best_eff,
                     **{f"k_{s}": k[s] for s in budget_srcs}})
        print(f"  B={B:6.1f}  recall={100 * R:6.2f}%  +{best:8s} eff={100 * best_eff:.4f}%/cand  {k}",
              flush=True)

    cur = {s: cutoffs(cfg)[s] for s in budget_srcs}
    Bc, Rc = totals(cur)
    _dump("frontier.json", {"path": path, "current": {"budget": Bc, "recall": Rc, **cur}})
    import csv
    with open(CV_DIR / "frontier.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for p in path for k in p}))
        w.writeheader(); w.writerows(path)
    return path


# ─────────────────────────── 輸出 ───────────────────────────

def _dump(name: str, obj) -> None:
    CV_DIR.mkdir(parents=True, exist_ok=True)
    (CV_DIR / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def report() -> None:
    """讀 step1_log.json、selected_config.json、frontier.json，寫 report.md。"""
    log = json.loads((CV_DIR / "step1_log.json").read_text())
    sel = json.loads((CV_DIR / "selected_config.json").read_text())
    fr = json.loads((CV_DIR / "frontier.json").read_text())
    pp = lambda x: f"{100 * x:+.3f}"
    L = ["# Recall cross-validation report", "",
         f"Validation folds: {', '.join(FOLDS)}. Metric: micro candidate recall. "
         f"Decision rule: mean Δ ≥ {100 * MIN_EFFECT:.1f} pp, ≥ {MIN_POSITIVE_FOLDS}/4 positive folds, "
         "pooled 95% CI > 0. All comparisons at equal candidate budget (segment cutoff adjusted).", "",
         "## Step 1: quality parameters", "",
         "| Round | Parameter | Current → Candidate | Mean Δ (pp) | 95% CI (pp) | Positive folds | Adopted |",
         "|---|---|---|---|---|---|---|"]
    for e in log:
        L.append(f"| {e['round']} | {e['param']} | {e['current']} → {e['value']} | {pp(e['delta_mean'])} | "
                 f"[{pp(e['ci95'][0])}, {pp(e['ci95'][1])}] | {e['positive_folds']}/4 | "
                 f"{'yes' if e['adopt'] else 'no'} |")
    changed = {k: (sel['start_config'].get(k), v) for k, v in sel['config'].items()
               if sel['start_config'].get(k) != v}
    L += ["", f"Selected configuration changes: {changed or 'none (current configuration retained)'}.", "",
          f"Recall at B = {sel['budget']:.1f} candidates/customer: mean {100 * sel['recall']['mean']:.2f}%.", "",
          "| Fold | Recall |", "|---|---|"]
    L += [f"| {t} | {100 * v:.2f}% |" for t, v in sel["recall"]["folds"].items()]
    L += ["", "| Recency stratum | Recall (pooled over folds) |", "|---|---|"]
    L += [f"| {k} | {100 * v:.2f}% |" for k, v in sel["recall"]["strata"].items()]
    path = fr["path"]
    L += ["", "## Step 2: budget allocation (greedy recall–budget frontier)", "",
          "| Target B | B | Recall | " + " | ".join(k for k in path[-1] if k.startswith("k_")) + " |",
          "|---|---|---|" + "---|" * sum(k.startswith("k_") for k in path[-1])]
    for tgt in FRONTIER_TARGETS:
        p = min(path, key=lambda x: abs(x["budget"] - tgt))
        L.append(f"| {tgt} | {p['budget']:.1f} | {100 * p['recall']:.2f}% | "
                 + " | ".join(str(p[k]) for k in path[-1] if k.startswith("k_")) + " |")
    c = fr["current"]
    lo = max((p for p in path if p["budget"] <= c["budget"]), key=lambda x: x["budget"])
    hi = min((p for p in path if p["budget"] >= c["budget"]), key=lambda x: x["budget"])
    w = 0 if hi["budget"] == lo["budget"] else (c["budget"] - lo["budget"]) / (hi["budget"] - lo["budget"])
    interp = lo["recall"] + w * (hi["recall"] - lo["recall"])
    L += ["", f"Current allocation: B = {c['budget']:.1f}, recall {100 * c['recall']:.2f}%. "
          f"Frontier interpolated at the same B: {100 * interp:.2f}% "
          f"(gap {100 * (interp - c['recall']):+.2f} pp).", ""]
    (CV_DIR / "report.md").write_text("\n".join(L))
    print(f"wrote {CV_DIR / 'report.md'}")


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "report":
        return report()
    t0 = time.time()
    gen_con = dataset.connect()
    folds = [Fold(t, gen_con) for t in FOLDS]
    print(f"folds: {[(f.t, f.NB, f.NT) for f in folds]}")
    if what in ("step1", "all"):
        print("== Step 1: quality parameters ==")
        step1(gen_con, folds)
    if what in ("step2", "all"):
        print("== Step 2: budget allocation ==")
        sel = json.loads((CV_DIR / "selected_config.json").read_text())
        step2(gen_con, folds, sel["config"])
    report()
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
