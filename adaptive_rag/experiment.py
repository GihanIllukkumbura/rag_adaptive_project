"""End-to-end experiment pipeline (all numbers come from real runs; nothing is simulated).

Stages (each writes CSV evidence into results/csv and logs a row in results/logs/run_log.csv):
  1 prepare()      load SQuAD, build retrievers, cache top-K ranked lists (dense, bm25, hybrid-RRF), features, splits
  2 evaluate()     sweep every strategy/parameter on the TEST split -> strategy_results.csv
  3 coverage()     conformal validity check (target vs achieved gold coverage) -> rcak_coverage.csv
  4 isobudget()    cluster-bootstrap differences vs fixed-k at the SAME mean budget -> isobudget_bootstrap.csv
  5 ablation()     feature-group ablation of RCAK -> ablation.csv
  6 seeds()        repeat with several article-level split seeds -> seed_robustness.csv
  7 latency()      measured per-query latency of retrieval and of the depth controller -> latency.csv
"""
from __future__ import annotations
import time
import numpy as np, pandas as pd
from .data import load_squad, normalize_answer, split_by_group, corpus_stats
from .retrieval import make_retriever, rrf_fuse, Ranked, time_single_queries
from .features import query_features, feature_groups
from .metrics import rel_matrix, ans_matrix, metrics_at_k
from .strategies import (fixed_k, rule_relative, rule_zscore, adaptive_k_gap, car_lite, oracle_k, RCAK, CGC)
from .utils import set_seed, env_info, save_json, results_dir, log_event, Timer

K_MAX = 50
FIXED_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
ALPHAS = [0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02]
TOK_PER_WORD = 1.3   # rough English tokenisation factor; real token counts are measured in the generation notebook

DEFAULT_CFG = dict(backend="lsa", st_model="sentence-transformers/all-MiniLM-L6-v2", lsa_dim=256,
                   split="train", n_queries=8000, n_articles=None, seed=0, alpha_main=0.10)

class Bundle:  # everything cached after prepare()
    pass

def prepare(cfg: dict) -> Bundle:
    cfg = {**DEFAULT_CFG, **cfg}; set_seed(cfg["seed"]); B = Bundle(); B.cfg = cfg
    with Timer() as t:
        corpus, queries = load_squad(cfg["split"], cfg["n_queries"], cfg["n_articles"], cfg["seed"])
    docs = corpus["text"].tolist(); qs = queries["question"].tolist()
    B.corpus, B.queries = corpus, queries
    B.words = corpus["text"].str.split().str.len().to_numpy()
    B.norm_texts = [normalize_answer(x) for x in docs]
    corpus_stats(corpus, queries).to_csv(results_dir("csv") / "dataset_stats.csv", index=False)
    timing = {}
    with Timer() as t: dense = make_retriever(cfg["backend"], model_name=cfg["st_model"], dim=cfg["lsa_dim"], seed=cfg["seed"]).fit(docs)
    timing["dense_index_s"] = t.elapsed
    with Timer() as t: bm = make_retriever("bm25").fit(docs)
    timing["bm25_index_s"] = t.elapsed
    with Timer() as t: Rd = dense.search(qs, K_MAX)
    timing["dense_batch_search_s"] = t.elapsed
    with Timer() as t: Rb = bm.search(qs, K_MAX)
    timing["bm25_batch_search_s"] = t.elapsed
    with Timer() as t: Rh = rrf_fuse(Rd, Rb, K_MAX)
    timing["rrf_s"] = t.elapsed
    B.dense_r, B.bm25_r, B.dense, B.bm25 = Rd, Rb, dense, bm
    B.lists = {dense.name: Rd, "bm25": Rb, "hybrid_rrf": Rh}
    B.dense_name = dense.name
    B.feat = query_features(Rd, Rb, qs)
    B.split = split_by_group(queries).to_numpy()
    rel_lists = queries["rel_pids"].tolist(); B.n_rel = np.array([len(r) for r in rel_lists])
    B.rel, B.ans, B.rank = {}, {}, {}
    for name, R in B.lists.items():
        rm = rel_matrix(R.ids, rel_lists); B.rel[name] = rm
        B.ans[name] = ans_matrix(R.ids, B.norm_texts, queries["answers"].tolist())
        B.rank[name] = np.where(rm.any(1), rm.argmax(1) + 1, K_MAX + 1)
    B.timing = timing
    save_json({"config": cfg, "env": env_info(), "timing_s": timing}, results_dir("logs") / "environment_and_config.json")
    log_event("prepare", **timing, n_passages=len(corpus), n_queries=len(queries), backend=dense.name)
    return B

def prepare_cached(cfg: dict) -> Bundle:
    """prepare() with a pickle cache of ranked lists/features (retriever objects are not cached; latency() refits them)."""
    import pickle, hashlib, json as _j
    c = {**DEFAULT_CFG, **cfg}; key = hashlib.md5(_j.dumps(c, sort_keys=True).encode()).hexdigest()[:10]
    path = results_dir("cache") / f"prep_{c['backend']}_{key}.pkl"
    if path.exists():
        with open(path, "rb") as f: return pickle.load(f)
    B = prepare(c); d, b = B.dense, B.bm25; B.dense = B.bm25 = None
    with open(path, "wb") as f: pickle.dump(B, f)
    B.dense, B.bm25 = d, b; return B

def _row(B, serve, mask_idx, k, strategy, param, extra=None):
    R = B.lists[serve]; w = B.words[R.ids[mask_idx]]
    m = metrics_at_k(B.rel[serve][mask_idx], B.ans[serve][mask_idx], w, B.n_rel[mask_idx], k)
    r = {"serve_retriever": serve, "strategy": strategy, "param": param, "n_queries": len(k),
         "mean_k": k.mean(), "median_k": np.median(k), "p95_k": np.percentile(k, 95),
         "hit_rate": m["hit"].mean(), "mrr": m["rr"].mean(), "ndcg": m["ndcg"].mean(),
         "answer_in_context": m["ans_in_ctx"].mean(), "mean_ctx_words": m["ctx_words"].mean(),
         "mean_ctx_tokens_est": m["ctx_words"].mean() * TOK_PER_WORD}
    if extra: r.update(extra)
    return r, m

def train_models(B, cols, seed=0, k1=3, k2=20, serve=None):
    tr, ca = B.split == "train", B.split == "val"
    Xtr, Xca = B.feat.loc[tr, cols].to_numpy(), B.feat.loc[ca, cols].to_numpy()
    out = {}
    for serve_name in B.lists:
        rk = B.rank[serve_name]
        rc = RCAK(K_MAX, seed=seed).fit(Xtr, rk[tr]); out[("rcak", serve_name)] = (rc, Xca, rk[ca])
        cg = CGC(k1, k2, seed=seed).fit(Xtr, rk[tr]); out[("cgc", serve_name)] = cg
    return out

def evaluate(B, cols=None, seed=0, return_per_query=False):
    cols = cols or [c for g in feature_groups().values() for c in g]
    te = np.where(B.split == "test")[0]; rows, perq = [], {}
    models = train_models(B, cols, seed)
    Xte = B.feat.loc[te, cols].to_numpy()
    for serve, R in B.lists.items():
        sc = R.scores[te]; n = len(te)
        ks = {}
        for k in FIXED_GRID: ks[("fixed_k", f"k={k}")] = fixed_k(n, k)
        for tau in [0.99, 0.97, 0.95, 0.92, 0.9, 0.85, 0.8, 0.7, 0.6]: ks[("rule_relative", f"tau={tau}")] = rule_relative(sc, tau)
        for z in [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]: ks[("rule_zscore", f"z={z}")] = rule_zscore(sc, z)
        for b in [0, 2, 5, 10]: ks[("adaptive_k_gap", f"buffer={b}")] = adaptive_k_gap(sc, buffer=b)
        ks[("car_lite", "default")] = car_lite(sc)
        ks[("oracle_k", "min_k")] = oracle_k(B.rank[serve][te], K_MAX)
        rc, Xca, rca = models[("rcak", serve)]
        for a in ALPHAS:
            rc.calibrate(Xca, rca, a); ks[("RCAK (proposed)", f"alpha={a}")] = rc.predict_k(Xte)
        cg = models[("cgc", serve)]
        for tau in [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]: ks[("CGC (proposed)", f"tau={tau}")] = cg.predict_k(Xte, tau)
        for (s, p), k in ks.items():
            r, m = _row(B, serve, te, k, s, p); rows.append(r)
            if return_per_query: perq[(serve, s, p)] = (k, m)
    df = pd.DataFrame(rows)
    return (df, perq, te) if return_per_query else df

def fixed_curve(B, serve, idx):
    """Per-query metric matrices for every fixed k=1..K_MAX -> dict name -> (n,K) array."""
    R = B.lists[serve]; rel = B.rel[serve][idx]; ans = B.ans[serve][idx]
    w = B.words[R.ids[idx]]; n = len(idx)
    hit = np.cumsum(rel, 1) > 0; aic = np.cumsum(ans, 1) > 0; cw = np.cumsum(w, 1)
    return {"hit_rate": hit.astype(float), "answer_in_context": aic.astype(float), "ctx_words": cw.astype(float)}

def interp_fixed(curve_mean_k_axis, y, x):
    return np.interp(x, np.arange(1, len(y) + 1), y)

def isobudget(B, serve=None, n_boot=1000, seed=0, cols=None):
    """Adaptive minus fixed-k at the same mean number of passages (cluster bootstrap over articles)."""
    serve = serve or "hybrid_rrf"
    df, perq, te = evaluate(B, cols, seed, return_per_query=True)
    fc = fixed_curve(B, serve, te); arts = B.queries["article_id"].to_numpy()[te]
    ua = np.unique(arts); idx_by = {a: np.where(arts == a)[0] for a in ua}
    rng = np.random.RandomState(seed); rows = []
    boots = [np.concatenate([idx_by[a] for a in rng.choice(ua, len(ua))]) for _ in range(n_boot)]
    for (sv, s, p), (k, m) in perq.items():
        if sv != serve or s in ("fixed_k", "oracle_k"): continue
        for metric, arr in (("answer_in_context", m["ans_in_ctx"]), ("hit_rate", m["hit"])):
            diffs = []
            for bi in boots:
                mk = k[bi].mean(); fm = fc[metric][bi].mean(0)
                diffs.append(arr[bi].mean() - np.interp(mk, np.arange(1, K_MAX + 1), fm))
            mk0 = k.mean(); f0 = np.interp(mk0, np.arange(1, K_MAX + 1), fc[metric].mean(0))
            rows.append({"serve_retriever": serve, "strategy": s, "param": p, "metric": metric, "mean_k": mk0,
                         "adaptive_value": arr.mean(), "fixed_value_at_same_mean_k": f0,
                         "diff": arr.mean() - f0, "ci95_lo": np.percentile(diffs, 2.5), "ci95_hi": np.percentile(diffs, 97.5),
                         "p_boot_diff_le_0": float(np.mean(np.array(diffs) <= 0))})
    return pd.DataFrame(rows)

def coverage(B, cols=None, seed=0):
    cols = cols or [c for g in feature_groups().values() for c in g]
    models = train_models(B, cols, seed); te = np.where(B.split == "test")[0]
    Xte = B.feat.loc[te, cols].to_numpy(); rows = []
    for serve in B.lists:
        rc, Xca, rca = models[("rcak", serve)]
        reach = float((B.rank[serve][te] <= K_MAX).mean())
        for a in ALPHAS + [0.01]:
            rc.calibrate(Xca, rca, a); k = rc.predict_k(Xte)
            cov = float((B.rank[serve][te] <= k).mean())
            rows.append({"serve_retriever": serve, "alpha": a, "target_coverage": 1 - a, "achieved_test_coverage": cov,
                         "pool_recall_at_K": reach, "mean_k": k.mean(), "calibration_shift": rc.shift})
    return pd.DataFrame(rows)

def ablation(B, serve="hybrid_rrf", seed=0, alpha=0.10, n_boot=300):
    groups = feature_groups(); allc = [c for g in groups.values() for c in g]
    variants = {"all features": allc}
    for g in groups: variants[f"without {g}"] = [c for c in allc if c not in groups[g]]
    for g in groups: variants[f"only {g}"] = groups[g]
    rows = []
    for name, cols in variants.items():
        iso = isobudget(B, serve, n_boot=n_boot, seed=seed, cols=cols)
        r = iso[(iso.strategy == "RCAK (proposed)") & (iso.param == f"alpha={alpha}") & (iso.metric == "answer_in_context")].iloc[0]
        rows.append({"variant": name, "n_features": len(cols), "mean_k": r.mean_k, "answer_in_context": r.adaptive_value,
                     "fixed_at_same_k": r.fixed_value_at_same_mean_k, "diff": r["diff"], "ci95_lo": r.ci95_lo, "ci95_hi": r.ci95_hi})
    return pd.DataFrame(rows)

def seeds(cfg, seed_list=(0, 1, 2, 3, 4), serve="hybrid_rrf", alpha=0.10):
    """Re-split (and re-sample queries) with different seeds; retrievers are re-fit each time for honesty."""
    rows = []
    for s in seed_list:
        B = prepare({**cfg, "seed": s}); df = evaluate(B, seed=s); iso = isobudget(B, serve, n_boot=200, seed=s)
        for strat, par in [("RCAK (proposed)", f"alpha={alpha}"), ("adaptive_k_gap", "buffer=5"), ("CGC (proposed)", "tau=0.7")]:
            r = iso[(iso.strategy == strat) & (iso.param == par) & (iso.metric == "answer_in_context")].iloc[0]
            rows.append({"seed": s, "strategy": strat, "param": par, "mean_k": r.mean_k, "answer_in_context": r.adaptive_value,
                         "fixed_at_same_k": r.fixed_value_at_same_mean_k, "diff": r["diff"]})
    return pd.DataFrame(rows)

def latency(B, n=300, alpha=0.10, cols=None):
    cols = cols or [c for g in feature_groups().values() for c in g]
    qs = B.queries["question"].tolist()[:n]; rows = []
    if B.dense is None:
        c = B.cfg; docs = B.corpus["text"].tolist()
        B.dense = make_retriever(c["backend"], model_name=c["st_model"], dim=c["lsa_dim"], seed=c["seed"]).fit(docs)
        B.bm25 = make_retriever("bm25").fit(docs)
    for nm, r in ((B.dense_name, B.dense), ("bm25", B.bm25)):
        a = time_single_queries(r, qs, K_MAX, n)
        rows.append({"component": f"{nm} retrieval (1 query, top-{K_MAX})", "mean_ms": a.mean() * 1e3, "p50_ms": np.median(a) * 1e3, "p95_ms": np.percentile(a, 95) * 1e3})
    models = train_models(B, cols, 0); rc, Xca, rca = models[("rcak", "hybrid_rrf")]; rc.calibrate(Xca, rca, alpha)
    te = np.where(B.split == "test")[0][:n]; t = []
    for i in te:
        t0 = time.perf_counter()
        f = query_features(Ranked(B.dense_r.ids[i:i+1], B.dense_r.scores[i:i+1]), Ranked(B.bm25_r.ids[i:i+1], B.bm25_r.scores[i:i+1]), [B.queries.question.iloc[i]])
        rc.predict_k(f[cols].to_numpy()); t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e3
    rows.append({"component": "RCAK controller (features + prediction, 1 query)", "mean_ms": t.mean(), "p50_ms": np.median(t), "p95_ms": np.percentile(t, 95)})
    return pd.DataFrame(rows)
