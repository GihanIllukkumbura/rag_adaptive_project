#!/usr/bin/env python
"""Run the complete experiment suite and write every CSV / figure / manifest.
usage: python scripts/run_all.py [--backend lsa|st] [--n_queries 8000] [--seeds 5] [--quick]"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from adaptive_rag import experiment as E, plots as P
from adaptive_rag.utils import results_dir, log_event, write_manifest, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="lsa"); ap.add_argument("--st_model", default="sentence-transformers/all-MiniLM-L6-v2")
ap.add_argument("--n_queries", type=int, default=8000); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
if a.quick: a.n_queries, a.seeds, a.boot = 2000, 2, 200
cfg = dict(backend=a.backend, st_model=a.st_model, n_queries=a.n_queries)
csv = results_dir("csv")
B = E.prepare(cfg); log_event("run_all_start", **cfg)
df, perq, te = E.evaluate(B, return_per_query=True); df.to_csv(csv / "strategy_results.csv", index=False)
pd.DataFrame({"qid": B.queries.qid.iloc[te].values, "article_id": B.queries.article_id.iloc[te].values,
              **{f"gold_rank_{k}": v[te] for k, v in B.rank.items()}}).to_csv(csv / "test_gold_ranks.csv", index=False)
cov = E.coverage(B); cov.to_csv(csv / "rcak_coverage.csv", index=False)
iso = pd.concat([E.isobudget(B, s, n_boot=a.boot) for s in B.lists]); iso.to_csv(csv / "isobudget_bootstrap.csv", index=False)
ab = E.ablation(B, n_boot=min(300, a.boot)); ab.to_csv(csv / "ablation.csv", index=False)
lat = E.latency(B); lat.to_csv(csv / "latency.csv", index=False)
for s in B.lists:
    P.fig_pareto(df, s, "answer_in_context"); P.fig_pareto(df, s, "hit_rate"); P.fig_coverage(cov, s); P.fig_isobudget(iso, s)
serve = "hybrid_rrf"
kd = {"RCAK alpha=0.10": (perq[(serve, "RCAK (proposed)", "alpha=0.1")][0], "RCAK (proposed)"),
      "Adaptive-k buffer=5": (perq[(serve, "adaptive_k_gap", "buffer=5")][0], "adaptive_k_gap"),
      "Oracle": (perq[(serve, "oracle_k", "min_k")][0], "oracle_k")}
P.fig_k_hist(kd, B.rank[serve][te])
P.fig_ablation(ab); P.fig_latency(lat)
sd = E.seeds(cfg, tuple(range(a.seeds))); sd.to_csv(csv / "seed_robustness.csv", index=False); P.fig_seeds(sd)
write_manifest(); log_event("run_all_done")
print("done ->", results_dir())
