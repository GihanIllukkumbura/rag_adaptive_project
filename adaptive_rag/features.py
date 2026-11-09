"""Query-level features computed from retrieval score distributions (no labels, no LLM calls)."""
from __future__ import annotations
import numpy as np, pandas as pd
from .retrieval import Ranked

QTYPES = ["who", "when", "where", "what", "which", "how", "why"]

def _entropy_z(s: np.ndarray) -> np.ndarray:
    """Entropy of softmax over scale-free scores z=(s-s1)/std. Scale-invariant, so comparable across encoders."""
    z = (s - s[:, :1]) / (s.std(1, keepdims=True) + 1e-6)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return -(p * np.log(p + 1e-12)).sum(1)

def feature_groups() -> dict[str, list[str]]:
    return {
        "dense_scores": ["d_s1", "d_gap12", "d_gap15", "d_gap110", "d_mean10", "d_std10", "d_ratio10", "d_entropy"],
        "bm25_scores": ["b_s1", "b_gap12", "b_mean10", "b_ratio10", "b_entropy"],
        "agreement": ["overlap5", "overlap10", "d1_rank_in_bm25", "b1_rank_in_dense"],
        "query": ["q_words"] + [f"qt_{t}" for t in QTYPES] + ["qt_other"],
    }

def query_features(dense: Ranked, bm25: Ranked, questions: list[str]) -> pd.DataFrame:
    n = dense.ids.shape[0]; d = dense.scores[:, :20].astype(np.float64); b = bm25.scores[:, :20].astype(np.float64)
    eps = 1e-6; f = {}
    f["d_s1"] = d[:, 0]; f["d_gap12"] = d[:, 0] - d[:, 1]; f["d_gap15"] = d[:, 0] - d[:, 4]; f["d_gap110"] = d[:, 0] - d[:, 9]
    f["d_mean10"] = d[:, :10].mean(1); f["d_std10"] = d[:, :10].std(1)
    f["d_ratio10"] = d[:, 9] / (np.abs(d[:, 0]) + eps); f["d_entropy"] = _entropy_z(d)
    f["b_s1"] = b[:, 0]; f["b_gap12"] = (b[:, 0] - b[:, 1]) / (b[:, 0] + eps)
    f["b_mean10"] = b[:, :10].mean(1) / (b[:, 0] + eps); f["b_ratio10"] = b[:, 9] / (b[:, 0] + eps); f["b_entropy"] = _entropy_z(b)
    o5 = np.zeros(n); o10 = np.zeros(n); r1 = np.full(n, 51.0); r2 = np.full(n, 51.0)
    for i in range(n):
        o5[i] = len(set(dense.ids[i, :5]) & set(bm25.ids[i, :5])) / 5.0
        o10[i] = len(set(dense.ids[i, :10]) & set(bm25.ids[i, :10])) / 10.0
        w = np.where(bm25.ids[i] == dense.ids[i, 0])[0]
        if len(w): r1[i] = w[0] + 1
        w = np.where(dense.ids[i] == bm25.ids[i, 0])[0]
        if len(w): r2[i] = w[0] + 1
    f["overlap5"], f["overlap10"], f["d1_rank_in_bm25"], f["b1_rank_in_dense"] = o5, o10, r1, r2
    toks = [q.lower().replace("?", " ").split() for q in questions]
    f["q_words"] = np.array([len(t) for t in toks], dtype=float)
    first = [next((w for w in t[:4] if w in QTYPES), None) for t in toks]
    for t in QTYPES: f[f"qt_{t}"] = np.array([1.0 if x == t else 0.0 for x in first])
    f["qt_other"] = np.array([1.0 if x is None else 0.0 for x in first])
    return pd.DataFrame(f)
