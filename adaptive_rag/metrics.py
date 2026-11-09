"""Per-query retrieval metrics computed from cached ranked lists (vectorised)."""
from __future__ import annotations
import numpy as np
from .data import normalize_answer

def rel_matrix(ids: np.ndarray, rel_lists) -> np.ndarray:
    out = np.zeros(ids.shape, dtype=bool)
    for i, r in enumerate(rel_lists): out[i] = np.isin(ids[i], r)
    return out

def ans_matrix(ids: np.ndarray, norm_texts: list[str], answers) -> np.ndarray:
    """True where the passage text contains (normalised) any gold answer string."""
    out = np.zeros(ids.shape, dtype=bool)
    for i, ans in enumerate(answers):
        na = [normalize_answer(a) for a in ans if normalize_answer(a)]
        if not na: continue
        for j, p in enumerate(ids[i]):
            t = norm_texts[p]
            out[i, j] = any(a in t for a in na)
    return out

def metrics_at_k(rel: np.ndarray, ans: np.ndarray, words: np.ndarray, n_rel: np.ndarray, k: np.ndarray) -> dict:
    """Metrics when query i is served with its top-k[i] passages. All inputs share the (n, K) layout."""
    n, K = rel.shape
    ranks = np.arange(K)[None, :]
    mask = ranks < k[:, None]
    r = rel & mask
    hit = r.any(1)
    first = np.where(hit, r.argmax(1), 0)
    rr = np.where(hit, 1.0 / (first + 1), 0.0)
    recall = r.sum(1) / np.maximum(n_rel, 1)
    disc = 1.0 / np.log2(ranks + 2.0)
    dcg = (r * disc).sum(1)
    cum_disc = np.cumsum(disc[0])
    ideal_n = np.minimum(n_rel, k).astype(int)
    idcg = np.where(ideal_n > 0, cum_disc[np.maximum(ideal_n - 1, 0)], 1.0)
    return {
        "k": k.astype(int), "hit": hit.astype(int), "recall": recall, "rr": rr, "ndcg": dcg / idcg,
        "ans_in_ctx": (ans & mask).any(1).astype(int), "ctx_words": (words * mask).sum(1),
    }
