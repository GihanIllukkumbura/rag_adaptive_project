"""Retrieval-depth strategies. Each returns an integer array k[i] = number of passages served for query i.

Baselines
  fixed_k             classic top-k
  rule_relative       keep passages with score >= tau * best score (rule-based)
  rule_zscore         keep passages whose z-score within the candidate pool >= z
  adaptive_k_gap      Adaptive-k (Taguchi et al., 2025): largest score gap in the top-90% of the pool + buffer
  car_lite            simplified cluster-based cut (after Xu et al., 2025): 2-means on scores, cut at the cluster boundary
Proposed (this project)
  RCAK  Risk-Controlled Adaptive-k: gradient-boosted *quantile* regression of log(rank of the first gold passage)
        from score-distribution features, followed by split-conformal calibration (CQR; Romano et al., 2019),
        giving a finite-sample, distribution-free guarantee  P(gold rank <= k_hat(x)) >= 1 - alpha  (exchangeability).
        alpha is an explicit, interpretable quality/cost dial.
  CGC   Confidence-Gated Cascade: a classifier predicts P(gold in top-k1); if confidence >= tau serve k1, else k2
        (two-stage, cheap, monotone budget control).
Oracle
  oracle_k            minimal k that contains the gold passage (unattainable upper bound; shows head-room)
"""
from __future__ import annotations
import math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

def fixed_k(n: int, k: int) -> np.ndarray:
    return np.full(n, k, dtype=int)

def rule_relative(scores: np.ndarray, tau: float, kmin: int = 1, kmax: int | None = None) -> np.ndarray:
    kmax = kmax or scores.shape[1]
    s1 = scores[:, :1]
    keep = scores >= tau * s1 if (s1 > 0).all() else (scores - s1) >= -(1 - tau) * np.abs(s1)
    k = keep.sum(1)
    return np.clip(k, kmin, kmax).astype(int)

def rule_zscore(scores: np.ndarray, z: float, kmin: int = 1, kmax: int | None = None) -> np.ndarray:
    kmax = kmax or scores.shape[1]
    zs = (scores - scores.mean(1, keepdims=True)) / (scores.std(1, keepdims=True) + 1e-9)
    return np.clip((zs >= z).sum(1), kmin, kmax).astype(int)

def adaptive_k_gap(scores: np.ndarray, buffer: int = 5, frac: float = 0.9, kmin: int = 1) -> np.ndarray:
    n, K = scores.shape
    m = max(2, int(frac * K))
    gaps = scores[:, :m - 1] - scores[:, 1:m]
    k = gaps.argmax(1) + 1 + buffer
    return np.clip(k, kmin, K).astype(int)

def car_lite(scores: np.ndarray, kmin: int = 1, pool: int = 20) -> np.ndarray:
    """Per query: 1-D 2-means on the top-`pool` normalised scores; k = size of the upper cluster (contiguous prefix)."""
    n, K = scores.shape; P = min(pool, K); out = np.empty(n, int)
    for i in range(n):
        s = scores[i, :P].astype(float); rng = s[0] - s[-1]
        if rng < 1e-9: out[i] = P; continue
        z = (s - s[-1]) / rng; c1, c0 = 1.0, 0.0
        for _ in range(20):
            hi = np.abs(z - c1) <= np.abs(z - c0)
            n1 = z[hi].mean() if hi.any() else c1; n0 = z[~hi].mean() if (~hi).any() else c0
            if abs(n1 - c1) + abs(n0 - c0) < 1e-6: break
            c1, c0 = n1, n0
        hi = np.abs(z - c1) <= np.abs(z - c0)
        k = int(np.argmin(hi)) if not hi.all() else P   # first index not in upper cluster
        out[i] = max(kmin, k)
    return out

def oracle_k(rank_of_gold: np.ndarray, K: int) -> np.ndarray:
    return np.where(rank_of_gold <= K, rank_of_gold, K).astype(int)

# ---------------------------------------------------------------- RCAK
class RCAK:
    """Risk-Controlled Adaptive-k via conformalised quantile regression of log gold-rank."""
    def __init__(self, K: int = 50, kmin: int = 1, quantile: float = 0.9, seed: int = 0, max_iter: int = 200):
        self.K, self.kmin, self.q, self.seed, self.max_iter = K, kmin, quantile, seed, max_iter
        self.shift = 0.0
    def fit(self, X_train: np.ndarray, rank_train: np.ndarray):
        y = np.log(np.minimum(rank_train, self.K + 1).astype(float))
        self.model = HistGradientBoostingRegressor(loss="quantile", quantile=self.q, max_iter=self.max_iter,
                                                   learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                                   l2_regularization=1.0, random_state=self.seed)
        self.model.fit(X_train, y); return self
    def calibrate(self, X_cal: np.ndarray, rank_cal: np.ndarray, alpha: float) -> float:
        y = np.log(np.minimum(rank_cal, self.K + 1).astype(float))
        s = y - self.model.predict(X_cal); n = len(s)
        level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
        self.shift = float(np.quantile(s, level, method="higher")); self.alpha = alpha
        return self.shift
    def predict_k(self, X: np.ndarray, shift: float | None = None) -> np.ndarray:
        sh = self.shift if shift is None else shift
        k = np.ceil(np.exp(self.model.predict(X) + sh) - 1e-9)
        return np.clip(k, self.kmin, self.K).astype(int)

# ---------------------------------------------------------------- CGC
class CGC:
    """Confidence-Gated Cascade: serve k1 if P(gold in top-k1) >= tau else k2."""
    def __init__(self, k1: int = 3, k2: int = 20, seed: int = 0):
        self.k1, self.k2, self.seed = k1, k2, seed
    def fit(self, X_train, rank_train):
        self.clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                  min_samples_leaf=40, l2_regularization=1.0, random_state=self.seed)
        self.clf.fit(X_train, (rank_train <= self.k1).astype(int)); return self
    def confidence(self, X): return self.clf.predict_proba(X)[:, 1]
    def predict_k(self, X, tau: float):
        return np.where(self.confidence(X) >= tau, self.k1, self.k2).astype(int)
