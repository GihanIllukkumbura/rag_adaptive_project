"""Retrievers + ranked-list cache.

Backends
  lsa : TF-IDF -> TruncatedSVD (LSA) dense vectors, exact FAISS inner-product search. Runs offline on CPU.
  st  : sentence-transformers dense embeddings (e.g. all-MiniLM-L6-v2, bge-small, e5-small), exact FAISS search.
  bm25: sparse Okapi BM25 (own implementation on scipy sparse matrices).
All searches are EXACT, so top-k of a cached top-K_MAX list is identical to a fresh top-k search.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import faiss
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

@dataclass
class Ranked:
    ids: np.ndarray      # (n_queries, K) int32 passage ids, best first
    scores: np.ndarray   # (n_queries, K) float32, higher = better

def _topk_dense(S: np.ndarray, k: int):
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(S, idx, 1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, 1).astype(np.int32), np.take_along_axis(part, order, 1).astype(np.float32)

class BM25Retriever:
    name = "bm25"
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
    def fit(self, docs: list[str]):
        self.cv = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w\w+\b", dtype=np.float32)
        X = self.cv.fit_transform(docs).tocsr()
        N = X.shape[0]; df = np.asarray((X > 0).sum(0)).ravel()
        idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(X.sum(1)).ravel(); avgdl = dl.mean()
        X = X.tocoo(); denom = X.data + self.k1 * (1 - self.b + self.b * dl[X.row] / avgdl)
        w = X.data * (self.k1 + 1) / denom * idf[X.col]
        self.W = sp.csr_matrix((w.astype(np.float32), (X.row, X.col)), shape=X.shape)
        self.WT = self.W.T.tocsr()
        return self
    def search(self, queries: list[str], k: int, chunk: int = 256) -> Ranked:
        Q = self.cv.transform(queries); Q.data[:] = 1.0
        ids, sc = [], []
        for i in range(0, Q.shape[0], chunk):
            S = (Q[i:i+chunk] @ self.WT).toarray()
            a, b = _topk_dense(S, k); ids.append(a); sc.append(b)
        return Ranked(np.vstack(ids), np.vstack(sc))

class _DenseBase:
    def _build(self, E: np.ndarray):
        self.E = np.ascontiguousarray(E.astype("float32"))
        self.index = faiss.IndexFlatIP(self.E.shape[1]); self.index.add(self.E)
    def search(self, queries: list[str], k: int) -> Ranked:
        Q = self.encode_queries(queries)
        sc, ids = self.index.search(Q, k)
        return Ranked(ids.astype(np.int32), sc.astype(np.float32))

class LSARetriever(_DenseBase):
    name = "lsa"
    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim, self.seed = dim, seed
    def fit(self, docs: list[str]):
        self.tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2), min_df=2, dtype=np.float32)
        X = self.tfidf.fit_transform(docs)
        self.svd = TruncatedSVD(n_components=self.dim, n_iter=5, random_state=self.seed)
        self._build(normalize(self.svd.fit_transform(X)))
        return self
    def encode_queries(self, queries):
        return np.ascontiguousarray(normalize(self.svd.transform(self.tfidf.transform(queries))).astype("float32"))

# prefix conventions for popular free embedding models
ST_PREFIX = {
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
}

class STRetriever(_DenseBase):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64, device: str | None = None):
        self.model_name, self.batch_size, self.device = model_name, batch_size, device
        self.name = "st:" + model_name.split("/")[-1]
        self.qp, self.dp = ST_PREFIX.get(model_name, ("", ""))
    def fit(self, docs: list[str]):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name, device=self.device)
        E = self.model.encode([self.dp + d for d in docs], batch_size=self.batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=True)
        self._build(E); return self
    def encode_queries(self, queries):
        return self.model.encode([self.qp + q for q in queries], batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")

def make_retriever(backend: str, **kw):
    if backend == "lsa": return LSARetriever(**{k: v for k, v in kw.items() if k in ("dim", "seed")})
    if backend == "bm25": return BM25Retriever()
    if backend == "st": return STRetriever(**{k: v for k, v in kw.items() if k in ("model_name", "batch_size", "device")})
    raise ValueError(backend)

def rrf_fuse(a: Ranked, b: Ranked, depth: int, c: int = 60) -> Ranked:
    """Reciprocal Rank Fusion (Cormack et al., 2009) of two ranked lists, per query."""
    n = a.ids.shape[0]; ids = np.zeros((n, depth), np.int32); sc = np.zeros((n, depth), np.float32)
    for i in range(n):
        s = {}
        for r, p in enumerate(a.ids[i]): s[p] = s.get(p, 0.0) + 1.0 / (c + r + 1)
        for r, p in enumerate(b.ids[i]): s[p] = s.get(p, 0.0) + 1.0 / (c + r + 1)
        top = sorted(s.items(), key=lambda t: -t[1])[:depth]
        ids[i, :len(top)] = [t[0] for t in top]; sc[i, :len(top)] = [t[1] for t in top]
    return Ranked(ids, sc)

def time_single_queries(retriever, queries: list[str], k: int, n: int = 300, warmup: int = 5) -> np.ndarray:
    """Wall-clock seconds per single-query (encode + exact search), batch size 1."""
    qs = queries[:n]
    for q in qs[:warmup]: retriever.search([q], k)
    out = np.empty(len(qs))
    for i, q in enumerate(qs):
        t = time.perf_counter(); retriever.search([q], k); out[i] = time.perf_counter() - t
    return out
