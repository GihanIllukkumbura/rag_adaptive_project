"""Dataset loading. Real, freely available data only (no synthetic data).

Primary:  SQuAD v2.0 (Rajpurkar et al., 2018) from the public GitHub repo (answerable questions only).
Optional: BEIR corpora via Hugging Face `datasets` (SciFact, NFCorpus, ...) - needs HF access (Colab).
"""
from __future__ import annotations
import hashlib, json, re, string, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
from .utils import project_root

SQUAD_URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/{}-v2.0.json"

def _download(split: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}-v2.0.json"
    if not path.exists() or path.stat().st_size < 1_000_000:
        urllib.request.urlretrieve(SQUAD_URL.format(split), path)
    return path

def load_squad(split: str = "train", n_queries: int | None = 4000, n_articles: int | None = None,
               seed: int = 0, cache_dir: str | Path | None = None):
    """Return (corpus_df, queries_df). Corpus unit = one SQuAD paragraph (passage).

    corpus_df : pid, article_id, title, text
    queries_df: qid, question, answers(list[str]), rel_pids(list[int]), article_id
    Only answerable questions are kept. If n_articles is set, the corpus is restricted to a random
    subset of articles (controls corpus size / difficulty). n_queries samples questions uniformly.
    """
    cache = Path(cache_dir) if cache_dir else project_root() / "data" / "raw"
    raw = json.load(open(_download(split, cache), encoding="utf-8"))["data"]
    rng = np.random.RandomState(seed)
    art_idx = list(range(len(raw)))
    if n_articles and n_articles < len(raw):
        art_idx = sorted(rng.choice(art_idx, n_articles, replace=False).tolist())
    docs, qs, pid = [], [], 0
    for ai in art_idx:
        art = raw[ai]
        for para in art["paragraphs"]:
            docs.append({"pid": pid, "article_id": ai, "title": art["title"], "text": para["context"]})
            for qa in para["qas"]:
                if qa.get("is_impossible"): continue
                answers = sorted({a["text"] for a in qa["answers"]})
                qs.append({"qid": qa["id"], "question": qa["question"], "answers": answers,
                           "rel_pids": [pid], "article_id": ai})
            pid += 1
    corpus, queries = pd.DataFrame(docs), pd.DataFrame(qs)
    if n_queries and n_queries < len(queries):
        queries = queries.sample(n=n_queries, random_state=seed).sort_values("qid").reset_index(drop=True)
    return corpus, queries

def load_beir_hf(name: str = "scifact", n_queries: int | None = None, seed: int = 0):
    """BEIR dataset through Hugging Face (BeIR/<name>, BeIR/<name>-qrels). Requires internet access to HF.
    Returns the same (corpus_df, queries_df) schema; `answers` is empty (no answer strings in BEIR)."""
    from datasets import load_dataset
    c = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    q = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qr = load_dataset(f"BeIR/{name}-qrels")["test"]
    corpus = pd.DataFrame({"pid": range(len(c)), "doc_id": c["_id"], "title": c["title"], "text": c["text"]})
    corpus["article_id"] = corpus["pid"]
    id2pid = dict(zip(corpus["doc_id"], corpus["pid"]))
    rel = {}
    for r in qr:
        if int(r["score"]) > 0 and str(r["corpus-id"]) in id2pid:
            rel.setdefault(str(r["query-id"]), []).append(id2pid[str(r["corpus-id"])])
    qdf = pd.DataFrame({"qid": q["_id"], "question": q["text"]})
    qdf = qdf[qdf["qid"].isin(rel)].copy()
    qdf["rel_pids"] = qdf["qid"].map(rel); qdf["answers"] = [[] for _ in range(len(qdf))]
    qdf["article_id"] = qdf["qid"].map(lambda s: int(hashlib.md5(str(s).encode()).hexdigest(), 16) % 10**6)
    if n_queries and n_queries < len(qdf):
        qdf = qdf.sample(n=n_queries, random_state=seed).reset_index(drop=True)
    return corpus, qdf.reset_index(drop=True)

def split_by_group(queries: pd.DataFrame, group: str = "article_id", fractions=(0.6, 0.2, 0.2),
                   seed: int = 0) -> pd.Series:
    """Leakage-safe split: all questions of one article go to the same split (train/val/test)."""
    groups = np.array(sorted(queries[group].unique()))
    rng = np.random.RandomState(seed); rng.shuffle(groups)
    n = len(groups); a = int(round(fractions[0]*n)); b = a + int(round(fractions[1]*n))
    m = {g: "train" for g in groups[:a]}; m.update({g: "val" for g in groups[a:b]}); m.update({g: "test" for g in groups[b:]})
    return queries[group].map(m)

# ---- answer normalisation (SQuAD official style) ----
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def corpus_stats(corpus: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    wl = corpus["text"].str.split().str.len()
    ql = queries["question"].str.split().str.len()
    rows = [("n_passages", len(corpus)), ("n_articles", corpus["article_id"].nunique()), ("n_queries", len(queries)),
            ("passage_words_mean", wl.mean()), ("passage_words_median", wl.median()), ("passage_words_p95", wl.quantile(.95)),
            ("passage_words_max", wl.max()), ("query_words_mean", ql.mean()), ("query_words_median", ql.median()),
            ("rel_passages_per_query_mean", queries["rel_pids"].str.len().mean())]
    return pd.DataFrame(rows, columns=["statistic", "value"])
