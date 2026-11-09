"""Generate the Colab/Jupyter notebooks (nbformat). Run: python scripts/build_notebooks.py"""
import nbformat as nbf, os
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks"); os.makedirs(OUT, exist_ok=True)

SETUP = '''# ---- Environment setup (works in Colab AND in a local venv / Jupyter) ----
import os, sys, pathlib, subprocess, zipfile
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu", "sentence-transformers", "python-docx", "psutil"], check=False)
    if not pathlib.Path("adaptive_rag").exists():
        from google.colab import files
        print("Upload rag_adaptive_project.zip (the project archive you downloaded)")
        up = files.upload(); z = [k for k in up if k.endswith(".zip")][0]
        zipfile.ZipFile(z).extractall("."); 
        root = [p for p in pathlib.Path(".").iterdir() if (p / "adaptive_rag").exists()][0]; os.chdir(root)
    try:
        from google.colab import drive; drive.mount("/content/drive")      # results persist on your Drive
        os.environ["RAG_RESULTS_DIR"] = "/content/drive/MyDrive/rag_adaptive_results"
    except Exception as e:
        os.environ["RAG_RESULTS_DIR"] = os.path.abspath("results")
    BACKEND = "st"          # real neural embeddings on Colab (GPU recommended: Runtime > Change runtime type > T4)
else:
    here = pathlib.Path.cwd()
    root = here.parent if here.name == "notebooks" else here
    sys.path.insert(0, str(root)); os.chdir(root)
    os.environ.setdefault("RAG_RESULTS_DIR", str(root / "results_notebooks"))
    BACKEND = "lsa"         # offline CPU baseline (TF-IDF+SVD). Set to "st" if sentence-transformers can download models.
sys.path.insert(0, os.getcwd())
CFG = dict(backend=BACKEND, st_model="sentence-transformers/all-MiniLM-L6-v2", n_queries=int(os.environ.get("RAG_NQ", 10000)), seed=0)
print("backend:", BACKEND, "| results ->", os.environ["RAG_RESULTS_DIR"])
'''
def nb(name, cells):
    n = new_notebook(cells=cells); n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, os.path.join(OUT, name))

nb("01_data_and_setup.ipynb", [
md("# 01 - Data and setup\n**Goal.** Load a free, real dataset (SQuAD v2.0, answerable questions), document it, and create leakage-safe train / calibration / test splits.\n\n**Why SQuAD?** Every question has a gold passage (`rel_pids`) and gold answer strings, so retrieval recall *and* answer-in-context can be measured without an LLM judge. Freely downloadable from GitHub (no login). Optional BEIR corpora (SciFact, NFCorpus) are supported via `load_beir_hf` (needs Hugging Face access, e.g. Colab).\n\n**Outputs:** `csv/dataset_stats.csv`, `csv/split_sizes.csv`, `figures/data_lengths.png`"),
code(SETUP),
code('''import numpy as np, pandas as pd, matplotlib.pyplot as plt
from adaptive_rag.data import load_squad, corpus_stats, split_by_group
from adaptive_rag.utils import results_dir, env_info, save_json, set_seed
set_seed(0)
corpus, queries = load_squad("train", n_queries=CFG["n_queries"], seed=0)
stats = corpus_stats(corpus, queries); stats.to_csv(results_dir("csv")/"dataset_stats.csv", index=False); display(stats)'''),
code('''queries["split"] = split_by_group(queries).to_numpy()
sp = queries.groupby("split").agg(n_queries=("qid","size"), n_articles=("article_id","nunique")).reset_index()
sp.to_csv(results_dir("csv")/"split_sizes.csv", index=False); display(sp)
# leakage check: no article appears in two splits
assert queries.groupby("article_id")["split"].nunique().max() == 1; print("no article-level leakage between splits")'''),
code('''fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].hist(corpus.text.str.split().str.len(), bins=60, color="#2a78d6"); ax[0].set_title("Passage length (words)")
ax[1].hist(queries.question.str.split().str.len(), bins=range(0, 40), color="#eb6834"); ax[1].set_title("Question length (words)")
plt.tight_layout(); plt.savefig(results_dir("figures")/"data_lengths.png", dpi=300); plt.show()
queries.sample(5, random_state=0)[["question","answers"]]'''),
code('''save_json(env_info(), results_dir("logs")/"environment_notebook01.json"); env_info()'''),
])

nb("02_retrieval_baselines.ipynb", [
md("# 02 - Retrieval back-ends and baseline recall\nBuilds three candidate lists per query (top-50, exact FAISS / BM25 search):\n* **dense** - `sentence-transformers` embeddings on Colab (or LSA offline)\n* **bm25** - Okapi BM25 (k1=1.2, b=0.75)\n* **hybrid_rrf** - Reciprocal Rank Fusion of both (Cormack et al., 2009)\n\nMeasures Recall@k / Hit@k, MRR, answer-in-context and index/search times. **Outputs:** `csv/retrieval_baselines.csv`, `figures/recall_at_k.png`"),
code(SETUP),
code('''import numpy as np, pandas as pd, matplotlib.pyplot as plt
from adaptive_rag import experiment as E
from adaptive_rag.utils import results_dir
B = E.prepare_cached(CFG)
pd.Series(B.timing, name="seconds").to_frame().to_csv(results_dir("csv")/"build_timing.csv"); display(pd.Series(B.timing, name="seconds"))'''),
code('''rows = []
for name in B.lists:
    for k in [1,3,5,10,20,50]:
        rows.append({"retriever": name, "k": k, "hit_rate": B.rel[name][:, :k].any(1).mean(), "answer_in_context": B.ans[name][:, :k].any(1).mean(),
                     "mean_rr@50": np.where(B.rank[name] <= 50, 1/B.rank[name], 0).mean() if k == 50 else np.nan})
rb = pd.DataFrame(rows); rb.to_csv(results_dir("csv")/"retrieval_baselines.csv", index=False)
display(rb.pivot(index="retriever", columns="k", values="hit_rate").round(3))
fig, ax = plt.subplots(figsize=(6.4, 4))
for name in B.lists:
    d = rb[rb.retriever == name]; ax.plot(d.k, d.hit_rate, "-o", label=name)
ax.set_xlabel("k"); ax.set_ylabel("gold-passage hit rate"); ax.set_title("Hit@k by retriever"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(results_dir("figures")/"recall_at_k.png", dpi=300); plt.show()'''),
md("**Reading the result.** The distribution of the *rank of the first gold passage* is what an adaptive-k method must predict. If most queries have rank 1-3 but a long tail exists, a fixed k wastes tokens on easy queries or misses hard ones. Below: the empirical distribution that motivates the study."),
code('''r = B.rank["hybrid_rrf"]
q = pd.Series({f"rank<= {k}": (r <= k).mean() for k in [1,2,3,5,10,20,50]}); q.to_csv(results_dir("csv")/"gold_rank_cdf_hybrid.csv"); display(q.round(3))
plt.figure(figsize=(6.4,3.8)); plt.hist(np.minimum(r, 51), bins=np.arange(1, 53)-.5, color="#2a78d6"); plt.yscale("log")
plt.xlabel("rank of gold passage (51 = not in top-50)"); plt.ylabel("queries (log)"); plt.title("Heavy-tailed gold rank (hybrid)")
plt.tight_layout(); plt.savefig(results_dir("figures")/"gold_rank_hist.png", dpi=300); plt.show()'''),
])

nb("03_adaptive_methods.ipynb", [
md('''# 03 - Adaptive retrieval depth: baselines and the proposed methods
| Family | Method | Idea |
|---|---|---|
| fixed | top-k | same k for every query |
| rule | relative score / z-score | keep passages above a threshold |
| prior art | **Adaptive-k** (Taguchi et al., 2025) | cut at the largest gap in the sorted similarity scores + buffer |
| prior art | **CAR-lite** (after Xu et al., 2025) | 2-means on scores, cut at cluster boundary |
| **proposed** | **RCAK** - Risk-Controlled Adaptive-k | gradient-boosted *quantile* regression of log(rank of first gold passage) from score-distribution features, then **split-conformal calibration**. Guarantees P(gold rank <= k_hat) >= 1 - alpha under exchangeability; alpha is an interpretable dial. |
| **proposed** | **CGC** - Confidence-Gated Cascade | classifier predicts P(gold in top-3); serve 3 if confident else 20 |
| bound | Oracle | smallest k containing the gold passage |

Features use **only retrieval-score statistics, dense/BM25 agreement and the question wh-word** - no LLM call, so the controller costs milliseconds.

**Outputs:** `csv/strategy_results.csv`, `csv/feature_matrix_sample.csv`'''),
code(SETUP),
code('''import numpy as np, pandas as pd
from adaptive_rag import experiment as E
from adaptive_rag.features import feature_groups
from adaptive_rag.utils import results_dir
B = E.prepare_cached(CFG)
B.feat.assign(split=B.split).head(200).to_csv(results_dir("csv")/"feature_matrix_sample.csv", index=False)
feature_groups()'''),
code('''df, perq, te = E.evaluate(B, return_per_query=True)
df.to_csv(results_dir("csv")/"strategy_results.csv", index=False)
cols = ["strategy","param","mean_k","hit_rate","answer_in_context","mean_ctx_tokens_est"]
display(df[df.serve_retriever=="hybrid_rrf"][cols].round(3).sort_values(["strategy","mean_k"]).reset_index(drop=True))'''),
md("### Matched-budget view (headline table)\nFor each adaptive method we find the fixed top-k with the **same mean number of passages** (linear interpolation along the fixed-k curve) and report the difference."),
code('''iso = E.isobudget(B, "hybrid_rrf", n_boot=500); iso.to_csv(results_dir("csv")/"isobudget_hybrid.csv", index=False)
best = iso[iso.metric=="answer_in_context"].sort_values("diff", ascending=False).head(12)
display(best[["strategy","param","mean_k","adaptive_value","fixed_value_at_same_mean_k","diff","ci95_lo","ci95_hi"]].round(3))'''),
])

nb("04_evaluation_figures.ipynb", [
md("# 04 - Full evaluation, statistics and publication figures\nRuns every strategy on **all three candidate lists**, with 95% cluster-bootstrap CIs (resampling *articles*), the conformal validity check, and exports PNG (300 dpi) + PDF + the plotted data as CSV.\n\n**Outputs:** `csv/strategy_results.csv`, `csv/isobudget_bootstrap.csv`, `csv/rcak_coverage.csv`, `figures/pareto_*`, `figures/isobudget_*`, `figures/coverage_*`, `figures/k_distribution.*`"),
code(SETUP),
code('''import pandas as pd, numpy as np
from adaptive_rag import experiment as E, plots as P
from adaptive_rag.utils import results_dir
B = E.prepare_cached(CFG)
df, perq, te = E.evaluate(B, return_per_query=True); df.to_csv(results_dir("csv")/"strategy_results.csv", index=False)
cov = E.coverage(B); cov.to_csv(results_dir("csv")/"rcak_coverage.csv", index=False)
iso = pd.concat([E.isobudget(B, s, n_boot=1000) for s in B.lists]); iso.to_csv(results_dir("csv")/"isobudget_bootstrap.csv", index=False)
for s in B.lists:
    P.fig_pareto(df, s, "answer_in_context"); P.fig_pareto(df, s, "hit_rate"); P.fig_coverage(cov, s); P.fig_isobudget(iso, s)
serve = "hybrid_rrf"
P.fig_k_hist({"RCAK alpha=0.10": (perq[(serve,"RCAK (proposed)","alpha=0.1")][0], "RCAK (proposed)"),
              "Adaptive-k buffer=5": (perq[(serve,"adaptive_k_gap","buffer=5")][0], "adaptive_k_gap"),
              "Oracle": (perq[(serve,"oracle_k","min_k")][0], "oracle_k")}, B.rank[serve][te])
print("figures written to", results_dir("figures"))'''),
code('''from IPython.display import Image, display
for f in ["pareto_answer_in_context_hybrid_rrf", "coverage_hybrid_rrf", "k_distribution"]:
    display(Image(str(results_dir("figures")/f"{f}.png"), width=620))'''),
md("**Interpretation guide for the thesis.** (1) Pareto plot: a method is better if its curve is up-and-left of fixed top-k. (2) Iso-budget plot: points right of zero with CIs not crossing zero are significant gains. (3) Coverage plot: achieved coverage should sit on/above the diagonal until it hits the retriever's recall ceiling (pool recall@50)."),
])

nb("05_ablation_robustness_latency.ipynb", [
md("# 05 - Ablation, seed robustness, latency, evidence manifest\n* Which feature group drives RCAK? (drop-one / only-one ablation, iso-budget metric)\n* Is the gain stable across 5 different article-level splits **and** query samples (retrievers refit each time)?\n* What does the controller cost in milliseconds per query?\n* SHA-256 manifest of every output file (tamper-evident evidence trail for the thesis).\n\n**Outputs:** `csv/ablation.csv`, `csv/seed_robustness.csv`, `csv/latency.csv`, `MANIFEST.csv`"),
code(SETUP),
code('''import pandas as pd
from adaptive_rag import experiment as E, plots as P
from adaptive_rag.utils import results_dir, write_manifest, log_event
B = E.prepare_cached(CFG)
ab = E.ablation(B, n_boot=300); ab.to_csv(results_dir("csv")/"ablation.csv", index=False); P.fig_ablation(ab); display(ab.round(3))'''),
code('''lat = E.latency(B); lat.to_csv(results_dir("csv")/"latency.csv", index=False); P.fig_latency(lat); display(lat.round(3))'''),
code('''sd = E.seeds(CFG, seed_list=(0,1,2,3,4)); sd.to_csv(results_dir("csv")/"seed_robustness.csv", index=False); P.fig_seeds(sd)
display(sd.groupby("strategy")[["mean_k","answer_in_context","fixed_at_same_k","diff"]].agg(["mean","std"]).round(3))'''),
code('''m = write_manifest(); log_event("notebook05_done"); display(pd.read_csv(m).head(20))'''),
])

nb("06_generation_with_llm_colab.ipynb", [
md('''# 06 - (Optional, Colab GPU) End-to-end generation with a free open LLM
Closes the loop from *retrieval depth* to **answer quality and real token usage**. Runs a small free reader (default `Qwen/Qwen2.5-0.5B-Instruct`; swap for `google/flan-t5-large`, `Qwen2.5-1.5B-Instruct`, ...) on **fixed top-k vs RCAK vs Adaptive-k**, logging per-query Exact Match, token-F1, prompt tokens (real tokenizer counts) and latency.

> This notebook needs Hugging Face model downloads, which are **blocked in the offline sandbox where the other notebooks were executed**, so it was *not* executed there. Run it on Colab (Runtime > T4 GPU). Use `N_EVAL = 300-500` to stay within free-tier limits.

**Outputs:** `csv/generation_per_query.csv`, `csv/generation_summary.csv`, `figures/generation_tradeoff.png`'''),
code(SETUP),
code('''N_EVAL = 400; MODEL = "Qwen/Qwen2.5-0.5B-Instruct"; MAX_NEW = 24
import numpy as np, pandas as pd, time, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from adaptive_rag import experiment as E
from adaptive_rag.data import normalize_answer
from adaptive_rag.features import feature_groups
from adaptive_rag.strategies import fixed_k, adaptive_k_gap
from adaptive_rag.utils import results_dir
B = E.prepare_cached(CFG)
tok = AutoTokenizer.from_pretrained(MODEL); lm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to("cuda" if torch.cuda.is_available() else "cpu").eval()'''),
code('''serve = "hybrid_rrf"; R = B.lists[serve]; te = np.where(B.split == "test")[0][:N_EVAL]
cols = [c for g in feature_groups().values() for c in g]
models = E.train_models(B, cols, 0); rc, Xca, rca = models[("rcak", serve)]; rc.calibrate(Xca, rca, 0.10)
K = {"fixed k=3": fixed_k(len(te), 3), "fixed k=10": fixed_k(len(te), 10), "fixed k=20": fixed_k(len(te), 20),
     "Adaptive-k (buffer 5)": adaptive_k_gap(R.scores[te], buffer=5), "RCAK alpha=0.10": rc.predict_k(B.feat.loc[te, cols].to_numpy())}
def f1(pred, golds):
    best = 0.0
    for g in golds:
        p, t = normalize_answer(pred).split(), normalize_answer(g).split()
        c = sum(min(p.count(w), t.count(w)) for w in set(p))
        if c: pr, rc_ = c/len(p), c/len(t); best = max(best, 2*pr*rc_/(pr+rc_))
    return best
@torch.no_grad()
def answer(q, passages):
    ctx = "\\n\\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    msgs = [{"role":"system","content":"Answer the question using ONLY the passages. Reply with a short answer span."},
            {"role":"user","content":f"Passages:\\n{ctx}\\n\\nQuestion: {q}\\nAnswer:"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(lm.device)
    t = time.perf_counter(); out = lm.generate(ids, max_new_tokens=MAX_NEW, do_sample=False)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), ids.shape[1], time.perf_counter() - t
rows = []
for name, ks in K.items():
    for j, i in enumerate(te):
        q = B.queries.question.iloc[i]; gold = B.queries.answers.iloc[i]
        passages = B.corpus.text.iloc[R.ids[i, :ks[j]]].tolist(); pred, ntok, dt = answer(q, passages)
        em = int(any(normalize_answer(pred) == normalize_answer(g) for g in gold))
        rows.append({"method": name, "qid": B.queries.qid.iloc[i], "k": int(ks[j]), "prompt_tokens": ntok, "latency_s": dt, "EM": em, "F1": f1(pred, gold),
                     "answer_in_context": int(B.ans[serve][i, :ks[j]].any()), "pred": pred})
gen = pd.DataFrame(rows); gen.to_csv(results_dir("csv")/"generation_per_query.csv", index=False)
summ = gen.groupby("method").agg(mean_k=("k","mean"), prompt_tokens=("prompt_tokens","mean"), latency_s=("latency_s","mean"), EM=("EM","mean"), F1=("F1","mean"), answer_in_context=("answer_in_context","mean")).reset_index()
summ.to_csv(results_dir("csv")/"generation_summary.csv", index=False); display(summ.round(3))'''),
code('''import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 4))
for _, r in summ.iterrows(): ax.scatter(r.prompt_tokens, r.F1, s=60); ax.annotate(r.method, (r.prompt_tokens, r.F1), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("mean prompt tokens (real tokenizer)"); ax.set_ylabel("token F1"); ax.set_title("Answer quality vs prompt cost"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(results_dir("figures")/"generation_tradeoff.png", dpi=300); plt.show()'''),
])
print("notebooks written:", sorted(os.listdir(OUT)))
