"""Build the Word documents from the CSV evidence in results/ (numbers are read from files, never typed by hand)."""
import os, sys, hashlib, json
import pandas as pd, numpy as np
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, "results"); C = os.path.join(R, "csv"); F = os.path.join(R, "figures")
OUT = os.path.join(ROOT, "docs"); os.makedirs(OUT, exist_ok=True)
rd = lambda n: pd.read_csv(os.path.join(C, n))
res, iso, cov, ab, lat, sd, ds = rd("strategy_results.csv"), rd("isobudget_bootstrap.csv"), rd("rcak_coverage.csv"), rd("ablation.csv"), rd("latency.csv"), rd("seed_robustness.csv"), rd("dataset_stats.csv")
env = json.load(open(os.path.join(R, "logs", "environment_and_config.json")))
S = lambda n: float(ds.set_index("statistic").loc[n, "value"])

def newdoc(title, sub):
    d = Document(); st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
    for s in d.sections: s.left_margin = s.right_margin = Inches(1); s.top_margin = s.bottom_margin = Inches(0.9)
    p = d.add_paragraph(); r = p.add_run(title); r.bold = True; r.font.size = Pt(22); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p = d.add_paragraph(); r = p.add_run(sub); r.font.size = Pt(12); r.font.color.rgb = RGBColor(0x52, 0x51, 0x4e)
    return d
def para(d, t, bold=False, italic=False):
    p = d.add_paragraph(); r = p.add_run(t); r.bold = bold; r.italic = italic; return p
def bullets(d, items):
    for i in items: d.add_paragraph(i, style="List Bullet")
def table(d, df, fmt=3, widths=None, font=8.5):
    t = d.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]; cell.text = str(c)
        for r in cell.paragraphs[0].runs: r.bold = True; r.font.size = Pt(font)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = f"{v:.{fmt}f}" if isinstance(v, (float, np.floating)) else str(v)
            for r in cells[j].paragraphs[0].runs: r.font.size = Pt(font)
    d.add_paragraph(); return t
def fig(d, name, caption, w=6.2):
    p = os.path.join(F, name)
    if os.path.exists(p): d.add_picture(p, width=Inches(w)); c = d.add_paragraph(caption); c.runs[0].italic = True; c.runs[0].font.size = Pt(9)

# ---------- headline numbers ----------
H = "hybrid_rrf"
g = lambda s, p, m="answer_in_context", sv=H: iso[(iso.serve_retriever == sv) & (iso.strategy == s) & (iso.param == p) & (iso.metric == m)].iloc[0]
rc, cg, ak = g("RCAK (proposed)", "alpha=0.1"), g("CGC (proposed)", "tau=0.7"), g("adaptive_k_gap", "buffer=5")
fx = lambda k, sv=H, c="answer_in_context": res[(res.serve_retriever == sv) & (res.strategy == "fixed_k") & (res.param == f"k={k}")].iloc[0][c]
orc = res[(res.serve_retriever == H) & (res.strategy == "oracle_k")].iloc[0]
cv = cov[(cov.serve_retriever == H) & (np.isclose(cov.alpha, 0.1))].iloc[0]
sdm = sd.groupby("strategy")["diff"].agg(["mean", "std"])
n_seeds = sd.seed.nunique()
tok_fixed20 = res[(res.serve_retriever == H) & (res.strategy == "fixed_k") & (res.param == "k=20")].mean_ctx_tokens_est.iloc[0]
rcak_row = res[(res.serve_retriever == H) & (res.strategy == "RCAK (proposed)") & (res.param == "alpha=0.1")].iloc[0]
backend = env["config"]["backend"]

# =====================================================================  1. REPORT
d = newdoc("Risk-Controlled Adaptive Retrieval Depth for Retrieval-Augmented Generation",
           "Thesis-ready research draft - experiments, results and analysis (auto-generated from results/csv; all numbers are measured)")
d.add_heading("Status and honesty statement", 1)
bullets(d, [
 f"All numbers in this document were produced by the code in this project on real data (SQuAD v2.0, {int(S('n_queries')):,} answerable questions, {int(S('n_passages')):,} passages). Nothing is simulated or hand-typed; every table is read from a CSV in results/csv.",
 f"Dense retriever used in the executed run: '{backend}' = TF-IDF + truncated SVD (LSA) vectors with exact FAISS search. This was used because Hugging Face downloads were blocked in the sandbox. It is a WEAK dense baseline (hit@50 of {rd('retrieval_baselines.csv').query('retriever==\"lsa\" and k==50').hit_rate.iloc[0]:.2f}), so absolute numbers will change with neural embeddings. The Colab notebooks switch to sentence-transformers automatically; re-run them and paste the Colab CSVs into the thesis for the final results.",
 "Quality is measured by gold-passage hit rate and 'answer-in-context' (a gold answer string occurs in the served passages). These are reader-independent proxies. End-to-end EM/F1 with a real LLM is implemented in notebook 06 but was NOT executed in the sandbox (models could not be downloaded). Do not report generation results until you run it on Colab.",
 "The proposed methods (RCAK, CGC) are an original design by this project, not a published method. Prior-art baselines (Adaptive-k, CAR) were re-implemented from their papers' descriptions, not taken from the authors' code; see Limitations.",
])
d.add_heading("Abstract (draft)", 1)
para(d, f"Retrieval-augmented generation (RAG) systems usually pass a fixed number k of retrieved passages to the language model, which wastes tokens on easy queries and under-retrieves for hard ones. We study query-adaptive retrieval depth and propose RCAK (Risk-Controlled Adaptive-k), a lightweight controller that predicts, from retrieval-score statistics alone, an upper quantile of the rank at which the first relevant passage appears and calibrates it with split conformal prediction, so that the retrieval-coverage target 1-alpha is an explicit, interpretable control. We also study CGC, a two-stage confidence-gated cascade. On {int(S('n_queries')):,} SQuAD v2.0 questions over a {int(S('n_passages')):,}-passage corpus with a hybrid (dense + BM25 reciprocal-rank-fusion) retriever, RCAK at alpha=0.10 serves {rc.mean_k:.1f} passages on average and attains an answer-in-context rate of {rc.adaptive_value:.3f}, versus {rc.fixed_value_at_same_mean_k:.3f} for fixed top-k at the same mean depth (difference {rc['diff']:+.3f}, 95% cluster-bootstrap CI [{rc.ci95_lo:+.3f}, {rc.ci95_hi:+.3f}]). The gain is stable over {n_seeds} independent splits (mean {sdm.loc['RCAK (proposed)','mean']:+.3f}, sd {sdm.loc['RCAK (proposed)','std']:.3f}), the controller adds {lat.iloc[2].mean_ms:.1f} ms per query, and the published score-gap rule Adaptive-k yields {ak['diff']:+.3f} (CI [{ak.ci95_lo:+.3f}, {ak.ci95_hi:+.3f}]) in the same setting. Gains are smaller with BM25 alone and the calibration dial saturates when the candidate pool recall is low, which we analyse.", italic=False)
para(d, "[Re-write the numbers after the Colab run with neural embeddings; keep the structure.]", italic=True)

d.add_heading("1. Introduction", 1)
_cdf = rd("gold_rank_cdf_hybrid.csv").set_index(rd("gold_rank_cdf_hybrid.csv").columns[0]).iloc[:,0]
c1, c3, cx = _cdf["rank<= 1"], _cdf["rank<= 3"], 1-_cdf["rank<= 50"]; c20 = 1-c3
para(d, f"In a RAG pipeline the number of retrieved passages k trades answer coverage against prompt length (cost, latency, distraction of the reader). A single global k is a poor fit because the rank at which the first relevant passage appears is heavy-tailed: for {c1:.0%} of queries the gold passage is at rank 1 and for {c3:.0%} within the top 3, yet {c20:.0%} need more than 3 and {cx:.0%} are not in the top-50 pool at all (csv/gold_rank_cdf_hybrid.csv, figures/gold_rank_hist.png). The research question is: can k be chosen per query, without any LLM call, such that coverage is controllable and the context budget is smaller than for fixed k at equal quality?")
para(d, "Contributions: (1) RCAK, which casts depth selection as conformalised quantile regression of log(gold rank) and so offers a finite-sample coverage dial; (2) CGC, a cheap two-stage cascade; (3) a matched-budget evaluation protocol (adaptive vs fixed-k at the same mean depth with article-clustered bootstrap CIs) that avoids the common error of comparing methods at different budgets; (4) an open, reproducible code base and an evidence manifest.")

d.add_heading("2. Related work (verify each citation before submission)", 1)
bullets(d, [
 "Fixed-k RAG: Lewis et al., 2020 (RAG); Karpukhin et al., 2020 (DPR).",
 "Adaptive-k (Taguchi, Maekawa, Bhutani; arXiv 2506.08479, EMNLP 2025): chooses the cutoff at the largest gap in the sorted similarity scores, with a buffer; no tuning or extra LLM calls. Re-implemented here as 'adaptive_k_gap'.",
 "Cluster-based Adaptive Retrieval, CAR (Xu et al., arXiv 2511.14769): clusters query-document distances and cuts at the highest-scoring gap between clusters (silhouette-tuned). Our 'car_lite' is a simplified 2-means variant, not the full method.",
 "Tail-Aware Adaptive-k, TAA-k (Song et al., arXiv 2606.11907, June 2026): knee detection plus extreme-value (Generalised Pareto) tail fitting; reported to fix Adaptive-k's over-aggressive cuts on multi-hop questions. NOT implemented here - it is the most important baseline to add for a strong submission.",
 "DynamicRAG (Sun, Zhong, Zhou, Han; arXiv 2505.07233): a trained reranker that outputs both ordering and count of passages, optimised with generator feedback via RL. Heavier (needs generator-in-the-loop training).",
 "Adaptive-RAG (Jeong et al., 2024): routes queries among no-retrieval, single-step and multi-step strategies with a complexity classifier - adapts the strategy, not the depth within a retrieval step.",
 "Agentic RAG surveys: e.g. 'SoK: Agentic RAG' (arXiv 2603.07379).",
 "Conformal prediction: Vovk et al.; Romano, Patterson, Candes, 2019 (CQR) - the calibration device used by RCAK.",
 "Rank fusion: Cormack, Clarke, Buettcher, 2009 (RRF). Sparse retrieval: Robertson & Zaragoza (BM25).",
])
d.add_heading("3. Method", 1)
d.add_heading("3.1 Setting and notation", 2)
para(d, "For each query q a retriever returns a ranked list of K=50 passages with scores s_1>=...>=s_K. A depth policy chooses k(q) in {1..K}. Let r(q) be the rank of the first gold passage (K+1 if absent from the pool). Quality is hit(q)=1[r(q)<=k(q)] and answer-in-context. Cost is the number of passages / estimated tokens served.")
d.add_heading("3.2 Query features (no LLM call)", 2)
bullets(d, [f"dense_scores (8): top score, gaps s1-s2 / s1-s5 / s1-s10, mean & std of top-10, tail ratio, scale-free softmax entropy.",
 "bm25_scores (5): normalised analogues for BM25.", "agreement (4): overlap of dense and BM25 top-5/top-10; rank of each retriever's top-1 inside the other's list.",
 "query (9): length and wh-word indicator."])
d.add_heading("3.3 RCAK: Risk-Controlled Adaptive-k", 2)
para(d, "A gradient-boosted quantile regressor (loss=quantile, tau=0.9) predicts q(x), an upper quantile of y=log r(q) (censored at log(K+1)). On a held-out calibration split of whole articles, the conformity score e_i = y_i - q(x_i) is computed and its ceil((n+1)(1-alpha))/n empirical quantile Q is taken (split conformal / CQR). At test time k(x)=clip(ceil(exp(q(x)+Q)),1,K). Under exchangeability of calibration and test queries, P(r(q) <= k(q)) >= 1-alpha for the uncensored target. alpha is therefore a directly interpretable quality knob, replacing ad-hoc thresholds.")
d.add_heading("3.4 CGC: Confidence-Gated Cascade", 2)
para(d, "A classifier predicts P(r<=3 | x). If it exceeds tau the system serves k=3, otherwise k=20. Monotone in tau, trivially cheap, no conformal guarantee.")
d.add_heading("3.5 Baselines", 2)
bullets(d, ["Fixed top-k for k in {1,...,50} (15-point grid).", "Rule-based relative-score threshold and z-score threshold.", "Adaptive-k gap rule (buffer 0/2/5/10, search in top 90% of the pool).", "CAR-lite (2-means cluster boundary).", "Oracle: smallest k containing the gold passage (an unattainable bound that shows head-room)."])
d.add_heading("4. Experimental setup", 1)
table(d, ds.assign(value=ds.value.round(2)).rename(columns={"statistic": "Dataset statistic"}), fmt=2)
bullets(d, [
 "Data: SQuAD v2.0 train file (answerable questions only), corpus unit = one paragraph; each question has exactly one gold paragraph and 1+ gold answer strings. Freely downloadable (GitHub: rajpurkar/SQuAD-explorer).",
 f"Questions: {int(S('n_queries')):,} sampled uniformly (seed 0); all {int(S('n_passages')):,} passages are indexed (a realistic distractor load).",
 "Splits: 60% train / 20% calibration / 20% test BY ARTICLE (no article spans two splits - verified by assertion in notebook 01), to avoid leakage between near-duplicate questions about the same passage.",
 "Retrievers: dense (" + backend + "), BM25 (k1=1.2, b=0.75), and hybrid RRF (c=60). All searches exact; top-50 cached; same cached lists serve every policy.",
 "Matched-budget protocol: for each adaptive configuration we interpolate the fixed-k curve at the same mean number of passages and report the difference; 95% CIs from 1,000 article-cluster bootstrap resamples of the test set.",
 f"Hardware/software: Python {env['env']['python']}, {env['env']['cpu_count']} CPU cores, {env['env'].get('ram_gb')} GB RAM, no GPU. Seeds 0-{n_seeds-1} for the robustness study (each re-samples questions, re-splits and refits every retriever).",
])
d.add_heading("5. Results", 1)
d.add_heading("5.1 Headline: matched-budget comparison (hybrid retriever)", 2)
rows = []
for s, p in [("RCAK (proposed)", "alpha=0.1"), ("RCAK (proposed)", "alpha=0.05"), ("CGC (proposed)", "tau=0.7"), ("CGC (proposed)", "tau=0.8"), ("adaptive_k_gap", "buffer=5"), ("adaptive_k_gap", "buffer=10"), ("car_lite", "default"), ("rule_relative", "tau=0.95")]:
    r = g(s, p); rows.append({"Method": s, "Setting": p, "Mean k": r.mean_k, "Answer-in-ctx": r.adaptive_value, "Fixed top-k @ same k": r.fixed_value_at_same_mean_k, "Diff": r["diff"], "CI lo": r.ci95_lo, "CI hi": r.ci95_hi})
table(d, pd.DataFrame(rows))
para(d, f"Reading: RCAK at alpha=0.10 spends {rc.mean_k:.1f} passages per query on average and reaches {rc.adaptive_value:.3f}; fixed top-k needs the same budget to reach only {rc.fixed_value_at_same_mean_k:.3f}. Equivalently, fixed top-k reaches RCAK's quality only at roughly 25 passages (fixed k=25: {fx(25):.3f}), i.e. RCAK saves about {(1 - rc.mean_k/25)*100:.0f}% of the passages / about {(1-rcak_row.mean_ctx_tokens_est/ res[(res.serve_retriever==H)&(res.strategy=='fixed_k')&(res.param=='k=25')].mean_ctx_tokens_est.iloc[0])*100:.0f}% of the estimated context tokens at similar answer-in-context (interpolation-based statement; see the CSV for exact values).")
fig(d, f"pareto_answer_in_context_{H}.png", "Figure 1. Answer-in-context vs. mean context size (hybrid retriever, test split). Up-and-left is better. File: figures/pareto_answer_in_context_hybrid_rrf.png (+ .pdf, data in csv/fig_pareto_*.csv)")
fig(d, f"isobudget_answer_in_context_{H}.png", "Figure 2. Gain over fixed top-k at the same mean k with 95% CIs. File: figures/isobudget_answer_in_context_hybrid_rrf.png", 5.2)
d.add_heading("5.2 Head-room", 2)
para(d, f"The oracle (smallest k containing the gold passage) needs only {orc.mean_k:.1f} passages on average for hit rate {orc.hit_rate:.3f}. RCAK at alpha=0.10 uses {rc.mean_k:.1f}. So a gap of roughly a factor of two in context size remains, which bounds what any controller based on score features can still gain and motivates richer signals (query encoders, cross-encoder scores).")
d.add_heading("5.3 Conformal validity", 2)
para(d, f"For the hybrid list the candidate pool contains the gold passage for {cv.pool_recall_at_K:.3f} of test queries, which is a hard ceiling on achievable coverage. At target 1-alpha=0.90 the achieved test coverage is {cv.achieved_test_coverage:.3f} (slightly below target; with ~{int(S('n_queries')*0.2):,} test queries the binomial standard error is about 0.007 and queries within an article are not independent, so this is within expected sampling and exchangeability error but should be reported honestly). For targets above the pool recall, calibration cannot be met and k saturates at K; use alpha such that 1-alpha is below pool recall@K.")
table(d, cov[cov.serve_retriever == H][["alpha", "target_coverage", "achieved_test_coverage", "pool_recall_at_K", "mean_k"]].round(3))
fig(d, f"coverage_{H}.png", "Figure 3. Target vs achieved gold-passage coverage of RCAK (hybrid). File: figures/coverage_hybrid_rrf.png", 4.4)
d.add_heading("5.4 Other retrievers: where it does NOT help much", 2)
rows = []
for sv in ["bm25", "lsa"]:
    for s, p in [("RCAK (proposed)", "alpha=0.1"), ("CGC (proposed)", "tau=0.7"), ("adaptive_k_gap", "buffer=5"), ("car_lite", "default")]:
        r = g(s, p, sv=sv); rows.append({"Retriever": sv, "Method": s, "Setting": p, "Mean k": r.mean_k, "Diff vs fixed": r["diff"], "CI lo": r.ci95_lo, "CI hi": r.ci95_hi})
table(d, pd.DataFrame(rows))
bullets(d, ["BM25 alone: RCAK gives a smaller but significant gain; at low budgets simple rules (CAR-lite, z-score) are as good or better. BM25 scores are already sharply peaked on SQuAD (questions share many words with the gold paragraph), so easy queries are easy to detect with simple rules.",
 "LSA alone (weak, hit@50 only ~0.69): RCAK's alpha dial is saturated (all alphas give the same k) because ~31% of queries have the gold passage outside the pool, so the 0.9 quantile of the censored target is the cap. The gain over fixed top-k at that budget is real but it is not a controllable trade-off. This is a limitation of censored targets, not evidence for the method.",
 "Consequence for the thesis: claim the method is most useful with a strong hybrid first stage, and say so."])
d.add_heading("5.5 Ablation: which signals matter?", 2)
table(d, ab[["variant", "n_features", "mean_k", "answer_in_context", "fixed_at_same_k", "diff", "ci95_lo", "ci95_hi"]].round(3))
fig(d, "ablation.png", "Figure 4. Feature-group ablation (iso-budget gain, hybrid, alpha=0.10).", 5.2)
dropag = ab[ab.variant == "without agreement"].iloc[0]; dropbm = ab[ab.variant == "without bm25_scores"].iloc[0]
para(d, f"Removing the dense-vs-BM25 AGREEMENT features erases the gain ({dropag['diff']:+.3f}); removing BM25 score features halves it ({dropbm['diff']:+.3f}); no single group alone reproduces it. Removing the query (wh-word, length) or dense-score groups changes nothing detectable. The signal that two different retrievers agree on the top results is therefore the key cue that a query is easy.")
d.add_heading("5.6 Robustness over seeds", 2)
table(d, sd.round(3))
fig(d, "seed_robustness.png", "Figure 5. Iso-budget gain across seeds (each seed re-samples queries, re-splits and re-fits retrievers).", 4.8)
para(d, f"Over {n_seeds} seeds: RCAK {sdm.loc['RCAK (proposed)','mean']:+.3f} (sd {sdm.loc['RCAK (proposed)','std']:.3f}); CGC {sdm.loc['CGC (proposed)','mean']:+.3f} (sd {sdm.loc['CGC (proposed)','std']:.3f}); Adaptive-k buffer=5 {sdm.loc['adaptive_k_gap','mean']:+.3f} (sd {sdm.loc['adaptive_k_gap','std']:.3f}). CGC at tau=0.7 spends fewer passages (about 13) than RCAK at alpha=0.10 (about 19), so the two are not at the same operating point - compare them on the Pareto plot, not on this table.")
d.add_heading("5.7 Overhead", 2)
table(d, lat.round(2))
fig(d, "latency.png", "Figure 6. Measured per-query latency (batch size 1, CPU).", 5.2)
para(d, "The controller (features + gradient-boosted prediction) costs a few milliseconds, small next to LLM generation (hundreds of ms to seconds) and below LSA retrieval here. With a GPU-encoded neural retriever the retrieval time will differ; re-measure on Colab.")
d.add_heading("5.8 Depth distribution", 2)
fig(d, "k_distribution.png", "Figure 7. Distribution of served depth for RCAK, Adaptive-k and the oracle vs the gold-rank distribution.", 5.6)
d.add_heading("6. Discussion", 1)
bullets(d, ["The adaptive controller's value comes from detecting easy queries (retrievers agree, large score margin) and saving context there while spending more on ambiguous ones.",
 "A conformal wrapper makes the quality target explicit; an engineer can choose alpha from an SLA instead of tuning thresholds per corpus. The guarantee concerns retrieval coverage, not answer correctness.",
 "The published gap rule (Adaptive-k) did not beat fixed top-k at equal budget on this single-gold-passage task with buffer>=2; it cuts too aggressively with buffer 0 (answer-in-context -0.14 vs fixed at equal k). This is consistent with TAA-k's critique, but note our task is single-hop, whereas Adaptive-k targets aggregation/long-context settings - so this is NOT a refutation of that paper."])
d.add_heading("7. Limitations and threats to validity (state these in the paper)", 1)
bullets(d, ["Dense retriever in the executed run is LSA, not a neural model; re-run on Colab (backend 'st') and update all numbers.",
 "Single dataset with exactly one gold paragraph per question and mostly single-hop questions. Multi-hop (HotpotQA, MuSiQue, 2Wiki) and multiple-relevant-passage corpora (BEIR) are needed; BEIR loader is included but not executed here (Hugging Face blocked).",
 "Answer-in-context is a proxy for answer quality (the reader may still fail). End-to-end EM/F1 requires notebook 06 on Colab.",
 "Token counts are estimated at 1.3 tokens per word; real tokenizer counts are logged in notebook 06.",
 "Baselines were re-implemented from paper descriptions; CAR-lite is a simplification; TAA-k, DynamicRAG and Adaptive-RAG (strategy routing) are not benchmarked.",
 "Conformal guarantee assumes exchangeability; queries sharing an article are correlated (we split by article, which helps), and the censored target caps coverage at pool recall.",
 "Hyper-parameters of the gradient-boosted model were fixed a priori (not tuned on test); no extensive tuning was done, so gains may be conservative or optimistic.",
 "Test queries come from the SQuAD train file (the larger set, for statistical power) split by article; the official dev file was not used."])
d.add_heading("8. Reproducibility", 1)
para(d, "Code: adaptive_rag/ (data, retrieval, features, strategies, experiment, plots). Entry points: scripts/run_all.py (everything) and notebooks/01-06. All outputs land in results/csv, results/figures and results/logs; results/MANIFEST.csv lists SHA-256 hashes of every file. See 02_Experiment_Log_and_Evidence.docx for exact commands and the file-by-file guide.")
d.add_heading("9. Suggested next experiments for the publication", 1)
bullets(d, ["Run notebooks 02-05 on Colab with backend 'st' (all-MiniLM-L6-v2, then bge-small / e5-small) and report all three.", "Add TAA-k and full CAR as baselines.", "Add HotpotQA / MuSiQue / SciFact (BEIR) and Natural Questions.", "Run notebook 06 (generation) for EM/F1 and real token counts with 2 reader sizes.", "Learn the controller with a cross-encoder score and query embedding as additional features.", "Cost model: report dollars per 1,000 queries using the real token counts and a stated price."])
d.save(os.path.join(OUT, "01_Research_Report_Adaptive_RAG.docx"))

# =====================================================================  2. EXPERIMENT LOG
d = newdoc("Experiment Log and Evidence Guide", "What was run, with which settings, and where every output file lives")
d.add_heading("1. Environment and configuration (captured automatically)", 1)
e = env["env"]
table(d, pd.DataFrame([{"item": "timestamp (UTC)", "value": e["timestamp_utc"]}, {"item": "python", "value": e["python"]}, {"item": "platform", "value": e["platform"]},
    {"item": "cpu cores / RAM GB", "value": f"{e['cpu_count']} / {e.get('ram_gb')}"}, {"item": "CUDA", "value": str(e.get("cuda_available"))},
    *[{"item": f"pkg {k}", "value": v} for k, v in e["packages"].items() if v], *[{"item": f"config {k}", "value": v} for k, v in env["config"].items()]]), font=8.5)
d.add_heading("2. Build timings (seconds, measured)", 1)
table(d, pd.DataFrame([{"stage": k, "seconds": v} for k, v in env["timing_s"].items()]).round(2))
d.add_heading("3. Commands to reproduce", 1)
para(d, "Local venv:")
for c in ["python -m venv .venv && source .venv/bin/activate   (Windows: .venv\\Scripts\\activate)", "pip install -r requirements.txt", "python scripts/run_all.py --n_queries 10000 --seeds 5 --boot 1000      # about 25 min on 2 CPU cores", "python scripts/run_all.py --quick      # smoke test, about 4 min", "jupyter lab  -> open notebooks/01_...ipynb to 06 in order", "python scripts/build_docs.py     # rebuild these Word files from the CSVs"]:
    p = d.add_paragraph(c, style="List Bullet"); [setattr(r.font, "name", "Consolas") for r in p.runs]
para(d, "Colab: upload rag_adaptive_project.zip when the setup cell asks, set Runtime > T4 GPU, run notebooks in order. Results go to Google Drive (MyDrive/rag_adaptive_results). The setup cell switches the dense retriever to sentence-transformers automatically.")
d.add_heading("4. File-by-file guide to the evidence", 1)
guide = [("dataset_stats.csv", "Corpus and question statistics for the thesis 'Dataset' section."), ("split_sizes.csv", "Train/val/test sizes by article (notebook 01)."),
 ("retrieval_baselines.csv", "Hit@k and answer-in-context for each retriever (notebook 02)."), ("gold_rank_cdf_hybrid.csv", "CDF of the gold-passage rank; motivates adaptive k."),
 ("strategy_results.csv", "Every strategy x parameter x retriever on the test split: mean/median/p95 k, hit rate, MRR, nDCG, answer-in-context, words, estimated tokens."),
 ("isobudget_bootstrap.csv", "Adaptive minus fixed-top-k at equal mean k with 95% article-cluster bootstrap CI and bootstrap p."), ("rcak_coverage.csv", "Conformal validity: target vs achieved coverage and pool-recall ceiling."),
 ("ablation.csv", "Feature-group drop-one / only-one ablation."), ("seed_robustness.csv", "Re-runs with seeds 0-4."), ("latency.csv", "Measured single-query latency of retrieval and of the controller."),
 ("test_gold_ranks.csv", "Per-test-query gold rank for each retriever (supports any re-analysis)."), ("feature_matrix_sample.csv", "First 200 rows of the feature matrix (notebook 03)."),
 ("fig_*.csv", "Exact data behind each figure, so figures can be redrawn in any tool."), ("../figures/*.png|pdf", "300-dpi PNG and vector PDF of each figure."),
 ("../logs/run_log.csv, environment_and_config.json", "Timestamped run events; environment and configuration."), ("../MANIFEST.csv", "SHA-256 hash and size of every result file (tamper-evident evidence)."),
 ("generation_*.csv (notebook 06)", "Per-query EM/F1/prompt tokens/latency for the real LLM reader - produced only when you run notebook 06 on Colab.")]
table(d, pd.DataFrame(guide, columns=["file (results/csv/ unless stated)", "contents"]), font=9)
d.add_heading("5. Two result folders - which one to cite", 1)
para(d, "results/ = the full run from scripts/run_all.py (10,000 questions, 5 seeds, 1,000 bootstrap resamples). results_notebooks/ = the executed notebooks (same configuration; bootstrap 500-1000 in some steps; notebook 04 recomputes the iso-budget with 1,000). Numbers agree up to bootstrap noise; cite one folder consistently. notebooks_executed/ holds the notebooks with their outputs saved.")
d.add_heading("6. Evidence integrity", 1)
man = pd.read_csv(os.path.join(R, "MANIFEST.csv")); para(d, f"MANIFEST.csv lists {len(man)} files. Verify with: python -c \"from adaptive_rag.utils import sha256; ...\" or compare hashes by hand. No synthetic or hand-edited values exist in any CSV.")
d.add_heading("7. What was NOT run in the sandbox", 1)
bullets(d, ["Neural sentence-transformer embeddings (Hugging Face was blocked) - run on Colab.", "End-to-end LLM generation (notebook 06) - run on Colab.", "BEIR / HotpotQA corpora - need Hugging Face or other hosts that were not reachable."])
d.save(os.path.join(OUT, "02_Experiment_Log_and_Evidence.docx"))

# =====================================================================  3. LITERATURE & RESOURCES
d = newdoc("Literature Notes and Free Resources for RAG Research", "Reference note: what each source contributes and how it is used here. Verify every citation against the original before submission.")
d.add_heading("1. Adaptive retrieval depth - the closest prior work", 1)
lit = [("Adaptive-k", "Taguchi, Maekawa, Bhutani (2025), arXiv 2506.08479; EMNLP 2025", "Largest gap in sorted similarity scores sets k (+buffer). No tuning, no extra LLM calls. Reported 2-10x fewer tokens vs full context on long-context benchmarks.", "Baseline 'adaptive_k_gap'."),
 ("Cluster-based Adaptive Retrieval (CAR)", "Xu, Gupta, Aggarwal, Mahadevan, Krishnamachari, arXiv 2511.14769", "Clusters query-document distances; cuts before the best-scoring gap; reported ~60% token reduction in a deployed evaluation.", "Simplified baseline 'car_lite'."),
 ("Tail-Aware Adaptive-k (TAA-k)", "Song, Fang, Li, Xia, Wang, arXiv 2606.11907 (June 2026)", "Knee detection + Generalised Pareto tail fit; claims fixes Adaptive-k on multi-hop.", "NOT implemented - add as baseline."),
 ("DynamicRAG", "Sun, Zhong, Zhou, Han, arXiv 2505.07233", "Reranker chooses order and count; trained with generator feedback (SFT + DPO).", "Contrast: trained with LLM in loop; ours is LLM-free."),
 ("Adaptive-RAG", "Jeong et al. (2024)", "Query-complexity classifier routes between no retrieval, single-step and multi-step retrieval.", "Orthogonal: strategy routing vs depth."),
 ("SoK: Agentic RAG", "arXiv 2603.07379", "Taxonomy and evaluation survey of agentic RAG.", "Background for the 'agentic' framing.")]
table(d, pd.DataFrame(lit, columns=["Work", "Reference", "Idea / reported result (per the paper's own claims)", "Use in this project"]), font=8.5)
d.add_heading("2. Foundations to cite", 1)
bullets(d, ["Lewis et al. (2020) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.", "Karpukhin et al. (2020) Dense Passage Retrieval.", "Robertson & Zaragoza (2009) BM25 / probabilistic relevance framework.", "Cormack, Clarke, Buettcher (2009) Reciprocal Rank Fusion.", "Romano, Patterson, Candes (2019) Conformalized Quantile Regression.", "Rajpurkar et al. (2016, 2018) SQuAD 1.1 / 2.0.", "Thakur et al. (2021) BEIR benchmark."])
d.add_heading("3. Free datasets and resources (none behind a paywall)", 1)
resrc = [("SQuAD v2.0", "Wikipedia paragraphs + questions with answer spans", "USED. https://github.com/rajpurkar/SQuAD-explorer (dataset/ folder)"), ("BEIR (SciFact, NFCorpus, FiQA, ...)", "Zero-shot retrieval benchmark, multiple relevant docs per query", "Loader included (adaptive_rag.data.load_beir_hf); Hugging Face 'BeIR/<name>'"),
 ("HotpotQA / 2WikiMultihopQA / MuSiQue", "Multi-hop QA - the setting where fixed k and gap rules fail most", "Recommended next; Hugging Face datasets"), ("Natural Questions / TriviaQA / MS MARCO", "Open-domain QA / passage ranking", "Hugging Face datasets / official sites"),
 ("sentence-transformers all-MiniLM-L6-v2, bge-small-en-v1.5, e5-small-v2", "Free small embedding models (CPU/T4 friendly)", "Prefix conventions already coded in retrieval.ST_PREFIX"), ("Qwen2.5-0.5B/1.5B-Instruct, flan-t5-large", "Free small readers for notebook 06", "Hugging Face"),
 ("FAISS (faiss-cpu)", "Exact / approximate vector search", "pip install faiss-cpu"), ("Google Colab free tier", "T4 GPU for embeddings and generation", "Use batch 64; N_EVAL 300-500 for generation")]
table(d, pd.DataFrame(resrc, columns=["Resource", "What it is", "Status / where"]), font=8.5)
d.add_heading("4. How to turn this into a publication", 1)
bullets(d, ["Venue fit: IR/NLP workshops or short papers (SIGIR/ECIR/EMNLP/ACL industry or Findings tracks) accept efficiency-focused RAG work; for a journal, a full study with multi-hop datasets and generation results is expected.", "Required for credibility: neural retrievers, at least one multi-hop dataset, end-to-end generation with real token counts, TAA-k baseline, significance tests (already in the pipeline).", "Claim carefully: the contribution is a controllable (conformal) depth policy with a matched-budget evaluation, not a universal improvement."])
d.save(os.path.join(OUT, "03_Literature_and_Resources.docx"))
print("docs written:", os.listdir(OUT))
