"""Publication figures (PNG 300 dpi + PDF + underlying CSV). Palette validated with the dataviz skill's validator;
legends + distinct markers/line-styles provide secondary encoding (colour is never the only channel)."""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .utils import results_dir

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
PAL = {"fixed_k": "#52514e", "rule_relative": "#eda100", "rule_zscore": "#e87ba4", "adaptive_k_gap": "#1baf7a",
       "car_lite": "#6250d6", "RCAK (proposed)": "#2a78d6", "CGC (proposed)": "#eb6834", "oracle_k": "#0b0b0b"}
MK = {"fixed_k": "o", "rule_relative": "s", "rule_zscore": "^", "adaptive_k_gap": "D", "car_lite": "P",
      "RCAK (proposed)": "o", "CGC (proposed)": "v", "oracle_k": "*"}
LS = {"fixed_k": "-", "rule_relative": "--", "rule_zscore": ":", "adaptive_k_gap": "-.", "car_lite": "-", "RCAK (proposed)": "-", "CGC (proposed)": "-", "oracle_k": "-"}
NICE = {"fixed_k": "Fixed top-k", "rule_relative": "Rule: relative score", "rule_zscore": "Rule: z-score", "adaptive_k_gap": "Adaptive-k (gap)",
        "car_lite": "CAR-lite (cluster)", "RCAK (proposed)": "RCAK (proposed)", "CGC (proposed)": "CGC (proposed)", "oracle_k": "Oracle (upper bound)"}

def _style():
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 110})

def _save(fig, name, data: pd.DataFrame | None = None):
    d = results_dir("figures"); fig.tight_layout()
    fig.savefig(d / f"{name}.png", dpi=300); fig.savefig(d / f"{name}.pdf"); plt.close(fig)
    if data is not None: data.to_csv(results_dir("csv") / f"fig_{name}.csv", index=False)

def fig_pareto(df, serve, metric="answer_in_context", xcol="mean_ctx_tokens_est", name=None, xlabel="Mean context passed to the reader (estimated tokens)"):
    _style(); d = df[df.serve_retriever == serve]; fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for s in ["fixed_k", "rule_relative", "rule_zscore", "adaptive_k_gap", "car_lite", "CGC (proposed)", "RCAK (proposed)", "oracle_k"]:
        x = d[d.strategy == s].sort_values(xcol)
        if x.empty: continue
        lw = 2.4 if "proposed" in s else 1.6
        ax.plot(x[xcol], x[metric], LS[s], color=PAL[s], marker=MK[s], ms=8 if s == "oracle_k" else 5.5, lw=lw, label=NICE[s],
                mec=SURF, mew=1.2, zorder=3 if "proposed" in s else 2)
    ax.set_xlabel(xlabel); ax.set_ylabel({"answer_in_context": "Answer-in-context rate", "hit_rate": "Gold-passage hit rate"}.get(metric, metric))
    ax.set_title(f"Quality vs. context budget - {serve} (test split)"); ax.legend(loc="lower right", fontsize=8.5)
    ax.set_ylim(0, 1.0)
    _save(fig, name or f"pareto_{metric}_{serve.replace(':','_')}", d[["strategy", "param", "mean_k", xcol, metric]])

def fig_coverage(cov, serve, name=None):
    _style(); c = cov[cov.serve_retriever == serve].sort_values("target_coverage"); fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0.35, 1], [0.35, 1], color=INK2, lw=1, ls="--", label="ideal (achieved = target)")
    ax.plot(c.target_coverage, c.achieved_test_coverage, "-o", color=PAL["RCAK (proposed)"], lw=2.2, ms=6, mec=SURF, label="RCAK on test")
    ax.axhline(c.pool_recall_at_K.iloc[0], color=PAL["rule_relative"], lw=1.5, ls=":", label=f"pool recall@{50} (ceiling)")
    ax.set_xlabel("Target coverage 1 - alpha"); ax.set_ylabel("Achieved gold-passage coverage"); ax.set_title("Conformal validity of RCAK")
    ax.legend(fontsize=8.5, loc="upper left"); _save(fig, name or f"coverage_{serve.replace(':','_')}", c)

def fig_k_hist(perq_k: dict, rank: np.ndarray, name="k_distribution"):
    _style(); fig, ax = plt.subplots(figsize=(7.2, 4.2)); bins = np.arange(0, 52, 2)
    for lab, (k, s) in perq_k.items():
        ax.hist(np.clip(k, 0, 50), bins=bins, histtype="step", lw=2.2, color=PAL[s], label=lab, ls=LS[s])
    ax.hist(np.minimum(rank, 51), bins=bins, histtype="stepfilled", alpha=0.10, color=INK2, label="rank of gold passage (retriever)")
    ax.set_xlabel("passages served (k) / gold rank"); ax.set_ylabel("queries"); ax.set_title("Distribution of per-query depth (test)")
    ax.legend(fontsize=8.5); _save(fig, name)

def fig_isobudget(iso, serve, metric="answer_in_context", name=None):
    _style(); d = iso[(iso.serve_retriever == serve) & (iso.metric == metric)].copy()
    d["lab"] = d.strategy.map(NICE) + "\n" + d.param; d = d.sort_values("diff")
    fig, ax = plt.subplots(figsize=(7.2, max(4, 0.26 * len(d) + 1)))
    cols = [PAL[s] for s in d.strategy]; y = np.arange(len(d))
    ax.hlines(y, d.ci95_lo, d.ci95_hi, color=INK2, lw=1.2)
    ax.scatter(d["diff"], y, c=cols, s=42, zorder=3, edgecolors=SURF, linewidths=1.2)
    ax.axvline(0, color=INK, lw=1); ax.set_yticks(y); ax.set_yticklabels(d.lab.str.replace("\n", " "), fontsize=7)
    ax.set_xlabel("Adaptive minus fixed top-k at the SAME mean k (95% cluster-bootstrap CI)"); ax.set_title(f"Iso-budget gain in {metric.replace('_',' ')} - {serve}")
    _save(fig, name or f"isobudget_{metric}_{serve.replace(':','_')}", d.drop(columns="lab"))

def fig_ablation(ab, name="ablation"):
    _style(); d = ab.sort_values("diff"); fig, ax = plt.subplots(figsize=(7.2, 4.4)); y = np.arange(len(d))
    ax.hlines(y, d.ci95_lo, d.ci95_hi, color=INK2, lw=1.2); ax.scatter(d["diff"], y, color=PAL["RCAK (proposed)"], s=46, zorder=3, edgecolors=SURF, linewidths=1.2)
    ax.axvline(0, color=INK, lw=1); ax.set_yticks(y); ax.set_yticklabels(d.variant, fontsize=8)
    ax.set_xlabel("Iso-budget gain in answer-in-context vs fixed top-k"); ax.set_title("RCAK feature-group ablation"); _save(fig, name, ab)

def fig_seeds(sd, name="seed_robustness"):
    _style(); fig, ax = plt.subplots(figsize=(6.4, 4.2)); strategies = sd.strategy.unique(); w = 0.8 / len(strategies)
    for i, s in enumerate(strategies):
        x = sd[sd.strategy == s].sort_values("seed"); key = s if s in PAL else "adaptive_k_gap"
        ax.bar(x.seed + i * w - 0.4 + w / 2, x["diff"], width=w * 0.92, color=PAL.get(s, PAL[key]), label=s)
    ax.axhline(0, color=INK, lw=1); ax.set_xlabel("split / sampling seed"); ax.set_ylabel("iso-budget gain (answer-in-context)")
    ax.set_title("Robustness across seeds"); ax.legend(fontsize=8.5); _save(fig, name, sd)

def fig_latency(lat, name="latency"):
    _style(); fig, ax = plt.subplots(figsize=(7.2, 3.2)); y = np.arange(len(lat))
    ax.barh(y, lat.mean_ms, color=PAL["RCAK (proposed)"], height=0.5); ax.set_yticks(y); ax.set_yticklabels(lat.component, fontsize=8)
    for yi, v in zip(y, lat.mean_ms): ax.text(v, yi, f" {v:.2f} ms", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("mean latency per query (ms)"); ax.set_title("Measured overhead of the depth controller"); _save(fig, name, lat)
