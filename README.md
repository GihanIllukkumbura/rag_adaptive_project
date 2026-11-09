# Adaptive retrieval depth for RAG - reproducible research package

Real data (SQuAD v2.0), real experiments, every output saved as CSV / PNG / PDF / DOCX.

## Layout
- `adaptive_rag/`  library: data, retrieval (BM25, LSA, sentence-transformers, RRF), features, strategies (fixed, rules, Adaptive-k, CAR-lite, **RCAK**, **CGC**, oracle), metrics, experiment pipeline, plots
- `notebooks/`  01 data -> 02 retrievers -> 03 methods -> 04 evaluation+figures -> 05 ablation/seeds/latency -> 06 LLM generation (Colab GPU)
- `notebooks_executed/`  notebooks 01-05 with outputs from the sandbox run (LSA backend)
- `scripts/run_all.py`  whole pipeline in one command; `scripts/build_docs.py` rebuilds the DOCX from CSVs
- `results/`  csv/, figures/, logs/, MANIFEST.csv (SHA-256) - the full run cited in the report
- `results_notebooks/`  outputs written while executing the notebooks
- `docs/`  01 research report, 02 experiment log and evidence guide, 03 literature and resources

## Local venv
```
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_all.py --quick        # ~4 min smoke test
python scripts/run_all.py                # full run (~25 min on 2 cores)
jupyter lab                              # run notebooks/01..06
```
## Colab
Upload the zip to Colab (or open a notebook and upload when asked), choose Runtime > T4 GPU, run notebooks in order.
The setup cell installs packages, mounts Drive (results saved in MyDrive/rag_adaptive_results) and uses neural embeddings (backend "st").
The dataset downloads automatically from GitHub.

## Important
The executed numbers use an LSA dense retriever (no Hugging Face access in the sandbox). Re-run on Colab with neural embeddings before publishing, and run notebook 06 for end-to-end EM/F1.
