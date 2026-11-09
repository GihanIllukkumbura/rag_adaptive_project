"""Utilities: seeding, environment capture, timing, CSV/JSON logging, manifests."""
from __future__ import annotations
import csv, hashlib, json, os, platform, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

def project_root() -> Path:
    return Path(os.environ.get("RAG_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

def results_dir(sub: str = "") -> Path:
    base = Path(os.environ.get("RAG_RESULTS_DIR", project_root() / "results"))
    p = base / sub if sub else base
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def _pkg_version(name: str):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None

def env_info() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": {p: _pkg_version(p) for p in
            ["numpy","pandas","scipy","scikit-learn","matplotlib","faiss-cpu","sentence-transformers",
             "transformers","torch","datasets","python-docx"]},
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total/1e9, 2)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                info["ram_gb"] = round(int(f.readline().split()[1])/1e6, 2)
        except Exception:
            info["ram_gb"] = None
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["cuda_available"] = False
    return info

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0

def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def log_event(event: str, **fields) -> None:
    """Append one row to results/logs/run_log.csv (fixed schema; details stored as JSON)."""
    path = results_dir("logs") / "run_log.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(["timestamp_utc", "event", "details_json"])
        w.writerow([datetime.now(timezone.utc).isoformat(), event, json.dumps(fields, default=str)])

def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_manifest(root: Path | None = None) -> Path:
    """SHA-256 manifest of every csv/figure/json in results/ -> tamper-evident evidence."""
    root = Path(root) if root else results_dir()
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.csv" and p.suffix.lower() in {".csv",".png",".pdf",".json",".svg"}:
            rows.append({"file": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    out = root / "MANIFEST.csv"
    import pandas as pd
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
