# -*- coding: utf-8 -*-
"""Compute standardized (z-score) MAE for every model under results/.

pred.npy/true.npy are stored on the ORIGINAL price scale. The framework's
reported `mae` (see log.txt) is on the standardized scale: each target channel
is normalized with the StandardScaler fit on the first 70% (train split) of the
dataset CSV (ddof=0), exactly as in recompute_backup_metrics.py.

standardized MAE = mean( |pred - true| / train_std[channel] )
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

EXOG_COLS = {"is_holiday", "in_CS", "is_CS", "is_need", "is_make", "is_active", "is_public"}

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATASET_DIR = ROOT / "dataset" / "bullet" / "collection_category"


def read_data_path_from_log(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(2000):
            line = f.readline()
            if not line:
                break
            line = line.lstrip("﻿").strip()
            if line.startswith("Data Path:"):
                v = line[len("Data Path:"):].strip().strip('"').strip("'")
                # log puts "Features:" on the same column block sometimes; cut it
                v = v.split("  ")[0].strip()
                return v or None
    return None


def read_log_mae(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.findall(r"mae:([0-9.eE+-]+)", txt)
    return float(m[-1]) if m else None


def norm_key(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z一-鿿]+", "", s).lower()


def infer_csv(exp_name: str) -> Path | None:
    prefix = exp_name.split("_", 1)[0]
    if prefix.startswith("V1") and "_" in exp_name:
        # names look like V17_7d_4570Govt_... -> token after the date tag
        toks = exp_name.split("_")
        prefix = toks[2] if len(toks) > 2 else prefix
    alias = {"Arrow": "箭矢", "300BLK": ".300BLK"}.get(prefix, prefix)
    key = norm_key(alias)
    for p in DATASET_DIR.glob("*.csv"):
        if norm_key(p.stem) == key:
            return p
    for p in DATASET_DIR.glob("*.csv"):
        if norm_key(p.stem).startswith(key):
            return p
    return None


def target_scaler(csv: Path):
    df = pd.read_csv(csv, encoding="utf-8-sig")
    cols = list(df.columns)
    cols_data = cols[1:] if cols and cols[0].strip().lower() == "date" else cols
    targets = [c for c in cols_data if c not in EXOG_COLS]
    n_train = int(len(df) * 0.7)
    train = df.loc[: max(n_train - 1, 0), cols_data].to_numpy(dtype=float, copy=False)
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=0)
    std = np.where(std == 0, 1.0, std)
    ti = [cols_data.index(c) for c in targets]
    return targets, mean[ti], std[ti]


def looks_standardized(v: np.ndarray) -> bool:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return False
    return float(np.max(np.abs(v))) < 20 and float(np.std(v)) < 10


rows = []
for exp in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
    pred_p, true_p = exp / "pred.npy", exp / "true.npy"
    if not pred_p.exists() or not true_p.exists():
        rows.append({"Model": exp.name, "err": "no pred/true"})
        continue
    pred = np.load(pred_p).astype(float)
    true = np.load(true_p).astype(float)

    log = exp / "log.txt"
    dp = read_data_path_from_log(log)
    csv = (DATASET_DIR / dp) if dp else None
    if csv is None or not csv.exists():
        csv = infer_csv(exp.name)

    if looks_standardized(true):
        # already standardized -> MAE direct
        std_mae = float(np.mean(np.abs(pred - true)))
        ch = [float(np.mean(np.abs(pred[..., c] - true[..., c]))) for c in range(true.shape[-1])]
        targets = [f"ch{c}" for c in range(true.shape[-1])]
    else:
        if csv is None or not csv.exists():
            rows.append({"Model": exp.name, "err": "csv not found"})
            continue
        targets, tmean, tstd = target_scaler(csv)
        ps = (pred - tmean) / tstd
        ts = (true - tmean) / tstd
        std_mae = float(np.mean(np.abs(ps - ts)))
        ch = [float(np.mean(np.abs(ps[..., c] - ts[..., c]))) for c in range(true.shape[-1])]

    rows.append({
        "Model": exp.name,
        "CSV": csv.name if csv else "(std npy)",
        "n_ch": true.shape[-1],
        "std_MAE": std_mae,
        "log_mae": read_log_mae(log),
        "per_ch": ", ".join(f"{x:.4f}" for x in ch),
    })

df = pd.DataFrame(rows)
ok = df[df.get("std_MAE").notna()].sort_values("std_MAE") if "std_MAE" in df else df
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 200)

print("\n=== Standardized MAE per model (sorted best->worst) ===\n")
show = ok[["Model", "std_MAE", "log_mae", "n_ch", "per_ch"]].copy()
show["std_MAE"] = show["std_MAE"].map(lambda x: f"{x:.4f}")
show["log_mae"] = show["log_mae"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
print(show.to_string(index=False))

if "std_MAE" in df and df["std_MAE"].notna().any():
    vals = df["std_MAE"].dropna()
    print(f"\nModels: {len(vals)} | mean={vals.mean():.4f}  median={vals.median():.4f}  "
          f"min={vals.min():.4f}  max={vals.max():.4f}")

err = df[df.get("err").notna()] if "err" in df else df.iloc[0:0]
if len(err):
    print("\n=== Skipped ===")
    print(err[["Model", "err"]].to_string(index=False))

out = ROOT / "standardized_mae_summary.csv"
ok.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\nSaved: {out}")
