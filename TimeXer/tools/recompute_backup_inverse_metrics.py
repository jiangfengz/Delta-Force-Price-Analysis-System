from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_loader import Dataset_Custom
from utils.metrics import metric, metric_per_channel


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def _parse_bool(s: str) -> Optional[bool]:
    if s is None:
        return None
    v = str(s).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    return None


def _search_one(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    g = m.group(1)
    return g.strip() if g is not None else None


def _norm_token(s: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(s).lower())


def _infer_bullet_folder(repo_root: Path, model_id: str) -> Optional[Path]:
    mid = str(model_id)
    if "Collection_Category" in mid:
        return repo_root / "dataset" / "bullet" / "collection_category"
    if "Collection_Solo" in mid:
        return repo_root / "dataset" / "bullet" / "collection_solo"
    if "Category" in mid:
        return repo_root / "dataset" / "bullet" / "category"
    if "Solo" in mid:
        return repo_root / "dataset" / "bullet" / "solo"
    return None


def _infer_token_from_model_id(model_id: str) -> str:
    mid = str(model_id)
    for prefix in ("V18_7d_", "V17_7d_", "V16_", "V15_", "V14_", "V13_", "V12_", "V11_", "V10_"):
        if mid.startswith(prefix):
            rest = mid[len(prefix) :]
            return rest.split("_", 1)[0]
    for task_prefix in (
        "long_term_forecast_",
        "short_term_forecast_",
        "imputation_",
        "anomaly_detection_",
        "classification_",
    ):
        if mid.startswith(task_prefix):
            rest = mid[len(task_prefix) :]
            for cut in ("_TimeXer_", "_TimesNet_", "_Transformer_", "_Autoformer_"):
                if cut in rest:
                    rest = rest.split(cut, 1)[0]
                    break
            parts = [p for p in rest.split("_") if p]
            token_parts = []
            stopwords = {
                "Category",
                "Solo",
                "Collection",
                "Exp",
                "Opt",
                "Opt2",
                "Deep",
                "Deep6",
                "Long",
                "LongCtx",
                "Short",
                "ShortReg",
                "Large",
                "TEST",
            }
            for p in parts:
                if p in ("M", "S", "MS"):
                    break
                if p.startswith(("ft", "sl", "ll", "pl", "dm", "nh", "el", "dl", "df", "eb", "dt")):
                    break
                if p in stopwords:
                    break
                if token_parts and "." in token_parts[0] and not any(ch.isdigit() for ch in p):
                    break
                if p.isdigit() and token_parts and token_parts[-1].isdigit():
                    break
                token_parts.append(p)
                if len(token_parts) >= 3:
                    break
            if token_parts:
                if token_parts[0].upper() in {"BE", "DE", "FR", "NP", "PJM"}:
                    return token_parts[0]
                return " ".join(token_parts)
    return mid.split("_", 1)[0]


def _infer_custom_data_from_meta(repo_root: Path, meta: Dict[str, object]) -> Optional[Tuple[str, str, str]]:
    model_id = str(meta.get("model_id") or "")
    token = _infer_token_from_model_id(model_id)
    mid_norm = _norm_token(model_id)
    token_norm = _norm_token(token)
    if "arrow" in mid_norm or token_norm.startswith("arrow"):
        if "arrowlyt" in mid_norm:
            token = "玻纤柳叶箭矢"
        elif "arrowcg" in mid_norm:
            token = "碳纤维刺骨箭矢"
        elif "arrowcj" in mid_norm:
            token = "碳纤维穿甲箭矢"
        elif "arrow3" in mid_norm:
            token = "玻纤柳叶箭矢"
        elif "arrow4" in mid_norm:
            token = "碳纤维刺骨箭矢"
        elif "arrow5" in mid_norm:
            token = "碳纤维穿甲箭矢"
        else:
            token = "箭矢"
    folder = _infer_bullet_folder(repo_root, model_id)
    if folder is None or not folder.exists():
        return None

    token_n = _norm_token(token)
    token_parts_n = [p for p in (_norm_token(x) for x in str(token).split()) if p]
    scored = []
    for p in folder.glob("*.csv"):
        stem_n = _norm_token(p.stem)
        if not token_n:
            continue
        if stem_n == token_n:
            scored.append((3, len(stem_n), p))
        elif token_parts_n and all(tp in stem_n for tp in token_parts_n):
            scored.append((2, len(stem_n), p))
        elif token_n in stem_n or stem_n in token_n:
            scored.append((1, len(stem_n), p))
    if not scored:
        return None
    scored.sort(reverse=True)
    best = scored[0]
    if len(scored) > 1 and scored[1][:2] == best[:2]:
        return None
    csv_path = best[2]
    root_path = str(csv_path.parent.resolve())
    data_path = csv_path.name
    target = csv_path.stem
    return root_path, data_path, target


def _candidate_bullet_folders(repo_root: Path, model_id: str) -> List[Path]:
    preferred = _infer_bullet_folder(repo_root, model_id)
    ordered: List[Path] = []
    if preferred is not None:
        ordered.append(preferred)
    for p in [
        repo_root / "dataset" / "bullet" / "collection_category",
        repo_root / "dataset" / "bullet" / "category",
        repo_root / "dataset" / "bullet" / "collection_solo",
        repo_root / "dataset" / "bullet" / "solo",
    ]:
        if p not in ordered:
            ordered.append(p)
    return [p for p in ordered if p.exists()]


def _infer_custom_data_from_model_id(repo_root: Path, model_id: str) -> Optional[Tuple[str, str, str]]:
    token = _infer_token_from_model_id(model_id)
    token_norm = _norm_token(token)
    mid_norm = _norm_token(model_id)
    epf_root = repo_root / "dataset" / "EPF"
    if token.strip().upper() in {"BE", "DE", "FR", "NP", "PJM"} and epf_root.exists():
        csv_path = epf_root / f"{token.strip().upper()}.csv"
        if csv_path.exists():
            return str(epf_root.resolve()), csv_path.name, "OT"

    arrow_alias = None
    if "arrow" in mid_norm or token_norm.startswith("arrow"):
        if "arrowlyt" in mid_norm:
            arrow_alias = "玻纤柳叶箭矢"
        elif "arrowcg" in mid_norm:
            arrow_alias = "碳纤维刺骨箭矢"
        elif "arrowcj" in mid_norm:
            arrow_alias = "碳纤维穿甲箭矢"
        elif "arrow3" in mid_norm:
            arrow_alias = "玻纤柳叶箭矢"
        elif "arrow4" in mid_norm:
            arrow_alias = "碳纤维刺骨箭矢"
        elif "arrow5" in mid_norm:
            arrow_alias = "碳纤维穿甲箭矢"
        else:
            arrow_alias = "箭矢"
        token = arrow_alias
    token_n = _norm_token(token)
    token_parts_n = [p for p in (_norm_token(x) for x in str(token).split()) if p]
    scored = []
    folders = _candidate_bullet_folders(repo_root, model_id)
    for rank, folder in enumerate(folders):
        for p in folder.glob("*.csv"):
            stem_n = _norm_token(p.stem)
            if not token_n:
                continue
            if stem_n == token_n:
                scored.append((3, len(stem_n), -rank, p))
            elif token_parts_n and all(tp in stem_n for tp in token_parts_n):
                scored.append((2, len(stem_n), -rank, p))
            elif token_n in stem_n or stem_n in token_n:
                scored.append((1, len(stem_n), -rank, p))
    if not scored:
        return None
    scored.sort(reverse=True)
    best = scored[0]
    if len(scored) > 1 and scored[1][:3] == best[:3]:
        return None
    csv_path = best[3]
    root_path = str(csv_path.parent.resolve())
    data_path = csv_path.name
    target = "OT"
    try:
        cols = list(pd.read_csv(str(csv_path), nrows=1).columns)
        if target not in cols:
            target = cols[-1] if cols else "OT"
    except Exception:
        pass
    return root_path, data_path, target
    return None


@dataclass(frozen=True)
class ExpArgs:
    data: str
    root_path: str
    data_path: str
    features: str
    target: str
    freq: str
    embed: str
    seq_len: int
    label_len: int
    pred_len: int
    inverse_logged: Optional[bool]


def parse_exp_args_from_model_id_only(model_id: str) -> Optional[ExpArgs]:
    mid = str(model_id)
    ft = re.search(r"ft([A-Z]+)", mid)
    sl = re.search(r"sl(\d+)", mid)
    ll = re.search(r"ll(\d+)", mid)
    pl = re.search(r"pl(\d+)", mid)
    eb = re.search(r"eb([A-Za-z0-9]+)", mid)
    data = "custom"
    features = ft.group(1) if ft else ""
    embed = eb.group(1) if eb else "timeF"
    freq = "h"
    inverse_logged = None
    if not (features and sl and ll and pl):
        return None
    inferred = _infer_custom_data_from_model_id(REPO_ROOT, mid)
    if inferred is None:
        return None
    root_path, data_path, target = inferred
    return ExpArgs(
        data=data.strip(),
        root_path=root_path.strip(),
        data_path=data_path.strip(),
        features=features.strip(),
        target=target.strip(),
        freq=freq.strip(),
        embed=embed.strip(),
        seq_len=int(sl.group(1)),
        label_len=int(ll.group(1)),
        pred_len=int(pl.group(1)),
        inverse_logged=inverse_logged,
    )


def parse_exp_args_from_log(log_path: Path, fallback_model_id: Optional[str] = None) -> Optional[ExpArgs]:
    try:
        raw = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    text = _strip_ansi(raw)
    data = _search_one(r"^\s*Data:\s*([^\r\n]+?)\s+Root Path:", text)
    root_path = _search_one(r"Root Path:\s*([^\r\n]+)", text)
    data_path = _search_one(r"Data Path:\s*([^\r\n]+?)\s*Features:", text)
    features = _search_one(r"Features:\s*([^\r\n]+)", text)
    target = _search_one(r"Target:\s*([^\r\n]+?)\s*Freq:", text)
    freq = _search_one(r"Freq:\s*([^\r\n]+)", text)
    embed = _search_one(r"\"embed\"\s*:\s*\"([^\"]+)\"", text) or _search_one(r"Embed:\s*([^\r\n]+)", text)
    seq_label = re.search(r"Seq Len:\s*(\d+).*Label Len:\s*(\d+)", text)
    pred = re.search(r"Pred Len:\s*(\d+)", text)
    inv = _search_one(r"Inverse:\s*([^\r\n]+)", text)
    inverse_logged = _parse_bool(inv) if inv is not None else None

    if not (data and root_path and data_path and features and target and freq and embed and seq_label and pred):
        meta_json = _search_one(r"^META:\s*(\{.*\})\s*$", text)
        if meta_json is None:
            if fallback_model_id is None:
                return None
            return parse_exp_args_from_model_id_only(str(fallback_model_id))
        try:
            meta = json.loads(meta_json)
        except Exception:
            return None

        data = str(meta.get("data") or "")
        features = str(meta.get("features") or "")
        embed = str(meta.get("embed") or "")
        freq = "h"
        inverse_logged = None

        seq_len = meta.get("seq_len")
        label_len = meta.get("label_len")
        pred_len = meta.get("pred_len")
        if not (data and features and embed and isinstance(seq_len, int) and isinstance(label_len, int) and isinstance(pred_len, int)):
            return None

        inferred = _infer_custom_data_from_meta(REPO_ROOT, meta)
        if inferred is None:
            return None
        root_path, data_path, target = inferred

        return ExpArgs(
            data=data.strip(),
            root_path=root_path.strip(),
            data_path=data_path.strip(),
            features=features.strip(),
            target=target.strip(),
            freq=freq.strip(),
            embed=embed.strip(),
            seq_len=int(seq_len),
            label_len=int(label_len),
            pred_len=int(pred_len),
            inverse_logged=inverse_logged,
        )

    seq_len = int(seq_label.group(1))
    label_len = int(seq_label.group(2))
    pred_len = int(pred.group(1))

    return ExpArgs(
        data=data.strip(),
        root_path=root_path.strip(),
        data_path=data_path.strip(),
        features=features.strip(),
        target=target.strip(),
        freq=freq.strip(),
        embed=embed.strip(),
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        inverse_logged=inverse_logged,
    )


def _build_custom_dataset(args: ExpArgs) -> Dataset_Custom:
    timeenc = 0 if args.embed != "timeF" else 1
    ds_args = SimpleNamespace(augmentation_ratio=0)
    return Dataset_Custom(
        args=ds_args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag="train",
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns="Monthly",
    )

def _load_raw_targets_for_custom(args: ExpArgs) -> Optional[np.ndarray]:
    csv_path = Path(args.root_path) / args.data_path
    if not csv_path.exists():
        return None
    df_raw = pd.read_csv(str(csv_path))
    cols = list(df_raw.columns)
    if "date" in cols:
        cols.remove("date")
    if args.target in cols:
        cols.remove(args.target)
        df_raw = df_raw[["date"] + cols + [args.target]] if "date" in df_raw.columns else df_raw[cols + [args.target]]
    else:
        df_raw = df_raw[["date"] + cols] if "date" in df_raw.columns else df_raw[cols]

    if args.features in ("M", "MS"):
        exog_cols = ["is_holiday", "in_CS", "is_CS", "is_need", "is_make", "is_active", "is_public"]
        present_exog = [c for c in exog_cols if c in df_raw.columns]
        cols_data = df_raw.columns[1:] if "date" in df_raw.columns else df_raw.columns
        target_cols = [c for c in cols_data if c not in present_exog]
        df_y = df_raw[target_cols]
    else:
        df_y = df_raw[[args.target]] if args.target in df_raw.columns else None

    if df_y is None:
        return None
    return df_y.values.astype(np.float64, copy=False)


def _inverse_y(dataset: Dataset_Custom, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if not hasattr(dataset, "target_indices"):
        flat = arr.reshape(-1, arr.shape[-1])
        inv = dataset.inverse_transform_y(flat)
        return np.asarray(inv).reshape(arr.shape)
    target_indices = list(getattr(dataset, "target_indices") or [])
    if not target_indices:
        flat = arr.reshape(-1, arr.shape[-1])
        inv = dataset.inverse_transform_y(flat)
        return np.asarray(inv).reshape(arr.shape)

    ch = int(arr.shape[-1])
    if ch == len(target_indices):
        flat = arr.reshape(-1, ch)
        inv = dataset.inverse_transform_y(flat)
        return np.asarray(inv).reshape(arr.shape)

    tail = target_indices[-ch:]
    means = dataset.scaler.mean_[tail]
    scales = dataset.scaler.scale_[tail]
    flat = arr.reshape(-1, ch).astype(np.float64, copy=False)
    inv = flat * scales + means
    return np.asarray(inv).reshape(arr.shape)


def _p99_abs(arr: np.ndarray) -> float:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return float("nan")
    return float(np.nanpercentile(np.abs(flat), 99))


def _within_ratio(x: float, ref: float, lo: float = 0.25, hi: float = 4.0) -> bool:
    if not np.isfinite(x) or not np.isfinite(ref) or ref <= 0:
        return False
    r = x / ref
    return lo <= r <= hi


def _infer_scale_state_for_custom(dataset: Dataset_Custom, exp_args: ExpArgs, trues: np.ndarray) -> Tuple[str, Dict[str, object]]:
    raw_y = _load_raw_targets_for_custom(exp_args)
    if raw_y is None:
        return "unknown", {"reason": "cannot_load_raw_targets"}

    ch = int(np.asarray(trues).shape[-1])
    if raw_y.ndim == 2 and raw_y.shape[1] >= ch and ch > 0:
        raw_y = raw_y[:, -ch:]
    ref = _p99_abs(raw_y)
    cur = _p99_abs(trues)
    if _within_ratio(cur, ref):
        return "original", {"ref_p99_abs": ref, "cur_p99_abs": cur}

    inv = _inverse_y(dataset, trues)
    inv_p99 = _p99_abs(inv)
    if _within_ratio(inv_p99, ref):
        return "needs_inverse", {"ref_p99_abs": ref, "cur_p99_abs": cur, "inv_p99_abs": inv_p99}

    return "unknown", {"ref_p99_abs": ref, "cur_p99_abs": cur, "inv_p99_abs": inv_p99, "reason": "ambiguous_scale"}


def _write_metrics_csv(path: Path, mae: float, mse: float, rmse: float, mape: float, mspe: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["MAE", "MSE", "RMSE", "MAPE", "MSPE"])
        writer.writeheader()
        writer.writerow(
            {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MAPE": mape,
                "MSPE": mspe,
            }
        )


def _write_metrics_detail_csv(path: Path, preds: np.ndarray, trues: np.ndarray) -> None:
    mae_d, mse_d, rmse_d, mape_d, mspe_d = metric_per_channel(preds, trues)
    num_channels = int(preds.shape[-1])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Channel", "MAE", "MSE", "RMSE", "MAPE", "MSPE"])
        writer.writeheader()
        for i in range(num_channels):
            writer.writerow(
                {
                    "Channel": f"Channel_{i}",
                    "MAE": float(mae_d[i]),
                    "MSE": float(mse_d[i]),
                    "RMSE": float(rmse_d[i]),
                    "MAPE": float(mape_d[i]),
                    "MSPE": float(mspe_d[i]),
                }
            )


def _find_log_near(dir_path: Path) -> Optional[Path]:
    candidates = [
        dir_path / "log.txt",
        dir_path.parent / "log.txt",
        dir_path.parent.parent / "log.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def recompute_one_dir(dir_path: Path, force: bool, rewrite_metrics: bool) -> Tuple[str, Dict[str, object]]:
    pred_path = dir_path / "pred.npy"
    true_path = dir_path / "true.npy"
    if not pred_path.exists() or not true_path.exists():
        return "skip", {"reason": "missing_pred_or_true"}

    metrics_missing = not (dir_path / "metrics.csv").exists()

    log_path = _find_log_near(dir_path)
    exp_args = None
    if log_path is not None:
        exp_args = parse_exp_args_from_log(log_path, fallback_model_id=dir_path.name)
    if exp_args is None:
        exp_args = parse_exp_args_from_model_id_only(dir_path.name)
    if exp_args is None:
        return "fail", {"reason": "cannot_infer_args"}

    if exp_args.data != "custom":
        return "skip", {"reason": f"unsupported_data={exp_args.data}"}

    if exp_args.inverse_logged is True:
        return "skip", {"reason": "already_inverse_logged_true"}

    preds = np.load(pred_path, allow_pickle=False)
    trues = np.load(true_path, allow_pickle=False)
    if preds.shape != trues.shape or preds.ndim != 3:
        return "fail", {"reason": f"shape_mismatch_or_ndim: pred={preds.shape}, true={trues.shape}"}

    dataset = _build_custom_dataset(exp_args)
    sliced = 0
    if hasattr(dataset, "target_indices"):
        target_indices = list(getattr(dataset, "target_indices") or [])
        if target_indices:
            max_idx = int(np.max(target_indices))
            if preds.shape[-1] != len(target_indices) and preds.shape[-1] > max_idx:
                preds = preds[:, :, target_indices]
                trues = trues[:, :, target_indices]
                sliced = 1
    state, state_info = _infer_scale_state_for_custom(dataset, exp_args, trues)
    if state == "original" and not force and not rewrite_metrics and not metrics_missing:
        return "skip", {"reason": "already_original", "sliced_to_targets": sliced, **state_info}
    if state == "unknown" and not force:
        return "fail", {"reason": "unknown_scale_state", "sliced_to_targets": sliced, **state_info}

    inverse_applied = 0
    if state == "needs_inverse" or (state == "original" and force):
        inv_preds = _inverse_y(dataset, preds)
        inv_trues = _inverse_y(dataset, trues)
        inverse_applied = 1
    else:
        inv_preds = preds
        inv_trues = trues

    mae, mse, rmse, mape, mspe = metric(inv_preds, inv_trues)
    metrics_npy = np.array([mae, mse, rmse, mape, mspe], dtype=np.float64)

    np.save(dir_path / "metrics.npy", metrics_npy)
    if inverse_applied:
        np.save(pred_path, inv_preds)
        np.save(true_path, inv_trues)
    _write_metrics_csv(dir_path / "metrics.csv", float(mae), float(mse), float(rmse), float(mape), float(mspe))
    _write_metrics_detail_csv(dir_path / "metrics_detail.csv", inv_preds, inv_trues)

    return "ok", {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
        "scale_state": state,
        "sliced_to_targets": sliced,
        "inverse_applied": inverse_applied,
        "metrics_missing_before": int(metrics_missing),
        **state_info,
    }


def find_pred_true_dirs(backup_root: Path) -> List[Path]:
    dirs: Dict[str, Path] = {}
    for pred_path in backup_root.rglob("pred.npy"):
        dir_path = pred_path.parent
        if not (dir_path / "true.npy").exists():
            continue
        key = str(dir_path.resolve())
        dirs[key] = dir_path
    return list(dirs.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup_root", type=str, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rewrite_metrics", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    backup_root = Path(args.backup_root).resolve() if args.backup_root else (repo_root / "backup")
    dirs = find_pred_true_dirs(backup_root)

    workers = int(args.workers or 0)
    if workers <= 0:
        cpu = os.cpu_count() or 8
        workers = max(4, min(16, cpu))

    report_rows = []
    ok = skip = fail = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(recompute_one_dir, d, bool(args.force), bool(args.rewrite_metrics)): d for d in dirs}
        for fut in as_completed(futures):
            rd = futures[fut]
            try:
                status, payload = fut.result()
            except Exception as e:
                status, payload = "fail", {"reason": f"exception={type(e).__name__}:{e}"}

            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1

            row = {"dir": str(rd), "status": status}
            row.update(payload)
            report_rows.append(row)

    report_path = backup_root / "_inverse_recompute_report.csv"
    report_rows.sort(key=lambda r: (r.get("status", ""), r.get("dir", "")))
    fieldnames = sorted({k for r in report_rows for k in r.keys()})
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print(
        f"Done. dirs={len(dirs)} ok={ok} skip={skip} fail={fail} report={report_path}"
    )


if __name__ == "__main__":
    main()
