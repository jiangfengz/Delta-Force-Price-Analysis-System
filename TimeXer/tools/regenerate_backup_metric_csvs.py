from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


METRIC_KEYS = ["MAE", "MSE", "RMSE", "MAPE", "MSPE"]


_LEGACY_PATTERN = re.compile(
    r"^(.*)_ft([^_]+)_sl(\d+)_ll(\d+)_pl(\d+)_dm(\d+)_nh(\d+)_el(\d+)_dl(\d+)_df(\d+)_expand(\d+)_dc(\d+)_fc(\d+)_eb([^_]+)_dt([^_]+)_(.*)_(\d+)$"
)
_KNOWN_TASKS = [
    "long_term_forecast",
    "short_term_forecast",
    "imputation",
    "classification",
    "anomaly_detection",
]


def _detect_encoding(path: Path) -> str:
    try:
        b = path.read_bytes()[:3]
        if b.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
    except Exception:
        pass
    return "utf-8"


def _to_float(v: object) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        x = float(s)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]], str]:
    enc = _detect_encoding(path)
    with path.open("r", encoding=enc, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows: List[Dict[str, str]] = []
        for r in reader:
            rows.append({k: ("" if r.get(k) is None else str(r.get(k))) for k in fieldnames})
    return fieldnames, rows, enc


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]], encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _read_metrics_csv(path: Path) -> Optional[Dict[str, float]]:
    try:
        enc = _detect_encoding(path)
        with path.open("r", encoding=enc, newline="") as f:
            reader = csv.DictReader(f)
            r = next(reader, None)
        if not r:
            return None
        out: Dict[str, float] = {}
        for k in METRIC_KEYS:
            v = _to_float(r.get(k))
            if v is None:
                return None
            out[k] = float(v)
        return out
    except Exception:
        return None


def _model_id_from_metrics_csv(metrics_csv: Path) -> str:
    if metrics_csv.parent.name.lower() == "results":
        return metrics_csv.parent.parent.name
    return metrics_csv.parent.name


def _dataset_key_from_model_id(model_id: str) -> str:
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
            for cut in (
                "_TimeXer_",
                "_TimesNet_",
                "_Transformer_",
                "_Autoformer_",
                "_iTransformer_",
                "_PatchTST_",
            ):
                if cut in rest:
                    rest = rest.split(cut, 1)[0]
                    break
            return rest.split("_", 1)[0]
    return mid.split("_", 1)[0]


def _build_metrics_index(search_root: Path) -> Dict[str, Dict[str, float]]:
    best: Dict[str, Tuple[int, Path]] = {}
    for p in search_root.rglob("metrics.csv"):
        mid = _model_id_from_metrics_csv(p)
        try:
            rel_len = len(p.relative_to(search_root).parts)
        except Exception:
            rel_len = 10**9
        prev = best.get(mid)
        if prev is None or rel_len < prev[0]:
            best[mid] = (rel_len, p)
    out: Dict[str, Dict[str, float]] = {}
    for mid, (_, p) in best.items():
        m = _read_metrics_csv(p)
        if m:
            out[mid] = m
    return out


def _read_meta_from_log(log_path: Path) -> Optional[Dict[str, object]]:
    if not log_path.exists():
        return None
    try:
        enc = _detect_encoding(log_path)
        with log_path.open("r", encoding=enc, errors="ignore") as f:
            for _ in range(800):
                line = f.readline()
                if not line:
                    break
                s = line.lstrip("\ufeff").strip()
                if s.startswith("META:"):
                    payload = s[len("META:") :].strip()
                    return json.loads(payload)
    except Exception:
        return None
    return None


def _build_metrics_index_by_meta(search_root: Path) -> Dict[str, Dict[str, float]]:
    best: Dict[str, Tuple[int, Path]] = {}
    for metrics_csv in search_root.rglob("metrics.csv"):
        candidates = [
            metrics_csv.parent / "log.txt",
            metrics_csv.parent.parent / "log.txt",
        ]
        meta = None
        for lp in candidates:
            meta = _read_meta_from_log(lp)
            if meta:
                break
        if not meta:
            continue
        mid = str(meta.get("model_id") or "").strip()
        if not mid:
            continue
        try:
            rel_len = len(metrics_csv.relative_to(search_root).parts)
        except Exception:
            rel_len = 10**9
        prev = best.get(mid)
        if prev is None or rel_len < prev[0]:
            best[mid] = (rel_len, metrics_csv)
    out: Dict[str, Dict[str, float]] = {}
    for mid, (_, p) in best.items():
        m = _read_metrics_csv(p)
        if m:
            out[mid] = m
    return out


def _legacy_model_id_from_dirname(folder_name: str) -> Optional[str]:
    m = _LEGACY_PATTERN.match(str(folder_name))
    if not m:
        return None
    prefix = m.group(1)
    parts = str(prefix).split("_")
    task_name = None
    remaining = parts
    for t in _KNOWN_TASKS:
        if str(prefix).startswith(t):
            task_name = t
            task_len = len(t.split("_"))
            remaining = parts[task_len:]
            break
    if not task_name:
        return None
    if len(remaining) >= 2:
        model_id = "_".join(remaining[:-2]) if len(remaining) > 2 else ""
        return model_id if model_id else "Default"
    if len(remaining) == 1:
        return "Default"
    return None


def _build_metrics_index_by_legacy_dirname(search_root: Path) -> Dict[str, Dict[str, float]]:
    best: Dict[str, Tuple[int, Path]] = {}
    for p in search_root.rglob("metrics.csv"):
        key = _legacy_model_id_from_dirname(p.parent.name)
        if not key:
            continue
        try:
            rel_len = len(p.relative_to(search_root).parts)
        except Exception:
            rel_len = 10**9
        prev = best.get(key)
        if prev is None or rel_len < prev[0]:
            best[key] = (rel_len, p)
    out: Dict[str, Dict[str, float]] = {}
    for k, (_, p) in best.items():
        m = _read_metrics_csv(p)
        if m:
            out[k] = m
    return out


def _update_summary_metrics(path: Path) -> Dict[str, object]:
    fieldnames, rows, enc = _read_csv_rows(path)
    id_key = None
    for k in ("Model ID", "model_id", "modelId", "model"):
        if k in fieldnames:
            id_key = k
            break
    if id_key is None:
        return {"file": str(path), "status": "skip", "reason": "missing_model_id_column"}

    metrics_index = _build_metrics_index(path.parent)
    if not metrics_index:
        metrics_index = _build_metrics_index_by_meta(path.parent)
    else:
        meta_index = _build_metrics_index_by_meta(path.parent)
        for k, v in meta_index.items():
            if k not in metrics_index:
                metrics_index[k] = v
    legacy_index = _build_metrics_index_by_legacy_dirname(path.parent)
    for k, v in legacy_index.items():
        if k not in metrics_index:
            metrics_index[k] = v
    updated = 0
    missing = 0

    for r in rows:
        mid = (r.get(id_key) or "").strip()
        if not mid:
            continue
        m = metrics_index.get(mid)
        if not m:
            missing += 1
            continue
        for k in METRIC_KEYS:
            if k in fieldnames:
                r[k] = str(m[k])
        updated += 1

    if missing:
        try:
            backup_root = path.parents[1]
        except Exception:
            backup_root = None
        if backup_root and backup_root.exists():
            broad = _build_metrics_index(backup_root)
            broad_legacy = _build_metrics_index_by_legacy_dirname(backup_root)
            for k, v in broad_legacy.items():
                if k not in broad:
                    broad[k] = v
            fixed = 0
            still_missing = 0
            for r in rows:
                mid = (r.get(id_key) or "").strip()
                if not mid:
                    continue
                if mid in metrics_index:
                    continue
                m = broad.get(mid)
                if not m:
                    still_missing += 1
                    continue
                for k in METRIC_KEYS:
                    if k in fieldnames:
                        r[k] = str(m[k])
                fixed += 1
            missing = still_missing
            updated += fixed

    avg_updated = 0
    for r in rows:
        mid = (r.get(id_key) or "").strip()
        if "_AVG_" not in mid:
            continue
        left, right = mid.split("_AVG_", 1)
        left = left.strip("_")
        right = right.strip("_")
        candidates: List[Dict[str, str]] = []
        for rr in rows:
            rr_mid = (rr.get(id_key) or "").strip()
            if not rr_mid or rr_mid == mid:
                continue
            if "_AVG_" in rr_mid:
                continue
            if left and not rr_mid.startswith(left + "_"):
                continue
            if right and not rr_mid.endswith("_" + right):
                continue
            candidates.append(rr)
        if not candidates:
            continue
        ok = True
        for mk in METRIC_KEYS:
            if mk not in fieldnames:
                continue
            vals: List[float] = []
            for rr in candidates:
                v = _to_float(rr.get(mk))
                if v is None:
                    ok = False
                    break
                vals.append(float(v))
            if not ok or not vals:
                ok = False
                break
            r[mk] = str(statistics.mean(vals))
        if ok:
            avg_updated += 1

    _write_csv_rows(path, fieldnames, rows, encoding=enc)
    return {
        "file": str(path),
        "status": "ok",
        "rows": len(rows),
        "updated_from_metrics": updated,
        "avg_rows_updated": avg_updated,
        "missing_metrics_for_rows": missing,
    }


def _update_combined_metrics(path: Path) -> Dict[str, object]:
    fieldnames, rows, enc = _read_csv_rows(path)
    if "Source" not in fieldnames:
        return {"file": str(path), "status": "skip", "reason": "missing_Source_column"}
    updated = 0
    missing = 0
    base = path.parent
    for r in rows:
        src = (r.get("Source") or "").strip()
        if not src:
            continue
        cand1 = base / src / "metrics.csv"
        cand2 = base / src / "results" / "metrics.csv"
        metrics_path = cand1 if cand1.exists() else cand2 if cand2.exists() else None
        if metrics_path is None:
            missing += 1
            continue
        m = _read_metrics_csv(metrics_path)
        if not m:
            missing += 1
            continue
        for k in METRIC_KEYS:
            if k in fieldnames:
                r[k] = str(m[k])
        updated += 1
    _write_csv_rows(path, fieldnames, rows, encoding=enc)
    return {"file": str(path), "status": "ok", "rows": len(rows), "updated": updated, "missing": missing}


def _read_order_from_metrics_compare(path: Path) -> Tuple[List[str], List[str]]:
    fieldnames, rows, _ = _read_csv_rows(path)
    ds_order: List[str] = []
    group_order: List[str] = []
    group_key = "family" if "family" in fieldnames else "model_type" if "model_type" in fieldnames else "model"
    for r in rows:
        ds = (r.get("dataset") or "").strip()
        gp = (r.get(group_key) or "").strip()
        if ds and ds not in ds_order:
            ds_order.append(ds)
        if gp and gp not in group_order:
            group_order.append(gp)
    return ds_order, group_order


def _scan_group_metrics(group_dir: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    results_dir = group_dir / "results"
    if not results_dir.exists():
        return out
    for metrics_csv in results_dir.glob("*/metrics.csv"):
        model_id = metrics_csv.parent.name
        m = _read_metrics_csv(metrics_csv)
        if not m:
            continue
        out.append({"dataset": _dataset_key_from_model_id(model_id), "model_id": model_id, **m})
    return out


def _regen_metrics_compare(root: Path, path: Path) -> Tuple[List[Dict[str, object]], List[str], List[str], str]:
    fieldnames, _, enc = _read_csv_rows(path)
    group_key = "family" if "family" in fieldnames else "model_type" if "model_type" in fieldnames else "model"
    ds_order, group_order = _read_order_from_metrics_compare(path)
    groups = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))]
    if not group_order:
        group_order = [d.name for d in groups]

    dataset_set = set(ds_order)
    grid: Dict[Tuple[str, str], Dict[str, object]] = {}
    for g in groups:
        gname = g.name
        for r in _scan_group_metrics(g):
            ds = str(r.get("dataset") or "")
            mid = str(r.get("model_id") or "")
            if not ds or not mid:
                continue
            dataset_set.add(ds)
            grid[(ds, gname)] = {**r, group_key: gname}

    if not ds_order:
        ds_order = sorted(dataset_set, key=lambda x: x.lower())
    else:
        for ds in sorted(dataset_set, key=lambda x: x.lower()):
            if ds not in ds_order:
                ds_order.append(ds)

    rows_out: List[Dict[str, object]] = []
    for ds in ds_order:
        for gname in group_order:
            cell = grid.get((ds, gname))
            if cell is None:
                rows_out.append(
                    {
                        "dataset": ds,
                        group_key: gname,
                        "model_id": "",
                        "MAE": "",
                        "MSE": "",
                        "RMSE": "",
                        "MAPE": "",
                        "MSPE": "",
                    }
                )
            else:
                rows_out.append(
                    {
                        "dataset": ds,
                        group_key: gname,
                        "model_id": cell.get("model_id", ""),
                        "MAE": cell.get("MAE", ""),
                        "MSE": cell.get("MSE", ""),
                        "RMSE": cell.get("RMSE", ""),
                        "MAPE": cell.get("MAPE", ""),
                        "MSPE": cell.get("MSPE", ""),
                    }
                )
    return rows_out, fieldnames, group_order, enc


def _pick_best_by_mse(rows: Iterable[Dict[str, object]], mse_key: str = "MSE") -> Optional[Dict[str, object]]:
    best = None
    best_mse = None
    best_mae = None
    for r in rows:
        mse = _to_float(r.get(mse_key))
        mae = _to_float(r.get("MAE"))
        if mse is None:
            continue
        if best is None:
            best = r
            best_mse = mse
            best_mae = mae
            continue
        if best_mse is None or mse < best_mse:
            best = r
            best_mse = mse
            best_mae = mae
            continue
        if mse == best_mse and mae is not None and best_mae is not None and mae < best_mae:
            best = r
            best_mse = mse
            best_mae = mae
    return best


def _regen_per_dataset_best(root: Path, metrics_compare_path: Path, per_best_path: Path) -> Dict[str, object]:
    out_fields, _, enc = _read_csv_rows(per_best_path)
    compare_fields, compare_rows, _ = _read_csv_rows(metrics_compare_path)
    group_key = "family" if "family" in compare_fields else "model_type" if "model_type" in compare_fields else "model"
    best_group_field = None
    for k in ("best_family", "best_model_type", "best_model", "best_group"):
        if k in out_fields:
            best_group_field = k
            break
    if best_group_field is None:
        return {"file": str(per_best_path), "status": "skip", "reason": "unknown_best_group_field"}

    by_ds: Dict[str, List[Dict[str, object]]] = {}
    for r in compare_rows:
        ds = (r.get("dataset") or "").strip()
        if not ds:
            continue
        if not (r.get("model_id") or "").strip():
            continue
        by_ds.setdefault(ds, []).append(r)

    rows_out: List[Dict[str, object]] = []
    for ds in sorted(by_ds.keys(), key=lambda x: x.lower()):
        best = _pick_best_by_mse(by_ds[ds])
        if best is None:
            rows_out.append({k: "" for k in out_fields})
            rows_out[-1]["dataset"] = ds
            continue
        rows_out.append(
            {
                "dataset": ds,
                best_group_field: (best.get(group_key) or ""),
                "best_model_id": (best.get("model_id") or ""),
                "best_MSE": (best.get("MSE") or ""),
                "best_MAE": (best.get("MAE") or ""),
            }
        )
    _write_csv_rows(per_best_path, out_fields, rows_out, encoding=enc)
    return {"file": str(per_best_path), "status": "ok", "rows": len(rows_out)}


def _regen_group_summary(
    metrics_compare_path: Path,
    per_best_path: Path,
    summary_path: Path,
) -> Dict[str, object]:
    out_fields, _, enc = _read_csv_rows(summary_path)
    compare_fields, compare_rows, _ = _read_csv_rows(metrics_compare_path)
    per_fields, per_rows, _ = _read_csv_rows(per_best_path)

    group_key = "family" if "family" in compare_fields else "model_type" if "model_type" in compare_fields else "model"
    out_group_field = out_fields[0] if out_fields else group_key

    wins_map: Dict[str, int] = {}
    for r in per_rows:
        ds = (r.get("dataset") or "").strip()
        if not ds:
            continue
        winner = None
        for k in ("best_family", "best_model_type", "best_model", "best_group", group_key):
            if k in per_fields:
                v = (r.get(k) or "").strip()
                if v:
                    winner = v
                    break
        if winner:
            wins_map[winner] = wins_map.get(winner, 0) + 1

    by_group_mse: Dict[str, List[float]] = {}
    by_group_mae: Dict[str, List[float]] = {}
    datasets_by_group: Dict[str, set] = {}
    for r in compare_rows:
        g = (r.get(group_key) or "").strip()
        ds = (r.get("dataset") or "").strip()
        mid = (r.get("model_id") or "").strip()
        if not g or not ds or not mid:
            continue
        datasets_by_group.setdefault(g, set()).add(ds)
        mse = _to_float(r.get("MSE"))
        mae = _to_float(r.get("MAE"))
        if mse is not None:
            by_group_mse.setdefault(g, []).append(float(mse))
        if mae is not None:
            by_group_mae.setdefault(g, []).append(float(mae))

    def stats(vals: List[float]) -> Dict[str, object]:
        if not vals:
            return {"mean": "", "median": "", "min": "", "max": ""}
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
        }

    groups = sorted(datasets_by_group.keys(), key=lambda x: x.lower())
    rows_out: List[Dict[str, object]] = []
    for g in groups:
        mse_stats = stats(by_group_mse.get(g, []))
        mae_stats = stats(by_group_mae.get(g, []))
        row: Dict[str, object] = {k: "" for k in out_fields}
        row[out_group_field] = g
        if "datasets_count" in out_fields:
            row["datasets_count"] = len(datasets_by_group.get(g, set()))
        if "count_with_MSE" in out_fields:
            row["count_with_MSE"] = len(by_group_mse.get(g, []))
        if "count_with_MAE" in out_fields:
            row["count_with_MAE"] = len(by_group_mae.get(g, []))
        if "wins_by_(MSE,MAE)" in out_fields:
            row["wins_by_(MSE,MAE)"] = wins_map.get(g, 0)
        if "win_count_by_MSE" in out_fields:
            row["win_count_by_MSE"] = wins_map.get(g, 0)
        if "mean_MSE" in out_fields:
            row["mean_MSE"] = mse_stats["mean"]
        if "median_MSE" in out_fields:
            row["median_MSE"] = mse_stats["median"]
        if "min_MSE" in out_fields:
            row["min_MSE"] = mse_stats["min"]
        if "max_MSE" in out_fields:
            row["max_MSE"] = mse_stats["max"]
        if "mean_MAE" in out_fields:
            row["mean_MAE"] = mae_stats["mean"]
        if "median_MAE" in out_fields:
            row["median_MAE"] = mae_stats["median"]
        if "min_MAE" in out_fields:
            row["min_MAE"] = mae_stats["min"]
        if "max_MAE" in out_fields:
            row["max_MAE"] = mae_stats["max"]
        rows_out.append(row)

    _write_csv_rows(summary_path, out_fields, rows_out, encoding=enc)
    return {"file": str(summary_path), "status": "ok", "rows": len(rows_out)}


def _load_best_baseline(best_root: Path) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    if not best_root.exists():
        return out
    for exp_dir in best_root.iterdir():
        if not exp_dir.is_dir():
            continue
        if exp_dir.name.startswith((".", "_")):
            continue
        metrics_csv = exp_dir / "results" / "metrics.csv"
        if not metrics_csv.exists():
            continue
        m = _read_metrics_csv(metrics_csv)
        if not m:
            continue
        ds = _dataset_key_from_model_id(exp_dir.name)
        cur = out.get(ds)
        if cur is None:
            out[ds] = {"model_id": exp_dir.name, **m}
            continue
        prev_mse = _to_float(cur.get("MSE"))
        now_mse = _to_float(m.get("MSE"))
        if prev_mse is None or (now_mse is not None and now_mse < prev_mse):
            out[ds] = {"model_id": exp_dir.name, **m}
    return out


def _regen_vs_best(per_best_path: Path, vs_best_detail: Path, vs_best_summary: Path, best_root: Path) -> List[Dict[str, object]]:
    baseline = _load_best_baseline(best_root)
    detail_fields, _, detail_enc = _read_csv_rows(vs_best_detail)
    sum_fields, _, sum_enc = _read_csv_rows(vs_best_summary)
    per_fields, per_rows, _ = _read_csv_rows(per_best_path)

    def get_group(r: Dict[str, str]) -> str:
        for k in ("best_family", "best_model_type", "best_model", "best_group", "best_family"):
            if k in per_fields:
                return (r.get(k) or "").strip()
        return ""

    detail_rows: List[Dict[str, object]] = []
    beats = 0
    improve_vals: List[float] = []
    regress_vals: List[float] = []
    with_best = 0
    missing_best = 0
    missing_datasets: List[str] = []

    for r in per_rows:
        ds = (r.get("dataset") or "").strip()
        if not ds:
            continue
        cand_model_id = (r.get("best_model_id") or "").strip()
        cand_mse = _to_float(r.get("best_MSE"))
        cand_mae = _to_float(r.get("best_MAE"))
        base = baseline.get(ds)
        base_model_id = (base.get("model_id") if base else "") or ""
        base_mse = _to_float(base.get("MSE") if base else None)
        base_mae = _to_float(base.get("MAE") if base else None)

        row: Dict[str, object] = {k: "" for k in detail_fields}
        row["dataset"] = ds
        row["best_model_id"] = base_model_id
        row["best_MSE"] = "" if base_mse is None else base_mse
        row["best_MAE"] = "" if base_mae is None else base_mae
        row["candidate_best_model_id"] = cand_model_id
        if "candidate_best_family" in detail_fields:
            row["candidate_best_family"] = get_group(r)
        if "candidate_best_model_type" in detail_fields:
            row["candidate_best_model_type"] = get_group(r)
        row["candidate_best_MSE"] = "" if cand_mse is None else cand_mse
        row["candidate_best_MAE"] = "" if cand_mae is None else cand_mae

        if base_mse is None or cand_mse is None:
            missing_best += 1
            missing_datasets.append(ds)
        else:
            with_best += 1
            delta_mse = cand_mse - base_mse
            row["delta_MSE"] = delta_mse
            row["relative_delta_MSE_percent"] = (delta_mse / base_mse * 100.0) if base_mse != 0 else ""
            if cand_mae is not None and base_mae is not None:
                delta_mae = cand_mae - base_mae
                row["delta_MAE"] = delta_mae
                row["relative_delta_MAE_percent"] = (delta_mae / base_mae * 100.0) if base_mae != 0 else ""
            beat = 1 if cand_mse < base_mse else 0
            row["beats_best"] = beat
            if beat:
                beats += 1
                improve_vals.append(-delta_mse)
            else:
                regress_vals.append(delta_mse)

        detail_rows.append(row)

    _write_csv_rows(vs_best_detail, detail_fields, detail_rows, encoding=detail_enc)

    summary_row: Dict[str, object] = {k: "" for k in sum_fields}
    summary_row["datasets_with_best"] = with_best
    summary_row["datasets_missing_best"] = missing_best
    summary_row["beat_count"] = beats
    summary_row["avg_MSE_improvement_when_beats"] = statistics.mean(improve_vals) if improve_vals else ""
    summary_row["avg_MSE_regression_when_not_beats"] = statistics.mean(regress_vals) if regress_vals else ""
    summary_row["missing_best_datasets"] = ";".join(missing_datasets)
    _write_csv_rows(vs_best_summary, sum_fields, [summary_row], encoding=sum_enc)

    return [
        {"file": str(vs_best_detail), "status": "ok", "rows": len(detail_rows)},
        {"file": str(vs_best_summary), "status": "ok", "rows": 1},
    ]


def _update_best_update_log(path: Path, backup_root: Path) -> Dict[str, object]:
    fieldnames, rows, enc = _read_csv_rows(path)
    metrics_index = _build_metrics_index(backup_root)
    updated_old = 0
    updated_new = 0
    for r in rows:
        old_id = (r.get("old_model_id") or "").strip()
        new_id = (r.get("new_model_id") or "").strip()
        if old_id and old_id in metrics_index:
            m = metrics_index[old_id]
            if "old_MSE" in fieldnames:
                r["old_MSE"] = str(m["MSE"])
            if "old_MAE" in fieldnames:
                r["old_MAE"] = str(m["MAE"])
            updated_old += 1
        if new_id and new_id in metrics_index:
            m = metrics_index[new_id]
            if "new_MSE" in fieldnames:
                r["new_MSE"] = str(m["MSE"])
            if "new_MAE" in fieldnames:
                r["new_MAE"] = str(m["MAE"])
            updated_new += 1
    _write_csv_rows(path, fieldnames, rows, encoding=enc)
    return {"file": str(path), "status": "ok", "rows": len(rows), "updated_old": updated_old, "updated_new": updated_new}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup_root", type=str, default=None)
    parser.add_argument("--baseline_root", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    backup_root = Path(args.backup_root).resolve() if args.backup_root else (repo_root / "backup")
    baseline_root = (
        Path(args.baseline_root).resolve()
        if args.baseline_root
        else (backup_root / "best in 7 days")
    )

    reports: List[Dict[str, object]] = []

    for p in backup_root.rglob("summary_metrics.csv"):
        reports.append(_update_summary_metrics(p))

    for p in backup_root.rglob("combined_metrics.csv"):
        reports.append(_update_combined_metrics(p))

    best_update_log = backup_root / "best_update_log.csv"
    if best_update_log.exists():
        reports.append(_update_best_update_log(best_update_log, backup_root=backup_root))

    for metrics_compare in backup_root.rglob("metrics_compare.csv"):
        root = metrics_compare.parent
        try:
            rows_out, fieldnames, _, enc = _regen_metrics_compare(root, metrics_compare)
            _write_csv_rows(metrics_compare, fieldnames, rows_out, encoding=enc)
            reports.append({"file": str(metrics_compare), "status": "ok", "rows": len(rows_out)})
        except Exception as e:
            reports.append({"file": str(metrics_compare), "status": "fail", "reason": f"{type(e).__name__}:{e}"})
            continue

        per_best = root / "per_dataset_best.csv"
        if per_best.exists():
            reports.append(_regen_per_dataset_best(root, metrics_compare, per_best))

        model_summary = root / "model_summary.csv"
        if model_summary.exists() and per_best.exists():
            reports.append(_regen_group_summary(metrics_compare, per_best, model_summary))

        best_model_summary = root / "best_model_summary.csv"
        if best_model_summary.exists() and per_best.exists():
            reports.append(_regen_group_summary(metrics_compare, per_best, best_model_summary))

        vs_best_detail = root / "vs_best_detail.csv"
        vs_best_summary = root / "vs_best_summary.csv"
        if per_best.exists() and vs_best_detail.exists() and vs_best_summary.exists():
            try:
                reports.extend(_regen_vs_best(per_best, vs_best_detail, vs_best_summary, best_root=baseline_root))
            except Exception as e:
                reports.append({"file": str(vs_best_detail), "status": "fail", "reason": f"{type(e).__name__}:{e}"})
                reports.append({"file": str(vs_best_summary), "status": "fail", "reason": f"{type(e).__name__}:{e}"})

    report_path = backup_root / "_metrics_csv_regenerate_report.csv"
    fieldnames = sorted({k for r in reports for k in r.keys()})
    _write_csv_rows(report_path, fieldnames, reports, encoding="utf-8")
    print(f"Done. report={report_path} rows={len(reports)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

