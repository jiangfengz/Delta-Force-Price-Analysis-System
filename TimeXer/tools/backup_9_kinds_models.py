from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


GROUP_3 = "3 kinds of models"
GROUP_6 = "6 kinds of models"
DEFAULT_DST = "9 kinds of models"

MODEL_DIRS_3 = ["Opt2_M_192_A", "Exp_Large_M_768_A", "Exp_Long_M_336_A"]
FAMILIES_6 = [
    "MidCtx_240",
    "LongCtx2_384",
    "MidLarge_640",
    "Deep6_240",
    "RegDrop_128",
    "Opt2_192_Plateau4",
]
MODEL_ORDER_9 = MODEL_DIRS_3 + FAMILIES_6

_RE_VERSION_PREFIX = re.compile(r"^V\d+_")


@dataclass(frozen=True)
class MetricsRow:
    dataset: str
    model: str
    model_id: str
    mae: Optional[float]
    mse: Optional[float]
    rmse: Optional[float]
    mape: Optional[float]
    mspe: Optional[float]


def _detect_repo_root(script_path: Path) -> Path:
    root = script_path.resolve()
    for _ in range(8):
        if (root / "run.py").exists() and (root / "backup").exists():
            return root
        root = root.parent
    raise RuntimeError("无法自动定位项目根目录（未找到 run.py 与 backup/）")


def _ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def _dataset_from_model_id(model_id: str) -> str:
    if "_Collection_Category_" in model_id:
        prefix = model_id.split("_Collection_Category_", 1)[0]
        prefix = _RE_VERSION_PREFIX.sub("", prefix)
        return prefix
    prefix = model_id.split("_", 1)[0]
    prefix = _RE_VERSION_PREFIX.sub("", prefix)
    return prefix


def _read_metrics_csv(
    metrics_csv: Path,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not metrics_csv.exists():
        return None, None, None, None, None
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        row = next(reader, None)
    if not header or not row:
        return None, None, None, None, None
    col_to_idx = {name.strip(): i for i, name in enumerate(header)}

    def get(name: str) -> Optional[float]:
        i = col_to_idx.get(name)
        if i is None or i >= len(row):
            return None
        try:
            v = float(row[i])
        except ValueError:
            return None
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    return get("MAE"), get("MSE"), get("RMSE"), get("MAPE"), get("MSPE")


def _iter_dirs(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    for p in path.iterdir():
        if p.is_dir():
            yield p


def _write_csv(path: Path, rows: List[Dict[str, object]], dry_run: bool) -> None:
    if dry_run:
        return
    _ensure_dir(path.parent, dry_run=False)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _move_dir(src: Path, dst: Path, dry_run: bool) -> None:
    if not src.exists():
        return
    if dst.exists():
        raise RuntimeError(f"目标目录已存在，无法迁移：{dst}")
    _ensure_dir(dst.parent, dry_run=dry_run)
    if dry_run:
        return
    shutil.move(str(src), str(dst))


def _delete_files(files: List[Path], dry_run: bool) -> None:
    if dry_run:
        return
    for f in files:
        try:
            if f.exists() and f.is_file():
                f.unlink()
        except OSError:
            continue


def _delete_tree(path: Path, dry_run: bool) -> None:
    if dry_run or not path.exists():
        return
    shutil.rmtree(path)


def _flatten_group_dir(dst_root: Path, group_dir: Path, dry_run: bool) -> None:
    if not group_dir.exists():
        return
    for model_dir in _iter_dirs(group_dir):
        dst_model_dir = dst_root / model_dir.name
        _move_dir(model_dir, dst_model_dir, dry_run=dry_run)
    _delete_tree(group_dir, dry_run=dry_run)


def _migrate_groups(repo_root: Path, dst_name: str, dry_run: bool) -> Path:
    backup_root = repo_root / "backup"
    dst_root = backup_root / dst_name
    _ensure_dir(dst_root, dry_run=dry_run)

    _flatten_group_dir(dst_root, backup_root / GROUP_3, dry_run=dry_run)
    _flatten_group_dir(dst_root, backup_root / GROUP_6, dry_run=dry_run)
    _flatten_group_dir(dst_root, dst_root / GROUP_3, dry_run=dry_run)
    _flatten_group_dir(dst_root, dst_root / GROUP_6, dry_run=dry_run)

    return dst_root


def _collect_rows(dst_root: Path) -> List[MetricsRow]:
    rows: List[MetricsRow] = []
    for model_root in _iter_dirs(dst_root):
        if model_root.name in {GROUP_3, GROUP_6}:
            continue
        results_root = model_root / "results"
        for exp_dir in _iter_dirs(results_root):
            model_id = exp_dir.name
            mae, mse, rmse, mape, mspe = _read_metrics_csv(exp_dir / "metrics.csv")
            dataset = _dataset_from_model_id(model_id)
            rows.append(
                MetricsRow(
                    dataset=dataset,
                    model=model_root.name,
                    model_id=model_id,
                    mae=mae,
                    mse=mse,
                    rmse=rmse,
                    mape=mape,
                    mspe=mspe,
                )
            )

    def model_rank(name: str) -> Tuple[int, str]:
        if name in MODEL_ORDER_9:
            return MODEL_ORDER_9.index(name), ""
        return 10_000, name.lower()

    rows.sort(key=lambda r: (r.dataset.lower(), model_rank(r.model), r.model_id.lower()))
    return rows


def _build_metrics_compare(rows: List[MetricsRow]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in rows:
        out.append(
            {
                "dataset": r.dataset,
                "model": r.model,
                "model_id": r.model_id,
                "MAE": r.mae,
                "MSE": r.mse,
                "RMSE": r.rmse,
                "MAPE": r.mape,
                "MSPE": r.mspe,
            }
        )
    return out


def _pick_best(rows: List[MetricsRow]) -> Optional[MetricsRow]:
    candidates = [r for r in rows if r.mse is not None and r.mae is not None]
    if not candidates:
        return None

    def model_rank(name: str) -> int:
        return MODEL_ORDER_9.index(name) if name in MODEL_ORDER_9 else 10_000

    return sorted(candidates, key=lambda r: (r.mse, r.mae, model_rank(r.model)))[0]


def _build_per_dataset_best(rows: List[MetricsRow]) -> List[Dict[str, object]]:
    by_dataset: Dict[str, List[MetricsRow]] = {}
    for r in rows:
        by_dataset.setdefault(r.dataset, []).append(r)

    out: List[Dict[str, object]] = []
    for dataset in sorted(by_dataset.keys(), key=lambda x: x.lower()):
        best = _pick_best(by_dataset[dataset])
        if best is None:
            out.append(
                {
                    "dataset": dataset,
                    "best_model": "",
                    "best_model_id": "",
                    "best_MSE": "",
                    "best_MAE": "",
                }
            )
            continue
        out.append(
            {
                "dataset": dataset,
                "best_model": best.model,
                "best_model_id": best.model_id,
                "best_MSE": best.mse,
                "best_MAE": best.mae,
            }
        )
    return out


def _build_model_summary(rows: List[MetricsRow]) -> List[Dict[str, object]]:
    per_best = _build_per_dataset_best(rows)
    win_map: Dict[str, int] = {}
    for r in per_best:
        m = str(r["best_model"])
        if not m:
            continue
        win_map[m] = win_map.get(m, 0) + 1

    by_model_mse: Dict[str, List[float]] = {}
    by_model_mae: Dict[str, List[float]] = {}
    datasets_by_model: Dict[str, set] = {}
    for r in rows:
        datasets_by_model.setdefault(r.model, set()).add(r.dataset)
        if r.mse is not None:
            by_model_mse.setdefault(r.model, []).append(r.mse)
        if r.mae is not None:
            by_model_mae.setdefault(r.model, []).append(r.mae)

    def safe_stats(vals: List[float]) -> Tuple[object, object, object, object]:
        if not vals:
            return "", "", "", ""
        return (
            statistics.mean(vals),
            statistics.median(vals),
            min(vals),
            max(vals),
        )

    models = sorted(datasets_by_model.keys(), key=lambda x: (MODEL_ORDER_9.index(x) if x in MODEL_ORDER_9 else 10_000, x.lower()))
    out: List[Dict[str, object]] = []
    for m in models:
        mse_mean, mse_median, mse_min, mse_max = safe_stats(by_model_mse.get(m, []))
        mae_mean, mae_median, mae_min, mae_max = safe_stats(by_model_mae.get(m, []))
        out.append(
            {
                "model": m,
                "datasets_count": len(datasets_by_model.get(m, set())),
                "count_with_MSE": len(by_model_mse.get(m, [])),
                "count_with_MAE": len(by_model_mae.get(m, [])),
                "wins_by_(MSE,MAE)": win_map.get(m, 0),
                "mean_MSE": mse_mean,
                "median_MSE": mse_median,
                "min_MSE": mse_min,
                "max_MSE": mse_max,
                "mean_MAE": mae_mean,
                "median_MAE": mae_median,
                "min_MAE": mae_min,
                "max_MAE": mae_max,
            }
        )
    return out


def _read_best_baseline(best_root: Path) -> Dict[str, Dict[str, object]]:
    entries: Dict[str, Dict[str, object]] = {}
    for exp_dir in _iter_dirs(best_root):
        metrics_csv = exp_dir / "results" / "metrics.csv"
        mae, mse, rmse, mape, mspe = _read_metrics_csv(metrics_csv)
        if mse is None or mae is None:
            continue
        dataset = _dataset_from_model_id(exp_dir.name)
        existing = entries.get(dataset)
        if existing is None or (mse, mae) < (float(existing["best_MSE"]), float(existing["best_MAE"])):
            entries[dataset] = {
                "dataset": dataset,
                "best_model_id": exp_dir.name,
                "best_MSE": mse,
                "best_MAE": mae,
                "best_RMSE": rmse,
                "best_MAPE": mape,
                "best_MSPE": mspe,
            }
    return entries


def _build_vs_best(
    per_dataset_best: List[Dict[str, object]],
    best_entries: Dict[str, Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    per_map: Dict[str, Dict[str, object]] = {str(d["dataset"]): d for d in per_dataset_best if str(d.get("dataset", ""))}
    detail: List[Dict[str, object]] = []
    missing_best: List[str] = []
    beat_count = 0
    improvements: List[float] = []
    regressions: List[float] = []

    for dataset in sorted(per_map.keys(), key=lambda x: x.lower()):
        cand = per_map[dataset]
        if cand.get("best_MSE", "") == "" or cand.get("best_MAE", "") == "":
            continue
        best = best_entries.get(dataset)
        if best is None:
            missing_best.append(dataset)
            detail.append(
                {
                    "dataset": dataset,
                    "best_model_id": "",
                    "best_MSE": "",
                    "best_MAE": "",
                    "candidate_best_model_id": cand["best_model_id"],
                    "candidate_best_model": cand["best_model"],
                    "candidate_best_MSE": cand["best_MSE"],
                    "candidate_best_MAE": cand["best_MAE"],
                    "delta_MSE": "",
                    "relative_delta_MSE_percent": "",
                    "delta_MAE": "",
                    "relative_delta_MAE_percent": "",
                    "beats_best": "",
                }
            )
            continue

        delta_mse = float(cand["best_MSE"]) - float(best["best_MSE"])
        delta_mae = float(cand["best_MAE"]) - float(best["best_MAE"])
        rel_mse = (delta_mse / float(best["best_MSE"])) * 100.0 if float(best["best_MSE"]) != 0 else ""
        rel_mae = (delta_mae / float(best["best_MAE"])) * 100.0 if float(best["best_MAE"]) != 0 else ""
        beats = (float(cand["best_MSE"]), float(cand["best_MAE"])) < (float(best["best_MSE"]), float(best["best_MAE"]))

        if beats:
            beat_count += 1
            improvements.append(-delta_mse)
        else:
            regressions.append(delta_mse)

        detail.append(
            {
                "dataset": dataset,
                "best_model_id": best["best_model_id"],
                "best_MSE": best["best_MSE"],
                "best_MAE": best["best_MAE"],
                "candidate_best_model_id": cand["best_model_id"],
                "candidate_best_model": cand["best_model"],
                "candidate_best_MSE": cand["best_MSE"],
                "candidate_best_MAE": cand["best_MAE"],
                "delta_MSE": delta_mse,
                "relative_delta_MSE_percent": rel_mse,
                "delta_MAE": delta_mae,
                "relative_delta_MAE_percent": rel_mae,
                "beats_best": bool(beats),
            }
        )

    summary: List[Dict[str, object]] = [
        {
            "datasets_with_best": len(detail) - len(missing_best),
            "datasets_missing_best": len(missing_best),
            "beat_count": beat_count,
            "avg_MSE_improvement_when_beats": (sum(improvements) / len(improvements)) if improvements else "",
            "avg_MSE_regression_when_not_beats": (sum(regressions) / len(regressions)) if regressions else "",
            "missing_best_datasets": ";".join(missing_best),
        }
    ]
    return detail, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dst-name", default=DEFAULT_DST)
    parser.add_argument("--no-migrate", action="store_true")
    args = parser.parse_args()

    repo_root = _detect_repo_root(Path(__file__))
    dst_root = repo_root / "backup" / args.dst_name

    if not args.no_migrate:
        dst_root = _migrate_groups(repo_root, dst_name=args.dst_name, dry_run=args.dry_run)
    else:
        _ensure_dir(dst_root, dry_run=args.dry_run)

    rows = _collect_rows(dst_root)
    metrics_compare = _build_metrics_compare(rows)
    per_dataset_best = _build_per_dataset_best(rows)
    model_summary = _build_model_summary(rows)
    best_entries = _read_best_baseline(repo_root / "backup" / "best")
    vs_best_detail, vs_best_summary = _build_vs_best(per_dataset_best, best_entries)

    _write_csv(dst_root / "metrics_compare.csv", metrics_compare, dry_run=args.dry_run)
    _write_csv(dst_root / "per_dataset_best.csv", per_dataset_best, dry_run=args.dry_run)
    _write_csv(dst_root / "model_summary.csv", model_summary, dry_run=args.dry_run)
    _write_csv(dst_root / "vs_best_detail.csv", vs_best_detail, dry_run=args.dry_run)
    _write_csv(dst_root / "vs_best_summary.csv", vs_best_summary, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
