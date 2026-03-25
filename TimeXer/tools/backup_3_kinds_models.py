from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MODEL_DIRS = ["Opt2_M_192_A", "Exp_Large_M_768_A", "Exp_Long_M_336_A"]


@dataclass(frozen=True)
class MetricsRow:
    dataset: str
    model_key: str
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


def _dataset_from_model_id(model_id: str) -> str:
    if "_Collection_Category_" in model_id:
        return model_id.split("_Collection_Category_", 1)[0]
    return model_id.split("_", 1)[0]


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


def _iter_result_dirs(results_root: Path) -> Iterable[Path]:
    if not results_root.exists():
        return []
    for p in results_root.iterdir():
        if p.is_dir():
            yield p


def _collect_rows(backup_root: Path) -> List[MetricsRow]:
    rows: List[MetricsRow] = []
    for model_key in MODEL_DIRS:
        results_root = backup_root / model_key / "results"
        for exp_dir in _iter_result_dirs(results_root):
            model_id = exp_dir.name
            metrics_csv = exp_dir / "metrics.csv"
            mae, mse, rmse, mape, mspe = _read_metrics_csv(metrics_csv)
            dataset = _dataset_from_model_id(model_id)
            rows.append(
                MetricsRow(
                    dataset=dataset,
                    model_key=model_key,
                    model_id=model_id,
                    mae=mae,
                    mse=mse,
                    rmse=rmse,
                    mape=mape,
                    mspe=mspe,
                )
            )
    rows.sort(key=lambda r: (r.dataset.lower(), MODEL_DIRS.index(r.model_key)))
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _build_metrics_compare(rows: List[MetricsRow]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in rows:
        out.append(
            {
                "dataset": r.dataset,
                "model_key": r.model_key,
                "model_id": r.model_id,
                "MAE": r.mae,
                "MSE": r.mse,
                "RMSE": r.rmse,
                "MAPE": r.mape,
                "MSPE": r.mspe,
            }
        )
    return out


def _build_per_dataset_best(rows: List[MetricsRow]) -> List[Dict[str, object]]:
    by_dataset: Dict[str, List[MetricsRow]] = {}
    for r in rows:
        by_dataset.setdefault(r.dataset, []).append(r)

    out: List[Dict[str, object]] = []
    for dataset in sorted(by_dataset.keys(), key=lambda x: x.lower()):
        candidates = [r for r in by_dataset[dataset] if r.mse is not None]
        if not candidates:
            out.append(
                {
                    "dataset": dataset,
                    "best_model_key": None,
                    "best_model_id": None,
                    "best_MSE": None,
                }
            )
            continue
        best = min(candidates, key=lambda r: (r.mse, MODEL_DIRS.index(r.model_key)))
        out.append(
            {
                "dataset": dataset,
                "best_model_key": best.model_key,
                "best_model_id": best.model_id,
                "best_MSE": best.mse,
            }
        )
    return out


def _build_best_model_summary(rows: List[MetricsRow]) -> List[Dict[str, object]]:
    per_best = _build_per_dataset_best(rows)
    wins: Dict[str, int] = {k: 0 for k in MODEL_DIRS}
    for r in per_best:
        k = r["best_model_key"]
        if k in wins:
            wins[k] += 1

    by_model: Dict[str, List[float]] = {k: [] for k in MODEL_DIRS}
    for r in rows:
        if r.mse is None:
            continue
        by_model[r.model_key].append(r.mse)

    out: List[Dict[str, object]] = []
    for k in MODEL_DIRS:
        mses = by_model.get(k) or []
        out.append(
            {
                "model_key": k,
                "wins_by_MSE": wins.get(k, 0),
                "count_with_MSE": len(mses),
                "mean_MSE": statistics.mean(mses) if mses else None,
                "median_MSE": statistics.median(mses) if mses else None,
                "min_MSE": min(mses) if mses else None,
                "max_MSE": max(mses) if mses else None,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-root", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = _detect_repo_root(Path(__file__))
    backup_root = Path(args.backup_root) if args.backup_root else (repo_root / "backup" / "3 kinds of models")
    backup_root = backup_root.resolve()

    rows = _collect_rows(backup_root)

    metrics_compare = _build_metrics_compare(rows)
    per_dataset_best = _build_per_dataset_best(rows)
    best_model_summary = _build_best_model_summary(rows)

    _write_csv(backup_root / "metrics_compare.csv", metrics_compare, dry_run=args.dry_run)
    _write_csv(backup_root / "per_dataset_best.csv", per_dataset_best, dry_run=args.dry_run)
    _write_csv(backup_root / "best_model_summary.csv", best_model_summary, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

