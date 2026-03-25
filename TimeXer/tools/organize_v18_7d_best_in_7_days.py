from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


VARIANT_TAGS = [
    "Best_Base",
    "Best_LRDown",
    "Best_LRUp",
    "Best_LongSeq",
    "Best_Patch32",
    "Alt_2ndBase",
]


@dataclass(frozen=True)
class Metrics:
    mae: Optional[float]
    mse: Optional[float]
    rmse: Optional[float]
    mape: Optional[float]
    mspe: Optional[float]

    def key(self) -> Tuple[float, float, float]:
        mse = self.mse if self.mse is not None and math.isfinite(self.mse) else float("inf")
        mae = self.mae if self.mae is not None and math.isfinite(self.mae) else float("inf")
        rmse = self.rmse if self.rmse is not None and math.isfinite(self.rmse) else float("inf")
        return mse, mae, rmse


@dataclass(frozen=True)
class Experiment:
    model_id: str
    dataset: str
    variant_letter: Optional[str]
    variant_tag: Optional[str]
    metrics: Metrics
    results_dir: Path
    checkpoint_path: Optional[Path]
    test_results_dir: Optional[Path]


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def read_metrics_csv(metrics_csv_path: Path) -> Metrics:
    with metrics_csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        if not row:
            return Metrics(None, None, None, None, None)
        return Metrics(
            mae=_safe_float(row.get("MAE")),
            mse=_safe_float(row.get("MSE")),
            rmse=_safe_float(row.get("RMSE")),
            mape=_safe_float(row.get("MAPE")),
            mspe=_safe_float(row.get("MSPE")),
        )


def parse_dataset_from_model_id(model_id: str) -> str:
    if "_Collection_" in model_id:
        prefix = model_id.split("_Collection_", 1)[0]
    else:
        prefix = model_id
    if prefix.startswith("V18_7d_"):
        return prefix[len("V18_7d_") :]
    if prefix.startswith("V17_7d_"):
        return prefix[len("V17_7d_") :]
    if prefix.startswith("V"):
        parts = prefix.split("_", 2)
        if len(parts) >= 3:
            return parts[2]
    return prefix


def parse_variant(model_id: str) -> Tuple[Optional[str], Optional[str]]:
    for tag in VARIANT_TAGS:
        needle = f"_{tag}_"
        idx = model_id.find(needle)
        if idx == -1:
            continue
        before = model_id[:idx]
        if not before:
            return None, tag
        last_us = before.rfind("_")
        if last_us == -1 or last_us == len(before) - 1:
            return None, tag
        letter = before[last_us + 1 :]
        if len(letter) == 1 and letter in "ABCDEF":
            return letter, tag
        return None, tag
    return None, None


def scan_v18_experiments(repo_root: Path) -> List[Experiment]:
    results_root = repo_root / "results"
    checkpoints_root = repo_root / "checkpoints"
    test_results_root = repo_root / "test_results"
    experiments: List[Experiment] = []

    if not results_root.exists():
        return experiments

    for exp_dir in results_root.iterdir():
        if not exp_dir.is_dir():
            continue
        model_id = exp_dir.name
        if not model_id.startswith("V18_7d_"):
            continue
        metrics_path = exp_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = read_metrics_csv(metrics_path)
        dataset = parse_dataset_from_model_id(model_id)
        letter, tag = parse_variant(model_id)

        ckpt = checkpoints_root / model_id / "checkpoint.pth"
        ckpt_path = ckpt if ckpt.exists() else None

        tr_dir = test_results_root / model_id
        tr_path = tr_dir if tr_dir.exists() and tr_dir.is_dir() else None

        experiments.append(
            Experiment(
                model_id=model_id,
                dataset=dataset,
                variant_letter=letter,
                variant_tag=tag,
                metrics=metrics,
                results_dir=exp_dir,
                checkpoint_path=ckpt_path,
                test_results_dir=tr_path,
            )
        )
    return experiments


def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def copy_tree_files(src_dir: Path, dst_dir: Path, dry_run: bool) -> int:
    if not src_dir.exists():
        return 0
    ensure_dir(dst_dir, dry_run=dry_run)
    copied = 0
    for item in src_dir.iterdir():
        if item.is_dir():
            copied += copy_tree_files(item, dst_dir / item.name, dry_run=dry_run)
        else:
            if not dry_run:
                shutil.copy2(item, dst_dir / item.name)
            copied += 1
    return copied


def copy_experiment_artifacts(exp: Experiment, dst_model_dir: Path, dry_run: bool) -> Dict[str, int]:
    counts = {"results": 0, "checkpoints": 0, "test_results": 0}
    ensure_dir(dst_model_dir, dry_run=dry_run)

    results_dst = dst_model_dir / "results"
    counts["results"] = copy_tree_files(exp.results_dir, results_dst, dry_run=dry_run)

    ckpt_dst_dir = dst_model_dir / "checkpoints"
    ensure_dir(ckpt_dst_dir, dry_run=dry_run)
    if exp.checkpoint_path is not None:
        if not dry_run:
            shutil.copy2(exp.checkpoint_path, ckpt_dst_dir / "checkpoint.pth")
        counts["checkpoints"] = 1

    tr_dst = dst_model_dir / "test_results"
    if exp.test_results_dir is not None:
        counts["test_results"] = copy_tree_files(exp.test_results_dir, tr_dst, dry_run=dry_run)

    return counts


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str], dry_run: bool) -> None:
    ensure_dir(path.parent, dry_run=dry_run)
    if dry_run:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_by_dataset(experiments: Iterable[Experiment]) -> Dict[str, List[Experiment]]:
    grouped: Dict[str, List[Experiment]] = {}
    for exp in experiments:
        grouped.setdefault(exp.dataset, []).append(exp)
    return grouped


def best_experiment(exps: List[Experiment]) -> Optional[Experiment]:
    if not exps:
        return None
    return min(exps, key=lambda e: e.metrics.key())


def sort_experiments_for_report(exps: List[Experiment]) -> List[Experiment]:
    letter_order = {c: i for i, c in enumerate("ABCDEF", start=1)}

    def key(e: Experiment) -> Tuple[int, str]:
        lo = letter_order.get(e.variant_letter or "", 99)
        return lo, e.model_id

    return sorted(exps, key=key)


def organize_microchange_backup(repo_root: Path, dry_run: bool) -> Dict[str, object]:
    dest_root = repo_root / "backup" / "best in 7 days 6 microchange"
    experiments = scan_v18_experiments(repo_root)
    grouped = group_by_dataset(experiments)

    all_rows: List[Dict[str, object]] = []
    dataset_best_rows: List[Dict[str, object]] = []
    missing_summary: List[Dict[str, object]] = []

    for dataset, exps in sorted(grouped.items(), key=lambda x: x[0].lower()):
        dataset_dir = dest_root / dataset
        ensure_dir(dataset_dir, dry_run=dry_run)

        exps_sorted = sort_experiments_for_report(exps)
        best = best_experiment(exps)

        seen_tags = set()
        for exp in exps_sorted:
            seen_tags.add(exp.variant_tag or "")
            dst_model_dir = dataset_dir / exp.model_id
            copy_experiment_artifacts(exp, dst_model_dir, dry_run=dry_run)

            row = {
                "dataset": dataset,
                "model_id": exp.model_id,
                "variant_letter": exp.variant_letter or "",
                "variant_tag": exp.variant_tag or "",
                "MAE": exp.metrics.mae,
                "MSE": exp.metrics.mse,
                "RMSE": exp.metrics.rmse,
                "MAPE": exp.metrics.mape,
                "MSPE": exp.metrics.mspe,
                "is_best_in_dataset": 1 if best is not None and best.model_id == exp.model_id else 0,
                "has_checkpoint": 1 if exp.checkpoint_path is not None else 0,
                "has_test_results": 1 if exp.test_results_dir is not None else 0,
            }
            all_rows.append(row)

        expected = set(VARIANT_TAGS)
        missing = sorted(expected - seen_tags)
        if missing:
            missing_summary.append({"dataset": dataset, "missing_variant_tags": ";".join(missing)})

        dataset_metrics_summary_path = dataset_dir / "metrics_summary.csv"
        dataset_rows = [r for r in all_rows if r["dataset"] == dataset]
        write_csv(
            dataset_metrics_summary_path,
            dataset_rows,
            fieldnames=list(dataset_rows[0].keys()) if dataset_rows else [],
            dry_run=dry_run,
        )

        if best is not None:
            dataset_best_rows.append(
                {
                    "dataset": dataset,
                    "best_model_id": best.model_id,
                    "best_variant_letter": best.variant_letter or "",
                    "best_variant_tag": best.variant_tag or "",
                    "best_MAE": best.metrics.mae,
                    "best_MSE": best.metrics.mse,
                    "best_RMSE": best.metrics.rmse,
                    "best_MAPE": best.metrics.mape,
                    "best_MSPE": best.metrics.mspe,
                    "num_experiments": len(exps),
                }
            )

    summary_all_path = dest_root / "summary_all_datasets.csv"
    if all_rows:
        write_csv(summary_all_path, all_rows, fieldnames=list(all_rows[0].keys()), dry_run=dry_run)

    summary_best_path = dest_root / "summary_best_by_dataset.csv"
    if dataset_best_rows:
        write_csv(summary_best_path, dataset_best_rows, fieldnames=list(dataset_best_rows[0].keys()), dry_run=dry_run)

    if missing_summary:
        missing_path = dest_root / "missing_variants.csv"
        write_csv(missing_path, missing_summary, fieldnames=list(missing_summary[0].keys()), dry_run=dry_run)

    return {
        "num_experiments": len(experiments),
        "num_datasets": len(grouped),
        "missing_datasets": len(missing_summary),
        "dest_root": str(dest_root),
    }


def scan_existing_best_in_7_days(best_root: Path) -> Dict[str, List[Tuple[str, Metrics]]]:
    existing: Dict[str, List[Tuple[str, Metrics]]] = {}
    if not best_root.exists():
        return existing
    for item in best_root.iterdir():
        if not item.is_dir():
            continue
        model_id = item.name
        dataset = parse_dataset_from_model_id(model_id)
        metrics_path = item / "results" / "metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = read_metrics_csv(metrics_path)
        existing.setdefault(dataset, []).append((model_id, metrics))
    return existing


def move_dir(src: Path, dst: Path, dry_run: bool) -> None:
    ensure_dir(dst.parent, dry_run=dry_run)
    if dry_run:
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))


def update_best_in_7_days(repo_root: Path, dry_run: bool) -> Dict[str, object]:
    best_root = repo_root / "backup" / "best in 7 days"
    replaced_root = best_root / f"_replaced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    v18_exps = scan_v18_experiments(repo_root)
    grouped = group_by_dataset(v18_exps)
    existing = scan_existing_best_in_7_days(best_root)

    report_rows: List[Dict[str, object]] = []
    replaced_count = 0
    added_count = 0

    for dataset, exps in sorted(grouped.items(), key=lambda x: x[0].lower()):
        v18_best = best_experiment(exps)
        if v18_best is None:
            continue

        existing_entries = existing.get(dataset, [])
        existing_best = None
        if existing_entries:
            existing_best = min(existing_entries, key=lambda t: t[1].key())

        should_replace = existing_best is None or v18_best.metrics.key() < existing_best[1].key()

        if should_replace:
            if existing_best is not None:
                old_model_id = existing_best[0]
                old_dir = best_root / old_model_id
                if old_dir.exists():
                    move_dir(old_dir, replaced_root / old_model_id, dry_run=dry_run)
                replaced_count += 1
            else:
                added_count += 1

            dst_model_dir = best_root / v18_best.model_id
            copy_experiment_artifacts(
                v18_best,
                dst_model_dir,
                dry_run=dry_run,
            )

        report_rows.append(
            {
                "dataset": dataset,
                "old_best_model_id": existing_best[0] if existing_best is not None else "",
                "old_best_MSE": existing_best[1].mse if existing_best is not None else None,
                "old_best_MAE": existing_best[1].mae if existing_best is not None else None,
                "new_best_model_id": v18_best.model_id,
                "new_best_MSE": v18_best.metrics.mse,
                "new_best_MAE": v18_best.metrics.mae,
                "replaced_or_added": 1 if should_replace else 0,
            }
        )

    report_path = repo_root / "backup" / "best in 7 days 6 microchange" / "best_replaced_report.csv"
    if report_rows:
        write_csv(report_path, report_rows, fieldnames=list(report_rows[0].keys()), dry_run=dry_run)

    return {
        "best_root": str(best_root),
        "replaced_root": str(replaced_root),
        "replaced_count": replaced_count,
        "added_count": added_count,
        "report_path": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_update_best", action="store_true")
    args = parser.parse_args()

    if args.repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    else:
        repo_root = Path(args.repo_root).resolve()

    micro = organize_microchange_backup(repo_root, dry_run=args.dry_run)
    best = None
    if not args.skip_update_best:
        best = update_best_in_7_days(repo_root, dry_run=args.dry_run)

    print("organize_microchange_backup:", micro)
    if best is not None:
        print("update_best_in_7_days:", best)


if __name__ == "__main__":
    main()

