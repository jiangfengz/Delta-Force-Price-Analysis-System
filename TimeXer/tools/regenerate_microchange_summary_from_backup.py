from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_row(rows: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not rows:
        return None

    def key(r: Dict[str, object]) -> Tuple[float, float, float]:
        m = _safe_float(r.get("MSE"))
        a = _safe_float(r.get("MAE"))
        rm = _safe_float(r.get("RMSE"))
        mse = m if m is not None and math.isfinite(m) else float("inf")
        mae = a if a is not None and math.isfinite(a) else float("inf")
        rmse = rm if rm is not None and math.isfinite(rm) else float("inf")
        return mse, mae, rmse

    return min(rows, key=key)


def scan_microchange_backup(micro_root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    all_rows: List[Dict[str, object]] = []
    missing_summary: List[Dict[str, object]] = []

    for dataset_dir in sorted([p for p in micro_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        dataset = dataset_dir.name
        model_dirs = [p for p in dataset_dir.iterdir() if p.is_dir() and (p / "results" / "metrics.csv").exists()]
        dataset_rows: List[Dict[str, object]] = []
        seen_tags = set()

        for model_dir in sorted(model_dirs, key=lambda p: p.name.lower()):
            model_id = model_dir.name
            metrics_path = model_dir / "results" / "metrics.csv"
            metrics = read_metrics_csv(metrics_path)
            letter, tag = parse_variant(model_id)
            seen_tags.add(tag or "")

            row = {
                "dataset": dataset,
                "model_id": model_id,
                "variant_letter": letter or "",
                "variant_tag": tag or "",
                "MAE": metrics.mae,
                "MSE": metrics.mse,
                "RMSE": metrics.rmse,
                "MAPE": metrics.mape,
                "MSPE": metrics.mspe,
                "is_best_in_dataset": 0,
                "has_checkpoint": 1 if (model_dir / "checkpoints" / "checkpoint.pth").exists() else 0,
                "has_test_results": 1 if (model_dir / "test_results").exists() else 0,
            }
            dataset_rows.append(row)

        best = best_row(dataset_rows)
        if best is not None:
            for r in dataset_rows:
                r["is_best_in_dataset"] = 1 if r["model_id"] == best["model_id"] else 0

        expected = set(VARIANT_TAGS)
        missing = sorted(expected - seen_tags)
        if missing:
            missing_summary.append({"dataset": dataset, "missing_variant_tags": ";".join(missing)})

        if dataset_rows:
            write_csv(dataset_dir / "metrics_summary.csv", dataset_rows, fieldnames=list(dataset_rows[0].keys()))
            all_rows.extend(dataset_rows)

    return all_rows, missing_summary


def build_best_by_dataset(all_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for r in all_rows:
        grouped.setdefault(str(r["dataset"]), []).append(r)

    out: List[Dict[str, object]] = []
    for dataset, rows in sorted(grouped.items(), key=lambda x: x[0].lower()):
        best = best_row(rows)
        if best is None:
            continue
        out.append(
            {
                "dataset": dataset,
                "best_model_id": best["model_id"],
                "best_variant_letter": best["variant_letter"],
                "best_variant_tag": best["variant_tag"],
                "best_MAE": best["MAE"],
                "best_MSE": best["MSE"],
                "best_RMSE": best["RMSE"],
                "best_MAPE": best["MAPE"],
                "best_MSPE": best["MSPE"],
                "num_experiments": len(rows),
            }
        )
    return out


def scan_existing_best(best_root: Path) -> Dict[str, List[Tuple[str, Metrics]]]:
    existing: Dict[str, List[Tuple[str, Metrics]]] = {}
    if not best_root.exists():
        return existing
    for item in best_root.iterdir():
        if not item.is_dir():
            continue
        model_id = item.name
        metrics_path = item / "results" / "metrics.csv"
        if not metrics_path.exists():
            continue
        dataset = model_id
        if "_Collection_" in model_id:
            dataset = model_id.split("_Collection_", 1)[0]
        if dataset.startswith("V18_7d_"):
            dataset = dataset[len("V18_7d_") :]
        metrics = read_metrics_csv(metrics_path)
        existing.setdefault(dataset, []).append((model_id, metrics))
    return existing


def best_tuple(entries: List[Tuple[str, Metrics]]) -> Optional[Tuple[str, Metrics]]:
    if not entries:
        return None
    return min(entries, key=lambda t: t[1].key())


def write_best_replaced_report(repo_root: Path, micro_root: Path, best_by_dataset: List[Dict[str, object]]) -> None:
    best_root = repo_root / "backup" / "best in 7 days"
    existing = scan_existing_best(best_root)
    report_rows: List[Dict[str, object]] = []

    for row in best_by_dataset:
        dataset = str(row["dataset"])
        new_id = str(row["best_model_id"])
        new_mse = _safe_float(row.get("best_MSE"))
        new_mae = _safe_float(row.get("best_MAE"))

        existing_entries = existing.get(dataset, [])
        old_best = best_tuple(existing_entries)

        should_replace = old_best is None or Metrics(new_mae, new_mse, None, None, None).key() < old_best[1].key()
        report_rows.append(
            {
                "dataset": dataset,
                "old_best_model_id": old_best[0] if old_best is not None else "",
                "old_best_MSE": old_best[1].mse if old_best is not None else None,
                "old_best_MAE": old_best[1].mae if old_best is not None else None,
                "new_best_model_id": new_id,
                "new_best_MSE": new_mse,
                "new_best_MAE": new_mae,
                "replaced_or_added": 1 if should_replace else 0,
            }
        )

    if report_rows:
        write_csv(micro_root / "best_replaced_report.csv", report_rows, fieldnames=list(report_rows[0].keys()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--micro_root", type=str, default=None)
    parser.add_argument("--skip_best_replaced_report", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    micro_root = (
        Path(args.micro_root).resolve()
        if args.micro_root
        else (repo_root / "backup" / "best in 7 days 6 microchange")
    )

    all_rows, missing = scan_microchange_backup(micro_root)

    if all_rows:
        write_csv(micro_root / "summary_all_datasets.csv", all_rows, fieldnames=list(all_rows[0].keys()))

    best_by_dataset = build_best_by_dataset(all_rows)
    if best_by_dataset:
        write_csv(micro_root / "summary_best_by_dataset.csv", best_by_dataset, fieldnames=list(best_by_dataset[0].keys()))

    if missing:
        write_csv(micro_root / "missing_variants.csv", missing, fieldnames=list(missing[0].keys()))

    if not args.skip_best_replaced_report and best_by_dataset:
        write_best_replaced_report(repo_root, micro_root, best_by_dataset)

    print(f"Done. micro_root={micro_root} rows={len(all_rows)} datasets={len(best_by_dataset)}")


if __name__ == "__main__":
    main()

