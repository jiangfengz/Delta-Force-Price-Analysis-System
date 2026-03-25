from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


FAMILIES_5 = [
    "Opt2_192_Plateau4",
    "Deep6_240",
    "MidLarge_640",
    "Exp_Long_M_336_A",
    "LongCtx2_384",
]

FAMILY_ORDER = list(FAMILIES_5)


@dataclass(frozen=True)
class MetricsRow:
    dataset: str
    family: str
    model_id: str
    mae: Optional[float]
    mse: Optional[float]
    rmse: Optional[float]
    mape: Optional[float]
    mspe: Optional[float]


def _detect_repo_root(script_path: Path) -> Path:
    root = script_path.resolve()
    for _ in range(7):
        if (root / "run.py").exists() and (root / "results").exists():
            return root
        root = root.parent
    raise RuntimeError("无法自动定位项目根目录（未找到 run.py 与 results/）")


def _ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def _copy_tree_merge(src_dir: Path, dst_dir: Path, conflicts: List[Dict[str, str]], dry_run: bool) -> bool:
    if not src_dir.exists():
        return False
    had_conflict = False
    for src_path in src_dir.rglob("*"):
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel
        if src_path.is_dir():
            _ensure_dir(dst_path, dry_run=dry_run)
            continue
        _ensure_dir(dst_path.parent, dry_run=dry_run)
        if dst_path.exists():
            try:
                same = (
                    dst_path.stat().st_size == src_path.stat().st_size
                    and int(dst_path.stat().st_mtime) == int(src_path.stat().st_mtime)
                )
            except OSError:
                same = False
            if same:
                continue
            conflicts.append({"src": str(src_path), "dst": str(dst_path)})
            had_conflict = True
            continue
        if not dry_run:
            shutil.copy2(src_path, dst_path)
    return had_conflict


def _delete_path(path: Path, dry_run: bool) -> None:
    if not path.exists() or dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _iter_experiment_ids(results_dir: Path) -> Iterable[str]:
    if not results_dir.exists():
        return []
    for p in results_dir.iterdir():
        if p.is_dir():
            yield p.name


_RE_VERSION_PREFIX = re.compile(r"^V\d+_")
_RE_DAYS_PREFIX = re.compile(r"^\d+d_")


def _extract_family(model_id: str) -> Optional[str]:
    if "_Collection_Category_" not in model_id:
        return None
    rest = model_id.split("_Collection_Category_", 1)[1]
    for family in sorted(FAMILIES_5, key=len, reverse=True):
        if rest.startswith(family + "_"):
            return family
    return None


def _dataset_key_from_model_id(model_id: str) -> Optional[str]:
    if "_Collection_Category_" not in model_id:
        return None
    prefix = model_id.split("_Collection_Category_", 1)[0]
    prefix = _RE_VERSION_PREFIX.sub("", prefix)
    prefix = _RE_DAYS_PREFIX.sub("", prefix)
    return prefix or None


def _read_metrics_csv(metrics_csv: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
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
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v

    return get("MAE"), get("MSE"), get("RMSE"), get("MAPE"), get("MSPE")


def _validate_copy(src: Path, dst: Path, required_rel: List[Path]) -> List[str]:
    missing: List[str] = []
    for rel in required_rel:
        if (src / rel).exists() and not (dst / rel).exists():
            missing.append(str(dst / rel))
    return missing


def _write_csv(path: Path, rows: List[Dict[str, object]], dry_run: bool) -> None:
    if not rows or dry_run:
        return
    _ensure_dir(path.parent, dry_run=False)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _pick_best_row(rows: List[MetricsRow]) -> Optional[MetricsRow]:
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    candidates = [r for r in rows if r.mse is not None and r.mae is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (x.mse, x.mae, rank.get(x.family, 10_000)))[0]


def _read_best_baseline(best_root: Path) -> Dict[str, Dict[str, object]]:
    entries: Dict[str, Dict[str, object]] = {}
    if not best_root.exists():
        return entries
    for exp_dir in best_root.iterdir():
        if not exp_dir.is_dir():
            continue
        metrics_csv = exp_dir / "results" / "metrics.csv"
        mae, mse, rmse, mape, mspe = _read_metrics_csv(metrics_csv)
        if mse is None or mae is None:
            continue
        dataset = _dataset_key_from_model_id(exp_dir.name)
        if not dataset:
            continue
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
                    "candidate_best_family": cand["best_family"],
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
                "candidate_best_family": cand["best_family"],
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
    parser.add_argument("--dst-name", default="5 models in 7 days")
    parser.add_argument("--experiment-prefix", default="V17_7d_")
    args = parser.parse_args()

    repo_root = _detect_repo_root(Path(__file__))
    src_results = repo_root / "results"
    src_checkpoints = repo_root / "checkpoints"
    src_test_results = repo_root / "test_results"

    dst_root = repo_root / "backup" / args.dst_name
    best_root = repo_root / "backup" / "best"

    conflicts: List[Dict[str, str]] = []
    copy_errors: List[Dict[str, object]] = []

    experiment_ids: List[str] = []
    id_to_family: Dict[str, str] = {}
    for model_id in _iter_experiment_ids(src_results):
        if args.experiment_prefix and not model_id.startswith(args.experiment_prefix):
            continue
        family = _extract_family(model_id)
        if family is None or family not in FAMILIES_5:
            continue
        experiment_ids.append(model_id)
        id_to_family[model_id] = family
    experiment_ids = sorted(set(experiment_ids))

    moved_ids: List[str] = []
    deleted_ids: List[str] = []
    skipped_delete_due_to_conflict: List[str] = []

    for model_id in experiment_ids:
        family = id_to_family[model_id]

        dst_family_root = dst_root / family
        dst_results_dir = dst_family_root / "results" / model_id
        dst_ckpt_dir = dst_family_root / "checkpoints" / model_id
        dst_test_dir = dst_family_root / "test_results" / model_id

        src_results_dir = src_results / model_id
        src_ckpt_dir = src_checkpoints / model_id
        src_test_dir = src_test_results / model_id

        model_has_conflict = False
        model_has_conflict |= _copy_tree_merge(src_results_dir, dst_results_dir, conflicts=conflicts, dry_run=args.dry_run)
        model_has_conflict |= _copy_tree_merge(src_ckpt_dir, dst_ckpt_dir, conflicts=conflicts, dry_run=args.dry_run)
        model_has_conflict |= _copy_tree_merge(src_test_dir, dst_test_dir, conflicts=conflicts, dry_run=args.dry_run)

        required_results = [Path("metrics.csv"), Path("log.txt")]
        required_ckpt = [Path("checkpoint.pth")]

        missing: List[str] = []
        if args.dry_run:
            for rel in required_results:
                if not (src_results_dir / rel).exists():
                    missing.append(str(src_results_dir / rel))
            for rel in required_ckpt:
                if not (src_ckpt_dir / rel).exists():
                    missing.append(str(src_ckpt_dir / rel))
        else:
            missing += _validate_copy(src_results_dir, dst_results_dir, required_results)
            missing += _validate_copy(src_ckpt_dir, dst_ckpt_dir, required_ckpt)
            src_pdfs = list(src_test_dir.glob("*.pdf")) if src_test_dir.exists() else []
            if src_pdfs and not list(dst_test_dir.glob("*.pdf")):
                missing.append(str(dst_test_dir))

        if missing:
            copy_errors.append({"model_id": model_id, "missing": missing})
            continue

        if not args.dry_run:
            if model_has_conflict:
                skipped_delete_due_to_conflict.append(model_id)
            else:
                _delete_path(src_results_dir, dry_run=False)
                _delete_path(src_ckpt_dir, dry_run=False)
                _delete_path(src_test_dir, dry_run=False)
                deleted_ids.append(model_id)

        moved_ids.append(model_id)

    if conflicts:
        _write_csv(dst_root / "conflicts.csv", conflicts, dry_run=args.dry_run)

    if copy_errors:
        if not args.dry_run:
            (dst_root / "copy_errors.json").write_text(
                json.dumps(copy_errors, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        raise RuntimeError(f"存在 {len(copy_errors)} 个实验复制校验失败，已输出 copy_errors.json")

    metrics_rows: List[MetricsRow] = []
    scan_root = dst_root
    if scan_root.exists():
        for family in FAMILY_ORDER:
            base = scan_root / family / "results"
            if not base.exists():
                continue
            for metrics_csv in base.glob("*/metrics.csv"):
                model_id = metrics_csv.parent.name
                dataset = _dataset_key_from_model_id(model_id)
                if dataset is None:
                    continue
                mae, mse, rmse, mape, mspe = _read_metrics_csv(metrics_csv)
                metrics_rows.append(
                    MetricsRow(
                        dataset=dataset,
                        family=family,
                        model_id=model_id,
                        mae=mae,
                        mse=mse,
                        rmse=rmse,
                        mape=mape,
                        mspe=mspe,
                    )
                )

    dataset_to_rows: Dict[str, Dict[str, MetricsRow]] = {}
    for r in metrics_rows:
        dataset_to_rows.setdefault(r.dataset, {})[r.family] = r

    compare_out: List[Dict[str, object]] = []
    per_dataset_best: List[Dict[str, object]] = []
    for dataset in sorted(dataset_to_rows.keys(), key=lambda x: x.lower()):
        for family in FAMILY_ORDER:
            r = dataset_to_rows[dataset].get(family)
            if r is None:
                compare_out.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "model_id": "",
                        "MAE": "",
                        "MSE": "",
                        "RMSE": "",
                        "MAPE": "",
                        "MSPE": "",
                    }
                )
                continue
            compare_out.append(
                {
                    "dataset": r.dataset,
                    "family": r.family,
                    "model_id": r.model_id,
                    "MAE": r.mae,
                    "MSE": r.mse,
                    "RMSE": r.rmse,
                    "MAPE": r.mape,
                    "MSPE": r.mspe,
                }
            )

        best = _pick_best_row(list(dataset_to_rows[dataset].values()))
        if best is None:
            per_dataset_best.append(
                {
                    "dataset": dataset,
                    "best_family": "",
                    "best_model_id": "",
                    "best_MSE": "",
                    "best_MAE": "",
                }
            )
            continue
        per_dataset_best.append(
            {
                "dataset": dataset,
                "best_family": best.family,
                "best_model_id": best.model_id,
                "best_MSE": best.mse,
                "best_MAE": best.mae,
            }
        )

    _write_csv(dst_root / "metrics_compare.csv", compare_out, dry_run=args.dry_run)
    _write_csv(dst_root / "per_dataset_best.csv", per_dataset_best, dry_run=args.dry_run)

    win_map: Dict[str, int] = {}
    for r in per_dataset_best:
        m = str(r.get("best_family", ""))
        if not m:
            continue
        win_map[m] = win_map.get(m, 0) + 1

    by_family_mse: Dict[str, List[float]] = {}
    by_family_mae: Dict[str, List[float]] = {}
    datasets_by_family: Dict[str, set] = {}
    for r in metrics_rows:
        datasets_by_family.setdefault(r.family, set()).add(r.dataset)
        if r.mse is not None:
            by_family_mse.setdefault(r.family, []).append(r.mse)
        if r.mae is not None:
            by_family_mae.setdefault(r.family, []).append(r.mae)

    def safe_stats(vals: List[float]) -> Tuple[object, object, object, object]:
        if not vals:
            return "", "", "", ""
        return (
            statistics.mean(vals),
            statistics.median(vals),
            min(vals),
            max(vals),
        )

    model_summary: List[Dict[str, object]] = []
    for family in FAMILY_ORDER:
        mse_mean, mse_median, mse_min, mse_max = safe_stats(by_family_mse.get(family, []))
        mae_mean, mae_median, mae_min, mae_max = safe_stats(by_family_mae.get(family, []))
        model_summary.append(
            {
                "family": family,
                "datasets_count": len(datasets_by_family.get(family, set())),
                "count_with_MSE": len(by_family_mse.get(family, [])),
                "count_with_MAE": len(by_family_mae.get(family, [])),
                "wins_by_(MSE,MAE)": win_map.get(family, 0),
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
    _write_csv(dst_root / "model_summary.csv", model_summary, dry_run=args.dry_run)

    best_entries = _read_best_baseline(best_root)
    vs_best_detail, vs_best_summary = _build_vs_best(per_dataset_best, best_entries)
    _write_csv(dst_root / "vs_best_detail.csv", vs_best_detail, dry_run=args.dry_run)
    _write_csv(dst_root / "vs_best_summary.csv", vs_best_summary, dry_run=args.dry_run)

    print(
        f"识别待转移实验数：{len(experiment_ids)}，本次写入备份：{len(moved_ids)}，本次删除源目录：{len(deleted_ids)}，冲突文件条目：{len(conflicts)}"
    )
    if skipped_delete_due_to_conflict:
        print(f"因冲突未删除源目录的实验数：{len(skipped_delete_due_to_conflict)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
