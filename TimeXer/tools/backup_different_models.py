from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MODEL_NAME_TO_TYPE = {
    "TimeXer": "timexer",
    "iTransformer": "itransformer",
    "PatchTST": "patchtst",
}

MODEL_TYPE_ORDER = ["patchtst", "timexer", "itransformer"]


@dataclass(frozen=True)
class MetricsRow:
    dataset: str
    model_type: str
    model_id: str
    mae: Optional[float]
    mse: Optional[float]
    rmse: Optional[float]
    mape: Optional[float]
    mspe: Optional[float]


def _detect_repo_root(script_path: Path) -> Path:
    root = script_path.resolve()
    for _ in range(6):
        if (root / "run.py").exists() and (root / "results").exists():
            return root
        root = root.parent
    raise RuntimeError("无法自动定位项目根目录（未找到 run.py 与 results/）")


def _ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def _copy_tree_merge(
    src_dir: Path, dst_dir: Path, conflicts: List[Dict[str, str]], dry_run: bool
) -> None:
    if not src_dir.exists():
        return
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
            continue
        if not dry_run:
            shutil.copy2(src_path, dst_path)


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


def _detect_model_type(model_id: str) -> Optional[str]:
    for model_name, model_type in MODEL_NAME_TO_TYPE.items():
        if f"_{model_name}_" in model_id or model_id.endswith(f"_{model_name}"):
            return model_type
    return None


def _dataset_from_model_id(model_id: str) -> str:
    if "_Collection_Category_" in model_id:
        return model_id.split("_Collection_Category_", 1)[0]
    return model_id.split("_", 1)[0]


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
            return float(row[i])
        except ValueError:
            return None
    return get("MAE"), get("MSE"), get("RMSE"), get("MAPE"), get("MSPE")


def _validate_copy(src: Path, dst: Path, required_rel: List[Path]) -> List[str]:
    missing: List[str] = []
    for rel in required_rel:
        if (src / rel).exists() and not (dst / rel).exists():
            missing.append(str(dst / rel))
    return missing


def _write_csv(path: Path, rows: List[Dict[str, object]], dry_run: bool) -> None:
    if not rows:
        return
    if dry_run:
        return
    _ensure_dir(path.parent, dry_run=False)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = _detect_repo_root(Path(__file__))
    src_results = repo_root / "results"
    src_checkpoints = repo_root / "checkpoints"
    src_test_results = repo_root / "test_results"

    dst_root = repo_root / "backup" / "different models"
    conflicts: List[Dict[str, str]] = []

    experiment_ids: List[str] = []
    for model_id in _iter_experiment_ids(src_results):
        model_type = _detect_model_type(model_id)
        if model_type is None:
            continue
        experiment_ids.append(model_id)
    experiment_ids = sorted(set(experiment_ids))

    moved_ids: List[str] = []
    copy_errors: List[Dict[str, object]] = []

    for model_id in experiment_ids:
        model_type = _detect_model_type(model_id)
        if model_type is None:
            continue

        dst_model_root = dst_root / model_type
        dst_results_dir = dst_model_root / "results" / model_id
        dst_ckpt_dir = dst_model_root / "checkpoints" / model_id
        dst_test_dir = dst_model_root / "test_results" / model_id

        src_results_dir = src_results / model_id
        src_ckpt_dir = src_checkpoints / model_id
        src_test_dir = src_test_results / model_id

        _copy_tree_merge(src_results_dir, dst_results_dir, conflicts=conflicts, dry_run=args.dry_run)
        _copy_tree_merge(src_ckpt_dir, dst_ckpt_dir, conflicts=conflicts, dry_run=args.dry_run)
        _copy_tree_merge(src_test_dir, dst_test_dir, conflicts=conflicts, dry_run=args.dry_run)

        required_results = [Path("metrics.csv"), Path("log.txt")]
        required_ckpt = [Path("checkpoint.pth")]

        missing: List[str] = []
        missing += _validate_copy(src_results_dir, dst_results_dir, required_results)
        missing += _validate_copy(src_ckpt_dir, dst_ckpt_dir, required_ckpt)

        src_pdfs = list(src_test_dir.glob("*.pdf")) if src_test_dir.exists() else []
        if src_pdfs:
            if not list(dst_test_dir.glob("*.pdf")):
                missing.append(str(dst_test_dir))

        if missing:
            copy_errors.append({"model_id": model_id, "missing": missing})
            continue

        if not args.dry_run:
            _delete_path(src_results_dir, dry_run=False)
            _delete_path(src_ckpt_dir, dry_run=False)
            _delete_path(src_test_dir, dry_run=False)

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
    for model_id in moved_ids:
        model_type = _detect_model_type(model_id)
        if model_type is None:
            continue
        metrics_csv = dst_root / model_type / "results" / model_id / "metrics.csv"
        mae, mse, rmse, mape, mspe = _read_metrics_csv(metrics_csv)
        metrics_rows.append(
            MetricsRow(
                dataset=_dataset_from_model_id(model_id),
                model_type=model_type,
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
        dataset_to_rows.setdefault(r.dataset, {})[r.model_type] = r

    compare_out: List[Dict[str, object]] = []
    per_dataset_best: List[Dict[str, object]] = []

    for dataset in sorted(dataset_to_rows.keys()):
        for model_type in MODEL_TYPE_ORDER:
            r = dataset_to_rows[dataset].get(model_type)
            if r is None:
                compare_out.append(
                    {
                        "dataset": dataset,
                        "model_type": model_type,
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
                    "model_type": r.model_type,
                    "model_id": r.model_id,
                    "MAE": r.mae,
                    "MSE": r.mse,
                    "RMSE": r.rmse,
                    "MAPE": r.mape,
                    "MSPE": r.mspe,
                }
            )

        rows_for_best = [
            dataset_to_rows[dataset].get(mt)
            for mt in MODEL_TYPE_ORDER
            if dataset_to_rows[dataset].get(mt) is not None and dataset_to_rows[dataset][mt].mse is not None
        ]
        if rows_for_best:
            best = sorted(
                rows_for_best,
                key=lambda x: (
                    float("inf") if x.mse is None else x.mse,
                    float("inf") if x.mae is None else x.mae,
                ),
            )[0]
            per_dataset_best.append(
                {"dataset": dataset, "best_model_type": best.model_type, "best_model_id": best.model_id, "best_MSE": best.mse, "best_MAE": best.mae}
            )

    _write_csv(dst_root / "metrics_compare.csv", compare_out, dry_run=args.dry_run)
    _write_csv(dst_root / "per_dataset_best.csv", per_dataset_best, dry_run=args.dry_run)

    agg: Dict[str, Dict[str, object]] = {mt: {"model_type": mt, "count": 0, "sum_MSE": 0.0, "sum_MAE": 0.0} for mt in MODEL_TYPE_ORDER}
    for r in metrics_rows:
        if r.mse is None or r.mae is None:
            continue
        agg[r.model_type]["count"] = int(agg[r.model_type]["count"]) + 1
        agg[r.model_type]["sum_MSE"] = float(agg[r.model_type]["sum_MSE"]) + float(r.mse)
        agg[r.model_type]["sum_MAE"] = float(agg[r.model_type]["sum_MAE"]) + float(r.mae)

    win_count = {mt: 0 for mt in MODEL_TYPE_ORDER}
    for d in per_dataset_best:
        win_count[d["best_model_type"]] += 1

    summary_rows: List[Dict[str, object]] = []
    for mt in MODEL_TYPE_ORDER:
        c = int(agg[mt]["count"])
        mean_mse = float(agg[mt]["sum_MSE"]) / c if c else None
        mean_mae = float(agg[mt]["sum_MAE"]) / c if c else None
        summary_rows.append(
            {
                "model_type": mt,
                "datasets_count": c,
                "mean_MSE": mean_mse,
                "mean_MAE": mean_mae,
                "win_count_by_MSE": win_count.get(mt, 0),
            }
        )
    _write_csv(dst_root / "best_model_summary.csv", summary_rows, dry_run=args.dry_run)

    if not args.dry_run:
        candidates = [r for r in summary_rows if r["mean_MSE"] is not None]
        best_overall = sorted(candidates, key=lambda x: (x["mean_MSE"], x["mean_MAE"]))[0] if candidates else None
        if best_overall:
            print(f"整体最优（按 mean(MSE)）：{best_overall['model_type']}  mean_MSE={best_overall['mean_MSE']}  mean_MAE={best_overall['mean_MAE']}  win_count={best_overall['win_count_by_MSE']}")

    print(f"已处理实验数：{len(experiment_ids)}，成功转移：{len(moved_ids)}，冲突文件条目：{len(conflicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

