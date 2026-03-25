from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


FAMILIES_6 = [
    "MidCtx_240",
    "LongCtx2_384",
    "MidLarge_640",
    "Deep6_240",
    "RegDrop_128",
    "Opt2_192_Plateau4",
]

FAMILY_ORDER = list(FAMILIES_6)


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


def _copy_tree_merge(src_dir: Path, dst_dir: Path, conflicts: List[Dict[str, str]], dry_run: bool) -> None:
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


_RE_FAMILY = re.compile(r"_Collection_Category_(?P<family>.+?)_M_")
_RE_VERSION_PREFIX = re.compile(r"^V\d+_")


def _extract_family(model_id: str) -> Optional[str]:
    m = _RE_FAMILY.search(model_id)
    if not m:
        return None
    return m.group("family")


def _dataset_key_from_model_id(model_id: str) -> Optional[str]:
    if "_Collection_Category_" not in model_id:
        return None
    prefix = model_id.split("_Collection_Category_", 1)[0]
    prefix = _RE_VERSION_PREFIX.sub("", prefix)
    return prefix


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
    if not rows or dry_run:
        return
    _ensure_dir(path.parent, dry_run=False)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _append_csv(path: Path, rows: List[Dict[str, object]], dry_run: bool) -> None:
    if not rows or dry_run:
        return
    _ensure_dir(path.parent, dry_run=False)
    file_exists = path.exists()
    fieldnames = list(rows[0].keys())
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _pick_best_row(rows: List[MetricsRow]) -> Optional[MetricsRow]:
    candidates = [r for r in rows if r.mse is not None and r.mae is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (x.mse, x.mae))[0]


def _unique_path(base: Path) -> Path:
    if not base.exists():
        return base
    for i in range(1, 1000):
        p = base.with_name(f"{base.name}_{i}")
        if not p.exists():
            return p
    raise RuntimeError(f"无法生成唯一目录名：{base}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dst-name", default="6 kinds of models")
    parser.add_argument("--migrate-from", default="5 kinds of models")
    args = parser.parse_args()

    repo_root = _detect_repo_root(Path(__file__))
    src_results = repo_root / "results"
    src_checkpoints = repo_root / "checkpoints"
    src_test_results = repo_root / "test_results"

    dst_root = repo_root / "backup" / args.dst_name
    old_root = repo_root / "backup" / args.migrate_from
    best_root = repo_root / "backup" / "best"
    best_replaced_root = repo_root / "backup" / "best_replaced"

    conflicts: List[Dict[str, str]] = []
    copy_errors: List[Dict[str, object]] = []

    if old_root.exists() and old_root.resolve() != dst_root.resolve():
        if not args.dry_run:
            if not dst_root.exists():
                dst_root.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_root), str(dst_root))
            else:
                _copy_tree_merge(old_root, dst_root, conflicts=conflicts, dry_run=False)
                shutil.rmtree(old_root)

    experiment_ids: List[str] = []
    id_to_family: Dict[str, str] = {}
    for model_id in _iter_experiment_ids(src_results):
        if not model_id.startswith("V16_"):
            continue
        family = _extract_family(model_id)
        if family is None or family not in FAMILIES_6:
            continue
        experiment_ids.append(model_id)
        id_to_family[model_id] = family
    experiment_ids = sorted(set(experiment_ids))

    moved_ids: List[str] = []
    for model_id in experiment_ids:
        family = id_to_family[model_id]

        dst_family_root = dst_root / family
        dst_results_dir = dst_family_root / "results" / model_id
        dst_ckpt_dir = dst_family_root / "checkpoints" / model_id
        dst_test_dir = dst_family_root / "test_results" / model_id

        src_results_dir = src_results / model_id
        src_ckpt_dir = src_checkpoints / model_id
        src_test_dir = src_test_results / model_id

        _copy_tree_merge(src_results_dir, dst_results_dir, conflicts=conflicts, dry_run=args.dry_run)
        _copy_tree_merge(src_ckpt_dir, dst_ckpt_dir, conflicts=conflicts, dry_run=args.dry_run)
        _copy_tree_merge(src_test_dir, dst_test_dir, conflicts=conflicts, dry_run=args.dry_run)

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
    seen_model_ids: set[str] = set()

    scan_root = dst_root if dst_root.exists() else old_root
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
                seen_model_ids.add(model_id)

    if args.dry_run:
        for model_id in experiment_ids:
            if model_id in seen_model_ids:
                continue
            family = id_to_family[model_id]
            dataset = _dataset_key_from_model_id(model_id)
            if dataset is None:
                continue
            metrics_csv = src_results / model_id / "metrics.csv"
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
            seen_model_ids.add(model_id)

    dataset_to_rows: Dict[str, Dict[str, MetricsRow]] = {}
    for r in metrics_rows:
        dataset_to_rows.setdefault(r.dataset, {})[r.family] = r

    compare_out: List[Dict[str, object]] = []
    per_dataset_best: List[Dict[str, object]] = []
    for dataset in sorted(dataset_to_rows.keys()):
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

        best = _pick_best_row([dataset_to_rows[dataset][f] for f in dataset_to_rows[dataset].keys()])
        if best is not None:
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

    agg: Dict[str, Dict[str, object]] = {f: {"family": f, "count": 0, "sum_MSE": 0.0, "sum_MAE": 0.0} for f in FAMILY_ORDER}
    for r in metrics_rows:
        if r.mse is None or r.mae is None:
            continue
        agg[r.family]["count"] = int(agg[r.family]["count"]) + 1
        agg[r.family]["sum_MSE"] = float(agg[r.family]["sum_MSE"]) + float(r.mse)
        agg[r.family]["sum_MAE"] = float(agg[r.family]["sum_MAE"]) + float(r.mae)

    win_count = {f: 0 for f in FAMILY_ORDER}
    for d in per_dataset_best:
        win_count[d["best_family"]] += 1

    summary_rows: List[Dict[str, object]] = []
    for f in FAMILY_ORDER:
        c = int(agg[f]["count"])
        mean_mse = float(agg[f]["sum_MSE"]) / c if c else None
        mean_mae = float(agg[f]["sum_MAE"]) / c if c else None
        summary_rows.append(
            {
                "family": f,
                "datasets_count": c,
                "mean_MSE": mean_mse,
                "mean_MAE": mean_mae,
                "win_count_by_MSE": win_count.get(f, 0),
            }
        )
    _write_csv(dst_root / "family_summary.csv", summary_rows, dry_run=args.dry_run)

    best_entries: Dict[str, Dict[str, object]] = {}
    if best_root.exists():
        for p in best_root.iterdir():
            if not p.is_dir():
                continue
            if "_Collection_Category_" not in p.name:
                continue
            dataset = p.name.split("_Collection_Category_", 1)[0]
            metrics_csv = p / "results" / "metrics.csv"
            mae, mse, rmse, mape, mspe = _read_metrics_csv(metrics_csv)
            if mse is None or mae is None:
                continue
            existing = best_entries.get(dataset)
            if existing is None or (mse, mae) < (float(existing["best_MSE"]), float(existing["best_MAE"])):
                best_entries[dataset] = {
                    "dataset": dataset,
                    "best_dir": str(p),
                    "best_model_id": p.name,
                    "best_MAE": mae,
                    "best_MSE": mse,
                    "best_RMSE": rmse,
                    "best_MAPE": mape,
                    "best_MSPE": mspe,
                }

    per_dataset_best_map: Dict[str, Dict[str, object]] = {d["dataset"]: d for d in per_dataset_best}
    vs_best_detail: List[Dict[str, object]] = []
    beat_count = 0
    missing_best: List[str] = []
    improvements: List[float] = []
    regressions: List[float] = []
    replacement_rows: List[Dict[str, object]] = []

    for dataset in sorted(per_dataset_best_map.keys()):
        cand = per_dataset_best_map[dataset]
        best = best_entries.get(dataset)
        if best is None:
            missing_best.append(dataset)
            vs_best_detail.append(
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
        rel_mse = (delta_mse / float(best["best_MSE"])) * 100.0 if float(best["best_MSE"]) != 0 else None
        rel_mae = (delta_mae / float(best["best_MAE"])) * 100.0 if float(best["best_MAE"]) != 0 else None
        beats = (float(cand["best_MSE"]), float(cand["best_MAE"])) < (float(best["best_MSE"]), float(best["best_MAE"]))

        if beats:
            beat_count += 1
            improvements.append(-delta_mse)
            replacement_rows.append(
                {
                    "dataset": dataset,
                    "old_model_id": best["best_model_id"],
                    "new_model_id": cand["best_model_id"],
                    "old_MSE": best["best_MSE"],
                    "new_MSE": cand["best_MSE"],
                    "old_MAE": best["best_MAE"],
                    "new_MAE": cand["best_MAE"],
                    "best_family": cand["best_family"],
                }
            )
        else:
            regressions.append(delta_mse)

        vs_best_detail.append(
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

    _write_csv(dst_root / "vs_best_detail.csv", vs_best_detail, dry_run=args.dry_run)

    vs_best_summary: List[Dict[str, object]] = [
        {
            "datasets_with_best": len(vs_best_detail) - len(missing_best),
            "datasets_missing_best": len(missing_best),
            "beat_count": beat_count,
            "avg_MSE_improvement_when_beats": (sum(improvements) / len(improvements)) if improvements else "",
            "avg_MSE_regression_when_not_beats": (sum(regressions) / len(regressions)) if regressions else "",
            "missing_best_datasets": ";".join(missing_best),
        }
    ]
    _write_csv(dst_root / "vs_best_summary.csv", vs_best_summary, dry_run=args.dry_run)

    if not args.dry_run and replacement_rows:
        for rep in replacement_rows:
            dataset = str(rep["dataset"])
            old_model_id = str(rep["old_model_id"])
            new_model_id = str(rep["new_model_id"])
            family = str(rep["best_family"])

            old_best_dir = best_root / old_model_id
            if not old_best_dir.exists():
                continue

            replaced_dst = _unique_path(best_replaced_root / dataset / old_model_id)
            _ensure_dir(replaced_dst.parent, dry_run=False)
            shutil.move(str(old_best_dir), str(replaced_dst))

            cand_base = dst_root / family
            cand_results = cand_base / "results" / new_model_id
            cand_ckpt = cand_base / "checkpoints" / new_model_id
            cand_test = cand_base / "test_results" / new_model_id

            new_best_dir = best_root / new_model_id
            _ensure_dir(new_best_dir, dry_run=False)
            _copy_tree_merge(cand_results, new_best_dir / "results", conflicts=[], dry_run=False)
            _copy_tree_merge(cand_ckpt, new_best_dir / "checkpoints", conflicts=[], dry_run=False)
            _copy_tree_merge(cand_test, new_best_dir / "test_results", conflicts=[], dry_run=False)

        _append_csv(repo_root / "backup" / "best_update_log.csv", replacement_rows, dry_run=False)

    best_overall = None
    candidates = [r for r in summary_rows if r["mean_MSE"] is not None]
    if candidates:
        best_overall = sorted(candidates, key=lambda x: (x["mean_MSE"], x["mean_MAE"]))[0]

    total_in_backup = len(seen_model_ids)
    print(
        f"识别待转移实验数：{len(experiment_ids)}，本次成功转移：{len(moved_ids)}，备份内实验总数：{total_in_backup}，冲突文件条目：{len(conflicts)}"
    )
    if best_overall:
        print(
            f"六组参数整体最优（按 mean(MSE)）：{best_overall['family']}  mean_MSE={best_overall['mean_MSE']}  mean_MAE={best_overall['mean_MAE']}  win_count={best_overall['win_count_by_MSE']}"
        )
    if args.dry_run:
        print(f"dry-run：将替换 best 的数据集数：{len(replacement_rows)}")
    else:
        print(f"已替换 best 的数据集数：{len(replacement_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
