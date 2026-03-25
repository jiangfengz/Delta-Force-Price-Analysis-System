from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def _detect_repo_root(script_path: Path) -> Path:
    root = script_path.resolve()
    for _ in range(7):
        if (root / "run.py").exists() and (root / "backup").exists():
            return root
        root = root.parent
    raise RuntimeError("无法自动定位项目根目录（未找到 run.py 与 backup/）")


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


def _read_per_dataset_best(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"未找到：{path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out: List[Dict[str, str]] = []
        for r in reader:
            out.append({k: (v or "").strip() for k, v in r.items()})
    return out


def _validate_required(src: Path, required_files: List[str]) -> List[str]:
    missing: List[str] = []
    for name in required_files:
        if not (src / name).exists():
            missing.append(str(src / name))
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--src-name", default="5 models in 7 days")
    parser.add_argument("--dst-name", default="best in 7 days")
    args = parser.parse_args()

    repo_root = _detect_repo_root(Path(__file__))
    src_root = repo_root / "backup" / args.src_name
    dst_root = repo_root / "backup" / args.dst_name

    per_best_csv = src_root / "per_dataset_best.csv"
    rows = _read_per_dataset_best(per_best_csv)

    conflicts: List[Dict[str, str]] = []
    copy_errors: List[Dict[str, object]] = []
    selected: List[Dict[str, object]] = []

    _ensure_dir(dst_root, dry_run=args.dry_run)

    moved = 0
    for r in rows:
        dataset = r.get("dataset", "")
        best_family = r.get("best_family", "")
        best_model_id = r.get("best_model_id", "")
        best_mse = r.get("best_MSE", "")
        best_mae = r.get("best_MAE", "")

        if not dataset or not best_family or not best_model_id:
            continue

        src_family_root = src_root / best_family
        src_results = src_family_root / "results" / best_model_id
        src_ckpt = src_family_root / "checkpoints" / best_model_id
        src_test = src_family_root / "test_results" / best_model_id

        missing: List[str] = []
        missing += _validate_required(src_results, ["metrics.csv", "log.txt"])
        missing += _validate_required(src_ckpt, ["checkpoint.pth"])
        if missing:
            copy_errors.append(
                {"dataset": dataset, "best_model_id": best_model_id, "best_family": best_family, "missing": missing}
            )
            continue

        dst_exp = dst_root / best_model_id
        dst_results = dst_exp / "results"
        dst_ckpt = dst_exp / "checkpoints"
        dst_test = dst_exp / "test_results"

        _copy_tree_merge(src_results, dst_results, conflicts=conflicts, dry_run=args.dry_run)
        _copy_tree_merge(src_ckpt, dst_ckpt, conflicts=conflicts, dry_run=args.dry_run)
        if src_test.exists():
            _copy_tree_merge(src_test, dst_test, conflicts=conflicts, dry_run=args.dry_run)

        selected.append(
            {
                "dataset": dataset,
                "best_family": best_family,
                "best_model_id": best_model_id,
                "best_MSE": best_mse,
                "best_MAE": best_mae,
            }
        )
        moved += 1

    if conflicts:
        _write_csv(dst_root / "conflicts.csv", conflicts, dry_run=args.dry_run)

    if copy_errors:
        if not args.dry_run:
            (dst_root / "copy_errors.json").write_text(
                json.dumps(copy_errors, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        raise RuntimeError(f"存在 {len(copy_errors)} 个实验缺少必要文件，已输出 copy_errors.json")

    print(f"已写入 best in 7 days：{moved} 个数据集最优实验；冲突文件条目：{len(conflicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

