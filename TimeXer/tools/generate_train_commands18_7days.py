from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    idx: int
    key: str
    data_path: str
    target: str
    title: str


@dataclass(frozen=True)
class FamilyBase:
    family: str
    seq_len: int
    label_len: int
    d_model: int
    e_layers: int
    learning_rate: float
    lradj: str
    plateau_patience: int | None


@dataclass(frozen=True)
class TrainCfg:
    tag: str
    cfg_name: str
    family: str
    seq_len: int
    label_len: int
    d_model: int
    e_layers: int
    learning_rate: float
    lradj: str
    plateau_patience: int | None
    patch_len: int
    dropout: float | None
    batch_size: int


DATASETS: List[DatasetSpec] = [
    DatasetSpec(1, "300BLK", ".300BLK.csv", ".300BLK", ".300BLK"),
    DatasetSpec(2, "357Magnum", ".357 Magnum.csv", ".357 Magnum", ".357 Magnum"),
    DatasetSpec(3, "45ACP", ".45 ACP.csv", ".45 ACP", ".45 ACP"),
    DatasetSpec(4, "50AE", ".50 AE.csv", ".50 AE", ".50 AE"),
    DatasetSpec(5, "12Gauge", "12 Gauge.csv", "12 Gauge", "12 Gauge"),
    DatasetSpec(6, "127x55mm", "12.7x55mm.csv", "12.7x55mm", "12.7x55mm"),
    DatasetSpec(7, "46x30mm", "4.6x30mm.csv", "4.6x30mm", "4.6x30mm"),
    DatasetSpec(8, "4570Govt", "45-70 Govt.csv", "45-70 Govt", "45-70 Govt"),
    DatasetSpec(9, "545x39mm", "5.45x39mm.csv", "5.45x39mm", "5.45x39mm"),
    DatasetSpec(10, "556x45mm", "5.56x45mm.csv", "5.56x45mm", "5.56x45mm"),
    DatasetSpec(11, "57x28mm", "5.7x28mm.csv", "5.7x28mm", "5.7x28mm"),
    DatasetSpec(12, "58x42mm", "5.8x42mm.csv", "5.8x42mm", "5.8x42mm"),
    DatasetSpec(13, "68x51mm", "6.8x51mm.csv", "6.8x51mm", "6.8x51mm"),
    DatasetSpec(14, "762x39mm", "7.62x39mm.csv", "7.62x39mm", "7.62x39mm"),
    DatasetSpec(15, "762x51mm", "7.62x51mm.csv", "7.62x51mm", "7.62x51mm"),
    DatasetSpec(16, "762x54R", "7.62x54R.csv", "7.62x54R", "7.62x54R"),
    DatasetSpec(17, "9x19mm", "9x19mm.csv", "9x19mm", "9x19mm"),
    DatasetSpec(18, "9x39mm", "9x39mm.csv", "9x39mm", "9x39mm"),
    DatasetSpec(19, "Arrow", "箭矢.csv", "箭矢", "箭矢"),
]

FAMILY_BASE: Dict[str, FamilyBase] = {
    "Opt2_192_Plateau4": FamilyBase("Opt2_192_Plateau4", 192, 96, 512, 4, 5e-4, "plateau", 3),
    "Deep6_240": FamilyBase("Deep6_240", 240, 120, 512, 6, 3e-4, "type3", None),
    "MidLarge_640": FamilyBase("MidLarge_640", 240, 120, 640, 4, 5e-4, "plateau", 2),
    "Exp_Long_M_336_A": FamilyBase("Exp_Long_M_336_A", 336, 168, 512, 4, 5e-4, "plateau", 2),
    "LongCtx2_384": FamilyBase("LongCtx2_384", 384, 192, 512, 4, 5e-4, "plateau", 2),
}


def _detect_repo_root(script_path: Path) -> Path:
    root = script_path.resolve()
    for _ in range(8):
        if (root / "run.py").exists() and (root / "训练命令").exists():
            return root
        root = root.parent
    raise RuntimeError("无法定位项目根目录（未找到 run.py 与 训练命令/）")


def _read_metrics_compare(path: Path) -> Dict[str, List[Tuple[float, float, str]]]:
    by: Dict[str, List[Tuple[float, float, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            dataset = (r.get("dataset") or "").strip()
            family = (r.get("family") or "").strip()
            try:
                mse = float(r.get("MSE") or "")
                mae = float(r.get("MAE") or "")
            except ValueError:
                continue
            by.setdefault(dataset, []).append((mse, mae, family))
    return by


def _pick_best_and_second(by_dataset: Dict[str, List[Tuple[float, float, str]]]) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    for dataset, rows in by_dataset.items():
        ranked = sorted(rows, key=lambda x: (x[0], x[1], x[2]))
        if len(ranked) < 2:
            continue
        best = ranked[0][2]
        second = ranked[1][2]
        out[dataset] = (best, second)
    return out


def _lr_down(lr: float) -> float:
    if lr <= 2e-4:
        return lr
    if lr <= 3e-4:
        return 2e-4
    return 3e-4


def _lr_up(lr: float) -> float:
    if lr >= 8e-4:
        return lr
    if lr <= 3e-4:
        return 5e-4
    return 8e-4


def _long_seq(seq_len: int) -> int:
    if seq_len <= 192:
        return 288
    if seq_len <= 240:
        return 336
    if seq_len <= 336:
        return 432
    return 480


def _batch_for_seq(seq_len: int, d_model: int) -> int:
    if seq_len >= 432:
        return 32
    if seq_len >= 336 and d_model >= 640:
        return 32
    if seq_len >= 480:
        return 32
    return 64


def build_6_configs(dataset_key: str, best_family: str, second_family: str) -> List[TrainCfg]:
    best = FAMILY_BASE[best_family]
    second = FAMILY_BASE[second_family]

    a = TrainCfg(
        tag="A",
        cfg_name="Best_Base",
        family=best.family,
        seq_len=best.seq_len,
        label_len=best.label_len,
        d_model=best.d_model,
        e_layers=best.e_layers,
        learning_rate=best.learning_rate,
        lradj=best.lradj,
        plateau_patience=best.plateau_patience,
        patch_len=16,
        dropout=None,
        batch_size=_batch_for_seq(best.seq_len, best.d_model),
    )
    b = TrainCfg(
        tag="B",
        cfg_name="Best_LRDown",
        family=best.family,
        seq_len=best.seq_len,
        label_len=best.label_len,
        d_model=best.d_model,
        e_layers=best.e_layers,
        learning_rate=_lr_down(best.learning_rate),
        lradj=best.lradj,
        plateau_patience=2 if best.lradj == "plateau" else None,
        patch_len=16,
        dropout=None,
        batch_size=_batch_for_seq(best.seq_len, best.d_model),
    )
    c = TrainCfg(
        tag="C",
        cfg_name="Best_LRUp",
        family=best.family,
        seq_len=best.seq_len,
        label_len=best.label_len,
        d_model=best.d_model,
        e_layers=best.e_layers,
        learning_rate=_lr_up(best.learning_rate),
        lradj=best.lradj,
        plateau_patience=2 if best.lradj == "plateau" else None,
        patch_len=16,
        dropout=None,
        batch_size=_batch_for_seq(best.seq_len, best.d_model),
    )

    seq_long = _long_seq(best.seq_len)
    if best.family.startswith("Exp_Long_M"):
        long_family_name = f"ExpLong_{seq_long}"
    elif best.family.startswith("Opt2_"):
        long_family_name = f"Opt2_{seq_long}"
    elif best.family.startswith("Deep6_"):
        long_family_name = f"Deep6_{seq_long}"
    elif best.family.startswith("MidLarge_"):
        long_family_name = f"MidLarge_{seq_long}"
    elif best.family.startswith("LongCtx2_"):
        long_family_name = f"LongCtx2_{seq_long}"
    else:
        long_family_name = f"{best.family}_{seq_long}"
    d = TrainCfg(
        tag="D",
        cfg_name="Best_LongSeq",
        family=long_family_name,
        seq_len=seq_long,
        label_len=seq_long // 2,
        d_model=best.d_model,
        e_layers=best.e_layers,
        learning_rate=best.learning_rate,
        lradj=best.lradj,
        plateau_patience=best.plateau_patience if best.lradj == "plateau" else None,
        patch_len=16,
        dropout=None,
        batch_size=_batch_for_seq(seq_long, best.d_model),
    )
    e = TrainCfg(
        tag="E",
        cfg_name="Best_Patch32",
        family=best.family,
        seq_len=best.seq_len,
        label_len=best.label_len,
        d_model=best.d_model,
        e_layers=best.e_layers,
        learning_rate=_lr_down(best.learning_rate),
        lradj=best.lradj,
        plateau_patience=2 if best.lradj == "plateau" else None,
        patch_len=32,
        dropout=0.15,
        batch_size=_batch_for_seq(best.seq_len, best.d_model),
    )
    f = TrainCfg(
        tag="F",
        cfg_name="Alt_2ndBase",
        family=second.family,
        seq_len=second.seq_len,
        label_len=second.label_len,
        d_model=second.d_model,
        e_layers=second.e_layers,
        learning_rate=second.learning_rate,
        lradj=second.lradj,
        plateau_patience=second.plateau_patience if second.lradj == "plateau" else None,
        patch_len=16,
        dropout=None,
        batch_size=_batch_for_seq(second.seq_len, second.d_model),
    )
    return [a, b, c, d, e, f]


def _fmt_lr(lr: float) -> str:
    s = f"{lr:.10f}".rstrip("0").rstrip(".")
    if s.startswith("."):
        s = "0" + s
    return s


def _write_header(best_rows: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    lines.append("# 训练命令18：7days（collection_category）六组参数批量训练（TimeXer + M）")
    lines.append("")
    lines.append("基于 `训练命令17_7days.md` 的 7 天跨度训练产物（`backup/5 models in 7 days`）整理得到的经验与下一轮改进方案。")
    lines.append("")
    lines.append("## 1. 训练结果经验总结（来自 V17_7d）")
    lines.append("")
    lines.append("### 1.1 五个参数族的整体表现（19 个数据集）")
    lines.append("V17_7d 的 5 个参数族分别为：")
    lines.append("- Opt2_192_Plateau4：seq=192,label=96,d_model=512,e_layers=4,lr=5e-4,lradj=plateau(plateau_patience=3)")
    lines.append("- Deep6_240：seq=240,label=120,d_model=512,e_layers=6,lr=3e-4,lradj=type3")
    lines.append("- MidLarge_640：seq=240,label=120,d_model=640,e_layers=4,lr=5e-4,lradj=plateau")
    lines.append("- Exp_Long_M_336_A：seq=336,label=168,d_model=512,e_layers=4,lr=5e-4,lradj=plateau")
    lines.append("- LongCtx2_384：seq=384,label=192,d_model=512,e_layers=4,lr=5e-4,lradj=plateau")
    lines.append("")
    lines.append("结论：Deep6/Opt2/Exp_Long 是主力结构族，但个别数据集需要更长上下文或更大 d_model。")
    lines.append("")
    lines.append("### 1.2 各数据集 V17_7d 最优参数族（best_family）")
    lines.append("| dataset | best_family | best_MSE | best_MAE |")
    lines.append("|---|---|---:|---:|")
    for r in best_rows:
        lines.append(f"| {r['dataset']} | {r['best_family']} | {r['best_MSE']} | {r['best_MAE']} |")
    lines.append("")
    lines.append("### 1.3 V18_7d 的 6 组改进策略（围绕最优结构做微调 + 次优结构对照）")
    lines.append("1. Best_Base：复现 V17_7d 最优结构（锚点）。")
    lines.append("2. Best_LRDown：同结构，学习率下调（更稳）。")
    lines.append("3. Best_LRUp：同结构，学习率上调（更快探索）。")
    lines.append("4. Best_LongSeq：同结构，增大 seq_len（更强上下文）。")
    lines.append("5. Best_Patch32：同结构，patch_len=32 + dropout=0.15（更强正则/更粗粒度 patch）。")
    lines.append("6. Alt_2ndBase：使用该数据集在 V17_7d 的次优结构（防局部最优）。")
    lines.append("")
    lines.append("## 2. 训练固定约束（必须满足）")
    lines.append("- --model TimeXer")
    lines.append("- --features M（M 模式）")
    lines.append("- --pred_len 168（7days）")
    lines.append("- --num_workers 0")
    lines.append("")
    lines.append("版本标记：V18_7d（model_id 末尾统一保留 _P168）")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _render_command(
    repo_root: Path,
    ds: DatasetSpec,
    cfg: TrainCfg,
    is_last_command_in_file: bool,
    gpu: int,
    train_epochs: int,
    patience: int,
) -> str:
    root_path = repo_root / "dataset" / "bullet" / "collection_category"
    model_id = f"V18_7d_{ds.key}_Collection_Category_{cfg.family}_{cfg.tag}_{cfg.cfg_name}_M_{cfg.seq_len}_D{cfg.d_model}_P168"
    des = f"Exp18_7d_{ds.key}_{cfg.family}_{cfg.tag}_{cfg.cfg_name}_M"

    lines: List[str] = []
    lines.append("python run.py `")
    lines.append("  --task_name long_term_forecast `")
    lines.append("  --is_training 1 `")
    lines.append(f"  --root_path \"{root_path}\" `")
    lines.append(f"  --data_path \"{ds.data_path}\" `")
    lines.append(f"  --model_id \"{model_id}\" `")
    lines.append("  --model TimeXer `")
    lines.append("  --data custom `")
    lines.append("  --features M `")
    lines.append(f"  --target \"{ds.target}\" `")
    lines.append("  --freq \"h\" `")
    lines.append("  --checkpoints \"./checkpoints/\" `")
    lines.append(f"  --seq_len {cfg.seq_len} `")
    lines.append(f"  --label_len {cfg.label_len} `")
    lines.append("  --pred_len 168 `")
    lines.append(f"  --e_layers {cfg.e_layers} `")
    lines.append("  --d_layers 2 `")
    lines.append("  --factor 3 `")
    lines.append(f"  --d_model {cfg.d_model} `")
    lines.append(f"  --patch_len {cfg.patch_len} `")
    if cfg.dropout is not None:
        lines.append(f"  --dropout {cfg.dropout} `")
    lines.append(f"  --des \"{des}\" `")
    lines.append("  --itr 1 `")
    lines.append(f"  --train_epochs {train_epochs} `")
    lines.append(f"  --patience {patience} `")
    lines.append(f"  --batch_size {cfg.batch_size} `")
    lines.append(f"  --learning_rate {_fmt_lr(cfg.learning_rate)} `")
    lines.append(f"  --lradj {cfg.lradj} `")
    if cfg.lradj == "plateau" and cfg.plateau_patience is not None:
        lines.append(f"  --plateau_patience {cfg.plateau_patience} `")
    lines.append("  --num_workers 0 `")
    lines.append("  --use_amp `")
    lines.append("  --use_gpu True `")
    if is_last_command_in_file:
        lines.append(f"  --gpu {gpu}")
    else:
        lines.append(f"  --gpu {gpu} && `")
    return "\n".join(lines)


def _read_per_dataset_best(path: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append({k: (v or "").strip() for k, v in r.items()})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train-epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--out",
        type=str,
        default="训练命令/训练命令18_7days.md",
    )
    args = parser.parse_args()

    repo_root = _detect_repo_root(Path(__file__))
    metrics_compare = repo_root / "backup" / "5 models in 7 days" / "metrics_compare.csv"
    per_best = repo_root / "backup" / "5 models in 7 days" / "per_dataset_best.csv"

    by_dataset = _read_metrics_compare(metrics_compare)
    best2 = _pick_best_and_second(by_dataset)

    best_rows = _read_per_dataset_best(per_best)
    header = _write_header(best_rows)

    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blocks: List[str] = [header, "```powershell"]
    total_cmd = len(DATASETS) * 6
    cmd_idx = 0
    for ds in DATASETS:
        if ds.key not in best2:
            raise RuntimeError(f"metrics_compare.csv 中未找到数据集：{ds.key}")
        best_family, second_family = best2[ds.key]
        cfgs = build_6_configs(ds.key, best_family, second_family)

        blocks.append("# =========================")
        blocks.append(f"# {ds.idx}) {ds.title}（M）")
        blocks.append("# =========================")
        blocks.append("")
        for cfg in cfgs:
            cmd_idx += 1
            is_last = cmd_idx == total_cmd
            blocks.append(
                _render_command(
                    repo_root=repo_root,
                    ds=ds,
                    cfg=cfg,
                    is_last_command_in_file=is_last,
                    gpu=args.gpu,
                    train_epochs=args.train_epochs,
                    patience=args.patience,
                )
            )
            blocks.append("")
    blocks.append("```")

    out_path.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")
    print(f"已生成：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
