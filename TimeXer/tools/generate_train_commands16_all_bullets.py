from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Config:
    tag: str
    seq_len: int
    label_len: int
    pred_len: int
    d_model: int
    e_layers: int
    d_layers: int
    factor: int
    learning_rate: float
    lradj: str
    dropout: Optional[float] = None
    plateau_patience: Optional[int] = None


def _detect_repo_root(script_path: Path) -> Path:
    root = script_path.resolve()
    for _ in range(8):
        if (root / "run.py").exists() and (root / "dataset").exists():
            return root
        root = root.parent
    raise RuntimeError("无法自动定位项目根目录（未找到 run.py 与 dataset/）")


def _sanitize_model_key(target: str) -> str:
    if target == "箭矢":
        return "Arrow"
    s = target.lstrip(".")
    s = re.sub(r"[^0-9A-Za-z]+", "", s)
    return s or "Unknown"


def _iter_dataset_files(dataset_dir: Path) -> List[Path]:
    files = sorted(dataset_dir.glob("*.csv"), key=lambda p: p.name.lower())
    if not files:
        raise RuntimeError(f"未找到任何 CSV：{dataset_dir}")
    return files


def _format_ps_command_lines(lines: Iterable[str]) -> str:
    return "\n".join(lines)


def _build_command(
    *,
    root_path: Path,
    data_path: str,
    target: str,
    model_id: str,
    des: str,
    cfg: Config,
    train_epochs: int,
    patience: int,
    batch_size: int,
    gpu: int,
    end_with_and: bool,
) -> str:
    cmd_lines: List[str] = [
        "python run.py `",
        "  --task_name long_term_forecast `",
        "  --is_training 1 `",
        f'  --root_path "{root_path}" `',
        f'  --data_path "{data_path}" `',
        f'  --model_id "{model_id}" `',
        "  --model TimeXer `",
        "  --data custom `",
        "  --features M `",
        f'  --target "{target}" `',
        '  --freq "h" `',
        '  --checkpoints "./checkpoints/" `',
        f"  --seq_len {cfg.seq_len} `",
        f"  --label_len {cfg.label_len} `",
        f"  --pred_len {cfg.pred_len} `",
        f"  --e_layers {cfg.e_layers} `",
        f"  --d_layers {cfg.d_layers} `",
        f"  --factor {cfg.factor} `",
        f"  --d_model {cfg.d_model} `",
        f'  --des "{des}" `',
        "  --itr 1 `",
        f"  --train_epochs {train_epochs} `",
        f"  --patience {patience} `",
        f"  --batch_size {batch_size} `",
        f"  --learning_rate {cfg.learning_rate} `",
        f"  --lradj {cfg.lradj} `",
        "  --num_workers 0 `",
        "  --use_amp `",
        "  --use_gpu True `",
    ]

    if cfg.lradj == "plateau" and cfg.plateau_patience is not None:
        cmd_lines.append(f"  --plateau_patience {cfg.plateau_patience} `")

    if cfg.dropout is not None:
        cmd_lines.append(f"  --dropout {cfg.dropout} `")

    if end_with_and:
        cmd_lines.append(f"  --gpu {gpu} && `")
    else:
        cmd_lines.append(f"  --gpu {gpu}")

    return _format_ps_command_lines(cmd_lines)


def _configs() -> List[Config]:
    return [
        Config(
            tag="Opt2_192_Plateau4",
            seq_len=192,
            label_len=96,
            pred_len=72,
            d_model=512,
            e_layers=4,
            d_layers=2,
            factor=3,
            learning_rate=0.0005,
            lradj="plateau",
            plateau_patience=3,
        ),
        Config(
            tag="MidCtx_240",
            seq_len=240,
            label_len=120,
            pred_len=72,
            d_model=512,
            e_layers=4,
            d_layers=2,
            factor=3,
            learning_rate=0.0005,
            lradj="plateau",
        ),
        Config(
            tag="LongCtx2_384",
            seq_len=384,
            label_len=192,
            pred_len=72,
            d_model=512,
            e_layers=4,
            d_layers=2,
            factor=3,
            learning_rate=0.0005,
            lradj="plateau",
        ),
        Config(
            tag="MidLarge_640",
            seq_len=240,
            label_len=120,
            pred_len=72,
            d_model=640,
            e_layers=4,
            d_layers=2,
            factor=3,
            learning_rate=0.0005,
            lradj="plateau",
        ),
        Config(
            tag="Deep6_240",
            seq_len=240,
            label_len=120,
            pred_len=72,
            d_model=512,
            e_layers=6,
            d_layers=2,
            factor=3,
            learning_rate=0.0003,
            lradj="type3",
        ),
        Config(
            tag="RegDrop_128",
            seq_len=128,
            label_len=64,
            pred_len=72,
            d_model=512,
            e_layers=4,
            d_layers=2,
            factor=3,
            learning_rate=0.0005,
            lradj="type1",
            dropout=0.3,
        ),
    ]


def _build_markdown(
    *,
    dataset_files: List[Path],
    dataset_dir: Path,
    out_path: Path,
    version_tag: str,
    gpu: int,
    train_epochs: int,
    patience: int,
    batch_size: int,
) -> str:
    header_lines = [
        "# 训练命令16：all bullets（collection_category）六组参数批量训练（ftM）",
        "",
        "说明：",
        "- 适用数据目录：`dataset/bullet/collection_category`（共 19 个 CSV）。",
        "- 本文件新增一组 Opt2 基准命令，用于与历史结果直接对照；其余 5 组为新参数族。",
        "- 训练固定：`--model TimeXer`、`--features M`、`--num_workers 0`、并显式指定 `train_epochs/patience/batch_size/use_amp`。",
        "",
        "六组参数族：",
        "- Opt2_192_Plateau4：seq=192,label=96,d_model=512,e_layers=4,lr=5e-4，plateau 无改进4个epoch减半",
        "- MidCtx_240：seq=240,label=120,d_model=512,e_layers=4,lr=5e-4",
        "- LongCtx2_384：seq=384,label=192,d_model=512,e_layers=4,lr=5e-4",
        "- MidLarge_640：seq=240,label=120,d_model=640,e_layers=4,lr=5e-4",
        "- Deep6_240：seq=240,label=120,d_model=512,e_layers=6,lr=3e-4,lradj=type3",
        "- RegDrop_128：seq=128,label=64,d_model=512,e_layers=4,dropout=0.3,lradj=type1",
        "",
        f"生成位置：`{out_path.relative_to(out_path.parents[1])}`",
        f"版本标记：`{version_tag}`",
        "",
        "---",
        "",
        "```powershell",
    ]

    cfgs = _configs()
    command_blocks: List[str] = []

    total_cmds = len(dataset_files) * len(cfgs)
    cmd_idx = 0

    for i, csv_path in enumerate(dataset_files, start=1):
        data_path = csv_path.name
        target = csv_path.stem
        model_key = _sanitize_model_key(target)
        command_blocks.append(f"# =========================")
        command_blocks.append(f"# {i}) {target}（ftM）")
        command_blocks.append(f"# =========================")
        command_blocks.append("")

        for cfg in cfgs:
            cmd_idx += 1
            end_with_and = cmd_idx < total_cmds
            model_id = f"{version_tag}_{model_key}_Collection_Category_{cfg.tag}_M_{cfg.seq_len}_D{cfg.d_model}"
            des = f"Exp16_{model_key}_{cfg.tag}_ftM"
            command_blocks.append(
                _build_command(
                    root_path=dataset_dir,
                    data_path=data_path,
                    target=target,
                    model_id=model_id,
                    des=des,
                    cfg=cfg,
                    train_epochs=train_epochs,
                    patience=patience,
                    batch_size=batch_size,
                    gpu=gpu,
                    end_with_and=end_with_and,
                )
            )
            command_blocks.append("")

    footer_lines = ["```", ""]
    return "\n".join(header_lines + command_blocks + footer_lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="V16", help="写入 model_id 的版本前缀")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    repo_root = _detect_repo_root(Path(__file__))
    dataset_dir = repo_root / "dataset" / "bullet" / "collection_category"
    out_dir = repo_root / "训练命令"
    out_path = out_dir / "训练命令16_all bullets.md"

    dataset_files = _iter_dataset_files(dataset_dir)
    md = _build_markdown(
        dataset_files=dataset_files,
        dataset_dir=dataset_dir,
        out_path=out_path,
        version_tag=args.version,
        gpu=args.gpu,
        train_epochs=args.train_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"已生成：{out_path}（datasets={len(dataset_files)}，cmds={len(dataset_files) * len(_configs())}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
