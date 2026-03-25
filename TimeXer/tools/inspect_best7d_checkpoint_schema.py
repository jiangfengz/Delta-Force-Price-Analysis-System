import argparse
import os
import re
from dataclasses import dataclass
from typing import Optional


EXOG_COLS = ['is_holiday', 'in_CS', 'is_CS', 'is_need', 'is_make', 'is_active', 'is_public']


ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


@dataclass
class LogInfo:
    model_id: str
    root_path: Optional[str]
    data_path: Optional[str]
    features: Optional[str]
    target: Optional[str]
    updated_enc_in: Optional[int]
    updated_dec_in: Optional[int]
    updated_c_out: Optional[int]


def _clean_line(line: str) -> str:
    return ANSI_RE.sub('', line).strip()


def _parse_log_file(log_path: str, model_id: str) -> LogInfo:
    root_path = None
    data_path = None
    features = None
    target = None
    updated_enc_in = None
    updated_dec_in = None
    updated_c_out = None

    if not os.path.exists(log_path):
        return LogInfo(
            model_id=model_id,
            root_path=None,
            data_path=None,
            features=None,
            target=None,
            updated_enc_in=None,
            updated_dec_in=None,
            updated_c_out=None,
        )

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for _ in range(600):
            line = f.readline()
            if not line:
                break
            s = _clean_line(line)
            if 'Root Path:' in s:
                m = re.search(r'Root Path:\s*(.+?)(\s{2,}|$)', s)
                if m:
                    root_path = m.group(1).strip()
            if 'Data Path:' in s:
                m = re.search(r'Data Path:\s*(.+?)(\s{2,}|$)', s)
                if m:
                    data_path = m.group(1).strip()
            if 'Features:' in s and features is None:
                m = re.search(r'Features:\s*([A-Za-z]+)(\s{2,}|$)', s)
                if m:
                    features = m.group(1).strip()
            if 'Target:' in s and target is None:
                m = re.search(r'Target:\s*(.+?)(\s{2,}|$)', s)
                if m:
                    target = m.group(1).strip()
            if s.startswith('Updated enc_in:'):
                m = re.search(r'Updated enc_in:\s*(\d+),\s*dec_in:\s*(\d+),\s*c_out:\s*(\d+)', s)
                if m:
                    updated_enc_in = int(m.group(1))
                    updated_dec_in = int(m.group(2))
                    updated_c_out = int(m.group(3))
                    break

    return LogInfo(
        model_id=model_id,
        root_path=root_path,
        data_path=data_path,
        features=features,
        target=target,
        updated_enc_in=updated_enc_in,
        updated_dec_in=updated_dec_in,
        updated_c_out=updated_c_out,
    )


def _read_csv_header(csv_path: str) -> list[str]:
    with open(csv_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        header_line = f.readline()
    header_line = header_line.strip('\n').strip('\r')
    if not header_line:
        return []
    return [c.strip() for c in header_line.split(',')]


def _infer_schema(columns: list[str]) -> tuple[list[str], list[str]]:
    if not columns:
        return [], []
    cols = columns[:]
    if cols[0] == 'date':
        cols = cols[1:]
    present_exog = [c for c in EXOG_COLS if c in cols]
    target_cols = [c for c in cols if c not in present_exog]
    return target_cols, present_exog


def _print_one(model_dir: str, info: LogInfo) -> None:
    print('=' * 100)
    print(f'model_id: {info.model_id}')
    if info.root_path and info.data_path:
        csv_path = os.path.join(info.root_path, info.data_path)
        print(f'data_csv: {csv_path}')
    else:
        csv_path = None
        print('data_csv: <unknown>')

    print(f'features: {info.features}')
    print(f'target_arg: {info.target}')
    if info.updated_enc_in is not None:
        print(f'updated_dims: enc_in={info.updated_enc_in} dec_in={info.updated_dec_in} c_out={info.updated_c_out}')
    else:
        print('updated_dims: <unknown>')

    if not csv_path or not os.path.exists(csv_path):
        print('csv_header: <missing>')
        return

    columns = _read_csv_header(csv_path)
    target_cols, exog_cols = _infer_schema(columns)
    inferred_enc_in = len(target_cols) + len(exog_cols)
    inferred_c_out = len(target_cols)

    print(f'csv_columns({len(columns)}): {", ".join(columns)}')
    print(f'target_cols({len(target_cols)}): {", ".join(target_cols)}')
    print(f'exog_cols({len(exog_cols)}): {", ".join(exog_cols)}')
    print(f'inferred_dims: enc_in={inferred_enc_in} c_out={inferred_c_out}')

    if info.updated_enc_in is not None and info.updated_c_out is not None:
        ok = (info.updated_enc_in == inferred_enc_in) and (info.updated_c_out == inferred_c_out)
        print(f'dims_match_log: {ok}')

    if columns:
        print('template_header:')
        print(','.join(columns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--best7d_dir',
        type=str,
        default=os.path.join('backup', 'best in 7 days'),
        help='best in 7 days 目录路径（默认：backup/best in 7 days）',
    )
    parser.add_argument('--model_id', type=str, default=None, help='只检查指定 model_id（可选）')
    args = parser.parse_args()

    base_dir = os.path.abspath(args.best7d_dir)
    if not os.path.isdir(base_dir):
        raise SystemExit(f'not a directory: {base_dir}')

    model_ids = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            model_ids.append(name)
    model_ids.sort()

    if args.model_id is not None:
        model_ids = [m for m in model_ids if m == args.model_id]

    if not model_ids:
        raise SystemExit('no model_id found')

    for model_id in model_ids:
        model_dir = os.path.join(base_dir, model_id)
        log_path = os.path.join(model_dir, 'results', 'log.txt')
        info = _parse_log_file(log_path, model_id=model_id)
        _print_one(model_dir, info)


if __name__ == '__main__':
    main()
