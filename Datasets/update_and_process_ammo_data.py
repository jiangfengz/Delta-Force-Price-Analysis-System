import requests
import json
import os
import csv
import re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import glob
import shutil
import warnings
import importlib.util
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from dotenv import load_dotenv

# --- Configuration & Constants ---
BASE_PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_PROJECT_DIR / ".env")
WORK_DIR = BASE_PROJECT_DIR / "Datasets"
DATA_DIR = WORK_DIR / "daily Dataset"
HISTORY_DIR = WORK_DIR / "historical Datasets"
OUTPUT_DIR = WORK_DIR / "Datasets"
EXOG_DIR = WORK_DIR / "Exogenous"
EXOG_OUTPUT_DIR = WORK_DIR / "Datasets with Exogenous"
EXOG_TEMPLATE_DIR = BASE_PROJECT_DIR / "TimeXer" / "dataset" / "bullet" / "collection_category"
LEGACY_EXOG_BUILDER_PATH = EXOG_DIR / "build_collection_category_exog.py"
ENABLE_EXOG_PARITY_CHECK = True

# Ignored Bullets List
IGNORED_BULLETS = {
    "12 Gauge 独头 APX",
    "5.45x39mm BT +P",
    "6.8x51mm AP",
    "7.62x51mm M61",
    "7.62x54R SNB",
    ".338 Lap Mag AP",
    ".45 ACP Super\u200b",
    ".50 AE AP"
}

# API Configuration
API_BASE_URL = "https://df-api.shallow.ink"
API_ENDPOINT = f"{API_BASE_URL}/df/object/ammo"
AUTHORIZATION_HEADER = f"Bearer {os.getenv('DF_API_KEY')}"
HEADERS = {
    "Authorization": AUTHORIZATION_HEADER,
    "Content-Type": "application/json"
}
PARAMS = {"days": 30}

warnings.filterwarnings("ignore", category=FutureWarning)

EXOG_COLS = [
    "is_holiday",
    "in_CS",
    "is_CS",
    "is_need",
    "is_make",
    "is_active",
    "is_public",
]

ALIAS_MAP = {
    "arrow 3": "玻纤柳叶箭矢",
    "玻纤柳叶箭矢": "arrow 3",
    "arrow 4": "碳纤维刺骨箭矢",
    "碳纤维刺骨箭矢": "arrow 4",
    "arrow 5": "碳纤维穿甲箭矢",
    "碳纤维穿甲箭矢": "arrow 5",
}


def _norm_text(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip().lstrip("\ufeff")


def parse_chinese_date(date_str):
    try:
        match = re.match(r"(\d+)年(\d+)月(\d+)日", str(date_str))
        if match:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except Exception:
        pass
    return None


def get_aliases(name: str) -> set[str]:
    names = {_norm_text(name)}
    if name in ALIAS_MAP:
        names.add(_norm_text(ALIAS_MAP[name]))
    return names


def load_configs(exo_dir: Path):
    holiday_path = exo_dir / "China_Holiday_2025_2026.csv"
    season_path = exo_dir / "赛季时间.csv"
    gun_event_path = exo_dir / "品枪时间.csv"
    prediction_path = exo_dir / "predictions.csv"

    holiday_map = {}
    try:
        holidays_df = pd.read_csv(holiday_path)
        for _, row in holidays_df.iterrows():
            try:
                d = pd.to_datetime(row["date"]).date()
                holiday_map[d] = int(row["is_holiday"])
            except Exception:
                pass
    except Exception:
        holiday_map = {}

    seasons = []
    try:
        seasons_df = pd.read_csv(season_path)
        for _, row in seasons_df.iterrows():
            s_name = str(row.get("赛季名称", ""))
            s_id = 0
            if "s6" in s_name.lower():
                s_id = 1
            if "s7" in s_name.lower():
                s_id = 2

            start_date = parse_chinese_date(row.get("赛季开始"))
            end_date_raw = parse_chinese_date(row.get("赛季结束"))
            if not start_date or not end_date_raw:
                continue
            end_date = end_date_raw + timedelta(days=1)

            needed_raw = row.get("赛季任务所需子弹")
            made_raw = row.get("制造子弹")
            needed = str(needed_raw).split(";") if pd.notna(needed_raw) else []
            made = str(made_raw).split(";") if pd.notna(made_raw) else []

            seasons.append(
                {
                    "id": s_id,
                    "start": start_date,
                    "end": end_date,
                    "needed": [str(x).strip() for x in needed if str(x).strip()],
                    "made": [str(x).strip() for x in made if str(x).strip()],
                }
            )
    except Exception:
        seasons = seasons

    gun_events = []
    try:
        gun_events_df = pd.read_csv(gun_event_path)
        for _, row in gun_events_df.iterrows():
            time_range = str(row.get("时间", ""))
            bullets = str(row.get("所用子弹", "")).strip()
            try:
                start_str, _ = time_range.split("-")
            except Exception:
                continue

            def parse_md(s: str):
                m, d = map(int, str(s).split("."))
                y = 2025 if m >= 9 else 2026
                return datetime(y, m, d)

            try:
                s_date = parse_md(start_str)
            except Exception:
                continue

            active_start = s_date - timedelta(days=2)
            active_end = s_date + timedelta(days=2)
            active_end_exclusive = active_end + timedelta(days=1)
            gun_events.append({"start": active_start, "end": active_end_exclusive, "bullet": bullets})
    except Exception:
        gun_events = gun_events

    predictions = []
    try:
        pred_df = pd.read_csv(prediction_path)
        for _, row in pred_df.iterrows():
            try:
                t = pd.to_datetime(row["time"])
                n = str(row["name"])
                predictions.append({"time": t.to_pydatetime(), "name": n})
            except Exception:
                pass
    except Exception:
        predictions = predictions

    return holiday_map, seasons, gun_events, predictions


def get_is_holiday(dt: datetime, holiday_map: dict) -> int:
    return int(holiday_map.get(dt.date(), 0))


def get_season_data(dt: datetime, bullet_name: str, seasons: list[dict]):
    in_cs = 0.0
    is_cs = 0
    is_need = 0
    is_make = 0
    aliases = get_aliases(bullet_name)
    for s in seasons:
        if s["start"] <= dt < s["end"]:
            total_duration = (s["end"] - s["start"]).total_seconds()
            elapsed = (dt - s["start"]).total_seconds()
            val = elapsed / total_duration if total_duration > 0 else 0
            val = max(0.0, min(1.0, val))
            in_cs = float(val)
            is_cs = int(s["id"])

            for nb in s["needed"]:
                for alias in aliases:
                    if str(nb) in str(alias):
                        is_need = 1
                        break
                if is_need:
                    break

            for mb in s["made"]:
                for alias in aliases:
                    if str(mb) in str(alias):
                        is_make = 1
                        break
                if is_make:
                    break
            break
    return in_cs, is_cs, is_need, is_make


def get_is_active(dt: datetime, bullet_name: str, gun_events: list[dict]) -> int:
    aliases = get_aliases(bullet_name)
    for ev in gun_events:
        if ev["start"] <= dt < ev["end"]:
            for alias in aliases:
                if str(ev["bullet"]) in str(alias):
                    return 1
    return 0


def get_is_public(dt: datetime, bullet_name: str, predictions: list[dict]) -> float:
    val = 0.0
    aliases = get_aliases(bullet_name)
    for p in predictions:
        match = False
        for alias in aliases:
            if str(p["name"]) in str(alias):
                match = True
                break
        if not match:
            continue
        delta = dt - p["time"]
        days_diff = delta.total_seconds() / (24 * 3600)
        if 0 <= days_diff <= 60:
            curr_val = 1.0 - (days_diff / 60.0)
            if curr_val > val:
                val = float(curr_val)
    return float(val)


def _read_template_info(template_csv: Path):
    df_head = pd.read_csv(template_csv, nrows=1)
    cols = [_norm_text(c) for c in df_head.columns if c is not None]
    if not cols or cols[0] != "date":
        raise RuntimeError(f"模板表头异常: {template_csv}")
    present_exog = [c for c in EXOG_COLS if c in cols]
    target_cols = [c for c in cols[1:] if c not in present_exog]
    if not target_cols:
        raise RuntimeError(f"模板未找到 targets 列: {template_csv}")
    return cols, target_cols, present_exog


def _load_single_bullet(raw_dir: Path, bullet_col: str) -> pd.DataFrame:
    p = raw_dir / f"{bullet_col}.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    cols = {_norm_text(c): c for c in df.columns}
    if "时间戳" not in cols or "均价" not in cols:
        raise RuntimeError(f"CSV 列不匹配: {p}")
    out = df[[cols["时间戳"], cols["均价"]]].copy()
    out.columns = ["ts", bullet_col]
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("Int64")
    out[bullet_col] = pd.to_numeric(out[bullet_col], errors="coerce")
    out = out.dropna(subset=["ts", bullet_col])
    out["ts"] = out["ts"].astype(np.int64)
    out = out.groupby("ts", as_index=False).last()
    return out


def _build_targets_frame(raw_dir: Path, target_cols: list[str]) -> pd.DataFrame:
    merged = None
    for col in target_cols:
        df_col = _load_single_bullet(raw_dir, col)
        merged = df_col if merged is None else pd.merge(merged, df_col, on="ts", how="outer")
    merged = merged.sort_values("ts").reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("合并后数据为空")

    start_ts = int(merged["ts"].min())
    end_ts = int(merged["ts"].max())
    step = 60 * 60 * 1000
    full_ts = np.arange(start_ts - (start_ts % step), end_ts + step, step, dtype=np.int64)

    merged = merged.set_index("ts").reindex(full_ts)
    merged.index.name = "ts"
    merged = merged.reset_index()
    for col in target_cols:
        merged[col] = merged[col].astype(float)
    merged[target_cols] = merged[target_cols].ffill().bfill()
    merged["date"] = pd.to_datetime(merged["ts"], unit="ms")
    return merged[["date"] + target_cols]


def _add_exog_columns(df: pd.DataFrame, category_name: str, holiday_map, seasons, gun_events, predictions) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"])

    is_holiday = [get_is_holiday(dt.to_pydatetime(), holiday_map) for dt in dates]
    in_cs = []
    is_cs = []
    is_need = []
    is_make = []
    is_active = []
    is_public = []
    for dt in dates:
        dtp = dt.to_pydatetime()
        inc, isc, isn, ism = get_season_data(dtp, category_name, seasons)
        in_cs.append(inc)
        is_cs.append(isc)
        is_need.append(isn)
        is_make.append(ism)
        is_active.append(get_is_active(dtp, category_name, gun_events))
        is_public.append(get_is_public(dtp, category_name, predictions))

    df = df.copy()
    df["is_holiday"] = is_holiday
    df["in_CS"] = in_cs
    df["is_CS"] = is_cs
    df["is_need"] = is_need
    df["is_make"] = is_make
    df["is_active"] = is_active
    df["is_public"] = is_public
    return df


def _validate_output_frame(df: pd.DataFrame, template_cols: list[str], target_cols: list[str], present_exog: list[str]):
    actual_cols = [_norm_text(c) for c in df.columns]
    expected_cols = [_norm_text(c) for c in template_cols]
    if actual_cols != expected_cols:
        raise RuntimeError(f"输出列顺序与模板不一致: expected={expected_cols}, got={actual_cols}")

    if "date" not in df.columns:
        raise RuntimeError("输出缺少 date 列")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        raise RuntimeError("date 列存在无法解析的时间")
    if not dates.is_monotonic_increasing:
        raise RuntimeError("date 列未按时间递增排序")
    if dates.duplicated().any():
        raise RuntimeError("date 列存在重复时间")

    num_cols = list(target_cols) + list(present_exog)
    for c in num_cols:
        if c not in df.columns:
            raise RuntimeError(f"输出缺少列: {c}")
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.isna().all():
            raise RuntimeError(f"列全为空或无法转为数值: {c}")
        if c in target_cols and coerced.isna().any():
            raise RuntimeError(f"targets 列存在缺失值(未填满): {c}")


def build_one_category(template_csv: Path, raw_dir: Path, exo_dir: Path, out_dir: Path, overwrite: bool):
    category_name = _norm_text(template_csv.stem)
    out_path = out_dir / f"{category_name}.csv"
    if out_path.exists() and not overwrite:
        return {"category": category_name, "ok": True, "skipped": True}

    template_cols, target_cols, present_exog = _read_template_info(template_csv)
    holiday_map, seasons, gun_events, predictions = load_configs(exo_dir)

    df = _build_targets_frame(raw_dir, target_cols)
    df = _add_exog_columns(df, category_name, holiday_map, seasons, gun_events, predictions)
    df = df[template_cols]
    _validate_output_frame(df, template_cols, target_cols, present_exog)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return {
        "category": category_name,
        "ok": True,
        "skipped": False,
        "rows": int(len(df)),
        "targets": int(len(target_cols)),
        "exog": int(len(present_exog)),
    }


def _clear_dir(p: Path):
    if p.exists():
        for item in p.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        p.mkdir(parents=True, exist_ok=True)


def build_collection_category_exog_all(raw_dir: Path, exo_dir: Path, template_dir: Path, out_dir: Path, workers: int, overwrite: bool):
    if not template_dir.exists():
        raise RuntimeError(f"未找到训练模板目录: {template_dir}")
    templates = sorted(template_dir.glob("*.csv"), key=lambda p: p.name)
    if not templates:
        raise RuntimeError(f"模板目录为空: {template_dir}")

    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(build_one_category, t, raw_dir, exo_dir, out_dir, bool(overwrite)) for t in templates]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                failures.append(str(e))

    ok_count = sum(1 for r in results if r.get("ok"))
    skipped = sum(1 for r in results if r.get("skipped"))
    print(f"Done. ok={ok_count}, skipped={skipped}, total={len(templates)}, failed={len(failures)}")
    if failures:
        for msg in failures:
            print(msg)
        raise RuntimeError("构建带外生变量的总子弹数据失败")


def _legacy_build_collection_category_exog_all(
    legacy_path: Path,
    raw_dir: Path,
    exo_dir: Path,
    template_dir: Path,
    out_dir: Path,
    workers: int,
    overwrite: bool,
):
    spec = importlib.util.spec_from_file_location("legacy_build_collection_category_exog", str(legacy_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载旧脚本: {legacy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not template_dir.exists():
        raise RuntimeError(f"未找到训练模板目录: {template_dir}")
    templates = sorted(template_dir.glob("*.csv"), key=lambda p: p.name)
    if not templates:
        raise RuntimeError(f"模板目录为空: {template_dir}")

    results = []
    failures = []
    for t in templates:
        try:
            results.append(module.build_one_category(t, raw_dir, exo_dir, out_dir, bool(overwrite)))
        except Exception as e:
            failures.append(str(e))

    ok_count = sum(1 for r in results if r.get("ok"))
    skipped = sum(1 for r in results if r.get("skipped"))
    print(f"Legacy Done. ok={ok_count}, skipped={skipped}, total={len(templates)}, failed={len(failures)}")
    if failures:
        for msg in failures:
            print(msg)
        raise RuntimeError("旧脚本构建失败")


def _assert_exog_outputs_equal(dir_a: Path, dir_b: Path, template_dir: Path):
    templates = sorted(template_dir.glob("*.csv"), key=lambda p: p.name)
    for t in templates:
        category_name = _norm_text(t.stem)
        pa = dir_a / f"{category_name}.csv"
        pb = dir_b / f"{category_name}.csv"
        if not pa.exists() or not pb.exists():
            raise RuntimeError(f"一致性校验缺文件: {pa} / {pb}")
        df_a = pd.read_csv(pa, encoding="utf-8-sig")
        df_b = pd.read_csv(pb, encoding="utf-8-sig")
        pd.testing.assert_frame_equal(df_a, df_b, check_dtype=False, check_exact=False, atol=1e-12, rtol=0)


# --- Part 1: Fetch Data Functions ---

def sanitize_filename(name):
    """
    清理文件名，移除或替换Windows文件系统不允许的字符
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name

def create_timestamp_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = DATA_DIR / timestamp
    folder_path.mkdir(parents=True, exist_ok=True)
    return timestamp, folder_path

def save_json_data(folder_path, data):
    json_file = folder_path / "ammo_raw_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return json_file

def convert_to_csv(folder_path, keywords):
    csv_files = []
    
    for item in keywords:
        object_name = item.get('objectName', 'Unknown')
        
        safe_name = sanitize_filename(object_name)
        safe_name = safe_name.strip()
        grade = item.get('grade', '')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if object_name == '.300BLK' and grade:
            csv_filename = f"{safe_name} {grade}_{timestamp}.csv"
        else:
            csv_filename = f"{safe_name}_{timestamp}.csv"
        csv_path = folder_path / csv_filename
        
        price_history = item.get('priceHistoryNdays', [])
        price_history = sorted(price_history, key=lambda x: x.get('timestamp', 0))
        
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow(['时间戳', '时间', '均价', '数据类型'])
            
            for record in price_history:
                ts = record.get('timestamp', 0)
                dt_obj = datetime.fromtimestamp(ts/1000)
                dt = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                avg_price = record.get('avgPrice', '')
                data_type = record.get('dataType', '')
                
                # Filter recent data: only keep integer hours (minute == 0)
                if data_type == 'recent' and dt_obj.minute != 0:
                    continue

                writer.writerow([ts, dt, avg_price, data_type])
        
        csv_files.append((object_name, csv_filename))
    
    return csv_files

def save_summary(folder_path, keywords, csv_files):
    summary_file = folder_path / "summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"弹药数据抓取报告\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"抓取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总弹药数量: {len(keywords)}\n")
        f.write(f"生成的CSV文件数: {len(csv_files)}\n")
        f.write(f"{'=' * 60}\n\n")
        
        f.write(f"生成的CSV文件列表:\n")
        for name, filename in csv_files:
            f.write(f"  - {filename}\n")
        
        f.write(f"\n按等级统计:\n")
        grade_stats = {}
        for item in keywords:
            grade = item.get('grade', '未知')
            grade_stats.setdefault(grade, 0)
            grade_stats[grade] += 1
        
        for grade in sorted(grade_stats.keys()):
            f.write(f"  {grade}级: {grade_stats[grade]}种\n")
        
        f.write(f"\n价格最贵的10种:\n")
        sorted_by_price = sorted(keywords, key=lambda x: x.get('avgPrice', 0), reverse=True)
        for i, item in enumerate(sorted_by_price[:10], 1):
            f.write(f"  {i}. {item.get('objectName', 'N/A')} - ¥{item.get('avgPrice', 0)}\n")

def fetch_data():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始抓取弹药数据...")
    print(f"保存目录: {DATA_DIR}")
    print("-" * 60)
    
    timestamp, folder_path = create_timestamp_folder()
    print(f"创建时间戳文件夹: {folder_path.name}")
    
    print(f"\n正在调用API: {API_ENDPOINT}?days=30")
    try:
        response = requests.get(API_ENDPOINT, headers=HEADERS, params=PARAMS, timeout=30)
    except Exception as e:
        print(f"❌ API请求异常: {e}")
        return False

    if response.status_code != 200:
        print(f"❌ API请求失败: {response.status_code}")
        return False
    
    data = response.json()
    
    if not data.get('success'):
        print(f"❌ API返回失败: {data.get('message')}")
        return False
    
    keywords = data.get('data', {}).get('keywords', [])
    print(f"✅ 获取到 {len(keywords)} 种弹药数据")
    
    json_file = save_json_data(folder_path, data)
    print(f"✅ 保存原始JSON数据: {json_file.name}")
    
    csv_files = convert_to_csv(folder_path, keywords)
    print(f"✅ 生成 {len(csv_files)} 个CSV文件")
    
    save_summary(folder_path, keywords, csv_files)
    print(f"✅ 保存摘要报告: summary.txt")
    
    print("\n" + "=" * 60)
    print(f"🎉 数据抓取完成!")
    print(f"📁 数据位置: {folder_path}")
    print("=" * 60)
    return True

# --- Part 2: Process & Merge Data Functions ---

def normalize_name(raw_name):
    name = re.sub(r'_\d{8}_\d{6}$', '', raw_name)
    
    if "9号霰射_鼠弹" in name:
        name = name.replace('9号霰射_鼠弹_', '9号霰射鼠弹') # case with trailing underscore
        name = name.replace('9号霰射_鼠弹', '9号霰射鼠弹')  # case without
    
    name = name.strip('_')
    name = name.strip()

    if name in ["7.62x51mm Ultra Nosler", "7.62x51mm UN"]:
        name = "7.62x51mm UN"
    
    return name

def parse_time_source2(time_str):
    # Format: 2025-09-08 23时
    if not isinstance(time_str, str):
        return None
    try:
        return datetime.strptime(time_str, '%Y-%m-%d %H时')
    except ValueError:
        pass
    try:
        return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass
    try:
        return datetime.strptime(time_str, '%Y-%m-%d %H:%M')
    except ValueError:
        return None

def process_data():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始汇总处理数据...")
    
    if OUTPUT_DIR.exists():
        print(f"清空目标文件夹: {OUTPUT_DIR}")
        for item in OUTPUT_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    bullet_files = {} # Key: Normalized Name, Value: List of file paths
    
    # Scan Source 1 (Recursive) - Using Path object
    print(f"Scanning {DATA_DIR}...")
    # Convert Path to string for os.walk if needed, but Path objects work with walk in newer python
    # For safety with glob/walk mixed usage:
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith('.csv'):
                continue
            if file in ['summary.txt', 'ammo_raw_data.json']:
                continue
                
            file_path = Path(root) / file
            raw_name = file_path.stem # filename without extension
            norm_name = normalize_name(raw_name)
            
            if norm_name not in bullet_files:
                bullet_files[norm_name] = []
            bullet_files[norm_name].append({'path': file_path, 'source': 1})

    # Scan Source 2 (Flat)
    print(f"Scanning {HISTORY_DIR}...")
    if HISTORY_DIR.exists():
        for file in HISTORY_DIR.iterdir():
            if not file.suffix == '.csv':
                continue
                
            raw_name = file.stem
            norm_name = normalize_name(raw_name)
            
            if norm_name not in bullet_files:
                bullet_files[norm_name] = []
            bullet_files[norm_name].append({'path': file, 'source': 2})

    print(f"Found {len(bullet_files)} bullet types.")

    # Process each bullet type
    intersection_missing = None
    processed_count = 0
    for bullet_name, files in bullet_files.items():
        if bullet_name in IGNORED_BULLETS:
            print(f"Skipping ignored bullet: {bullet_name}")
            continue

        dfs = []
        for file_info in files:
            try:
                # Try reading with different encodings
                try:
                    df = pd.read_csv(file_info['path'], encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_info['path'], encoding='gbk')
                
                # Standardize Columns
                if file_info['source'] == 1:
                    # Columns: 时间戳,时间,均价,数据类型
                    if '时间' in df.columns and '均价' in df.columns:
                        cols_to_keep = ['时间', '均价']
                        if '时间戳' in df.columns:
                            cols_to_keep.insert(0, '时间戳')
                        
                        df = df[cols_to_keep].copy()
                        df['时间'] = pd.to_datetime(df['时间'])
                        
                        # Ensure timestamp exists
                        if '时间戳' not in df.columns:
                             df['时间戳'] = df['时间'].apply(lambda x: int(x.timestamp() * 1000) if pd.notnull(x) else 0)
                        
                        df['时间戳'] = df['时间戳'].astype('int64')
                        dfs.append(df)
                    else:
                        print(f"Skipping {file_info['path']}: Missing columns. Found: {df.columns}")
                        
                elif file_info['source'] == 2:
                    # Columns: 时间,价格
                    price_col = None
                    if '价格' in df.columns:
                        price_col = '价格'
                    elif '均价' in df.columns:
                        price_col = '均价'
                    if '时间' in df.columns and price_col:
                        df['时间'] = df['时间'].apply(parse_time_source2)
                        df['均价'] = df[price_col]
                        # Calculate timestamp (ms)
                        df['时间戳'] = df['时间'].apply(lambda x: int(x.timestamp() * 1000) if pd.notnull(x) else 0)
                        df['时间戳'] = df['时间戳'].astype('int64')
                        df = df[['时间戳', '时间', '均价']]
                        dfs.append(df)
                    else:
                        print(f"Skipping {file_info['path']}: Missing columns. Found: {df.columns}")
                
            except Exception as e:
                path = file_info.get('path', 'Memory Data')
                print(f"Error reading {path}: {e}")

        if not dfs:
            continue

        # Merge
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Remove invalid times
        merged_df = merged_df.dropna(subset=['时间'])
        
        # Drop duplicates
        merged_df = merged_df.drop_duplicates(subset=['时间'], keep='last')
        
        # Sort
        merged_df = merged_df.sort_values(by='时间')
        
        if not merged_df.empty:
            merged_df = merged_df[merged_df['时间'].apply(lambda x: pd.notnull(x) and x.minute == 0)]
            merged_df.set_index('时间', inplace=True)
            
            expected_index = pd.date_range(merged_df.index.min(), merged_df.index.max(), freq='1h')
            missing_index = expected_index.difference(merged_df.index)
            missing_count = missing_index.size
            if intersection_missing is None:
                intersection_missing = missing_index
            else:
                intersection_missing = intersection_missing.intersection(missing_index)
            processed_count += 1
            if missing_count > 0:
                missing_ranges = []
                if missing_count > 0:
                    start = missing_index[0]
                    prev = start
                    for current in missing_index[1:]:
                        if (current - prev) != pd.Timedelta(hours=1):
                            missing_ranges.append((start, prev))
                            start = current
                        prev = current
                    missing_ranges.append((start, prev))
                range_stats = []
                for s, e in missing_ranges:
                    hours = int((e - s) / pd.Timedelta(hours=1)) + 1
                    range_stats.append((s, e, hours))
                range_stats.sort(key=lambda x: x[2], reverse=True)
                earliest_missing = min(range_stats, key=lambda x: x[0])[0]
                latest_missing = max(range_stats, key=lambda x: x[1])[1]
                longest_s, longest_e, longest_h = range_stats[0]
                print(f"🔧 触发插值: {bullet_name} 缺失 {missing_count} 个小时点")
                print(f"🕒 缺失总区间: {earliest_missing.strftime('%Y-%m-%d %H:%M:%S')} ~ {latest_missing.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"🧩 最大连续缺失: {longest_s.strftime('%Y-%m-%d %H:%M:%S')} ~ {longest_e.strftime('%Y-%m-%d %H:%M:%S')} ({longest_h}小时)")
                merged_df = merged_df.resample('1h').interpolate(method='linear')

            merged_df.reset_index(inplace=True)
            
            merged_df['时间戳'] = merged_df['时间'].apply(lambda x: int(x.timestamp() * 1000) if pd.notnull(x) else 0)

        # Ensure timestamp is int64
        merged_df['时间戳'] = merged_df['时间戳'].astype('int64')
        
        output_path = OUTPUT_DIR / f"{bullet_name}.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    if processed_count > 0:
        if intersection_missing is None or intersection_missing.size == 0:
            print("🧾 全局共同缺失区间: 无")
        else:
            global_ranges = []
            start = intersection_missing[0]
            prev = start
            for current in intersection_missing[1:]:
                if (current - prev) != pd.Timedelta(hours=1):
                    global_ranges.append((start, prev))
                    start = current
                prev = current
            global_ranges.append((start, prev))
            print(f"🧾 全局共同缺失区间数: {len(global_ranges)}")
            for s, e in global_ranges:
                hours = int((e - s) / pd.Timedelta(hours=1)) + 1
                print(f"🧾 全局共同缺失区间: {s.strftime('%Y-%m-%d %H:%M:%S')} ~ {e.strftime('%Y-%m-%d %H:%M:%S')} ({hours}小时)")
    
    print(f"✅ 数据汇总完成! 文件已保存至: {OUTPUT_DIR}")
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始构建带外生变量的总子弹数据...")
    workers = min(16, os.cpu_count() or 1)
    _clear_dir(EXOG_OUTPUT_DIR)
    build_collection_category_exog_all(
        raw_dir=OUTPUT_DIR,
        exo_dir=EXOG_DIR,
        template_dir=EXOG_TEMPLATE_DIR,
        out_dir=EXOG_OUTPUT_DIR,
        workers=workers,
        overwrite=True,
    )

    if ENABLE_EXOG_PARITY_CHECK and LEGACY_EXOG_BUILDER_PATH.exists():
        parity_dir = EXOG_OUTPUT_DIR.parent / f"{EXOG_OUTPUT_DIR.name}__parity_tmp__"
        _clear_dir(parity_dir)
        _legacy_build_collection_category_exog_all(
            legacy_path=LEGACY_EXOG_BUILDER_PATH,
            raw_dir=OUTPUT_DIR,
            exo_dir=EXOG_DIR,
            template_dir=EXOG_TEMPLATE_DIR,
            out_dir=parity_dir,
            workers=workers,
            overwrite=True,
        )
        _assert_exog_outputs_equal(EXOG_OUTPUT_DIR, parity_dir, EXOG_TEMPLATE_DIR)
        shutil.rmtree(parity_dir, ignore_errors=True)
        print("✅ 外生输出一致性校验通过")

# --- Main Execution ---

def main():
    print("🚀 启动自动更新流程...")
    
    # Step 1: Fetch new data
    success = fetch_data()
    if not success:
        print("❌ 数据抓取失败，终止后续处理。")
        return

    # Step 2: Process and merge data
    process_data()
    
    print("\n✨ 所有任务执行完毕!")

if __name__ == "__main__":
    main()
