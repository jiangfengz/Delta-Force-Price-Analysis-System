import os
import csv
import json
import re
import html as _html
import argparse
from datetime import datetime, timedelta
import sys
import asyncio
import traceback
import random
import warnings
import urllib.request
import urllib.error
import urllib.parse
from dotenv import load_dotenv

# Filter out the specific pin_memory warning from torch dataloader when using CPU
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Configuration
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), ".env"), override=True)
BASE_URL = "https://df-api.shallow.ink"
_df_key = os.getenv('DF_API_KEY')
HEADERS = {
    "Authorization": f"Bearer {_df_key.strip() if _df_key else ''}"
}
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = DEEPSEEK_API_KEY.strip()
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
STANDARD_NAMES_PATH = os.path.join(PROJECT_DIR, "standard_names.csv")
OUTPUT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, os.pardir))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predictions.csv")
DEBUG_DIR = os.path.join(PROJECT_DIR, "调试")

# Concurrency Limit
CONCURRENCY_LIMIT = 5
OCR_CONCURRENCY_LIMIT = 1
OCR_IMAGE_MAX_DIM = 0
OCR_READTEXT_TIMEOUT_SEC = 0

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def _append_debug(debug_lock, debug_log_path, text):
    if not debug_log_path:
        return
    async with debug_lock:
        await asyncio.to_thread(_append_debug_sync, debug_log_path, text)

def _append_debug_sync(debug_log_path, text):
    with open(debug_log_path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")

def _sep(title):
    return f"\n{'=' * 20} {title} {'=' * 20}\n"

def _build_url(url, params=None):
    if not params:
        return url
    qs = urllib.parse.urlencode(params, doseq=True)
    if "?" in url:
        return url + "&" + qs
    return url + "?" + qs

def _http_request(url, method="GET", headers=None, data=None, timeout=30):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), resp.read(), resp.headers

def _http_get_json(url, headers=None, params=None, timeout=30):
    full_url = _build_url(url, params=params)
    status, body, _ = _http_request(full_url, method="GET", headers=headers, timeout=timeout)
    text = body.decode("utf-8", errors="replace")
    return status, json.loads(text)

def _http_get_text(url, headers=None, params=None, timeout=30):
    full_url = _build_url(url, params=params)
    status, body, resp_headers = _http_request(full_url, method="GET", headers=headers, timeout=timeout)
    encoding = getattr(resp_headers, "get_content_charset", lambda: None)() or "utf-8"
    return status, body.decode(encoding, errors="replace")

def _http_get_bytes(url, headers=None, params=None, timeout=60):
    full_url = _build_url(url, params=params)
    status, body, _ = _http_request(full_url, method="GET", headers=headers, timeout=timeout)
    return status, body

def _http_post_json(url, payload, headers=None, timeout=180):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    status, resp_body, _ = _http_request(url, method="POST", headers=h, data=body, timeout=timeout)
    return status, resp_body.decode("utf-8", errors="replace")

def clean_html(raw_html):
    """Remove HTML tags from string."""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def normalize_text(text):
    if not text:
        return ""
    t = _html.unescape(text)
    t = t.replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def meaningful_char_count(text):
    if not text:
        return 0
    t = normalize_text(text)
    t = re.sub(r"\s+", "", t)
    return len(t)

def extract_image_urls(html_text):
    if not html_text:
        return []
    urls = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html_text, flags=re.IGNORECASE)
    seen = set()
    out = []
    for u in urls:
        u = (u or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out

_easyocr_reader = None
_easyocr_init_lock = asyncio.Lock()
_ocr_unavailable_warned = False
_ocr_read_semaphore = asyncio.Semaphore(OCR_CONCURRENCY_LIMIT)

async def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is False:
        return None
    if _easyocr_reader is not None:
        return _easyocr_reader
    async with _easyocr_init_lock:
        if _easyocr_reader is False:
            return None
        if _easyocr_reader is not None:
            return _easyocr_reader
        try:
            try:
                import bidi
                if not hasattr(bidi, "get_display"):
                    from bidi.algorithm import get_display as _bidi_get_display
                    bidi.get_display = _bidi_get_display
            except Exception:
                pass
            import easyocr
        except Exception:
            _easyocr_reader = False
            return None
        
        gpu_available = False
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                print("GPU detected: Using CUDA for OCR.", flush=True)
            else:
                print("GPU not detected (torch.cuda.is_available() is False). Using CPU for OCR.", flush=True)
        except Exception as e:
            print(f"Error checking GPU availability: {e}. Using CPU for OCR.", flush=True)
            gpu_available = False
            
        try:
            _easyocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=gpu_available)
            return _easyocr_reader
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}", flush=True)
            _easyocr_reader = False
            return None

async def ocr_images_from_html(html_text, debug_lock, debug_log_path, article_meta=None, max_images=2):
    urls = extract_image_urls(html_text)
    if not urls:
        return ""

    reader = await _get_easyocr_reader()
    if reader is None:
        global _ocr_unavailable_warned
        if not _ocr_unavailable_warned:
            _ocr_unavailable_warned = True
            print("OCR 不可用：未安装/初始化 easyocr，将跳过图片公告识别（可 pip install easyocr）", flush=True)
        await _append_debug(
            debug_lock,
            debug_log_path,
            _sep("OCR UNAVAILABLE")
            + f"logged_at: {_now_str()}\n"
            + (f"article_meta: {json.dumps(article_meta, ensure_ascii=False)}\n" if article_meta else "")
            + "reason: easyocr not installed or init failed\n"
            + f"image_urls: {json.dumps(urls[:max_images], ensure_ascii=False)}\n"
        )
        return ""

    combined_lines = []
    use_urls = urls[:max_images]
    for idx, url in enumerate(use_urls, start=1):
        try:
            print(f"  OCR downloading image {idx}/{len(use_urls)}...", flush=True)
            status, img_bytes = await asyncio.to_thread(_http_get_bytes, url, None, None, 60)
            if status >= 400:
                raise urllib.error.HTTPError(url, status, "HTTP error", None, None)
        except Exception as e:
            await _append_debug(
                debug_lock,
                debug_log_path,
                _sep("OCR IMAGE DOWNLOAD ERROR")
                + f"logged_at: {_now_str()}\n"
                + (f"article_meta: {json.dumps(article_meta, ensure_ascii=False)}\n" if article_meta else "")
                + f"url: {url}\n"
                + f"error: {e}\n"
                + f"traceback:\n{traceback.format_exc()}\n"
            )
            continue

        try:
            import numpy as np
            import cv2
            buf = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("opencv decode failed")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            print(f"  OCR running image {idx}/{len(use_urls)} (shape={img_rgb.shape[1]}x{img_rgb.shape[0]})...", flush=True)
            async with _ocr_read_semaphore:
                if OCR_READTEXT_TIMEOUT_SEC and OCR_READTEXT_TIMEOUT_SEC > 0:
                    results = await asyncio.wait_for(
                        asyncio.to_thread(reader.readtext, img_rgb, detail=0, paragraph=True),
                        timeout=OCR_READTEXT_TIMEOUT_SEC
                    )
                else:
                    results = await asyncio.to_thread(reader.readtext, img_rgb, detail=0, paragraph=True)
            if results:
                combined_lines.extend([str(x).strip() for x in results if str(x).strip()])
        except Exception as e:
            await _append_debug(
                debug_lock,
                debug_log_path,
                _sep("OCR ERROR")
                + f"logged_at: {_now_str()}\n"
                + (f"article_meta: {json.dumps(article_meta, ensure_ascii=False)}\n" if article_meta else "")
                + f"url: {url}\n"
                + f"error: {e}\n"
                + f"traceback:\n{traceback.format_exc()}\n"
            )

    text = "\n".join(combined_lines).strip()
    await _append_debug(
        debug_lock,
        debug_log_path,
        _sep("OCR RESULT")
        + f"logged_at: {_now_str()}\n"
        + (f"article_meta: {json.dumps(article_meta, ensure_ascii=False)}\n" if article_meta else "")
        + f"image_count_used: {min(len(urls), max_images)}\n"
        + f"text_len: {len(text)}\n"
        + f"text:\n{text}\n"
    )
    return text
    
class SimpleWebCrawler:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Encoding": "identity"
        }

    async def fetch_search_results(self, keywords, limit=5):
        """Fetch search results from Bing/Baidu."""
        if not BeautifulSoup:
            print("BeautifulSoup not installed, skipping web crawl.")
            return ""

        results = []
        
        # Try Bing first
        try:
            q = urllib.parse.quote_plus(keywords or "", safe="")
            url = f"https://cn.bing.com/search?q={q}"
            status, html = await asyncio.to_thread(_http_get_text, url, self.headers, None, 10)
            if status == 200:
                soup = BeautifulSoup(html, 'html.parser')
                items = soup.select('li.b_algo')
                for item in items[:limit]:
                    title_el = item.select_one('h2')
                    snip_el = item.select_one('p') or item.select_one('.b_caption p')
                    if title_el:
                        title = title_el.get_text().strip()
                        snippet = snip_el.get_text().strip() if snip_el else ""
                        results.append(f"标题：{title}\n摘要：{snippet}\n来源：Bing Search")
        except Exception as e:
            print(f"Bing search failed: {e}")

        # If few results, try Baidu
        if len(results) < 2:
            try:
                q = urllib.parse.quote_plus(keywords or "", safe="")
                url = f"https://www.baidu.com/s?wd={q}"
                status, html = await asyncio.to_thread(_http_get_text, url, self.headers, None, 10)
                if status == 200:
                    soup = BeautifulSoup(html, 'html.parser')
                    items = soup.select('div.result.c-container')
                    for item in items[:limit]:
                        title_el = item.select_one('h3')
                        snip_el = item.select_one('.c-abstract') or item.select_one('.content-right_8Zs40')
                        if title_el:
                            title = title_el.get_text().strip()
                            snippet = snip_el.get_text().strip() if snip_el else ""
                            results.append(f"标题：{title}\n摘要：{snippet}\n来源：Baidu Search")
            except Exception as e:
                print(f"Baidu search failed: {e}")

        return "\n---\n".join(results)

def load_bullet_gun_data():
    """Load standard bullet names and bullet<->gun mapping from CSV."""
    bullet_names = []
    bullet_to_guns = {}
    gun_to_bullets = {}
    if not os.path.exists(STANDARD_NAMES_PATH):
        print(f"Error: Standard names file not found at {STANDARD_NAMES_PATH}")
        return [], {}, {}
        
    try:
        with open(STANDARD_NAMES_PATH, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "standard_name" not in reader.fieldnames:
                f.seek(0)
                reader2 = csv.reader(f)
                for row in reader2:
                    if not row:
                        continue
                    name = row[0].strip()
                    if name:
                        bullet_names.append(name)
                        bullet_to_guns.setdefault(name, [])
                return bullet_names, bullet_to_guns, gun_to_bullets

            for row in reader:
                name = (row.get("standard_name") or "").strip()
                if not name:
                    continue
                guns_raw = (row.get("guns") or "").strip()
                guns = [g.strip() for g in guns_raw.split("|") if g.strip()] if guns_raw else []

                bullet_names.append(name)
                bullet_to_guns[name] = guns

                for g in guns:
                    gun_to_bullets.setdefault(g, []).append(name)

        return bullet_names, bullet_to_guns, gun_to_bullets
    except Exception as e:
        print(f"Error reading standard names: {e}")
        return [], {}, {}

async def get_articles_async():
    """Fetch article list from API."""
    url = f"{BASE_URL}/df/tools/article/list"
    print(f"Fetching article list from {url}...")
    try:
        _, data = await asyncio.to_thread(_http_get_json, url, HEADERS, None, 30)
        
        articles = []
        # Structure: data['data']['articles']['list'] -> dict of lists
        if 'data' in data and 'articles' in data['data'] and 'list' in data['data']['articles']:
            article_lists = data['data']['articles']['list']
            for cat_id, art_list in article_lists.items():
                if isinstance(art_list, list):
                    articles.extend(art_list)
        return articles
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return []

async def get_article_detail_async(thread_id):
    """Fetch article detail asynchronously."""
    url = f"{BASE_URL}/df/tools/article/detail"
    params = {"threadID": thread_id}
    
    try:
        _, data = await asyncio.to_thread(_http_get_json, url, HEADERS, params, 30)
        if 'data' in data and 'article' in data['data']:
            article = data['data']['article']
            if 'content' in article and 'text' in article['content']:
                return article['content']['text']
        return ""
    except Exception as e:
        print(f"Error fetching detail for {thread_id}: {e}")
        return ""

def find_relevant_guns(article_text, all_guns):
    if not article_text or not all_guns:
        return []
    matched = []
    for g in all_guns:
        if g and g in article_text:
            matched.append(g)
    matched.sort(key=len, reverse=True)
    return matched

def build_relevant_context(article_text, bullet_names, bullet_to_guns, gun_to_bullets, max_guns=40):
    relevant_guns = find_relevant_guns(article_text, list(gun_to_bullets.keys()))
    if len(relevant_guns) > max_guns:
        relevant_guns = relevant_guns[:max_guns]

    relevant_bullets = set()
    for g in relevant_guns:
        for b in gun_to_bullets.get(g, []):
            relevant_bullets.add(b)

    for b in bullet_names:
        if b and b in article_text:
            relevant_bullets.add(b)

    if not relevant_bullets:
        return bullet_names, relevant_guns

    relevant_bullets_list = [b for b in bullet_names if b in relevant_bullets]
    return relevant_bullets_list, relevant_guns

async def call_deepseek_async(article_text, article_time, bullet_names, bullet_to_guns, gun_to_bullets, debug_lock, debug_log_path, article_meta=None, internet_info=""):
    """Analyze article content using DeepSeek API asynchronously."""
    if not article_text or not bullet_names:
        return []

    if len(article_text) > 30000:
        article_text = article_text[:30000] + "...(truncated)"

    scoped_bullets, scoped_guns = build_relevant_context(
        article_text=article_text,
        bullet_names=bullet_names,
        bullet_to_guns=bullet_to_guns,
        gun_to_bullets=gun_to_bullets,
        max_guns=40
    )
    names_str = ", ".join(scoped_bullets)

    gun_to_bullets_lines = []
    for g in scoped_guns:
        bullets = gun_to_bullets.get(g, [])
        if bullets:
            gun_to_bullets_lines.append(f"{g}: {', '.join(bullets)}")
    gun_to_bullets_str = "\n".join(gun_to_bullets_lines)
    
    system_prompt = (
        "你是一个《三角洲行动》游戏经济系统的专家分析师。你的任务是根据游戏更新公告或文章，并结合互联网市场情报，分析子弹价格的未来趋势。"
    )
    
    user_prompt = f"""
请分析以下文章内容，判断哪些子弹的价格可能会受到影响。

**标准子弹名称列表**（只能从此列表中选择，名称必须完全匹配）：
{names_str}

**枪支与子弹对应关系**（用于把“枪支调整/增强/削弱”等信息映射到相关子弹）：
{gun_to_bullets_str if gun_to_bullets_str else "（文章中未匹配到明确枪支名称，可忽略此部分）"}

**文章信息**：
发布时间：{article_time}
内容：
{article_text}

**互联网市场情报（仅供参考，反映当前玩家讨论热点）**：
{internet_info if internet_info else "（无互联网情报）"}

**分析要求**：
1. 仔细阅读文章，寻找关于武器平衡、子弹调整、新赛季活动、新地图等可能影响子弹供需的信息。
2. 结合互联网情报，验证市场情绪（例如：官方增强了某枪，玩家是否热议该枪变强？如果是，则对应子弹需求更确定上涨）。
3. 对于每一个在列表中且受影响的子弹，预测其价格趋势。
4. 如果文章提到某把枪支“加强/改强/上调/新增强力配件/热门玩法强化”等，会提升该枪支使用率，从而提升其可用子弹的需求；对应子弹趋势应偏向上涨。
5. 如果文章提到某把枪支“削弱/改弱/下调/退环境”等，会降低该枪支使用率；对应子弹趋势应偏向下跌。
6. 趋势强度（Trend）范围从 -1.0 到 1.0。请务必使用该区间内的**任意连续浮点数**（例如 0.15, 0.8, -0.3, -0.75 等）来精细表达影响的力度，**不要仅限于离散值**：
   - 越接近 1.0 代表极度看涨（如：核心机制改变、大幅增强、需求激增）
   - 大于 0.0 但较小的值（如 0.2, 0.3）代表轻微看涨（如：小幅数值微调）
   - 接近 0.0 代表中性/无明显影响
   - 小于 0.0 但较大的值（如 -0.2, -0.3）代表轻微看跌（如：小幅数值下调）
   - 越接近 -1.0 代表极度看跌（如：核心武器大幅削弱、产出过剩、退环境）
   - 请严格根据文章中对武器/子弹改动的“文字描述强烈程度”（微调、重做、大幅等）和受众面，给出细化的浮点数评估。
7. 如果文章中没有提及任何能推断出价格变化的信息，请返回空数组。

**输出格式**：
请仅返回一个纯净的 JSON 数组，不要包含任何 Markdown 格式（如 ```json ... ```）或解释性文字。
JSON 数组中的每个对象应包含以下字段：
- "time": 文章发布时间（直接使用输入的时间字符串）
- "name": 子弹标准名称
- "trend": 预测的趋势值（浮点数）
"""

    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    
    print(f"Calling DeepSeek (Async) for article time: {article_time}...", flush=True)
    try:
        await _append_debug(
            debug_lock,
            debug_log_path,
            _sep("DEEPSEEK REQUEST")
            + f"logged_at: {_now_str()}\n"
            + (f"article_meta: {json.dumps(article_meta, ensure_ascii=False)}\n" if article_meta else "")
            + f"article_time: {article_time}\n"
            + f"system_prompt:\n{system_prompt}\n"
            + f"user_prompt:\n{user_prompt}\n"
            + f"payload:\n{json.dumps(payload, ensure_ascii=False)}\n"
        )

        status, raw_text = await asyncio.to_thread(
            _http_post_json,
            DEEPSEEK_URL,
            payload,
            {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            180
        )
        print(f"  DeepSeek Response Status for {article_time}: {status}", flush=True)
        if status >= 400:
            raise urllib.error.HTTPError(DEEPSEEK_URL, status, "HTTP error", None, None)

        try:
            result = json.loads(raw_text)
        except Exception:
            result = None

        await _append_debug(
            debug_lock,
            debug_log_path,
            _sep("DEEPSEEK RAW RESPONSE")
            + f"logged_at: {_now_str()}\n"
            + (f"article_meta: {json.dumps(article_meta, ensure_ascii=False)}\n" if article_meta else "")
            + f"article_time: {article_time}\n"
            + f"status: {status}\n"
            + f"raw_text:\n{raw_text}\n"
        )

        if isinstance(result, dict) and 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            try:
                data = json.loads(content)
                if isinstance(data, list):
                    print(f"  Success: Parsed {len(data)} predictions for {article_time}", flush=True)
                    await _append_debug(
                        debug_lock,
                        debug_log_path,
                        _sep("DEEPSEEK PARSED RESULT")
                        + f"logged_at: {_now_str()}\n"
                        + (f"article_meta: {json.dumps(article_meta, ensure_ascii=False)}\n" if article_meta else "")
                        + f"article_time: {article_time}\n"
                        + f"assistant_content:\n{content}\n"
                        + f"parsed_json:\n{json.dumps(data, ensure_ascii=False)}\n"
                    )
                    return data
                else:
                    print(f"  DeepSeek returned valid JSON but not a list for {article_time}.", flush=True)
                    await _append_debug(
                        debug_lock,
                        debug_log_path,
                        _sep("DEEPSEEK INVALID SHAPE")
                        + f"logged_at: {_now_str()}\n"
                        + (f"article_meta: {json.dumps(article_meta, ensure_ascii=False)}\n" if article_meta else "")
                        + f"article_time: {article_time}\n"
                        + f"assistant_content:\n{content}\n"
                    )
                    return []
            except json.JSONDecodeError:
                print(f"  DeepSeek returned invalid JSON for {article_time}: {content[:100]}...", flush=True)
                await _append_debug(
                    debug_lock,
                    debug_log_path,
                    _sep("DEEPSEEK INVALID JSON")
                    + f"logged_at: {_now_str()}\n"
                    + (f"article_meta: {json.dumps(article_meta, ensure_ascii=False)}\n" if article_meta else "")
                    + f"article_time: {article_time}\n"
                    + f"assistant_content:\n{content}\n"
                )
                return []
        else:
            print(f"  DeepSeek returned no choices for {article_time}.", flush=True)
            await _append_debug(
                debug_lock,
                debug_log_path,
                _sep("DEEPSEEK NO CHOICES")
                + f"logged_at: {_now_str()}\n"
                + (f"article_meta: {json.dumps(article_meta, ensure_ascii=False)}\n" if article_meta else "")
                + f"article_time: {article_time}\n"
                + f"parsed_result_type: {type(result).__name__}\n"
                + (f"parsed_result:\n{json.dumps(result, ensure_ascii=False)}\n" if isinstance(result, (dict, list)) else "")
            )
            return []
            
    except Exception as e:
        print(f"  Error calling DeepSeek for {article_time}: {e}", flush=True)
        traceback.print_exc()
        await _append_debug(
            debug_lock,
            debug_log_path,
            _sep("DEEPSEEK CALL EXCEPTION")
            + f"logged_at: {_now_str()}\n"
            + (f"article_meta: {json.dumps(article_meta, ensure_ascii=False)}\n" if article_meta else "")
            + f"article_time: {article_time}\n"
            + f"error: {e}\n"
            + f"traceback:\n{traceback.format_exc()}\n"
        )
        return []

async def process_article(art, bullet_names, bullet_to_guns, gun_to_bullets, semaphore, debug_lock, debug_log_path, internet_info=""):
    """Process a single article with semaphore for concurrency control."""
    async with semaphore:
        title = art.get('title', 'No Title')
        thread_id = art.get('threadID')
        created_at = art.get('createdAt')
        
        print(f"Processing: {title} ({created_at})")
        article_meta = {"title": title, "threadID": thread_id, "createdAt": created_at}
        await _append_debug(
            debug_lock,
            debug_log_path,
            _sep("ARTICLE META")
            + f"logged_at: {_now_str()}\n"
            + f"{json.dumps(article_meta, ensure_ascii=False)}\n"
        )
        
        content_html = await get_article_detail_async(thread_id)
        if not content_html:
            print(f"  Skipping {title}: No content found.")
            await _append_debug(
                debug_lock,
                debug_log_path,
                _sep("ARTICLE CONTENT MISSING")
                + f"logged_at: {_now_str()}\n"
                + f"{json.dumps(article_meta, ensure_ascii=False)}\n"
            )
            return []
            
        content_text = normalize_text(clean_html(content_html))
        await _append_debug(
            debug_lock,
            debug_log_path,
            _sep("ARTICLE CONTENT")
            + f"logged_at: {_now_str()}\n"
            + f"{json.dumps(article_meta, ensure_ascii=False)}\n"
            + f"html:\n{content_html}\n"
            + f"text:\n{content_text}\n"
        )

        # Always attempt OCR if images are present (User request)
        ocr_text = await ocr_images_from_html(
            html_text=content_html,
            debug_lock=debug_lock,
            debug_log_path=debug_log_path,
            article_meta=article_meta,
            max_images=3
        )
        
        if ocr_text:
            print(f"  OCR extracted {len(ocr_text)} chars from images.")
            content_text += "\n" + ocr_text

        analysis_text = f"标题：{title}\n\n{content_text}".strip()
        if meaningful_char_count(analysis_text) < 10:
            print(f"  Skipping {title}: Content too short (even with OCR).")
            await _append_debug(
                debug_lock,
                debug_log_path,
                _sep("ARTICLE CONTENT TOO SHORT")
                + f"logged_at: {_now_str()}\n"
                + f"{json.dumps(article_meta, ensure_ascii=False)}\n"
                + f"text_len: {len(analysis_text)}\n"
                + f"meaningful_char_count: {meaningful_char_count(analysis_text)}\n"
            )
            return []
            
        print(f"  Sending {len(analysis_text)} chars to DeepSeek for analysis...", flush=True)
        return await call_deepseek_async(
            analysis_text,
            created_at,
            bullet_names,
            bullet_to_guns,
            gun_to_bullets,
            debug_lock,
            debug_log_path,
            article_meta=article_meta,
            internet_info=internet_info
        )

async def main_async():
    parser = argparse.ArgumentParser(description="Analyze bullet price trends from articles.")
    parser.add_argument("--time", help="Current time point (YYYY-MM-DD HH:MM:SS). Defaults to now.")
    parser.add_argument("--max-articles", type=int, default=0, help="Limit number of articles to process (0 means no limit).")
    args = parser.parse_args()
    
    # 1. Determine time range
    if args.time:
        try:
            current_time = datetime.strptime(args.time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("Error: Invalid time format. Use YYYY-MM-DD HH:MM:SS")
            sys.exit(1)
    else:
        current_time = datetime.now()
        
    cutoff_time = datetime(2025, 6, 1, 0, 0, 0)
    print(f"Current Time: {current_time}")
    print(f"Time Window: {cutoff_time} to {current_time}")

    _ensure_dir(DEBUG_DIR)
    # Use a fixed debug log file to avoid creating new files every run
    debug_log_path = os.path.join(DEBUG_DIR, "debug.log")
    
    # Optional: Rotate debug log if it gets too big, or just append. 
    # For now, we append, but maybe add a separator for new run.
    debug_lock = asyncio.Lock()
    await _append_debug(
        debug_lock,
        debug_log_path,
        "\n" + "#" * 50 + "\n" + 
        _sep("RUN START")
        + f"logged_at: {_now_str()}\n"
        + f"current_time: {current_time}\n"
        + f"cutoff_time: {cutoff_time}\n"
        + f"concurrency_limit: {CONCURRENCY_LIMIT}\n"
    )
    
    # 2. Load standard names and bullet-gun mapping
    print("Loading standard bullet names...")
    bullet_names, bullet_to_guns, gun_to_bullets = load_bullet_gun_data()
    if not bullet_names:
        print("No bullet names found. Exiting.")
        sys.exit(1)
    
    articles = await get_articles_async()
    target_articles = []

    for art in articles:
        created_at_str = art.get('createdAt')
        if not created_at_str:
            continue
        try:
            created_at = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
            if cutoff_time <= created_at <= current_time:
                target_articles.append(art)
        except ValueError:
            continue

    print(f"Articles in time range: {len(target_articles)}")

    if not target_articles:
        print("No articles found in the specified time range.")
    elif args.max_articles and args.max_articles > 0:
        def _art_time(a):
            s = a.get("createdAt")
            if not s:
                return datetime.min
            try:
                return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return datetime.min
        target_articles.sort(key=_art_time, reverse=True)
        target_articles = target_articles[:args.max_articles]
        print(f"Limited articles to process: {len(target_articles)}")

    print("Crawling web for market sentiment (this may take a few seconds)...")
    internet_info = ""
    crawler = SimpleWebCrawler()
    crawl_tasks = [
        crawler.fetch_search_results("三角洲行动 子弹 价格", limit=3),
        crawler.fetch_search_results("三角洲行动 涨跌 趋势", limit=3),
        crawler.fetch_search_results("三角洲行动 倒卖 赚钱", limit=2)
    ]
    crawl_results = await asyncio.gather(*crawl_tasks)
    internet_info = "\n\n".join([r for r in crawl_results if r])
    
    if internet_info:
        print("Web crawl successful. Market sentiment data gathered.")
        await _append_debug(
            debug_lock,
            debug_log_path,
            _sep("WEB CRAWL INFO")
            + f"logged_at: {_now_str()}\n"
            + f"{internet_info}\n"
        )
    else:
        print("Web crawl yielded no results (or failed). Continuing without it.")

    # 5. Analyze articles concurrently
    all_predictions = []
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    tasks = [
        process_article(art, bullet_names, bullet_to_guns, gun_to_bullets, semaphore, debug_lock, debug_log_path, internet_info)
        for art in target_articles
    ]
    results = await asyncio.gather(*tasks)
    
    for res in results:
        if res:
            all_predictions.extend(res)

    # 6. Save results (Append mode with Deduplication)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    try:
        existing_records = set()
        file_exists = os.path.exists(OUTPUT_FILE)
        
        if file_exists:
            try:
                with open(OUTPUT_FILE, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None) # Skip header
                    for row in reader:
                        if len(row) >= 3:
                            # Store as tuple for hashing: (time, name, str(trend))
                            existing_records.add((row[0].strip(), row[1].strip(), str(row[2]).strip()))
            except Exception as read_e:
                print(f"Warning: Could not read existing file for deduplication: {read_e}")

        mode = 'a' if file_exists else 'w'
        
        with open(OUTPUT_FILE, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["time", "name", "trend"])
            
            count = 0
            skipped = 0
            for p in all_predictions:
                if 'time' in p and 'name' in p and 'trend' in p:
                    time_val = str(p['time']).strip()
                    name_val = str(p['name']).strip()
                    trend_val = str(p['trend']).strip()
                    
                    record_key = (time_val, name_val, trend_val)
                    
                    if record_key not in existing_records:
                        writer.writerow([time_val, name_val, trend_val])
                        existing_records.add(record_key) # Update set to avoid duplicates within the current batch too
                        count += 1
                    else:
                        skipped += 1
        
        if file_exists:
             print(f"Successfully appended {count} predictions to {OUTPUT_FILE} (Skipped {skipped} duplicates)")
        else:
             print(f"Successfully created {OUTPUT_FILE} with {count} predictions (Skipped {skipped} duplicates)")
             
    except Exception as e:
        print(f"Error saving CSV: {e}")
        await _append_debug(
            debug_lock,
            debug_log_path,
            _sep("CSV SAVE ERROR")
            + f"logged_at: {_now_str()}\n"
            + f"error: {e}\n"
            + f"traceback:\n{traceback.format_exc()}\n"
        )
    await _append_debug(
        debug_lock,
        debug_log_path,
        _sep("RUN END")
        + f"logged_at: {_now_str()}\n"
        + f"output_file: {OUTPUT_FILE}\n"
        + f"debug_log: {debug_log_path}\n"
        + f"predictions_count: {len(all_predictions)}\n"
    )

if __name__ == "__main__":
    # Windows-specific event loop policy to avoid issues
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async())
