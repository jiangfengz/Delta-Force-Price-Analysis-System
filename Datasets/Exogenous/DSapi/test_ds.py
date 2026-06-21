import os
import urllib.request
import json

# 从环境变量读取密钥，切勿将真实密钥硬编码并提交到仓库
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

if not DEEPSEEK_API_KEY:
    raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY（可参考根目录 .env.example）")

payload = {
    "model": "deepseek-reasoner",
    "messages": [
        {"role": "system", "content": "你是一个分析师。"},
        {"role": "user", "content": "请返回一个JSON数组，包含两个对象，每个对象有 'name' (子弹名) 和 'trend' (范围在-1.0到1.0之间的浮点数)。只返回JSON，不要markdown。"}
    ],
    "stream": False
}

body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}

req = urllib.request.Request(DEEPSEEK_URL, data=body, headers=headers, method="POST")
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print("Content:", res['choices'][0]['message']['content'])
        print("Reasoning Content:", res['choices'][0]['message'].get('reasoning_content', 'N/A'))
except Exception as e:
    print("Error:", e)
