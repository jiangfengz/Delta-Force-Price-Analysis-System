import re

with open("e:/project/三角洲行动物价捕获分析/TimeXer/run_exp19.ps1", "r", encoding="utf-8") as f:
    content = f.read()

# Make sure all lines ending with --num_workers 0 have a backtick
content = re.sub(r'--num_workers 0\s*\n', '--num_workers 0 `\n', content)

with open("e:/project/三角洲行动物价捕获分析/TimeXer/run_exp19.ps1", "w", encoding="utf-8") as f:
    f.write(content)
