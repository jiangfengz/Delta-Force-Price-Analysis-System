# 三角洲行动物价捕获分析 (Delta Force Price Analysis System)

这是一个围绕「三角洲行动」游戏内**子弹物价**的端到端分析系统。系统能够从 API 抓取与清洗价格时间序列，构建带外生变量的数据集，训练 TimeXer 预测模型，并提供 Web 服务展示趋势/预测，最终生成"倒卖计划"（买卖时机 + 仓位分配）。

## 🚀 项目目标

- **持续采集与归一化**：将分散、格式不一的子弹价格数据统一汇总为按小时对齐的时间序列 CSV。
- **可解释的预测与决策辅助**：利用 TimeXer（支持外生变量）预测未来 7 天（168 小时）价格，并基于预测序列自动提取可执行的买卖区间。
- **可交付的可视化与接口**：提供 Web 页面查看历史曲线、预测曲线与"倒卖计划"，并通过 HTTP API 供外部调用。
- **模型评估与版本管理**：完整的实验管理工具链（指标收集、模型对比、外生变量贡献分析、最优模型筛选与微调）。

## 🏗️ 总体架构

项目由三条主链路组成：**数据更新链路**、**离线训练链路**、**在线服务链路**。

```text
            ┌────────────────────────────┐
            │ 远端 API / 公告 / 外部信息   │
            └──────────────┬─────────────┘
                           │
                    (抓取/解析/清洗)
                           │
┌──────────────────────────▼──────────────────────────────┐
│ 数据层：Datasets/                                        │
│ - Datasets/{子弹名}.csv（小时级单子弹 CSV）               │
│ - Datasets with Exogenous/{口径}.csv（多变量+外生变量）    │
│ - Exogenous/（节假日/赛季/品枪活动/公告预测等配置）        │
│ - bullets information.csv（子弹元信息表）                 │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
         (离线训练)                  (在线推理服务读取)
               │                          │
┌──────────────▼──────────────┐   ┌──────▼──────────────────────┐
│ TimeXer（PyTorch）            │   │ Python 推理 Worker            │
│ - run.py 训练/测试入口        │   │ - 加载 19 个口径 checkpoints  │
│ - 数据预处理 → 训练 → 评估    │   │ - 构造多变量+外生变量输入      │
│ - 工具链：指标/对比/可视化等  │   │ - 输出 predPoints（168 小时）   │
└──────────────┬──────────────┘   └──────┬──────────────────────┘
               │                          │ JSONL stdin/stdout
               │                          │
               │                    ┌─────▼──────────────────────────┐
               │                    │ Node.js (Express) Web 后端      │
               │                    │ - API：series/forecast/tradeplan │
               │                    │ - 静态页服务（Chart.js）         │
               │                    └─────┬──────────────────────────┘
               │                          │
               │                    ┌─────▼───────────┐
               └───────────────────►│ 浏览器前端页面    │
                                    └─────────────────┘
```

## 📂 目录结构

- **`web/`**：在线服务与页面
  - `server.js`：Express 后端入口，负责 API 路由及管理 Python 推理子进程。
  - `dataParser.js`：CSV 流式解析器，从 `Datasets/Datasets/` 读取价格数据。
  - `tradePlan.js`：倒卖计划核心逻辑（DP 提取买卖区间 + 置信度映射 + 仓位分配）。
  - `technicalIndicators.js`：技术指标计算模块（SMA/EMA/BOLL/RSI/MACD/KDJ，当前预留）。
  - `timexer_worker.py`：Python 推理 Worker，加载 TimeXer 模型，通过 JSONL（stdin/stdout）与 Node 通信。
  - `public/`：前端静态资源（`index.html`、`trade-plan.html`、CSS 主题体系）。
  - `docs/trade-plan-algo.md`：倒卖计划算法详细说明文档。

- **`TimeXer/`**：模型训练、评估与实验管理
  - 基于 TimeXer 官方实现，当前已训练 **19 个口径**模型（V17/V18，pred_len=168）。
  - `run.py`：训练/测试入口（支持 CLI + `.env` 参数覆盖）。
  - `data_preprocess_all.py`：外生变量合并与数据集构建。
  - `collect_metrics.py`：汇总各实验指标。
  - `compare_metrics.py`：新旧模型对比。
  - `exog_contrib.py`：外生变量贡献度分析（逐一遮蔽法）。
  - `visualize_results.py`：预测 vs 真实值可视化。
  - `tools/`：实验管理工具集（模型备份、最优筛选、微调命令生成、指标重算/审计）。

- **`Datasets/`**：数据与更新脚本
  - `update_and_process_ammo_data.py`：核心数据更新脚本（API 抓取 → 合并历史 → 小时对齐 → 外生变量构建）。
  - `Datasets/`：合并对齐后的单子弹小时级 CSV。
  - `Datasets with Exogenous/`：带 7 列外生变量的品类级 CSV。
  - `Exogenous/`：节假日、赛季、品枪活动、公告预测等外生变量配置文件。
  - `Exogenous/DSapi/`：DeepSeek 公告趋势分析子模块（OCR + LLM）。

## 🛠️ 安装与运行

### 1. 环境准备

- **Node.js**：用于运行 Web 后端。
- **Python 3.9+**：用于数据处理与模型推理。
- **PyTorch**：如果需要 GPU 加速推理，请安装对应的 CUDA 版本。

### 2. 依赖安装

**Python 依赖**：
```bash
pip install -r requirements.txt
```

**Web 依赖**：
```bash
npm run install:web
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件，并配置以下密钥（请替换为实际 Key）：

```env
DF_API_KEY=your_delta_force_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

常用可选环境变量（均有默认值，可按需覆盖）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `3000` | Web 服务端口 |
| `PYTHON` | `python` | Python 解释器路径 |
| `BULLET_DATA_DIR` | `../Datasets/Datasets` | 单子弹 CSV 目录 |
| `SLOTS_TOTAL` | `1100` | 倒卖计划总仓位 |
| `SELL_FEE_RATE` | `0.15` | 卖出手续费率 |
| `PLAN_STRATEGY` | `aggressive` | 策略档位 |

### 4. 启动服务

**启动 Web 服务**：
```bash
# 开发模式（nodemon 热重载）
npm run dev

# 生产模式
npm run start
```
服务默认运行在 `http://localhost:3000`。

**更新数据**：
运行以下脚本抓取最新数据并生成模型所需的输入文件：
```bash
python Datasets/update_and_process_ammo_data.py
```

**运行自检**：
```bash
npm --prefix web run selftest:trade-plan
```

## 📊 核心功能

### 1. 数据更新流
从 API 拉取近 30 天数据，合并历史记录，进行小时级对齐与线性插值补齐，最后结合节假日、赛季、品枪活动、公告预测等信息生成带 7 列外生变量的品类级 CSV 文件。

### 2. 在线预测
Web 后端通过 `timexer_worker.py` JSONL 子进程协议调用训练好的 TimeXer 模型，实时输出未来 168 小时（7天）的价格预测。当前覆盖 19 种子弹口径。

### 3. 倒卖计划 (Trade Plan)
基于预测曲线，使用两状态动态规划提取最佳买卖区间，并结合模型误差（MAPE）计算置信度，自动生成买入/卖出建议与仓位分配（最大余数法）。支持 aggressive/balanced/conservative 三档策略。

### 4. 模型实验管理
`TimeXer/tools/` 提供完整的实验管理工具链：训练命令生成、模型备份/筛选、最优模型自动替换、外生变量贡献度分析、指标重算与审计。

## 📝 许可证

[MIT License](LICENSE) (如适用)
