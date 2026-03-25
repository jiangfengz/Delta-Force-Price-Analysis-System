# 三角洲行动物价捕获分析 (Delta Force Price Analysis System)

这是一个围绕「三角洲行动」游戏内**子弹物价**的端到端分析系统。系统能够从 API 抓取与清洗价格时间序列，构建带外生变量的数据集，训练 TimeXer 预测模型，并提供 Web 服务展示趋势/预测，最终生成“倒卖计划”（买卖时机 + 仓位分配）。

## 🚀 项目目标

- **持续采集与归一化**：将分散、格式不一的子弹价格数据统一汇总为按小时对齐的时间序列 CSV。
- **可解释的预测与决策辅助**：利用 TimeXer（支持外生变量）预测未来价格，并基于预测序列自动提取可执行的买卖区间。
- **可交付的可视化与接口**：提供 Web 页面查看历史曲线、预测曲线与“倒卖计划”，并通过 HTTP API 供外部调用。

## 🏗️ 总体架构

项目由三条主链路组成：**数据更新链路**、**离线训练链路**、**在线服务链路**。

```text
            ┌────────────────────────────┐
            │ 远端 API / 公告 / 外部信息   │
            └──────────────┬─────────────┘
                           │
                    (抓取/解析/清洗)
                           │
┌──────────────────────────▼──────────────────────────┐
│ 数据目录：Datasets                                         │
│ - daily Dataset/historical Datasets/Datasets（小时级 CSV）      │
│ - Exogenous（节假日/赛季/活动/公告预测等）            │
│ - Datasets with Exogenous（用于推理/训练）                │
└──────────────┬──────────────────────────┬───────────┘
               │                          │
         (离线训练)                  (在线推理服务读取)
               │                          │
┌──────────────▼──────────────┐   ┌──────▼─────────────────────────┐
│ TimeXer（PyTorch）            │   │ Python 推理 Worker              │
│ - 训练/评估/导出 results       │   │ - 读取 results/checkpoints     │
│ - 生成 metrics_detail(MAPE)    │   │ - 输出未来 predPoints           │
└──────────────┬──────────────┘   └──────┬─────────────────────────┘
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
  - `server.js`: Express 后端入口，负责 API 路由及管理 Python 推理进程。
  - `tradePlan.js`: 倒卖计划核心逻辑（动态规划提取买卖区间）。
  - `timexer_worker.py`: Python 推理 Worker，加载 TimeXer 模型进行预测。
  - `public/`: 前端静态资源。

- **`TimeXer/`**：模型训练与评估
  - 包含 TimeXer 官方实现及本项目的训练/数据脚本。
  - `run.py`: 训练/测试入口。
  - `data_preprocess_all.py`: 外生变量合并与数据集构建。

- **`Datasets/`**：数据与更新脚本
  - `update_and_process_ammo_data.py`: 核心数据更新脚本（抓取 -> 清洗 -> 对齐 -> 生成外生变量）。
  - `Exogenous/`: 节假日、赛季、活动、公告预测等配置。

## 🛠️ 安装与运行

### 1. 环境准备

- **Node.js**: 用于运行 Web 后端。
- **Python 3.9+**: 用于数据处理与模型推理。
- **PyTorch**: 如果需要 GPU 加速推理，请安装对应的 CUDA 版本。

### 2. 依赖安装

**Python 依赖**:
```bash
pip install -r requirements.txt
```

**Web 依赖**:
```bash
cd web
npm install
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件，并配置以下密钥（请替换为实际 Key）：

```env
DF_API_KEY=your_delta_force_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### 4. 启动服务

**启动 Web 服务**:
```bash
# 开发模式
npm run dev

# 生产模式
npm run start
```
服务默认运行在 `http://localhost:3000`。

**更新数据**:
运行以下脚本抓取最新数据并生成模型所需的输入文件：
```bash
python 测试新子弹价格数据/update_and_process_ammo_data.py
```

## 📊 核心功能

### 1. 数据更新流
从 API 拉取数据，合并历史记录，进行小时级对齐与插值补齐，最后结合节假日、赛季等信息生成带外生变量的 CSV 文件。

### 2. 在线预测
Web 后端通过 `timexer_worker.py` 调用训练好的 TimeXer 模型，实时输出未来 168 小时（7天）的价格预测。

### 3. 倒卖计划 (Trade Plan)
基于预测曲线，使用动态规划算法提取最佳买卖区间，并结合模型误差（MAPE）计算置信度，自动生成买入/卖出建议与仓位分配。

## 📝 许可证

[MIT License](LICENSE) (如适用)
