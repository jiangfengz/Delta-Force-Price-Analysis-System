# 倒卖计划：算法说明

本文档解释“倒卖计划”页面生成候选子弹、提取买卖时机、筛选与仓位分配的完整流程。输出结果用于辅助决策，不构成收益保证。

## 输入与数据来源

- **价格预测**：由 TimeXer 推理服务返回未来一段时间的价格序列（按时间戳升序的点集）。
- **子弹基础信息**：用于确定每个“格子(slot)”对应的堆叠数量（默认 60 或 20），并可携带物品 ID。
- **模型误差（MAPE）**：从 `TimeXer/results/{modelId}/metrics_detail.csv` 读取到对应子弹通道的 MAPE，用于计算置信度与保守估计。

## 关键概念与指标

- **predPoints**：预测点序列 `[{ ts, price }, ...]`，长度为 `predLen`。
- **sellFeeRate**：卖出手续费率（页面参数），卖出净价为 `sellNet = price * (1 - sellFeeRate)`。
- **profitPerUnit**：按“单发/单位”计算的预测净收益（把所有提取出来的交易段的净收益相加）。
- **stackSize**：一个 slot 的堆叠数量（一般为 60，部分子弹为 20）。
- **profitPerSlot**：`profitPerSlot = stackSize * profitPerUnit`。
- **MAPE 与置信度**：
  - `base = clamp(1 - mape / mapeCap, 0, 1)`
  - `confidence = base ^ confGamma`
  - 当没有 MAPE 时，置信度会趋向 0（更保守）。
- **保守估计（worst-case）**：将预测价按 MAPE 进行缩放后，构造更保守的买入成本与卖出收入，计算 `deltaConservative` / `roiConservative`（用于展示风险，不参与候选交易段提取）。

## 总体流程

1. **枚举候选子弹**：从数据集中取出支持的口径分类下所有子弹，逐一获取预测。
2. **逐子弹生成计划**：对每个预测序列，使用动态规划提取买卖时机，计算收益、置信度与评分。
3. **全局筛选与排序**：过滤掉收益为 0 或置信度不足的候选，按评分排序并截断为前 N 个子弹。
4. **仓位分配**：按“收益 × 置信度”的权重把总 slots 分配到各子弹，得到建议持仓数量。

## 逐子弹：交易段提取（动态规划）

目标：在预测价格序列上，找到若干个不重叠的买入/卖出区间，使得“卖出扣费后的净收益总和”最大。随后再用阈值过滤出真正可执行的交易段。

### 1) 构造买入成本与卖出收入序列

对每个小时点 `t = 1..T`（`T = predLen`）：

- 买入成本：`buyCost[t] = price[t]`
- 卖出收入（扣费）：`sellRev[t] = price[t] * (1 - sellFeeRate)`

### 2) 两状态 DP：现金/持有

定义：

- `cash[t]`：处理到第 `t` 个点后，手上不持仓时的最大收益
- `hold[t]`：处理到第 `t` 个点后，手上持仓时的最大收益

转移（并记录前驱用于回溯交易段）：

```
cash[t] = max(cash[t-1], hold[t-1] + sellRev[t])        // 不动 或 卖出
hold[t] = max(hold[t-1], cash[t-1] - buyCost[t])        // 不动 或 买入
```

该 DP 等价于在允许多次买卖（但同一时刻只能持有或空仓）的条件下，找到最大净收益路径。

### 3) 回溯得到交易段

从 `t = T` 反向根据前驱指针找出所有 `buyHour -> sellHour` 区间，并映射到对应的时间戳与预测价格：

- `buyTs / sellTs`
- `buyPricePred / sellPricePred`

### 4) 计算每段交易的指标并做阈值过滤

每段交易计算“预测净收益”与“预测 ROI”：

- `sellPredNet = sellPricePred * (1 - sellFeeRate)`
- `deltaPredNet = sellPredNet - buyPricePred`
- `roiPredNet = deltaPredNet / buyPricePred`

过滤规则（任意不满足即丢弃该交易段）：

- `deltaPredNet > 0`
- `deltaPredNet >= minDeltaAbs`（若 `minDeltaAbs > 0`）
- `roiPredNet >= minRoi`（若 `minRoi > 0`）

最后将剩余交易段的 `deltaPredNet` 累加得到：

- `profitPerUnit = Σ deltaPredNet`

## 保守估计（worst-case）与策略缩放

为展示风险，会对买入/卖出价格进行按 MAPE 缩放的保守估计：

- `mapeScale` 由策略决定：
  - `aggressive = 0.25`
  - `balanced = 0.5`
  - `conservative = 1`
- `mapeUsed = mape * mapeScale`
- 保守买入：`buyWorst = buyPricePred * (1 + mapeUsed)`
- 保守卖出：`sellWorst = sellPricePred * max(0, 1 - mapeUsed)`
- 保守卖出净价：`sellWorstNet = sellWorst * (1 - sellFeeRate)`
- `deltaConservative = sellWorstNet - buyWorst`
- `roiConservative = deltaConservative / buyWorst`

当前页面默认使用 `aggressive` 策略。

## 候选打分与全局筛选

每个子弹生成计划后，计算：

- `profitPerSlot = stackSize * profitPerUnit`（预期单格利润，未折算置信度）
- `confidence`：由 MAPE 映射得到
- `scoreRaw = max(0, profitPerSlot) * confidence^γ`（**综合「置信度」与「预期单格利润」**；γ 由策略决定：标准=1、平衡=1.5、保守=2。标准策略下置信度同样参与评分）

然后进行全局筛选：

- 只保留 `ok && profitPerSlot > 0 && confidence >= minConfidence` 的子弹
- 按 `scoreRaw` 降序排序
- 取前 `maxBullets` 个（若 `maxBullets > 0`）
- 展示用分数：按候选集最大值归一化到 `0..100`（`score = scoreRaw / max(scoreRaw) * 100`，仅用于展示与比较；仓位分配使用未归一化的 `scoreRaw`，见下）

## 仓位分配（allocateSlots）

设总 slots 为 `S = slotsTotal`，对每个候选子弹计算权重：

- `weight = scoreRaw = max(0, profitPerSlot) * confidence^γ`（使用**未归一化**的评分，使持仓与 `score` 严格成正比）
- 若全部权重为 0，则退化为等权分配

分配方法：

1. 先按比例分配 `exact = S * weight / totalWeight`，取整得到 `baseSlots = floor(exact)`。
2. 将剩余 slots 按小数部分从大到小逐个补齐（最大余数法）。

最终得到：

- `slots`：该子弹分配到的格子数
- `units = slots * stackSize`：建议持仓数量
- `positionRatio = slots / slotsTotal`：仓位占比

## 参数说明（默认值）

参数来自环境变量，页面也可通过查询参数覆盖：

- `slotsTotal`（默认 1100）
- `sellFeeRate`（默认 0.15）
- `mapeCap`（默认 0.2）
- `confGamma`（默认 1.0）
- `minRoi`（默认 0.05）
- `minDeltaAbs`（默认 0）
- `minConfidence`（默认 0.5）
- `maxBullets`（默认 20）
- `maxSlotsPerBullet`（默认 200）
- `planConcurrency`（默认 8，用于并发生成计划）

## 结果解读建议

- **优先看 profitPerSlot 与交易段数量**：收益为 0 的通常意味着未来窗口内缺少可执行的涨跌段。
- **把 confidence 当作“模型稳定性”**：置信度越低，仓位分配权重越低。
- **用保守 ROI/收益衡量风险**：当 `deltaConservative` 为负时，说明在误差放大场景下可能不划算。
