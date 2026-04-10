const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function numberFromEnv(name, fallback) {
  const raw = process.env[name];
  if (raw == null || raw === '') return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

function intFromEnv(name, fallback) {
  const raw = process.env[name];
  if (raw == null || raw === '') return fallback;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) ? n : fallback;
}

function strategyFromEnv(name, fallback) {
  const raw = process.env[name];
  if (raw == null || raw === '') return fallback;
  const v = String(raw).trim().toLowerCase();
  if (v === 'aggressive' || v === 'balanced' || v === 'conservative') return v;
  return fallback;
}

function getPlanParams() {
  return {
    slotsTotal: intFromEnv('SLOTS_TOTAL', 1100),
    sellFeeRate: numberFromEnv('SELL_FEE_RATE', 0.15),
    mapeCap: numberFromEnv('MAPE_CAP', 0.2),
    confGamma: numberFromEnv('CONF_GAMMA', 1.0),
    minRoi: numberFromEnv('MIN_ROI', 0.05),
    minDeltaAbs: numberFromEnv('MIN_DELTA_ABS', 0),
    triggerTheta: numberFromEnv('TRIGGER_THETA', 0.8),
    kappaMape: numberFromEnv('KAPPA_MAPE', 0),
    minConfidence: numberFromEnv('MIN_CONFIDENCE', 0.5),
    strategy: strategyFromEnv('PLAN_STRATEGY', 'aggressive'),
    maxBullets: intFromEnv('MAX_BULLETS', 20),
    maxSlotsPerBullet: intFromEnv('MAX_SLOTS_PER_BULLET', 200),
    maxSlotsPerGroup: numberFromEnv('MAX_SLOTS_PER_GROUP', 0.3),
    planConcurrency: intFromEnv('PLAN_CONCURRENCY', 8)
  };
}

function buildPlanParams(paramsOverride) {
  const params = getPlanParams();
  const o = paramsOverride && typeof paramsOverride === 'object' ? paramsOverride : null;
  if (!o) return params;

  const setNumber = (key) => {
    const v = Number(o[key]);
    if (Number.isFinite(v)) params[key] = v;
  };
  const setInt = (key) => {
    const v = Number.parseInt(o[key], 10);
    if (Number.isFinite(v)) params[key] = v;
  };
  const setStrategy = () => {
    if (o.strategy == null) return;
    const v = String(o.strategy).trim().toLowerCase();
    if (v === 'aggressive' || v === 'balanced' || v === 'conservative') params.strategy = v;
  };

  setInt('slotsTotal');
  setNumber('sellFeeRate');
  setNumber('mapeCap');
  setNumber('confGamma');
  setNumber('minRoi');
  setNumber('minDeltaAbs');
  setNumber('triggerTheta');
  setNumber('kappaMape');
  setNumber('minConfidence');
  setStrategy();
  setInt('maxBullets');
  setInt('maxSlotsPerBullet');
  setNumber('maxSlotsPerGroup');
  setInt('planConcurrency');

  return params;
}

function localDayKey(tsMs) {
  const d = new Date(tsMs);
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

async function readFirstLineUtf8(filePath) {
  return new Promise((resolve, reject) => {
    const stream = fs.createReadStream(filePath, { encoding: 'utf8' });
    let buf = '';
    let resolved = false;
    const cleanup = () => {
      stream.removeAllListeners();
      stream.destroy();
    };
    stream.on('data', (chunk) => {
      if (resolved) return;
      buf += chunk;
      const idx = buf.search(/\r?\n/);
      if (idx >= 0) {
        resolved = true;
        const line = buf.slice(0, idx).replace(/^\ufeff/, '');
        cleanup();
        resolve(line);
      }
    });
    stream.on('error', (e) => {
      if (resolved) return;
      resolved = true;
      cleanup();
      reject(e);
    });
    stream.on('end', () => {
      if (resolved) return;
      resolved = true;
      cleanup();
      resolve(buf.replace(/^\ufeff/, '').trim());
    });
  });
}

function parseCsvLine(line) {
  if (!line) return [];
  return line.split(',').map((s) => s.trim().replace(/^\ufeff/, ''));
}

function approxEqual(a, b, eps = 1e-9) {
  return Math.abs(a - b) <= eps;
}

function computeStackSizeFromProfitCoefficient(profitCoefficient) {
  if (!Number.isFinite(profitCoefficient)) return 60;
  return approxEqual(profitCoefficient, 0.333, 1e-6) ? 20 : 60;
}

let bulletInfoPromise = null;
let bulletInfoCache = null;

async function loadBulletInfoMap(rootDir) {
  if (bulletInfoCache) return bulletInfoCache;
  if (bulletInfoPromise) return bulletInfoPromise;

  const csvPath = process.env.BULLET_INFO_CSV
    ? path.resolve(process.env.BULLET_INFO_CSV)
    : path.resolve(rootDir, '测试新子弹价格数据', '子弹信息对齐.csv');

  bulletInfoPromise = new Promise((resolve) => {
    if (!fs.existsSync(csvPath)) {
      resolve(new Map());
      return;
    }
    const map = new Map();
    fs.createReadStream(csvPath)
      .pipe(csv({ mapHeaders: ({ header }) => header.trim().replace(/^\ufeff/, '') }))
      .on('data', (row) => {
        const itemIdRaw = row['物品ID'];
        const standardName = row['standard_name'];
        const newName = row['newname'];
        const itemName = row['物品名称'];
        const profitCoeffRaw = row['profit_coefficient'];
        const itemId = itemIdRaw == null ? null : String(itemIdRaw).trim();
        const profitCoefficient = profitCoeffRaw == null ? Number.NaN : Number(profitCoeffRaw);
        const stackSize = computeStackSizeFromProfitCoefficient(profitCoefficient);
        const record = { itemId, profitCoefficient, stackSize };
        for (const key of [standardName, newName, itemName]) {
          if (!key) continue;
          const k = String(key).trim();
          if (!k) continue;
          if (!map.has(k)) map.set(k, record);
        }
      })
      .on('end', () => resolve(map))
      .on('error', () => resolve(new Map()));
  }).then((map) => {
    bulletInfoCache = map;
    bulletInfoPromise = null;
    return map;
  });

  return bulletInfoPromise;
}

const headerCache = new Map();

async function loadTargetColumnIndexMap(rootDir, dataParser, modelGroup) {
  if (headerCache.has(modelGroup)) return headerCache.get(modelGroup);
  const filePath = path.resolve(rootDir, 'TimeXer', 'dataset', 'bullet', 'collection_category', `${modelGroup}.csv`);
  if (!fs.existsSync(filePath)) {
    headerCache.set(modelGroup, null);
    return null;
  }
  const line = await readFirstLineUtf8(filePath);
  const header = parseCsvLine(line);
  const columns = header.slice(1);
  const bulletsInCategory = dataParser.getBulletsByCategory(modelGroup);
  const bulletSet = new Set(bulletsInCategory);
  const targetColumns = [];
  for (const col of columns) {
    if (bulletSet.has(col)) targetColumns.push(col);
  }
  const indexMap = new Map();
  for (let i = 0; i < targetColumns.length; i += 1) {
    indexMap.set(targetColumns[i], i);
  }
  headerCache.set(modelGroup, indexMap);
  return indexMap;
}

const metricsCache = new Map();

async function loadMetricsMap(rootDir, modelId) {
  if (metricsCache.has(modelId)) return metricsCache.get(modelId);
  const metricsPath = path.resolve(rootDir, 'TimeXer', 'results', modelId, 'metrics_detail.csv');
  if (!fs.existsSync(metricsPath)) {
    metricsCache.set(modelId, null);
    return null;
  }
  const text = await fs.promises.readFile(metricsPath, 'utf8');
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) {
    metricsCache.set(modelId, null);
    return null;
  }
  const header = parseCsvLine(lines[0]);
  const mapeIndex = header.findIndex((h) => h === 'MAPE');
  const channelIndex = header.findIndex((h) => h === 'Channel');
  if (mapeIndex < 0 || channelIndex < 0) {
    metricsCache.set(modelId, null);
    return null;
  }
  const map = new Map();
  for (let i = 1; i < lines.length; i += 1) {
    const parts = parseCsvLine(lines[i]);
    const channel = parts[channelIndex];
    const mape = Number(parts[mapeIndex]);
    if (!channel || !Number.isFinite(mape)) continue;
    map.set(channel, mape);
  }
  metricsCache.set(modelId, map);
  return map;
}

function confidenceFromMape(mape, mapeCap, gamma) {
  if (!Number.isFinite(mape) || mape < 0) return 0;
  const base = clamp(1 - mape / mapeCap, 0, 1);
  return gamma === 1 ? base : Math.pow(base, gamma);
}

function mapeScaleFromStrategy(strategy) {
  if (strategy === 'aggressive') return 0.25;
  if (strategy === 'balanced') return 0.5;
  return 1;
}

function buildWorstCaseSeries(predPrices, mape, sellFeeRate, mapeScale) {
  const buyCost = [0];
  const sellRev = [0];
  const scale = Number.isFinite(mapeScale) && mapeScale >= 0 ? mapeScale : 1;
  const mapeUsed = (Number.isFinite(mape) && mape >= 0 ? mape : 0) * scale;
  const sellFactor = Math.max(0, 1 - mapeUsed);
  for (let i = 0; i < predPrices.length; i += 1) {
    const p = predPrices[i];
    buyCost.push(p * (1 + mapeUsed));
    sellRev.push(p * sellFactor * (1 - sellFeeRate));
  }
  return { buyCost, sellRev };
}

function buildPredictedNetSeries(predPrices, sellFeeRate) {
  const buyCost = [0];
  const sellRev = [0];
  const fee = Number.isFinite(sellFeeRate) && sellFeeRate >= 0 ? sellFeeRate : 0;
  for (let i = 0; i < predPrices.length; i += 1) {
    const p = predPrices[i];
    buyCost.push(p);
    sellRev.push(p * (1 - fee));
  }
  return { buyCost, sellRev };
}

function backtrackTradeHours(predPoints, cashPrev, holdPrev) {
  let state = 'cash';
  let t = predPoints.length;
  let pendingSellHour = null;
  const tradesRev = [];
  while (t > 0) {
    if (state === 'cash') {
      const prev = cashPrev[t];
      if (prev && prev.from === 'hold' && prev.action === 'sell') {
        pendingSellHour = t;
        state = 'hold';
        t = prev.t;
        continue;
      }
      t = prev ? prev.t : t - 1;
      continue;
    }
    const prev = holdPrev[t];
    if (prev && prev.from === 'cash' && prev.action === 'buy') {
      const buyHour = t;
      const sellHour = pendingSellHour;
      if (sellHour != null && buyHour < sellHour) {
        tradesRev.push({
          buyHour,
          sellHour,
          buyTs: predPoints[buyHour - 1].ts,
          sellTs: predPoints[sellHour - 1].ts,
          buyPricePred: predPoints[buyHour - 1].price,
          sellPricePred: predPoints[sellHour - 1].price
        });
      }
      pendingSellHour = null;
      state = 'cash';
      t = prev.t;
      continue;
    }
    t = prev ? prev.t : t - 1;
  }
  return tradesRev.reverse();
}

function attachTradeMetrics(tr, mape, params) {
  const fee = Number.isFinite(params.sellFeeRate) && params.sellFeeRate >= 0 ? params.sellFeeRate : 0;
  const buyPred = tr.buyPricePred;
  const sellPred = tr.sellPricePred;
  const sellPredNet = sellPred * (1 - fee);
  const buyPredCost = buyPred;
  const deltaPredNet = sellPredNet - buyPredCost;
  const roiPredNet = buyPredCost > 0 ? deltaPredNet / buyPredCost : 0;

  const mapeScale = mapeScaleFromStrategy(params.strategy);
  const mapeForWorst = Number.isFinite(mape) && mape >= 0 ? mape : params.mapeCap;
  const mapeUsed = (Number.isFinite(mapeForWorst) && mapeForWorst >= 0 ? mapeForWorst : 0) * mapeScale;
  const sellFactor = Math.max(0, 1 - mapeUsed);
  const buyWorst = buyPred * (1 + mapeUsed);
  const sellWorst = sellPred * sellFactor;
  const sellWorstNet = sellWorst * (1 - fee);
  const deltaConservative = sellWorstNet - buyWorst;
  const roiConservative = buyWorst > 0 ? deltaConservative / buyWorst : 0;

  return {
    ...tr,
    buyPredCost,
    sellPredNet,
    deltaPredNet,
    roiPredNet,
    buyWorst,
    sellWorst,
    sellWorstNet,
    deltaConservative,
    roiConservative
  };
}

function extractTradesDp(predPoints, mape, params) {
  const prices = predPoints.map((p) => p.price);
  const { buyCost, sellRev } = buildPredictedNetSeries(prices, params.sellFeeRate);
  const T = prices.length;

  const cash = new Array(T + 1).fill(0);
  const hold = new Array(T + 1).fill(Number.NEGATIVE_INFINITY);
  const cashPrev = new Array(T + 1).fill(null);
  const holdPrev = new Array(T + 1).fill(null);

  cashPrev[0] = { from: null, t: 0, action: 'init' };
  holdPrev[0] = { from: null, t: 0, action: 'init' };

  for (let t = 1; t <= T; t += 1) {
    const stayCash = cash[t - 1];
    const sellCash = hold[t - 1] + sellRev[t];
    if (sellCash > stayCash) {
      cash[t] = sellCash;
      cashPrev[t] = { from: 'hold', t: t - 1, action: 'sell' };
    } else {
      cash[t] = stayCash;
      cashPrev[t] = { from: 'cash', t: t - 1, action: 'stay' };
    }

    const stayHold = hold[t - 1];
    const buyHold = cash[t - 1] - buyCost[t];
    if (buyHold > stayHold) {
      hold[t] = buyHold;
      holdPrev[t] = { from: 'cash', t: t - 1, action: 'buy' };
    } else {
      hold[t] = stayHold;
      holdPrev[t] = { from: 'hold', t: t - 1, action: 'stay' };
    }
  }

  const trades = backtrackTradeHours(predPoints, cashPrev, holdPrev).map((tr) => attachTradeMetrics(tr, mape, params));
  const filteredTrades = trades.filter((tr) => {
    if (!Number.isFinite(tr.deltaPredNet) || tr.deltaPredNet <= 0) return false;
    if (params.minDeltaAbs > 0 && tr.deltaPredNet < params.minDeltaAbs) return false;
    if (params.minRoi > 0 && tr.roiPredNet < params.minRoi) return false;
    return true;
  });

  const profitPerUnit = filteredTrades.reduce((sum, tr) => sum + tr.deltaPredNet, 0);
  return { trades: filteredTrades, profitPerUnit };
}

function scoreFromRiskAdjusted(riskAdjustedProfitPerSlot, mape, kappaMape) {
  if (!Number.isFinite(riskAdjustedProfitPerSlot)) return Number.NEGATIVE_INFINITY;
  if (kappaMape > 0 && Number.isFinite(mape) && mape >= 0) {
    return riskAdjustedProfitPerSlot / (1 + kappaMape * mape);
  }
  return riskAdjustedProfitPerSlot;
}

function allocateSlots(candidates, params) {
  const S = Math.max(0, Number.parseInt(String(params.slotsTotal), 10) || 0);
  if (!Array.isArray(candidates) || candidates.length === 0 || S <= 0) {
    return { slotsTotal: S, slotsUsed: 0, plans: [] };
  }

  const items = candidates.map((c, idx) => {
    const profitPerSlot = Number(c && c.profitPerSlot);
    const confidence = clamp(Number(c && c.confidence), 0, 1);
    const score = Number(c && c.score);
    const weightedProfit = (Number.isFinite(profitPerSlot) ? Math.max(0, profitPerSlot) : 0) * confidence;
    const weight = Number.isFinite(score) && score > 0 ? score : weightedProfit;
    return { idx, weight };
  });

  let totalWeight = items.reduce((sum, it) => sum + it.weight, 0);
  if (!(totalWeight > 0)) {
    for (const it of items) it.weight = 1;
    totalWeight = items.length;
  }

  const baseSlots = new Array(items.length).fill(0);
  const remainders = items.map((it, i) => {
    const exact = (S * it.weight) / totalWeight;
    const base = Math.floor(exact);
    baseSlots[i] = base;
    return { i, rem: exact - base };
  });

  let used = baseSlots.reduce((sum, v) => sum + v, 0);
  let remaining = S - used;
  remainders.sort((a, b) => {
    if (b.rem !== a.rem) return b.rem - a.rem;
    const wa = items[a.i].weight;
    const wb = items[b.i].weight;
    if (wb !== wa) return wb - wa;
    return a.i - b.i;
  });
  for (let k = 0; k < remainders.length && remaining > 0; k += 1) {
    baseSlots[remainders[k].i] += 1;
    remaining -= 1;
  }

  const allocated = candidates
    .map((c, i) => {
      const slots = baseSlots[i] || 0;
      const positionWeight = items[i].weight;
      if (slots <= 0) return null;
      return {
        ...c,
        slots,
        units: slots * c.stackSize,
        positionWeight,
        positionRatio: S > 0 ? slots / S : 0
      };
    })
    .filter(Boolean);

  used = allocated.reduce((sum, p) => sum + p.slots, 0);
  return { slotsTotal: S, slotsUsed: used, plans: allocated };
}

async function asyncPool(items, concurrency, mapper) {
  const results = new Array(items.length);
  let nextIndex = 0;

  async function worker() {
    while (true) {
      const idx = nextIndex;
      nextIndex += 1;
      if (idx >= items.length) return;
      try {
        results[idx] = await mapper(items[idx], idx);
      } catch (e) {
        results[idx] = { ok: false, error: e };
      }
    }
  }

  const n = Math.max(1, Math.min(concurrency, items.length));
  await Promise.all(new Array(n).fill(0).map(() => worker()));
  return results;
}

async function createBulletPlan({ rootDir, dataParser, forecast, params }) {
  const bullet = forecast.bullet;
  const modelGroup = forecast.modelGroup;
  const modelId = forecast.modelId;
  const predPoints = Array.isArray(forecast.points) ? forecast.points.slice().sort((a, b) => a.ts - b.ts) : [];
  const predLen = predPoints.length;
  if (!modelGroup || !modelId || predLen <= 0) {
    return {
      ok: false,
      bullet,
      modelGroup: modelGroup || null,
      reason: '预测数据不完整'
    };
  }

  const bulletInfoMap = await loadBulletInfoMap(rootDir);
  const bulletInfo = bulletInfoMap.get(bullet) || { itemId: null, stackSize: 60, profitCoefficient: Number.NaN };

  const indexMap = await loadTargetColumnIndexMap(rootDir, dataParser, modelGroup);
  const colIndex = indexMap ? indexMap.get(bullet) : null;
  const channel = colIndex == null ? null : `Channel_${colIndex}`;

  const metricsMap = await loadMetricsMap(rootDir, modelId);
  const mape = channel && metricsMap ? metricsMap.get(channel) : null;
  const confidence = confidenceFromMape(mape, params.mapeCap, params.confGamma);

  let trades = [];
  let profitPerUnit = 0;
  if (predLen > 0) {
    const extracted = extractTradesDp(predPoints, mape, params);
    trades = extracted.trades;
    profitPerUnit = extracted.profitPerUnit;
  }

  const profitPerSlot = bulletInfo.stackSize * profitPerUnit;
  const riskAdjustedProfitPerSlot = params.strategy === 'aggressive' ? profitPerSlot : profitPerSlot * confidence;
  const score = params.strategy === 'aggressive'
    ? (Number.isFinite(profitPerSlot) ? profitPerSlot : 0)
    : (Number.isFinite(confidence) ? confidence : 0) * (Number.isFinite(profitPerSlot) ? profitPerSlot : 0);

  return {
    ok: true,
    bullet,
    itemId: bulletInfo.itemId,
    modelGroup,
    modelId,
    predLen,
    stackSize: bulletInfo.stackSize,
    mape,
    confidence,
    strategy: params.strategy,
    score,
    profitPerUnit,
    profitPerSlot,
    riskAdjustedProfitPerSlot,
    trades
  };
}

async function createTradePlanAll({ dataParser, forecastProvider, forceRefresh = false, paramsOverride = null }) {
  const params = buildPlanParams(paramsOverride);
  const rootDir = path.resolve(__dirname, '..');

  const supportedModelGroups = new Set([
    '5.56x45mm',
    '.300BLK',
    '9x19mm',
    '9x39mm',
    '7.62x39mm',
    '7.62x51mm',
    '7.62x54R',
    '5.45x39mm',
    '5.7x28mm',
    '5.8x42mm',
    '6.8x51mm',
    '4.6x30mm',
    '12.7x55mm',
    '12 Gauge',
    '.357 Magnum',
    '45-70 Govt',
    '.45 ACP',
    '.50 AE',
    '箭矢'
  ]);

  const bullets = [];
  for (const category of dataParser.getCategories()) {
    if (!supportedModelGroups.has(category)) continue;
    for (const bullet of dataParser.getBulletsByCategory(category)) {
      bullets.push({ bullet, modelGroup: category });
    }
  }

  const forecasts = await asyncPool(bullets, params.planConcurrency, async ({ bullet, modelGroup }) => {
    const f = await forecastProvider(bullet, modelGroup);
    return f;
  });

  const plansRaw = await asyncPool(forecasts.filter(Boolean), params.planConcurrency, async (forecast) => {
    if (!forecast || !forecast.available) return null;
    return createBulletPlan({ rootDir, dataParser, forecast, params });
  });

  const plansBuilt = plansRaw.filter(Boolean);
  const okFlagPlans = plansBuilt.filter((p) => p && p.ok);
  const profitPlans = okFlagPlans.filter((p) => p.profitPerSlot > 0);

  // 当严格阈值导致空计划时，自动放宽置信度阈值，避免前端“无结果”。
  const requestedMinConfidence = Number.isFinite(params.minConfidence) ? params.minConfidence : 0;
  let appliedMinConfidence = requestedMinConfidence;
  let okPlans = profitPlans.filter((p) => Number(p.confidence) >= appliedMinConfidence);
  if (okPlans.length === 0 && profitPlans.length > 0) {
    const fallbackThresholds = [0.4, 0.3, 0.2, 0.1, 0];
    for (const t of fallbackThresholds) {
      if (t >= appliedMinConfidence) continue;
      const relaxed = profitPlans.filter((p) => Number(p.confidence) >= t);
      if (relaxed.length > 0) {
        appliedMinConfidence = t;
        okPlans = relaxed;
        break;
      }
    }
  }

  okPlans.sort((a, b) => b.score - a.score);
  const limitedRaw = params.maxBullets > 0 ? okPlans.slice(0, params.maxBullets) : okPlans;
  let minRaw = null;
  let maxRaw = null;
  for (const p of limitedRaw) {
    const v = Number(p && p.score);
    if (!Number.isFinite(v)) continue;
    if (minRaw == null || v < minRaw) minRaw = v;
    if (maxRaw == null || v > maxRaw) maxRaw = v;
  }
  const limited = limitedRaw.map((p) => {
    const raw = Number(p && p.score);
    const v = Number.isFinite(raw) ? raw : 0;
    if (minRaw == null || maxRaw == null) return { ...p, score: 0 };
    if (maxRaw === minRaw) return { ...p, score: 100 };
    const norm = ((v - minRaw) / (maxRaw - minRaw)) * 100;
    return { ...p, score: clamp(norm, 0, 100) };
  }).sort((a, b) => b.score - a.score);

  const allocated = allocateSlots(limited, params);
  const allocatedByBullet = new Map();
  for (const p of allocated.plans) allocatedByBullet.set(p.bullet, p);
  const mergedPlans = limited.map((p) => {
    const alloc = allocatedByBullet.get(p.bullet);
    if (alloc) return alloc;
    return {
      ...p,
      slots: 0,
      units: 0,
      positionWeight: 0,
      positionRatio: 0
    };
  });

  const availableForecasts = forecasts.filter((f) => f && f.available);
  let bestRaps = null;
  let bestConfidence = null;
  let bestProfitPerUnit = null;
  for (const p of okFlagPlans) {
    if (bestRaps == null || p.riskAdjustedProfitPerSlot > bestRaps) bestRaps = p.riskAdjustedProfitPerSlot;
    if (bestConfidence == null || p.confidence > bestConfidence) bestConfidence = p.confidence;
    if (bestProfitPerUnit == null || p.profitPerUnit > bestProfitPerUnit) bestProfitPerUnit = p.profitPerUnit;
  }

  const result = {
    cached: false,
    generatedAt: Date.now(),
    params,
    slotsTotal: allocated.slotsTotal,
    slotsUsed: allocated.slotsUsed,
    plans: mergedPlans,
    debug: {
      bulletsTotal: bullets.length,
      forecastsTotal: forecasts.length,
      forecastsAvailable: availableForecasts.length,
      plansBuilt: plansBuilt.length,
      plansOkFlag: okFlagPlans.length,
      plansProfitPositive: profitPlans.length,
      plansConfidenceOk: okPlans.length,
      minConfidenceRequested: requestedMinConfidence,
      minConfidenceApplied: appliedMinConfidence,
      bestRiskAdjustedProfitPerSlot: bestRaps == null ? null : bestRaps,
      bestConfidence: bestConfidence == null ? null : bestConfidence,
      bestProfitPerUnit: bestProfitPerUnit == null ? null : bestProfitPerUnit
    }
  };
  return result;
}

async function createTradePlanForBullet({ dataParser, forecast, paramsOverride = null }) {
  const params = buildPlanParams(paramsOverride);
  const rootDir = path.resolve(__dirname, '..');
  return createBulletPlan({ rootDir, dataParser, forecast, params });
}

module.exports = {
  getPlanParams,
  createTradePlanAll,
  createTradePlanForBullet
};
