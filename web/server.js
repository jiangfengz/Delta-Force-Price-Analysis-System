const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const DataParser = require('./dataParser');
const { createTradePlanAll, createTradePlanForBullet } = require('./tradePlan');

const app = express();
const PORT = process.env.PORT || 3000;
const dataParser = new DataParser();

// 中间件
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));
const thesisDocsCandidates = ['毕设文档', '毕设文档集'];
for (const name of thesisDocsCandidates) {
  const dir = path.resolve(__dirname, '..', name);
  if (fs.existsSync(dir)) {
    app.use('/thesis', express.static(dir));
    break;
  }
}

app.get('/vendor/chart.js', (req, res) => {
  res.sendFile(path.join(__dirname, 'node_modules', 'chart.js', 'dist', 'chart.umd.min.js'));
});

const originalListen = app.listen.bind(app);
app.listen = (...args) => {
  const server = originalListen(...args);
  server.on('close', () => {
    if (typeof app.shutdown === 'function') {
      void app.shutdown();
    }
  });
  return server;
};

let timexerWorker = null;
let timexerWorkerReady = false;
const timexerPending = new Map();
let timexerStdoutBuffer = '';

function rejectAllTimexerPending(error) {
  for (const [, pending] of timexerPending.entries()) {
    pending.reject(error);
  }
  timexerPending.clear();
}

const SUPPORTED_MODEL_GROUPS = new Set([
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

function getModelGroupForBullet(bulletName) {
  const category = dataParser.getBulletCategory(bulletName);
  return SUPPORTED_MODEL_GROUPS.has(category) ? category : null;
}

function ensureTimeXerWorker() {
  if (timexerWorker && !timexerWorker.killed) return;

  timexerWorkerReady = false;
  const pythonCmd = process.env.PYTHON || 'python';
  const workerPath = path.join(__dirname, 'timexer_worker.py');
  timexerWorker = spawn(pythonCmd, [workerPath], {
    cwd: path.resolve(__dirname, '..'),
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      ...process.env,
      PYTHONIOENCODING: 'utf-8',
      PYTHONUTF8: '1'
    }
  });

  timexerWorker.on('error', (error) => {
    timexerWorkerReady = false;
    rejectAllTimexerPending(new Error(`TimeXer 推理进程启动失败：${error.message || String(error)}`));
  });

  timexerWorker.stdin.on('error', (error) => {
    timexerWorkerReady = false;
    rejectAllTimexerPending(new Error(`TimeXer 通信失败：${error.message || String(error)}`));
  });

  timexerWorker.stdout.setEncoding('utf8');
  timexerWorker.stdout.on('data', (chunk) => {
    timexerStdoutBuffer += chunk;
    const parts = timexerStdoutBuffer.split(/\r?\n/);
    timexerStdoutBuffer = parts.pop() || '';
    const lines = parts.filter(Boolean);
    for (const line of lines) {
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        continue;
      }
      if (msg && msg.type === 'ready') {
        timexerWorkerReady = true;
        continue;
      }
      const requestId = msg && msg.requestId;
      if (!requestId) continue;
      const pending = timexerPending.get(requestId);
      if (!pending) continue;
      timexerPending.delete(requestId);
      if (msg.ok) pending.resolve(msg);
      else pending.reject(new Error(msg.error || 'TimeXer 推理失败'));
    }
  });

  timexerWorker.stderr.setEncoding('utf8');
  timexerWorker.stderr.on('data', (chunk) => {
    const lines = String(chunk).trim().split(/\r?\n/).filter(Boolean);
    for (const line of lines) console.error(`[TimeXer Worker] ${line}`);
  });

  timexerWorker.on('exit', () => {
    timexerWorkerReady = false;
    rejectAllTimexerPending(new Error('TimeXer 推理进程已退出'));
  });
}

function timexerRequest(payload, timeoutMs = 60_000) {
  ensureTimeXerWorker();
  if (!timexerWorker || timexerWorker.killed) {
    return Promise.reject(new Error('TimeXer 推理进程不可用'));
  }
  const requestId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const message = JSON.stringify({ requestId, ...payload });
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      timexerPending.delete(requestId);
      reject(new Error('TimeXer 推理超时'));
    }, timeoutMs);
    if (typeof timer.unref === 'function') timer.unref();
    timexerPending.set(requestId, {
      resolve: (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      reject: (e) => {
        clearTimeout(timer);
        reject(e);
      }
    });
    try {
      timexerWorker.stdin.write(message + '\n');
    } catch (error) {
      timexerPending.delete(requestId);
      clearTimeout(timer);
      reject(new Error(`TimeXer 通信失败：${error.message || String(error)}`));
    }
  });
}

function shutdownTimeXerWorker() {
  if (!timexerWorker || timexerWorker.killed) return Promise.resolve();

  const child = timexerWorker;
  timexerWorker = null;
  timexerWorkerReady = false;
  timexerStdoutBuffer = '';
  rejectAllTimexerPending(new Error('TimeXer 推理进程已关闭'));

  return new Promise((resolve) => {
    const timer = setTimeout(resolve, 1500);
    if (typeof timer.unref === 'function') timer.unref();
    child.once('exit', () => {
      clearTimeout(timer);
      resolve();
    });
    try {
      child.kill();
    } catch {
      resolve();
    }
  });
}

app.get('/api/categories', (req, res) => {
  try {
    res.json({ categories: dataParser.getCategories() });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/bullets', (req, res) => {
  try {
    const { category } = req.query;
    if (!category) return res.status(400).json({ error: '缺少 category' });
    res.json({ category, bullets: dataParser.getBulletsByCategory(category) });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/series', async (req, res) => {
  try {
    const bullet = req.query.bullet;
    const range = req.query.range || 'all';
    if (!bullet) return res.status(400).json({ error: '缺少 bullet' });

    const points = await dataParser.getBulletPriceSeries(bullet);
    let filtered = points;
    if (range !== 'all' && points.length > 0) {
      const days = range === '3d' ? 3 : range === '7d' ? 7 : range === '30d' ? 30 : null;
      if (days != null) {
        // 历史数据非实时，可能落后于当前时钟；以最新数据点为基准向前取 N 天，
        // 避免「近 N 天」相对 Date.now() 落空、被迫回退到全部区间（导致历史范围看似无法切换）。
        const latestTs = points[points.length - 1].ts;
        const minTs = latestTs - days * 24 * 60 * 60 * 1000;
        filtered = points.filter(p => p.ts >= minTs);
      }
    }

    res.json({
      bullet,
      category: dataParser.getBulletCategory(bullet),
      points: filtered
    });
  } catch (error) {
    if (error.code === 'BULLET_NOT_FOUND') return res.status(404).json({ error: error.message });
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/forecast', async (req, res) => {
  const bullet = req.query.bullet;
  try {
    if (!bullet) return res.status(400).json({ error: '缺少 bullet' });

    const modelGroup = getModelGroupForBullet(bullet);
    if (!modelGroup) {
      return res.json({ available: false, bullet, reason: '该子弹暂无已训练模型' });
    }

    if (!timexerWorkerReady) {
      const pingOk = await timexerRequest({ type: 'ping' }, 10_000)
        .then(() => true)
        .catch(() => false);
      if (!pingOk) {
        return res.json({ available: false, bullet, modelGroup, reason: 'TimeXer 推理服务不可用' });
      }
    }

    const result = await timexerRequest({ type: 'forecast', modelGroup, bullet }, 120_000);
    res.json({
      available: true,
      bullet,
      modelGroup,
      modelId: result.modelId,
      predLen: result.predLen,
      points: result.points
    });
  } catch (error) {
    res.json({ available: false, bullet: bullet || null, reason: error.message || '预测失败' });
  }
});

app.get('/api/docs/trade-plan-algo', async (req, res) => {
  try {
    const rootDir = path.resolve(__dirname, '..');
    const docCandidates = [
      path.resolve(__dirname, 'docs', 'trade-plan-algo.md'),
      path.resolve(rootDir, 'profits', '倒卖子弹买卖计划算法.md')
    ];
    const filePath = docCandidates.find((p) => fs.existsSync(p));
    if (!filePath) return res.status(404).json({ error: '算法文档不存在' });
    const text = await fs.promises.readFile(filePath, 'utf8');
    res.set('Content-Type', 'text/markdown; charset=utf-8');
    res.send(text.replace(/^\ufeff/, ''));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/trade-plan', async (req, res) => {
  try {
    const refresh = req.query.refresh === '1' || req.query.refresh === 'true';
    const paramsOverride = {};
    const numberKeys = ['sellFeeRate', 'minRoi', 'minConfidence', 'minDeltaAbs', 'triggerTheta'];
    const intKeys = ['maxBullets', 'maxSlotsPerBullet', 'slotsTotal'];
    const strategyRaw = req.query.strategy == null ? '' : String(req.query.strategy);
    const strategy = strategyRaw.trim().toLowerCase();
    if (strategy === 'aggressive' || strategy === 'balanced' || strategy === 'conservative') {
      paramsOverride.strategy = strategy;
    }
    for (const k of numberKeys) {
      if (req.query[k] == null) continue;
      const v = Number(req.query[k]);
      if (Number.isFinite(v)) paramsOverride[k] = v;
    }
    for (const k of intKeys) {
      if (req.query[k] == null) continue;
      const v = Number.parseInt(String(req.query[k]), 10);
      if (Number.isFinite(v)) paramsOverride[k] = v;
    }
    const result = await createTradePlanAll({
      dataParser,
      forecastProvider: async (bullet, modelGroupHint) => {
        const modelGroup = modelGroupHint || getModelGroupForBullet(bullet);
        try {
          if (!modelGroup) {
            return { available: false, bullet, reason: '该子弹暂无已训练模型' };
          }
          if (!timexerWorkerReady) {
            const pingOk = await timexerRequest({ type: 'ping' }, 10_000)
              .then(() => true)
              .catch(() => false);
            if (!pingOk) {
              return { available: false, bullet, modelGroup, reason: 'TimeXer 推理服务不可用' };
            }
          }
          const r = await timexerRequest({ type: 'forecast', modelGroup, bullet }, 120_000);
          return {
            available: true,
            bullet,
            modelGroup,
            modelId: r.modelId,
            predLen: r.predLen,
            points: r.points
          };
        } catch (error) {
          return { available: false, bullet, modelGroup, reason: error.message || '预测失败' };
        }
      },
      forceRefresh: refresh,
      paramsOverride: Object.keys(paramsOverride).length ? paramsOverride : null
    });
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/trade-plan/bullet', async (req, res) => {
  try {
    const bullet = req.query.bullet;
    if (!bullet) return res.status(400).json({ error: '缺少 bullet' });
    const paramsOverride = {};
    const numberKeys = ['sellFeeRate', 'minRoi', 'minConfidence', 'minDeltaAbs', 'triggerTheta'];
    const strategyRaw = req.query.strategy == null ? '' : String(req.query.strategy);
    const strategy = strategyRaw.trim().toLowerCase();
    if (strategy === 'aggressive' || strategy === 'balanced' || strategy === 'conservative') {
      paramsOverride.strategy = strategy;
    }
    for (const k of numberKeys) {
      if (req.query[k] == null) continue;
      const v = Number(req.query[k]);
      if (Number.isFinite(v)) paramsOverride[k] = v;
    }

    const modelGroup = getModelGroupForBullet(bullet);
    if (!modelGroup) {
      return res.json({ ok: false, available: false, bullet, reason: '该子弹暂无已训练模型' });
    }

    if (!timexerWorkerReady) {
      const pingOk = await timexerRequest({ type: 'ping' }, 10_000)
        .then(() => true)
        .catch(() => false);
      if (!pingOk) {
        return res.json({ ok: false, available: false, bullet, modelGroup, reason: 'TimeXer 推理服务不可用' });
      }
    }
    const r = await timexerRequest({ type: 'forecast', modelGroup, bullet }, 120_000);
    const plan = await createTradePlanForBullet({
      dataParser,
      forecast: {
        available: true,
        bullet,
        modelGroup,
        modelId: r.modelId,
        predLen: r.predLen,
        points: r.points
      },
      paramsOverride: Object.keys(paramsOverride).length ? paramsOverride : null
    });
    res.json(plan);
  } catch (error) {
    res.json({ ok: false, available: false, error: error.message || '生成倒卖计划失败' });
  }
});

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.shutdown = async () => {
  await shutdownTimeXerWorker();
};

// 启动服务器（端口被占用时自动顺延，避免 npm start 直接失败）
if (require.main === module) {
  const maxPortRetry = 20;
  let shuttingDown = false;
  let activeServer = null;

  const startWithPortRetry = (basePort) => {
    return new Promise((resolve, reject) => {
      const tryListen = (port, attempt) => {
        const server = app.listen(port, () => {
          resolve({ server, port });
        });
        server.once('error', (error) => {
          if (error && error.code === 'EADDRINUSE' && attempt < maxPortRetry) {
            tryListen(port + 1, attempt + 1);
            return;
          }
          reject(error);
        });
      };
      tryListen(basePort, 0);
    });
  };

  startWithPortRetry(PORT)
    .then(({ server, port }) => {
      activeServer = server;
      if (port !== PORT) {
        console.warn(`端口 ${PORT} 被占用，已切换到 http://localhost:${port}`);
      } else {
        console.log(`服务器运行在 http://localhost:${port}`);
      }
    })
    .catch((error) => {
      console.error(`服务器启动失败：${error.message || String(error)}`);
      process.exit(1);
    });

  const gracefulExit = async () => {
    if (shuttingDown) return;
    shuttingDown = true;
    if (activeServer) {
      await new Promise((resolve) => activeServer.close(resolve));
    }
    await app.shutdown();
    process.exit(0);
  };

  process.on('SIGINT', () => {
    void gracefulExit();
  });
  process.on('SIGTERM', () => {
    void gracefulExit();
  });
}

module.exports = app;
