const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

class DataParser {
  constructor() {
    this.bulletDataPath = process.env.BULLET_DATA_DIR
      ? path.resolve(process.env.BULLET_DATA_DIR)
      : path.resolve(__dirname, '..', 'Datasets', 'Datasets');

    this._indexCache = null;
    this._indexCacheDirMtimeMs = null;
  }

  _getCategoryFromBulletName(bulletName) {
    if (bulletName.startsWith('12 Gauge')) return '12 Gauge';
    if (bulletName.includes(' ')) {
      const parts = bulletName.split(' ').filter(Boolean);
      if (parts.length >= 2) {
        const suffix2 = new Set(['Magnum', 'ACP', 'AE', 'Gauge', 'Govt']);
        if (suffix2.has(parts[1])) return `${parts[0]} ${parts[1]}`;
      }
      return parts[0];
    }
    if (bulletName.endsWith('箭矢')) {
      return '箭矢';
    }
    return '其他';
  }

  _ensureIndex() {
    const stat = fs.statSync(this.bulletDataPath);
    const dirMtimeMs = stat.mtimeMs;
    if (this._indexCache && this._indexCacheDirMtimeMs === dirMtimeMs) {
      return this._indexCache;
    }

    const files = fs.readdirSync(this.bulletDataPath, { withFileTypes: true })
      .filter(d => d.isFile())
      .map(d => d.name)
      .filter(name => name.toLowerCase().endsWith('.csv'));

    const bullets = files
      .map(file => path.basename(file, '.csv'))
      .sort((a, b) => a.localeCompare(b, 'zh-Hans-CN'));

    const categoriesMap = new Map();
    for (const bulletName of bullets) {
      const category = this._getCategoryFromBulletName(bulletName);
      if (!categoriesMap.has(category)) categoriesMap.set(category, []);
      categoriesMap.get(category).push(bulletName);
    }

    for (const list of categoriesMap.values()) {
      list.sort((a, b) => a.localeCompare(b, 'zh-Hans-CN'));
    }

    const categories = Array.from(categoriesMap.keys()).sort((a, b) => a.localeCompare(b, 'zh-Hans-CN'));

    this._indexCache = { bullets, categories, categoriesMap };
    this._indexCacheDirMtimeMs = dirMtimeMs;
    return this._indexCache;
  }

  getCategories() {
    const { categories } = this._ensureIndex();
    return categories;
  }

  getBulletsByCategory(category) {
    const { categoriesMap } = this._ensureIndex();
    return categoriesMap.get(category) || [];
  }

  getBulletCategory(bulletName) {
    return this._getCategoryFromBulletName(bulletName);
  }

  async getBulletPriceSeries(bulletName) {
    const filePath = path.join(this.bulletDataPath, `${bulletName}.csv`);
    if (!fs.existsSync(filePath)) {
      const err = new Error(`未找到子弹数据文件: ${bulletName}.csv`);
      err.code = 'BULLET_NOT_FOUND';
      throw err;
    }

    return new Promise((resolve, reject) => {
      const points = [];
      fs.createReadStream(filePath)
        .pipe(csv({ mapHeaders: ({ header }) => header.trim().replace(/^\ufeff/, '') }))
        .on('data', (row) => {
          const tsRaw = row['时间戳'];
          const priceRaw = row['均价'];
          if (tsRaw == null || priceRaw == null) return;

          const ts = Number(tsRaw);
          const price = Number(priceRaw);
          if (!Number.isFinite(ts) || !Number.isFinite(price)) return;
          points.push({ ts, price });
        })
        .on('end', () => {
          points.sort((a, b) => a.ts - b.ts);
          resolve(points);
        })
        .on('error', reject);
    });
  }
}

module.exports = DataParser;
