<template>
  <div class="test-platform">
    <div class="sidebar-container">
      <div class="sidebar-header">
        <span>算法测试控制台</span>
      </div>
      <el-card class="sidebar-card" :body-style="{ padding: '15px', maxHeight: 'calc(100vh - 220px)', overflowY: 'auto' }">
        <el-form :model="form" label-width="100px" size="default">
          <div class="section-title">参数配置</div>
          <el-form-item label="模型文件">
            <el-select v-model="form.model_name" placeholder="请选择模型" style="width: 100%">
              <el-option v-for="m in models" :key="m" :label="m" :value="m" />
            </el-select>
          </el-form-item>

          <el-form-item label="电池组编号">
            <el-input-number
              v-model="form.cell_id"
              :min="0"
              :max="123"
              :precision="0"
              :step="1"
              :step-strictly="true"
              style="width: 100%"
              :disabled="isPredicting"
              placeholder="0–123"
            />
          </el-form-item>

          <el-form-item label="预测步长">
            <el-input-number
              v-model="form.step"
              :min="1"
              :max="100000"
              :precision="0"
              :step="1"
              :step-strictly="true"
              style="width: 100%"
              :disabled="isPredicting"
            />
          </el-form-item>

          <div class="action-buttons">
            <el-button type="primary" :loading="starting" :disabled="isPredicting" @click="handleStartPredict" icon="VideoPlay" class="w-full">开始预测</el-button>
            <el-button type="danger" :disabled="!isPredicting" @click="handleStopPredict" icon="VideoPause" class="w-full">停止预测</el-button>
            <el-button :disabled="loadingModels || isPredicting" @click="refreshModels" icon="Refresh" class="w-full">刷新模型列表</el-button>
          </div>
        </el-form>

        <el-alert v-if="errorMsg" type="error" show-icon :closable="false" class="mt-3">
          <template #title>预测失败</template>
          <template #default>
            <pre class="error-pre">{{ errorMsg }}</pre>
          </template>
        </el-alert>

        <el-divider content-position="left">预测状态</el-divider>

        <div class="info-box">
          <div class="info-row">
            <span class="label">当前状态:</span>
            <el-tag :type="isPredicting ? 'success' : 'info'" size="small">{{ isPredicting ? '预测中' : '空闲' }}</el-tag>
          </div>
          <div class="info-row">
            <span class="label">模型:</span>
            <span class="value" :title="displayModelFile">{{ displayModelFile || '-' }}</span>
          </div>
          <div class="info-row">
            <span class="label">电池组:</span>
            <span class="value">{{ String(form.cell_id ?? '-') }}</span>
          </div>
          <div class="info-row">
            <span class="label">步长:</span>
            <span class="value">{{ String(form.step || '-') }}</span>
          </div>
        </div>

        <el-divider content-position="left">结果导出</el-divider>

        <div class="export-section">
          <el-radio-group v-model="exportFormat" size="small" class="w-full flex mb-2">
            <el-radio-button label="csv" class="flex-1">CSV</el-radio-button>
            <el-radio-button label="excel" class="flex-1">Excel</el-radio-button>
          </el-radio-group>
          <el-button type="success" :disabled="!canExport" @click="handleExport" icon="Download" class="w-full">导出结果</el-button>
          <div v-if="!canExport" class="hint text-center mt-2">预测完成后可导出数据</div>
        </div>
      </el-card>
    </div>

    <div class="main-content">
      <el-card header="曲线对比（预测 vs 实际）" class="mb-4">
        <el-empty v-if="cellsData.length === 0" description="开始预测后展示 PCL/RUL 曲线对比" />

        <el-tabs v-else v-model="activeCellTab" class="cell-tabs">
          <el-tab-pane v-for="c in cellsData" :key="String(c.cell_id)" :label="`电池组 ${c.cell_id}`" :name="String(c.cell_id)">
            <div class="chart-grid">
              <div class="chart-item">
                <div class="chart-title">PCL（容量衰减）</div>
                <div :ref="(el) => setChartRef(c.cell_id, 'pcl', el)" class="chart-canvas"></div>
              </div>
              <div class="chart-item">
                <div class="chart-title">RUL（剩余寿命）</div>
                <div :ref="(el) => setChartRef(c.cell_id, 'rul', el)" class="chart-canvas"></div>
              </div>
            </div>
          </el-tab-pane>
        </el-tabs>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onUnmounted, reactive, ref, watch } from 'vue';
import { ElMessage } from 'element-plus';
import * as echarts from 'echarts';
import { listPredictModels, runPredict } from '../api/battery';

const form = reactive({
  model_name: '',
  cell_id: 100,
  step: 1,
});

const starting = ref(false);
const loadingModels = ref(false);
const isPredicting = ref(false);
const errorMsg = ref('');

const models = ref([]);
const cellsData = ref([]);
const activeCellTab = ref('');

const exportFormat = ref('csv');
const canExport = computed(() => !isPredicting.value && Array.isArray(cellsData.value) && cellsData.value.length > 0);

let abortController = null;
let animTimer = null;
let animIndex = 0;
let animMax = 0;

const chartRefs = ref({});
const chartInstances = new Map();

const displayModelFile = computed(() => {
  return String(form.model_name || '').trim();
});

const resolveModelFile = () => String(form.model_name || '').trim();

const setChartRef = (cellId, type, el) => {
  if (!el) return;
  const key = `${cellId}_${type}`;
  chartRefs.value[key] = el;
};

const disposeCharts = () => {
  chartInstances.forEach((c) => c.dispose());
  chartInstances.clear();
  chartRefs.value = {};
};

const ensureChart = (cellId, type) => {
  const key = `${cellId}_${type}`;
  const el = chartRefs.value[key];
  if (!el) return null;
  let chart = chartInstances.get(key);
  if (chart && typeof chart.isDisposed === 'function' && chart.isDisposed()) {
    chartInstances.delete(key);
    chart = null;
  }
  if (!chart) {
    chart = echarts.init(el);
    chartInstances.set(key, chart);
  }
  return chart;
};

const buildPclOption = (cell, endIdx) => {
  const cycles = (cell.cycles || []).slice(0, endIdx);
  const pclTrue = (cell.pcl_true || []).slice(0, endIdx);
  const pclPred = (cell.pcl_pred || []).slice(0, endIdx);
  return {
    tooltip: { trigger: 'axis' },
    legend: { top: 8, left: 'center' },
    grid: { left: '10%', right: '6%', top: 40, bottom: '10%', containLabel: true },
    xAxis: { type: 'category', data: cycles, name: 'cycle' },
    yAxis: { type: 'value', scale: true },
    series: [
      { name: '实际', type: 'line', data: pclTrue, showSymbol: false, smooth: true, lineStyle: { width: 2 } },
      { name: '预测', type: 'line', data: pclPred, showSymbol: false, smooth: true, lineStyle: { width: 2 } },
    ],
  };
};

const buildRulOption = (cell, endIdx) => {
  const cycles = (cell.cycles || []).slice(0, endIdx);
  const rulPred = (cell.rul_pred || []).slice(0, endIdx);
  const hasTrue = Array.isArray(cell.rul_true) && cell.rul_true.length > 0;
  const series = [];
  if (hasTrue) {
    series.push({ name: '实际', type: 'line', data: (cell.rul_true || []).slice(0, endIdx), showSymbol: false, smooth: true, lineStyle: { width: 2 } });
  }
  series.push({ name: '预测', type: 'line', data: rulPred, showSymbol: false, smooth: true, lineStyle: { width: 2 } });
  return {
    tooltip: { trigger: 'axis' },
    legend: { top: 8, left: 'center' },
    grid: { left: '10%', right: '6%', top: 40, bottom: '10%', containLabel: true },
    xAxis: { type: 'category', data: cycles, name: 'cycle' },
    yAxis: { type: 'value', scale: true },
    series,
  };
};

const renderCharts = async (endIdx) => {
  await nextTick();
  cellsData.value.forEach((cell) => {
    const pclChart = ensureChart(cell.cell_id, 'pcl');
    if (pclChart) {
      pclChart.setOption(buildPclOption(cell, endIdx), true);
      pclChart.resize();
    }
    const rulChart = ensureChart(cell.cell_id, 'rul');
    if (rulChart) {
      rulChart.setOption(buildRulOption(cell, endIdx), true);
      rulChart.resize();
    }
  });
};

const stopAnimation = () => {
  if (animTimer) {
    clearInterval(animTimer);
    animTimer = null;
  }
};

const startAnimation = () => {
  stopAnimation();
  animIndex = 0;
  const lengths = cellsData.value.map((c) => (Array.isArray(c.cycles) ? c.cycles.length : 0));
  animMax = Math.max(0, ...lengths);
  if (animMax <= 0) {
    isPredicting.value = false;
    return;
  }
  const step = Math.max(1, Math.floor(animMax / 200));
  animTimer = setInterval(async () => {
    animIndex = Math.min(animMax, animIndex + step);
    await renderCharts(animIndex);
    if (animIndex >= animMax) {
      stopAnimation();
      isPredicting.value = false;
    }
  }, 80);
};

const refreshModels = async () => {
  loadingModels.value = true;
  try {
    const data = await listPredictModels();
    models.value = Array.isArray(data?.models) ? data.models : [];
    if (!form.model_name && models.value.length > 0) {
      form.model_name = models.value[0];
    }
  } catch (e) {
    const msg = e?.response?.data?.detail || e?.message || String(e);
    ElMessage.error(`获取模型列表失败：${msg}`);
  } finally {
    loadingModels.value = false;
  }
};

const validatePayload = () => {
  const modelFile = resolveModelFile();
  if (!modelFile) throw new Error('请选择模型文件');
  const cellId = Number(form.cell_id);
  if (!Number.isInteger(cellId) || cellId < 0 || cellId > 123) throw new Error('电池组编号必须为 0–123 的整数');
  const step = Number(form.step);
  if (!Number.isInteger(step) || step <= 0) throw new Error('预测步长必须为正整数');
  return { modelFile, cellId, step };
};

const handleStartPredict = async () => {
  starting.value = true;
  errorMsg.value = '';
  stopAnimation();
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
  try {
    const v = validatePayload();
    isPredicting.value = true;
    cellsData.value = [];
    activeCellTab.value = '';
    disposeCharts();

    abortController = new AbortController();
    const payload = {
      model_name: v.modelFile,
      cell_ids: [v.cellId],
      step: v.step,
    };
    const data = await runPredict(payload, { signal: abortController.signal });
    const arr = Array.isArray(data?.cells) ? data.cells : [];
    if (arr.length === 0) {
      throw new Error('未返回可用的预测结果');
    }
    cellsData.value = arr;
    activeCellTab.value = String(arr[0].cell_id);
    await renderCharts(1);
    startAnimation();
    ElMessage.success('已开始在线生成曲线');
  } catch (e) {
    const msg = e?.name === 'CanceledError' ? '请求已取消' : (e?.response?.data?.detail || e?.message || String(e));
    errorMsg.value = msg;
    isPredicting.value = false;
  } finally {
    starting.value = false;
  }
};

const handleStopPredict = () => {
  stopAnimation();
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
  isPredicting.value = false;
  ElMessage.info('已停止预测');
};

const sanitizeName = (s) => String(s || '').replace(/[\\/:*?"<>|]/g, '_').trim();
const buildFilename = () => {
  const cell = Number(form.cell_id);
  const model = sanitizeName(String(form.model_name || 'model'));
  return `prediction_battery${cell}_${model}`;
};

const buildCsvContent = (cell) => {
  const cycles = Array.isArray(cell.cycles) ? cell.cycles : [];
  const pclTrue = Array.isArray(cell.pcl_true) ? cell.pcl_true : [];
  const pclPred = Array.isArray(cell.pcl_pred) ? cell.pcl_pred : [];
  const rulTrue = Array.isArray(cell.rul_true) ? cell.rul_true : [];
  const rulPred = Array.isArray(cell.rul_pred) ? cell.rul_pred : [];
  const n = Math.max(cycles.length, pclTrue.length, pclPred.length, rulTrue.length, rulPred.length);
  const lines = ['cycle,pcl_true,pcl_pred,rul_true,rul_pred'];
  for (let i = 0; i < n; i++) {
    const row = [
      cycles[i] ?? '',
      pclTrue[i] ?? '',
      pclPred[i] ?? '',
      rulTrue[i] ?? '',
      rulPred[i] ?? '',
    ];
    lines.push(row.join(','));
  }
  return lines.join('\n');
};

const buildExcelHtml = (cell) => {
  const cycles = Array.isArray(cell.cycles) ? cell.cycles : [];
  const pclTrue = Array.isArray(cell.pcl_true) ? cell.pcl_true : [];
  const pclPred = Array.isArray(cell.pcl_pred) ? cell.pcl_pred : [];
  const rulTrue = Array.isArray(cell.rul_true) ? cell.rul_true : [];
  const rulPred = Array.isArray(cell.rul_pred) ? cell.rul_pred : [];
  const n = Math.max(cycles.length, pclTrue.length, pclPred.length, rulTrue.length, rulPred.length);
  const header = '<tr><th>cycle</th><th>pcl_true</th><th>pcl_pred</th><th>rul_true</th><th>rul_pred</th></tr>';
  let rows = '';
  for (let i = 0; i < n; i++) {
    rows += `<tr><td>${cycles[i] ?? ''}</td><td>${pclTrue[i] ?? ''}</td><td>${pclPred[i] ?? ''}</td><td>${rulTrue[i] ?? ''}</td><td>${rulPred[i] ?? ''}</td></tr>`;
  }
  const table = `<table>${header}${rows}</table>`;
  const html = `
    <html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel" xmlns="http://www.w3.org/TR/REC-html40">
    <head><meta charset="UTF-8"></head><body>${table}</body></html>`;
  return html;
};

const triggerDownload = (blob, filename) => {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

const handleExport = () => {
  if (!canExport.value) {
    ElMessage.info('预测尚未完成，无法导出');
    return;
  }
  const cell = cellsData.value[0];
  const base = buildFilename();
  if (exportFormat.value === 'csv') {
    const content = buildCsvContent(cell);
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8' });
    triggerDownload(blob, `${base}.csv`);
    ElMessage.success('已导出 CSV');
    return;
  }
  // excel
  const html = buildExcelHtml(cell);
  const blob = new Blob([html], { type: 'application/vnd.ms-excel;charset=utf-8' });
  triggerDownload(blob, `${base}.xls`);
  ElMessage.success('已导出 Excel');
};

onUnmounted(() => {
  handleStopPredict();
  disposeCharts();
});

watch(activeCellTab, async () => {
  await nextTick();
  chartInstances.forEach((c) => c.resize());
});

refreshModels();
</script>

<style scoped>
.test-platform {
  position: relative;
  min-height: 100vh;
}

.sidebar-container {
  position: fixed;
  top: 130px;
  left: 0;
  bottom: 0;
  width: 320px;
  background-color: #fff;
  border-right: 1px solid #e4e7ed;
  z-index: 900;
  padding: 20px;
  box-shadow: 2px 0 12px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  font-size: 18px;
  font-weight: 600;
  color: #303133;
  padding: 0 10px 20px;
  border-bottom: 1px solid #ebeef5;
  margin-bottom: 15px;
  display: flex;
  align-items: center;
}

.sidebar-card {
  border: none;
  box-shadow: none !important;
  flex: 1;
}

.main-content {
  margin-left: 340px;
  padding: 20px;
  min-height: 100vh;
  background-color: #f5f7fa;
}

.section-title {
  font-size: 14px;
  font-weight: 500;
  color: #909399;
  margin: 10px 0 15px;
  padding-left: 5px;
  border-left: 3px solid #409eff;
  line-height: 1;
}

.action-buttons {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 10px;
}
.action-buttons .el-button {
  margin-left: 0 !important;
  margin-bottom: 0 !important;
}

.w-full {
  width: 100%;
}
.flex {
  display: flex;
}
.flex-1 {
  flex: 1;
}
.mb-2 {
  margin-bottom: 8px;
}
.mb-4 {
  margin-bottom: 16px;
}
.mt-2 {
  margin-top: 8px;
}
.mt-3 {
  margin-top: 12px;
}
.text-center {
  text-align: center;
}

.error-pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: monospace;
  font-size: 12px;
  line-height: 1.4;
}

.hint {
  color: #909399;
  font-size: 12px;
  line-height: 1.5;
}

.info-box {
  background: #f8f9fa;
  border-radius: 4px;
  padding: 12px;
  margin-bottom: 10px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 13px;
}
.info-row:last-child {
  margin-bottom: 0;
}
.info-row .label {
  color: #606266;
}
.info-row .value {
  color: #303133;
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 150px;
}

.chart-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.chart-item {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 16px;
  background: #fff;
  transition: all 0.3s ease;
}
.chart-item:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

.chart-title {
  font-size: 15px;
  font-weight: 500;
  color: #303133;
  margin-bottom: 12px;
  border-left: 4px solid #409eff;
  padding-left: 8px;
}

.chart-canvas {
  width: 100%;
  height: 320px;
}

.export-section {
  padding: 0 5px;
}
</style>
