<template>
  <div class="test-platform">
    <el-row :gutter="20">
      <el-col :span="10">
        <el-card header="算法测试平台" class="mb-4">
          <el-form :model="form" label-width="110px">
            <el-form-item label="模型来源">
              <el-radio-group v-model="form.model_source">
                <el-radio label="server">从后端列表选择</el-radio>
                <el-radio label="local">选择本地模型文件</el-radio>
              </el-radio-group>
            </el-form-item>

            <el-form-item v-if="form.model_source === 'server'" label="模型文件">
              <el-select v-model="form.model_file" placeholder="请选择模型" style="width: 100%">
                <el-option v-for="m in models" :key="m.file" :label="m.file" :value="m.file" />
              </el-select>
            </el-form-item>

            <el-form-item v-else label="本地模型">
              <el-upload
                :auto-upload="false"
                :show-file-list="false"
                accept=".pth"
                :disabled="isPredicting"
                @change="handleLocalModelChange"
              >
                <el-button :disabled="isPredicting">选择 .pth 文件</el-button>
              </el-upload>
              <div class="hint">{{ localModelHint }}</div>
            </el-form-item>

            <el-form-item label="电池组编号">
              <el-input v-model="form.cells" placeholder="例如：100 或 91-95,100" :disabled="isPredicting" />
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

            <el-form-item>
              <el-button type="primary" :loading="starting" :disabled="isPredicting" @click="handleStartPredict">开始预测</el-button>
              <el-button type="danger" :disabled="!isPredicting" @click="handleStopPredict">停止预测</el-button>
              <el-button :disabled="loadingModels || isPredicting" @click="refreshModels">刷新模型列表</el-button>
            </el-form-item>
          </el-form>

          <el-alert v-if="errorMsg" type="error" show-icon :closable="false" class="mt-3">
            <template #title>预测失败</template>
            <template #default>
              <pre class="error-pre">{{ errorMsg }}</pre>
            </template>
          </el-alert>
        </el-card>

        <el-card header="预测信息">
          <el-descriptions :column="1" border>
            <el-descriptions-item label="状态">{{ isPredicting ? '预测中' : '空闲' }}</el-descriptions-item>
            <el-descriptions-item label="模型">{{ displayModelFile || '-' }}</el-descriptions-item>
            <el-descriptions-item label="电池组">{{ displayCells || '-' }}</el-descriptions-item>
            <el-descriptions-item label="步长">{{ String(form.step || '-') }}</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>

      <el-col :span="14">
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

        <el-card v-if="exportInfo.url" header="导出结果">
          <el-space wrap>
            <el-link :href="exportInfo.url" target="_blank" type="primary">下载最近导出文件</el-link>
          </el-space>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { computed, nextTick, onUnmounted, reactive, ref, watch } from 'vue';
import { ElMessage } from 'element-plus';
import * as echarts from 'echarts';
import { listPredictModels, runPredict } from '../api/battery';

const form = reactive({
  model_source: 'server',
  model_file: '',
  local_model_name: '',
  cells: '100',
  step: 1,
});

const starting = ref(false);
const loadingModels = ref(false);
const isPredicting = ref(false);
const errorMsg = ref('');

const models = ref([]);
const cellsData = ref([]);
const activeCellTab = ref('');

const exportInfo = reactive({
  url: '',
});

let abortController = null;
let animTimer = null;
let animIndex = 0;
let animMax = 0;

const chartRefs = ref({});
const chartInstances = new Map();

const localModelHint = computed(() => {
  const name = String(form.local_model_name || '').trim();
  return name ? `已选择：${name}` : '仅使用文件名触发预测（后端需能找到同名 .pth）';
});

const displayModelFile = computed(() => {
  if (form.model_source === 'server') return String(form.model_file || '').trim();
  return String(form.local_model_name || '').trim();
});

const displayCells = computed(() => String(form.cells || '').trim());

const resolveModelFile = () => {
  if (form.model_source === 'server') return String(form.model_file || '').trim();
  return String(form.local_model_name || '').trim();
};

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
    if (!form.model_file && models.value.length > 0) {
      form.model_file = models.value[0].file;
    }
  } catch (e) {
    const msg = e?.response?.data?.detail || e?.message || String(e);
    ElMessage.error(`获取模型列表失败：${msg}`);
  } finally {
    loadingModels.value = false;
  }
};

const handleLocalModelChange = (file) => {
  const name = String(file?.name || '').trim();
  form.local_model_name = name;
};

const validatePayload = () => {
  const modelFile = resolveModelFile();
  if (!modelFile) throw new Error('请选择模型文件');
  const cells = String(form.cells || '').trim();
  if (!cells) throw new Error('请输入电池组编号');
  const step = Number(form.step);
  if (!Number.isInteger(step) || step <= 0) throw new Error('预测步长必须为正整数');
  return { modelFile, cells, step };
};

const handleStartPredict = async () => {
  starting.value = true;
  errorMsg.value = '';
  exportInfo.url = '';
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
      model_file: v.modelFile,
      cells: v.cells,
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
.mb-4 {
  margin-bottom: 16px;
}

.mt-3 {
  margin-top: 12px;
}

.error-pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
}

.hint {
  color: #909399;
  font-size: 12px;
  margin-left: 10px;
  line-height: 32px;
}

.chart-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
}

.chart-item {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 12px;
  background: #fff;
}

.chart-title {
  font-size: 14px;
  color: #303133;
  margin-bottom: 8px;
}

.chart-canvas {
  width: 100%;
  height: 320px;
}
</style>
