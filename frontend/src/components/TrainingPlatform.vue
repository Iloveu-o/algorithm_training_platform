<template>
  <div class="training-platform">
    <!-- 侧边导航栏：配置与控制 -->
    <div class="sidebar-container">
      <el-card class="sidebar-card" :body-style="{ padding: '15px', maxHeight: 'calc(100vh - 180px)', overflowY: 'auto' }">
        <el-form :model="form" label-width="100px" size="small">
          <el-form-item label="模型类型">
            <el-select v-model="form.model_type" placeholder="请选择" style="width: 100%">
              <el-option label="Baseline" value="Baseline" />
              <el-option label="BiLSTM" value="BiLSTM" />
              <el-option label="DeepHPM" value="DeepHPM" />
            </el-select>
          </el-form-item>

          <el-form-item label="Epochs">
            <el-input-number
              v-model="form.epochs"
              :min="1"
              :max="100000"
              :step="1"
              :precision="0"
              :step-strictly="true"
              style="width: 100%"
            />
          </el-form-item>

          <el-form-item label="Batch Size">
            <el-input-number
              v-model="form.batch_size"
              :min="1"
              :max="100000"
              :step="1"
              :precision="0"
              :step-strictly="true"
              style="width: 100%"
            />
          </el-form-item>

          <el-form-item label="学习率">
            <el-input-number v-model="form.lr" :min="0.00000001" :step="0.0001" :precision="8" style="width: 100%" />
          </el-form-item>

          <el-form-item label="优化器">
            <el-select v-model="form.optimizer" placeholder="请选择" style="width: 100%">
              <el-option label="Adam" value="Adam" />
              <el-option label="SGD" value="SGD" />
            </el-select>
          </el-form-item>

          <el-form-item v-if="showStructure" label="Layers">
            <el-input v-model="form.layers" placeholder="例如：32,32" />
          </el-form-item>

          <el-form-item v-if="showStructure" label="Activation">
            <el-select v-model="form.activation" placeholder="请选择" style="width: 100%">
              <el-option label="Tanh" value="Tanh" />
              <el-option label="ReLU" value="ReLU" />
              <el-option label="Sigmoid" value="Sigmoid" />
              <el-option label="LeakyReLU" value="LeakyReLU" />
              <el-option label="Sin" value="Sin" />
            </el-select>
          </el-form-item>

          <el-alert
            v-if="form.model_type === 'BiLSTM'"
            type="info"
            show-icon
            :closable="false"
            title="BiLSTM 结构参数由后端默认配置"
            class="mb-3"
          />

          <el-form-item>
            <el-button type="primary" :loading="submitting" :disabled="isTraining" @click="handleStartTraining" style="width: 100%; margin-bottom: 10px;">开始训练</el-button>
            <el-button type="danger" :disabled="!isTraining" :loading="canceling" @click="handleCancelTraining" style="width: 100%; margin-left: 0;">停止训练</el-button>
          </el-form-item>
        </el-form>
      </el-card>
    </div>

    <!-- 右侧主内容：状态与图表 -->
    <div class="main-content">
      <!-- 简化后的训练任务状态栏 -->
      <el-card shadow="hover" class="mb-4 task-status-card" v-if="job.job_id">
        <div class="status-header">
          <span class="font-bold">任务 ID: {{ job.job_id }}</span>
          <el-tag :type="statusTagType" effect="dark">{{ job.status }}</el-tag>
        </div>
        
        <div class="mt-3 progress-section">
          <div class="progress-info">
            <span>进度: {{ progressText }}</span>
            <span v-if="job.finished_at">结束于: {{ job.finished_at }}</span>
          </div>
          <el-progress 
            :percentage="progressPercentage" 
            :status="progressStatus"
            :stroke-width="18"
            text-inside 
          />
        </div>

        <div class="mt-3" v-if="job.artifacts">
          <el-space wrap>
            <el-link v-if="job.artifacts.pth" :href="job.artifacts.pth" target="_blank" type="primary" :underline="false">
              <el-button type="primary" link icon="Download">下载模型 (.pth)</el-button>
            </el-link>
            <el-link v-if="job.artifacts.txt" :href="job.artifacts.txt" target="_blank" type="primary" :underline="false">
              <el-button type="primary" link icon="Document">查看日志 (.txt)</el-button>
            </el-link>
          </el-space>
        </div>

        <el-alert v-if="job.error" type="error" show-icon :closable="false" class="mt-3">
          <template #title>训练失败</template>
          <template #default>
            <pre class="error-pre">{{ job.error }}</pre>
          </template>
        </el-alert>
      </el-card>

      <el-empty v-else description="请在左侧配置参数并开始训练" />

      <div v-if="hasHistory" class="charts-wrapper">
        <el-card header="训练曲线" class="charts-card">
          <div class="charts-grid">
            <div v-for="d in chartDefs" :key="d.key" class="chart-item">
              <div :ref="(el) => setChartRef(el, d.key)" class="chart-canvas"></div>
            </div>
          </div>
        </el-card>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onUnmounted, reactive, ref, watch } from 'vue';
import { ElMessage } from 'element-plus';
import { cancelUnifiedTrainingJob, getUnifiedTrainingJob, startUnifiedTraining } from '../api/battery';
import * as echarts from 'echarts';

const form = reactive({
  model_type: 'DeepHPM',
  epochs: 100,
  lr: 0.0001,
  optimizer: 'SGD',
  layers: '32,32',
  activation: 'Tanh',
  batch_size: 128,
});

const submitting = ref(false);

const job = reactive({
  job_id: '',
  status: '',
  created_at: '',
  started_at: '',
  finished_at: '',
  error: '',
  config: null,
  progress: null,
  history: null,
  metrics: null,
  save_path: '',
  artifacts: null,
});

const polling = ref(false);
let pollTimer = null;

const showStructure = computed(() => form.model_type === 'Baseline' || form.model_type === 'DeepHPM');
const canceling = computed(() => job.status === 'canceling');
const isTraining = computed(() => ['queued', 'running', 'canceling'].includes(job.status));

watch(
  () => form.model_type,
  (v) => {
    if (!showStructure.value) {
      form.layers = '32,32';
      form.activation = 'Tanh';
    } else if (v === 'DeepHPM' && !form.activation) {
      form.activation = 'Tanh';
    }
  }
);

const statusTagType = computed(() => {
  switch (job.status) {
    case 'running': return 'primary';
    case 'succeeded': return 'success';
    case 'failed': return 'danger';
    case 'canceled': return 'info';
    case 'queued': return 'warning';
    default: return 'info';
  }
});

const progressPercentage = computed(() => {
  const p = job.progress;
  if (!p || !p.epoch || !p.epochs) return 0;
  return Math.min(100, Math.floor((p.epoch / p.epochs) * 100));
});

const progressStatus = computed(() => {
  if (job.status === 'succeeded') return 'success';
  if (job.status === 'failed' || job.status === 'canceled') return 'exception';
  return '';
});

const progressText = computed(() => {
  const p = job.progress;
  if (!p || !p.epoch || !p.epochs) return '-';
  return `${p.epoch} / ${p.epochs} Epochs`;
});

const hasHistory = computed(() => {
  const h = job.history;
  return Boolean(h && Array.isArray(h.epoch) && h.epoch.length > 0);
});

const chartDefs = [
  {
    key: 'loss',
    title: 'Loss',
    series: [
      { name: 'train', dataKey: 'loss_train' },
      { name: 'val', dataKey: 'loss_val' },
    ],
  },
  {
    key: 'loss_u',
    title: 'Loss_U',
    series: [
      { name: 'train', dataKey: 'loss_u_train' },
      { name: 'val', dataKey: 'loss_u_val' },
    ],
  },
  {
    key: 'loss_f',
    title: 'Loss_F',
    series: [
      { name: 'train', dataKey: 'loss_f_train' },
      { name: 'val', dataKey: 'loss_f_val' },
    ],
  },
  {
    key: 'loss_f_t',
    title: 'Loss_F_t',
    series: [
      { name: 'train', dataKey: 'loss_f_t_train' },
      { name: 'val', dataKey: 'loss_f_t_val' },
    ],
  },
  { key: 'mae', title: 'MAE', series: [{ name: 'val', dataKey: 'mae_val' }] },
  { key: 'mse', title: 'MSE', series: [{ name: 'val', dataKey: 'mse_val' }] },
  { key: 'rmspe', title: 'RMSPE', series: [{ name: 'val', dataKey: 'rmspe_val' }] },
  { key: 'r2', title: 'R2', series: [{ name: 'val', dataKey: 'r2_val' }] },
];

const chartRefs = ref({});
const chartInstances = new Map();

const setChartRef = (el, key) => {
  if (el) chartRefs.value[key] = el;
};

const resetCharts = () => {
  chartInstances.forEach((c) => c.dispose());
  chartInstances.clear();
  chartRefs.value = {};
};

const getHistoryArray = (key) => {
  const h = job.history;
  if (!h || !Array.isArray(h[key])) return [];
  return h[key];
};

const buildChartOption = (def) => {
  const epochs = getHistoryArray('epoch');
  const series = def.series.map((s) => ({
    name: s.name,
    type: 'line',
    data: getHistoryArray(s.dataKey),
    showSymbol: false,
    smooth: true,
    connectNulls: false,
    lineStyle: { width: 2 },
  }));

  return {
    title: { text: def.title, left: 'center', top: 8, textStyle: { fontSize: 14 } },
    tooltip: { trigger: 'axis' },
    legend: series.length > 1 ? { top: 32, left: 'center' } : undefined,
    grid: { left: '8%', right: '6%', top: series.length > 1 ? 60 : 48, bottom: '10%', containLabel: true },
    xAxis: { type: 'category', data: epochs, name: 'epoch' },
    yAxis: { type: 'value', scale: true },
    series,
  };
};

const updateCharts = async () => {
  if (!hasHistory.value) return;
  await nextTick();

  chartDefs.forEach((def) => {
    const el = chartRefs.value[def.key];
    if (!el) return;

    let chart = chartInstances.get(def.key);
    if (chart && ((typeof chart.isDisposed === 'function' && chart.isDisposed()) || (typeof chart.getDom === 'function' && chart.getDom() !== el))) {
      chart.dispose();
      chartInstances.delete(def.key);
      chart = null;
    }
    if (!chart) {
      chart = echarts.init(el);
      chartInstances.set(def.key, chart);
    }
    chart.setOption(buildChartOption(def), true);
    chart.resize();
  });
};

const setJob = (data) => {
  if (data?.job_id && job.job_id && data.job_id !== job.job_id) {
    resetCharts();
  }
  job.job_id = data.job_id || '';
  job.status = data.status || '';
  job.created_at = data.created_at || '';
  job.started_at = data.started_at || '';
  job.finished_at = data.finished_at || '';
  job.error = data.error || '';
  job.config = data.config || null;
  job.progress = data.progress || null;
  job.history = data.history || null;
  job.metrics = data.metrics || null;
  job.save_path = data.save_path || '';
  job.artifacts = data.artifacts || null;
};

const buildPayload = () => {
  const epochs = Number(form.epochs);
  const batchSize = Number(form.batch_size);
  const lr = Number(form.lr);

  if (!Number.isInteger(epochs) || epochs <= 0) {
    throw new Error('Epochs 必须为正整数');
  }
  if (!Number.isInteger(batchSize) || batchSize <= 0) {
    throw new Error('Batch Size 必须为正整数');
  }
  if (!Number.isFinite(lr) || lr <= 0) {
    throw new Error('学习率必须为正数');
  }

  const payload = {
    model_type: form.model_type,
    epochs,
    lr,
    optimizer: form.optimizer,
    batch_size: batchSize,
  };

  if (showStructure.value) {
    payload.layers = String(form.layers || '').trim();
    payload.activation = String(form.activation || '').trim();
  }

  return payload;
};

const handleStartTraining = async () => {
  submitting.value = true;
  stopPolling();
  resetCharts();
  try {
    const payload = buildPayload();
    const data = await startUnifiedTraining(payload);
    setJob(data);
    ElMessage.success(`已提交训练任务：${data.job_id}`);
    await updateCharts();
    startPolling();
  } catch (e) {
    const msg = e?.response?.data?.detail || e?.message || String(e);
    ElMessage.error(`提交失败：${msg}`);
  } finally {
    submitting.value = false;
  }
};

const fetchJob = async () => {
  if (!job.job_id) return;
  const data = await getUnifiedTrainingJob(job.job_id);
  setJob(data);
  await updateCharts();
  if (data.status === 'succeeded' || data.status === 'failed' || data.status === 'canceled') {
    stopPolling();
  }
};

const handleCancelTraining = async () => {
  if (!job.job_id) return;
  try {
    const data = await cancelUnifiedTrainingJob(job.job_id);
    setJob(data);
    ElMessage.success('已请求停止训练');
    if (!polling.value) startPolling();
  } catch (e) {
    const msg = e?.response?.data?.detail || e?.message || String(e);
    ElMessage.error(`停止失败：${msg}`);
  }
};

const startPolling = () => {
  if (!job.job_id) return;
  polling.value = true;
  pollTimer = setInterval(async () => {
    try {
      await fetchJob();
    } catch (e) {
      stopPolling();
    }
  }, 1200);
};

const stopPolling = () => {
  polling.value = false;
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
};

watch(hasHistory, async (v) => {
  if (v) {
    await updateCharts();
  }
});

onUnmounted(() => {
  stopPolling();
  resetCharts();
});
</script>

<style scoped>
.training-platform {
  position: relative;
  min-height: 100vh;
}

.sidebar-container {
  position: fixed;
  top: 130px;
  left: 0;
  bottom: 0;
  width: 250px;
  background-color: #fff;
  border-right: 1px solid #dcdfe6;
  z-index: 900;
  padding: 20px;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.05);
  overflow-y: hidden;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.sidebar-card {
  border: none;
  box-shadow: none !important;
}

.main-content {
  margin-left: 300px;
  padding: 20px;
  min-height: 100vh;
  background-color: #f5f7fa;
}

.mb-3 {
  margin-bottom: 12px;
}

.mb-4 {
  margin-bottom: 16px;
}

.mt-3 {
  margin-top: 12px;
}

.charts-wrapper {
  margin-top: 16px;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.chart-item {
  border: 1px solid #ebeef5;
  border-radius: 6px;
  padding: 6px;
  background: #fff;
}

.chart-canvas {
  width: 100%;
  height: 260px;
}

.error-pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 12px;
  line-height: 1.4;
}

.task-status-card .status-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.task-status-card .progress-section {
  margin-top: 16px;
}

.task-status-card .progress-info {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 14px;
  color: #606266;
}

.font-bold {
  font-weight: bold;
}

.charts-card :deep(.el-card__body) {
  padding: 12px;
}
</style>
