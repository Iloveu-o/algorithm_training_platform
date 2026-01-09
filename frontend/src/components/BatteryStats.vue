<template>
  <div class="battery-stats">
    <el-card class="search-card">
      <div class="search-box">
        <el-input 
          v-model="batteryId" 
          placeholder="请输入电池组编号 (例如: 1)" 
          class="input-id" 
          clearable
          @keyup.enter="handleAnalyze"
        >
          <template #prepend>电池组编号</template>
        </el-input>
        <el-button type="primary" :loading="loading" @click="handleAnalyze">开始分析</el-button>
      </div>
    </el-card>

    <div v-if="result" class="result-container">
      <!-- 概览信息 -->
      <el-row :gutter="20" class="mb-4">
        <el-col :span="8">
          <el-card shadow="hover">
            <template #header>基本信息</template>
            <div class="stat-item">
              <span>样本数量：</span>
              <span class="value">{{ result.sample_count }}</span>
            </div>
            <div class="stat-item">
              <span>电池编号：</span>
              <span class="value">#{{ result.battery_id }}</span>
            </div>
          </el-card>
        </el-col>

      </el-row>

      <!-- 特征统计表格 -->
      <el-card class="mb-4" header="特征统计与相关性分析">
        <el-table :data="featureTableData" style="width: 100%" stripe border>
          <el-table-column prop="name" label="特征名称" width="100" fixed />
          <el-table-column prop="mean" label="均值" width="120">
            <template #default="scope">{{ scope.row.mean?.toFixed(4) ?? '-' }}</template>
          </el-table-column>
          <el-table-column prop="var" label="方差" width="120">
            <template #default="scope">{{ scope.row.var?.toFixed(6) ?? '-' }}</template>
          </el-table-column>
          <el-table-column prop="min" label="最小值" width="120">
            <template #default="scope">{{ scope.row.min?.toFixed(4) ?? '-' }}</template>
          </el-table-column>
          <el-table-column prop="max" label="最大值" width="120">
            <template #default="scope">{{ scope.row.max?.toFixed(4) ?? '-' }}</template>
          </el-table-column>
          <el-table-column prop="skew" label="偏度" width="120">
            <template #default="scope">{{ scope.row.skew?.toFixed(4) ?? '-' }}</template>
          </el-table-column>
          <el-table-column prop="kurtosis" label="峰度" width="120">
            <template #default="scope">{{ scope.row.kurtosis?.toFixed(4) ?? '-' }}</template>
          </el-table-column>
          <el-table-column prop="corr_rul" label="RUL 相关性" width="120">
            <template #default="scope">
              <el-tag :type="Math.abs(scope.row.corr_rul) > 0.5 ? 'danger' : 'info'">
                {{ scope.row.corr_rul?.toFixed(4) ?? '-' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="corr_pcl" label="PCL 相关性" width="120">
            <template #default="scope">
              <el-tag :type="Math.abs(scope.row.corr_pcl) > 0.5 ? 'warning' : 'info'">
                {{ scope.row.corr_pcl?.toFixed(4) ?? '-' }}
              </el-tag>
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <!-- 可视化图表 -->
      <el-card header="数据可视化图表">
        <!-- 交互式图表 - 8个特征独立图 -->
        <h3 style="text-align: center; color: #606266; margin-bottom: 20px;">特征随循环次数变化趋势</h3>
        <div class="features-grid mb-4" v-if="result.raw_data">
            <div v-for="f in Object.keys(result.raw_data.features)" :key="f" class="feature-chart-item">
                 <div :ref="el => setFeatureChartRef(el, f)" style="width: 100%; height: 300px;"></div>
            </div>
        </div>

        <h3 style="text-align: center; color: #606266; margin: 20px 0;">特征 vs RUL</h3>
        <div class="features-grid mb-4" v-if="result.raw_data">
            <div v-for="f in Object.keys(result.raw_data.features)" :key="f" class="feature-chart-item">
                 <div :ref="el => setRulChartRef(el, f)" style="width: 100%; height: 300px;"></div>
            </div>
        </div>

        <h3 style="text-align: center; color: #606266; margin: 20px 0;">特征数据分布 (直方图)</h3>
        <div class="features-grid mb-4" v-if="result.raw_data">
            <div v-for="f in Object.keys(result.raw_data.features)" :key="f" class="feature-chart-item">
                 <div :ref="el => setFeatureDistChartRef(el, f)" style="width: 100%; height: 300px;"></div>
            </div>
        </div>

        <div class="charts-grid">
          <div class="chart-item">
            <div ref="pclChartRef" style="width: 100%; height: 300px;"></div>
          </div>
          <div class="chart-item-full">
            <div ref="heatmapChartRef" style="width: 100%; height: 400px;"></div>
          </div>
        </div>
      </el-card>
    </div>

    <el-empty v-else description="请输入电池编号并点击分析" />
  </div>
</template>

<script setup>
// Fixed rulChart reference
import { ref, computed, watch, nextTick, onUnmounted } from 'vue';
import { ElMessage } from 'element-plus';
import { getBatteryAnalysis } from '../api/battery';
import * as echarts from 'echarts';

const batteryId = ref('');
const loading = ref(false);
const result = ref(null);

// Charts refs
const featureChartRefs = ref({});
const rulChartRefs = ref({});
const featureDistChartRefs = ref({});
const pclChartRef = ref(null);
const heatmapChartRef = ref(null);

// Chart instances
let featureCharts = [];
let rulCharts = [];
let featureDistCharts = [];
let pclChart = null;
let heatmapChart = null;

const setFeatureChartRef = (el, key) => {
  if (el) {
    featureChartRefs.value[key] = el;
  }
};

const setRulChartRef = (el, key) => {
  if (el) {
    rulChartRefs.value[key] = el;
  }
};

const setFeatureDistChartRef = (el, key) => {
  if (el) {
    featureDistChartRefs.value[key] = el;
  }
};

const featureTableData = computed(() => {
  if (!result.value) return [];
  return Object.entries(result.value.features).map(([key, value]) => ({
    name: key,
    ...value
  }));
});

// Helper function to calculate Pearson correlation
const calculateCorrelation = (x, y) => {
    const n = x.length;
    if (n !== y.length || n === 0) return 0;
    
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
};

const initCharts = () => {
  if (!result.value?.raw_data) return;
  const rawData = result.value.raw_data;
  const features = Object.keys(rawData.features);

  // 1. Features vs Cycles (8 separate Line Charts)
  // Clear old instances
  featureCharts.forEach(c => c.dispose());
  featureCharts = [];

  features.forEach(f => {
      const el = featureChartRefs.value[f];
      if (el) {
          const chart = echarts.init(el);
          chart.setOption({
            title: { text: f, left: 'center', top: 10, textStyle: { fontSize: 14 } },
            tooltip: { trigger: 'axis' },
            grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true, top: 40 },
            xAxis: { type: 'category', boundaryGap: false, data: rawData.cycles },
            yAxis: { type: 'value', scale: true }, // scale: true makes it adapt to min/max
            series: [{
                name: f,
                type: 'line',
                data: rawData.features[f],
                showSymbol: false,
                smooth: true,
                lineStyle: { width: 2 }
            }]
          });
          featureCharts.push(chart);
      }
  });

  // 2. Features vs RUL (Scatter Plot)
  // Clear old instances
  rulCharts.forEach(c => c.dispose());
  rulCharts = [];

  features.forEach(f => {
      const el = rulChartRefs.value[f];
      if (el) {
          const chart = echarts.init(el);
          // Pair feature value with RUL
          const data = rawData.features[f].map((val, idx) => [rawData.rul[idx], val]);
          
          chart.setOption({
              title: { text: f, left: 'center', top: 10, textStyle: { fontSize: 14 } },
              tooltip: {
                  trigger: 'item',
                  formatter: (params) => `RUL: ${params.value[0]}<br/>${f}: ${params.value[1]}`
              },
              grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true, top: 40 },
              xAxis: { type: 'value', name: 'RUL', scale: true },
              yAxis: { type: 'value', scale: true },
              series: [{
                  name: f,
                  type: 'scatter',
                  symbolSize: 5,
                  data: data
              }]
          });
          rulCharts.push(chart);
      }
  });

  // 3. Feature Distribution (Histograms)
  featureDistCharts.forEach(c => c.dispose());
  featureDistCharts = [];

  features.forEach(f => {
      const el = featureDistChartRefs.value[f];
      // Get stats from result.features
      const stats = result.value.features[f];
      
      if (el && stats && stats.bins && stats.counts) {
          const chart = echarts.init(el);
          
          // Format bin labels (bins has N+1 edges for N counts)
          const binLabels = [];
          if (stats.bins.length === stats.counts.length + 1) {
              for (let i = 0; i < stats.counts.length; i++) {
                  const start = stats.bins[i];
                  const end = stats.bins[i+1];
                  binLabels.push((start + end) / 2); // Use center value for simpler display
              }
          } else {
             // Fallback if mismatch
             binLabels.push(...stats.bins);
          }
          
          // Improve label formatting
          const formattedLabels = binLabels.map(v => typeof v === 'number' ? v.toFixed(2) : v);

          chart.setOption({
              title: { text: f, left: 'center', top: 10, textStyle: { fontSize: 14 } },
              tooltip: { 
                  trigger: 'axis',
                  formatter: (params) => {
                      const idx = params[0].dataIndex;
                      const count = params[0].value;
                      const start = stats.bins[idx]?.toFixed(2);
                      const end = stats.bins[idx+1]?.toFixed(2);
                      return `${f}<br/>Range: [${start}, ${end})<br/>Count: ${count}`;
                  }
              },
              grid: { left: '3%', right: '4%', bottom: '10%', containLabel: true, top: 40 },
              xAxis: { 
                  type: 'category', 
                  data: formattedLabels, 
                  name: 'Value',
                  axisLabel: { interval: 'auto' } 
              },
              yAxis: { type: 'value', name: 'Count' },
              series: [{
                  name: f,
                  type: 'bar',
                  barWidth: '95%',
                  data: stats.counts,
                  itemStyle: { color: '#409EFF' }
              }]
          });
          featureDistCharts.push(chart);
      }
  });

  // 4. PCL Distribution (Histogram)
  if (pclChartRef.value) {
      if (pclChart) pclChart.dispose();
      pclChart = echarts.init(pclChartRef.value);
      
      // Calculate histogram data
      const pclData = rawData.pcl.filter(v => v !== null && v !== undefined);
      // Simple histogram logic or use ECharts histogram if supported (usually via bar chart with custom transform or pre-processing)
      // Here we use a simple pre-processing to get bins
      if (pclData.length > 0) {
          const min = Math.min(...pclData);
          const max = Math.max(...pclData);
          const binCount = 10;
          const step = (max - min) / binCount || 1;
          const bins = new Array(binCount).fill(0);
          const binLabels = [];
          
          for(let i=0; i<binCount; i++) {
              binLabels.push((min + i * step).toFixed(2));
          }

          pclData.forEach(val => {
              const idx = Math.min(Math.floor((val - min) / step), binCount - 1);
              bins[idx]++;
          });

          pclChart.setOption({
              title: { text: 'PCL 分布直方图', left: 'center' },
              tooltip: { trigger: 'axis' },
              xAxis: { type: 'category', data: binLabels, name: 'PCL 区间' },
              yAxis: { type: 'value', name: '频数' },
              series: [{
                  data: bins,
                  type: 'bar',
                  barWidth: '90%'
              }]
          });
      }
  }

  // 4. Feature Correlation Heatmap
  if (heatmapChartRef.value) {
      if (heatmapChart) heatmapChart.dispose();
      heatmapChart = echarts.init(heatmapChartRef.value);

      const variables = [...features, 'rul', 'pcl'];
      const correlationData = [];
      
      // Calculate correlation matrix
      for (let i = 0; i < variables.length; i++) {
          for (let j = 0; j < variables.length; j++) {
              const var1 = variables[i];
              const var2 = variables[j];
              
              let data1, data2;
              if (features.includes(var1)) data1 = rawData.features[var1];
              else data1 = rawData[var1];
              
              if (features.includes(var2)) data2 = rawData.features[var2];
              else data2 = rawData[var2];

              const corr = calculateCorrelation(data1, data2);
              correlationData.push([i, j, parseFloat(corr.toFixed(2))]);
          }
      }

      heatmapChart.setOption({
          title: { text: '特征相关性热力图', left: 'center' },
          tooltip: {
              position: 'top',
              formatter: (params) => {
                  return `${variables[params.value[1]]} vs ${variables[params.value[0]]}: ${params.value[2]}`;
              }
          },
          grid: { top: 60, bottom: 60, left: 80, right: 60 },
          xAxis: { type: 'category', data: variables, splitArea: { show: true } },
          yAxis: { type: 'category', data: variables, splitArea: { show: true } },
          visualMap: {
              min: -1,
              max: 1,
              calculable: true,
              orient: 'horizontal',
              left: 'center',
              bottom: '0%',
              inRange: {
                  color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
              }
          },
          series: [{
              name: 'Correlation',
              type: 'heatmap',
              data: correlationData,
              label: { show: true },
              emphasis: {
                  itemStyle: {
                      shadowBlur: 10,
                      shadowColor: 'rgba(0, 0, 0, 0.5)'
                  }
              }
          }]
      });
  }
};

watch(result, () => {
  nextTick(() => {
    initCharts();
  });
});

// Resize chart on window resize
const handleResize = () => {
  featureCharts.forEach(c => c.resize());
  rulCharts.forEach(c => c.resize());
  featureDistCharts.forEach(c => c.resize());
  pclChart?.resize();
  heatmapChart?.resize();
};
window.addEventListener('resize', handleResize);

onUnmounted(() => {
  window.removeEventListener('resize', handleResize);
  featureCharts.forEach(c => c.dispose());
  rulCharts.forEach(c => c.dispose());
  featureDistCharts.forEach(c => c.dispose());
  pclChart?.dispose();
  heatmapChart?.dispose();
});

const handleAnalyze = async () => {
  if (!batteryId.value) {
    ElMessage.warning('请输入电池组编号');
    return;
  }

  const id = parseInt(batteryId.value);
  if (isNaN(id)) {
    ElMessage.error('请输入有效的数字编号');
    return;
  }

  loading.value = true;
  result.value = null;

  try {
    const data = await getBatteryAnalysis(id);
    result.value = data;
    ElMessage.success('分析完成');
  } catch (error) {
    console.error(error);
    const msg = error.response?.data?.detail || '请求失败，请检查后端服务';
    ElMessage.error(msg);
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.battery-stats {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.search-card {
  margin-bottom: 20px;
}

.search-box {
  display: flex;
  gap: 10px;
  max-width: 500px;
  margin: 0 auto;
}

.input-id {
  flex: 1;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10px;
  font-size: 16px;
}

.stat-item .value {
  font-weight: bold;
  color: #409EFF;
}

.mb-4 {
  margin-bottom: 20px;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
}

.chart-item {
  border: 1px solid #eee;
  padding: 10px;
  border-radius: 4px;
  text-align: center;
}

.charts-grid-full {
  width: 100%;
  margin-bottom: 20px;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
}

.feature-chart-item {
  border: 1px solid #eee;
  padding: 10px;
  border-radius: 4px;
}

.chart-item-full {
  grid-column: span 2;
  border: 1px solid #eee;
  padding: 10px;
  border-radius: 4px;
}

.chart-container {
  border: 1px solid #eee;
  padding: 10px;
  border-radius: 4px;
}
</style>
