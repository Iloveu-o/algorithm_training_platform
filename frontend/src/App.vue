<script setup>
import BatteryStats from './components/BatteryStats.vue'
import TrainingPlatform from './components/TrainingPlatform.vue'
import AlgorithmTestPlatform from './components/AlgorithmTestPlatform.vue'
import { ref } from 'vue'

const activeTab = ref('analysis')
</script>

<template>
  <div class="app-container">
    <header class="app-header">
      <h1>储能电池寿命分析及算法测试平台</h1>
      <el-tabs v-model="activeTab" class="tabs">
        <el-tab-pane label="电池分析" name="analysis">
        </el-tab-pane>
        <el-tab-pane label="算法训练" name="training">
        </el-tab-pane>
        <el-tab-pane label="算法测试平台" name="testing">
        </el-tab-pane>
      </el-tabs>
    </header>
    <main class="main-content-wrapper">
      <keep-alive>
        <component :is="activeTab === 'analysis' ? BatteryStats : activeTab === 'training' ? TrainingPlatform : AlgorithmTestPlatform" />
      </keep-alive>
    </main>
  </div>
</template>

<style scoped>
.app-container {
  min-height: 100vh;
  background-color: #f5f7fa;
  padding-top: 130px; /* Header height + spacing */
}

.app-header {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  background-color: #fff;
  padding: 1rem 2rem 0;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  z-index: 2000;
  display: flex;
  flex-direction: column;
  align-items: center;
}

h1 {
  margin: 0 0 1rem 0;
  color: #303133;
  font-size: 1.5rem;
  text-align: center;
}

.tabs {
  width: 100%;
  max-width: 600px; /* Control tabs width */
}

:deep(.el-tabs__nav-wrap) {
  display: flex;
  justify-content: center;
}

:deep(.el-tabs__nav) {
  float: none; /* Override Element Plus float */
}
</style>
