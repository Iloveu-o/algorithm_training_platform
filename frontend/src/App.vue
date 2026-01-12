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
      <h1>算法训练与电池分析平台</h1>
    </header>
    <main>
      <el-tabs v-model="activeTab" class="tabs">
        <el-tab-pane label="电池分析" name="analysis">
          <BatteryStats />
        </el-tab-pane>
        <el-tab-pane label="算法训练" name="training">
          <TrainingPlatform />
        </el-tab-pane>
        <el-tab-pane label="算法测试平台" name="testing">
          <AlgorithmTestPlatform />
        </el-tab-pane>
      </el-tabs>
    </main>
  </div>
</template>

<style scoped>
.app-container {
  min-height: 100vh;
  background-color: #f5f7fa;
}

.app-header {
  background-color: #fff;
  padding: 1rem 2rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  margin-bottom: 20px;
}

h1 {
  margin: 0;
  color: #303133;
  font-size: 1.5rem;
}

.tabs {
  padding: 0 20px;
}
</style>
