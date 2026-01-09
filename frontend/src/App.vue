<script setup>
import BatteryStats from './components/BatteryStats.vue'
</script>

<template>
  <div class="app-container">
    <header class="app-header">
      <h1>电池数据分析平台</h1>
    </header>
    <main>
      <BatteryStats />
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
</style>
