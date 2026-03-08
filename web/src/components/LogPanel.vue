<script setup>
// web/src/components/LogPanel.vue
// 3) 日志区：
// - 固定高度、可滚动
// - 自动滚动到最新日志
// - INFO/WARNING/ERROR 使用不同颜色标签

import { nextTick, onMounted, ref, watch } from 'vue'
import { Document } from '@element-plus/icons-vue'

const props = defineProps({
  // logs: [{ time: 'HH:mm:ss', level: 'INFO'|'WARNING'|'ERROR', message: string }]
  logs: {
    type: Array,
    default: () => [],
  },
})

const logContainerRef = ref(null)

function levelTagType(level) {
  if (level === 'ERROR') return 'danger'
  if (level === 'WARNING') return 'warning'
  return 'info'
}

async function scrollToBottom() {
  await nextTick()
  const el = logContainerRef.value
  if (!el) return
  el.scrollTop = el.scrollHeight
}

watch(
  () => props.logs.length,
  () => {
    scrollToBottom()
  }
)

onMounted(() => {
  scrollToBottom()
})
</script>

<template>
  <el-card shadow="hover" class="panel">
    <template #header>
      <div class="panel__header">
        <div class="panel__title">
          <el-icon><Document /></el-icon>
          <span>处理日志</span>
        </div>
        <div class="panel__hint">自动滚动到最新日志</div>
      </div>
    </template>

    <div ref="logContainerRef" class="log-container">
      <div v-if="logs.length === 0" class="log-empty">暂无日志</div>

      <div v-for="(item, idx) in logs" :key="idx" class="log-row">
        <div class="log-time">{{ item.time }}</div>
        <el-tag class="log-level" size="small" :type="levelTagType(item.level)">{{ item.level }}</el-tag>
        <div class="log-msg">{{ item.message }}</div>
      </div>
    </div>
  </el-card>
</template>

<style scoped>
.panel__header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}

.panel__title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.panel__hint {
  font-size: 12px;
  color: #909399;
}

.log-container {
  height: 320px;
  overflow-y: auto;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #ebeef5;
  background: #fff;
}

.log-empty {
  color: #909399;
  display: grid;
  place-items: center;
  height: 100%;
}

.log-row {
  display: grid;
  grid-template-columns: 78px 92px 1fr;
  gap: 10px;
  align-items: start;
  padding: 8px 6px;
  border-bottom: 1px dashed #f0f2f5;
}

.log-row:last-child {
  border-bottom: none;
}

.log-time {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
  color: #606266;
}

.log-level {
  width: fit-content;
}

.log-msg {
  color: var(--deep-gray);
  line-height: 1.4;
}

@media (max-width: 1200px) {
  .log-container {
    height: 260px;
  }
}
</style>
