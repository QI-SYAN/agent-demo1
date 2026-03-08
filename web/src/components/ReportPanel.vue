<script setup>
// web/src/components/ReportPanel.vue
// 4) 报告下载区：
// - 报告类型选择（PDF/Word）
// - 展示后端已生成的下载链接
// - 保留“生成/下载”交互，便于切换格式

import { computed } from 'vue'
import { Download, DocumentAdd } from '@element-plus/icons-vue'

const props = defineProps({
  reportType: {
    type: String,
    default: 'PDF',
  },
  status: {
    type: String,
    default: '待生成',
  },
  link: {
    type: String,
    default: '',
  },
  canGenerate: {
    type: Boolean,
    default: false,
  },
  canDownload: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits(['update:reportType', 'generate', 'download'])

const localType = computed({
  get: () => props.reportType,
  set: (v) => emit('update:reportType', v),
})

function statusTagType(status) {
  if (status === '已完成') return 'success'
  if (status === '生成中') return 'warning'
  return 'info'
}
</script>

<template>
  <el-card shadow="hover" class="panel">
    <template #header>
      <div class="panel__header">
        <div class="panel__title">
          <el-icon><DocumentAdd /></el-icon>
          <span>报告下载</span>
        </div>
        <div class="panel__hint">支持 PDF / Word（后端固定模板导出）</div>
      </div>
    </template>

    <div class="row">
      <div class="row__label">报告类型</div>
      <el-radio-group v-model="localType">
        <el-radio-button label="PDF">PDF</el-radio-button>
        <el-radio-button label="WORD">Word</el-radio-button>
      </el-radio-group>
    </div>

    <div class="row">
      <div class="row__label">生成状态</div>
      <el-tag :type="statusTagType(status)">{{ status }}</el-tag>
    </div>

    <div class="actions">
      <el-button
        type="primary"
        :disabled="!canGenerate"
        @click="$emit('generate')"
      >
        生成检测报告
      </el-button>

      <el-button
        type="success"
        :icon="Download"
        :disabled="!canDownload"
        @click="$emit('download')"
      >
        下载报告
      </el-button>
    </div>

    <div class="link">
      <div class="link__label">下载链接</div>
      <div v-if="link" class="link__value">
        <a :href="link" target="_blank" rel="noreferrer">{{ link }}</a>
      </div>
      <div v-else class="link__empty">（生成后会在此处显示链接）</div>
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

.row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.row__label {
  width: 72px;
  color: #606266;
  font-size: 12px;
}

.actions {
  display: flex;
  gap: 12px;
  margin-top: 10px;
}

.link {
  margin-top: 14px;
  padding: 12px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  background: #fff;
}

.link__label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 6px;
}

.link__value {
  word-break: break-all;
}

.link__empty {
  color: #909399;
}
</style>
