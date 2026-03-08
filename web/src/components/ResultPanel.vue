<script setup>
// web/src/components/ResultPanel.vue
// 2) 结果展示区：
// - 用 el-card 承载结果
// - 展示：篡改类型、置信度（progress）、风险等级（tag）
// - 展示后端返回的证据图片对（前帧 / 起始帧）
// - 用红色框标注可疑片段时间点（如有）

import { computed } from 'vue'
import { Picture, Warning } from '@element-plus/icons-vue'

const props = defineProps({
  fileName: {
    type: String,
    default: '',
  },
  videoUrl: {
    type: String,
    default: '',
  },
  // result: { tamperingType, confidence, riskLevel, suspiciousSegments, evidenceFrames }
  result: {
    type: Object,
    default: null,
  },
})

const hasResult = computed(() => !!props.result)
const evidenceFrames = computed(() => props.result?.evidenceFrames || [])

function riskTagType(risk) {
  if (risk === '高') return 'danger'
  if (risk === '中') return 'warning'
  return 'success'
}
</script>

<template>
  <el-card shadow="hover" class="panel">
    <template #header>
      <div class="panel__header">
        <div class="panel__title">
          <el-icon><Picture /></el-icon>
          <span>结果展示</span>
        </div>
        <div class="panel__hint">展示后端返回的证据截图与置信度</div>
      </div>
    </template>

    <div v-if="!hasResult" class="empty">
      <el-icon class="empty__icon"><Warning /></el-icon>
      <div class="empty__text">暂无结果：请先上传视频并点击「开始检测」</div>
    </div>

    <div v-else class="content">
      <div class="summary">
        <div class="summary__item">
          <div class="summary__label">篡改类型</div>
          <div class="summary__value">{{ result.tamperingType }}</div>
        </div>

        <div class="summary__item">
          <div class="summary__label">风险等级</div>
          <div class="summary__value">
            <el-tag :type="riskTagType(result.riskLevel)">{{ result.riskLevel }}</el-tag>
          </div>
        </div>

        <div class="summary__item summary__item--wide">
          <div class="summary__label">置信度</div>
          <div class="summary__value">
            <el-progress :percentage="result.confidence" :stroke-width="10" />
          </div>
        </div>
      </div>

      <div class="evidence">
        <div class="evidence__title">疑似篡改证据帧</div>
        <div v-if="evidenceFrames.length" class="evidence__grid">
          <div v-for="(item, idx) in evidenceFrames" :key="`${item.track_id}-${item.start_frame}-${idx}`" class="evidence__pair">
            <div class="evidence__pair-header">
              <span>Track {{ item.track_id }}</span>
              <span>起始帧 {{ item.start_frame }}</span>
            </div>

            <div class="evidence__images">
              <div class="evidence__image-card">
                <div class="evidence__image-label">前一帧</div>
                <img v-if="item.pre_image_url" class="evidence__image" :src="item.pre_image_url" alt="前一帧证据图" />
                <div v-else class="evidence__placeholder">无前一帧图片</div>
              </div>

              <div class="evidence__image-card evidence__image-card--alert">
                <div class="evidence__image-label">疑似篡改起始帧</div>
                <img v-if="item.start_image_url" class="evidence__image" :src="item.start_image_url" alt="起始帧证据图" />
                <div v-else class="evidence__placeholder">无起始帧图片</div>
              </div>
            </div>
          </div>
        </div>
        <div v-else class="evidence__empty">当前未返回证据图片，可先检查后端是否提取到了 evidence_images。</div>

        <div class="evidence__file">文件：{{ fileName || '（未选择）' }}</div>
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

.empty {
  height: 320px;
  display: grid;
  place-items: center;
  background: #fff;
  border: 1px dashed #dcdfe6;
  border-radius: 8px;
}

.empty__icon {
  font-size: 28px;
  color: #909399;
}

.empty__text {
  margin-top: 8px;
  color: #909399;
}

.summary {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 12px;
}

.summary__item {
  padding: 10px 12px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  background: #fff;
}

.summary__item--wide {
  grid-column: 1 / -1;
}

.summary__label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 6px;
}

.summary__value {
  color: var(--deep-gray);
  font-weight: 600;
}

.evidence {
  padding: 12px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  background: #fff;
}

.evidence__title {
  font-weight: 600;
  margin-bottom: 8px;
}

.evidence__grid {
  display: grid;
  gap: 12px;
}

.evidence__pair {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 10px;
  background: #fafafa;
}

.evidence__pair-header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 12px;
  color: #606266;
  margin-bottom: 8px;
}

.evidence__images {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.evidence__image-card {
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  overflow: hidden;
  background: #fff;
}

.evidence__image-card--alert {
  border-color: #f56c6c;
}

.evidence__image-label {
  padding: 8px 10px;
  font-size: 12px;
  color: #606266;
  border-bottom: 1px solid #ebeef5;
}

.evidence__image {
  display: block;
  width: 100%;
  max-height: 320px;
  object-fit: contain;
  background: #111827;
}

.evidence__placeholder,
.evidence__empty {
  min-height: 160px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  background: #f2f3f5;
  color: #909399;
  padding: 12px;
  text-align: center;
}

.evidence__file {
  margin-top: 8px;
  font-size: 12px;
  color: #909399;
}

@media (max-width: 1200px) {
  .empty {
    height: 240px;
  }

  .evidence__images {
    grid-template-columns: 1fr;
  }
}
</style>
