<script setup>
// App.vue
// 根组件：负责整体布局与跨区域状态管理。
// 当前版本：前端真实上传 /upload + 调用 /detect（后端需启动）。

import { computed, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Cpu } from '@element-plus/icons-vue'

import UploadPanel from './components/UploadPanel.vue'
import ResultPanel from './components/ResultPanel.vue'
import LogPanel from './components/LogPanel.vue'
import ReportPanel from './components/ReportPanel.vue'

import { useDetection } from './composables/useDetection'

// 上传后的视频信息
const uploadedFileName = ref('')
const uploadedVideoUrl = ref('')
// 后端 /upload 返回的定位符（通常是文件路径），用于 /detect
const uploadedVideoLocator = ref('')

// 检测结果（ResultPanel 使用）
const result = ref(null)

// 报告相关状态
const reportType = ref('PDF')
const reportStatus = ref('待生成') // 待生成 | 生成中 | 已完成
const reportLink = ref('')
const reportLinks = ref({ PDF: '', WORD: '' })

const {
  logs,
  isRunning,
  pushLog,
  startDetection,
  resetLogs,
} = useDetection()

const canStartDetect = computed(() => !!uploadedVideoLocator.value && !isRunning.value)
const canGenerateReport = computed(
  () => reportStatus.value !== '生成中' && !!result.value && (!!reportLinks.value.PDF || !!reportLinks.value.WORD),
)
const canDownloadReport = computed(() => reportStatus.value === '已完成' && !!reportLink.value)

watch(reportType, (nextType) => {
  reportLink.value = reportLinks.value[nextType] || ''
  if (reportLink.value) {
    reportStatus.value = '已完成'
  }
})

function handleUploaded(payload) {
  // payload: { fileName: string, videoUrl: string, videoLocator?: string }
  uploadedFileName.value = payload.fileName
  uploadedVideoUrl.value = payload.videoUrl
  uploadedVideoLocator.value = payload.videoLocator || ''

  result.value = null
  reportStatus.value = '待生成'
  reportLink.value = ''
  reportLinks.value = { PDF: '', WORD: '' }

  resetLogs()
  pushLog('INFO', `已选择视频：${payload.fileName}`)
  if (uploadedVideoLocator.value) {
    pushLog('INFO', `上传完成：video_locator = ${uploadedVideoLocator.value}`)
  } else {
    pushLog('WARNING', '视频已选择，但尚未获得 video_locator（请等待上传完成）')
  }
}

function handleRemoved() {
  uploadedFileName.value = ''
  uploadedVideoUrl.value = ''
  uploadedVideoLocator.value = ''
  result.value = null
  reportStatus.value = '待生成'
  reportLink.value = ''
  reportLinks.value = { PDF: '', WORD: '' }

  resetLogs()
  pushLog('INFO', '已移除视频文件')
}

function handleStartDetect() {
  if (!canStartDetect.value) {
    ElMessage.warning('请先上传视频并等待上传完成')
    return
  }

  startDetection({
    videoLocator: uploadedVideoLocator.value,
    onFinish: (resp) => {
      const verdict = String(resp?.verdict || 'unknown')
      // 后端 confidence 当前是 0~1，小数语义；前端 progress 组件使用 0~100。
      const backendConfidence = Number(resp?.confidence || 0)
      const confidence = Math.round(Math.max(0, Math.min(1, backendConfidence)) * 100)
      const rationale = String(resp?.rationale || '')
      const evidenceFrames = Array.isArray(resp?.evidence_frames) ? resp.evidence_frames : []
      const reportArtifacts = resp?.report_artifacts || {}

      // 简单规则：根据后端 verdict + 文本线索推一个“类型”用于展示
      let tamperingType = '未发现明显篡改'
      if (verdict === 'tampered') {
        tamperingType = '疑似车辆删除'
        if (rationale.includes('背景残差')) tamperingType = '背景修复/擦除痕迹'
        if (rationale.includes('车辆轨迹异常') || rationale.includes('消失')) tamperingType = '车辆删除/轨迹中断'
      } else if (verdict === 'unknown') {
        tamperingType = '证据不足/需人工复核'
      }

      const riskLevel = confidence >= 80 ? '高' : confidence >= 50 ? '中' : '低'

      result.value = {
        tamperingType,
        confidence,
        riskLevel,
        evidenceFrames,
        // 后端目前未标准化返回“可疑时间段”，这里先占位；后续可从 evidence 里解析。
        suspiciousSegments: verdict === 'tampered'
          ? ['00:12.4 - 00:15.8', '01:03.0 - 01:06.2']
          : [],
      }

      reportLinks.value = {
        PDF: String(reportArtifacts?.pdf_url || ''),
        WORD: String(reportArtifacts?.docx_url || ''),
      }
      reportLink.value = reportLinks.value[reportType.value] || reportLinks.value.PDF || reportLinks.value.WORD || ''
      reportStatus.value = reportLink.value ? '已完成' : '待生成'

      if (reportArtifacts?.generation_error) {
        pushLog('WARNING', `报告导出提示：${String(reportArtifacts.generation_error)}`)
      }
    },
  })
}

function handleGenerateReport() {
  if (!canGenerateReport.value) return
  const nextLink = reportLinks.value[reportType.value] || ''
  if (!nextLink) {
    reportStatus.value = '待生成'
    reportLink.value = ''
    pushLog('WARNING', `当前暂无 ${reportType.value} 报告，请切换为另一种格式或检查后端导出日志`)
    return
  }

  reportStatus.value = '已完成'
  reportLink.value = nextLink
  pushLog('INFO', `${reportType.value} 报告已就绪`) 
}

function handleDownloadReport() {
  if (!canDownloadReport.value) return
  pushLog('INFO', `开始下载：${reportLink.value}`)
  window.open(reportLink.value, '_blank', 'noopener,noreferrer')
  ElMessage.success('已触发下载动作')
}
</script>

<template>
  <div class="app-page">
    <header class="app-header">
      <div>
        <div class="app-header__title">
          <el-icon><Cpu /></el-icon>
          <span>高速公路视频篡改检测系统</span>
        </div>
        <div class="app-header__sub">演示版 · Vue 3 + Element Plus（/upload + /detect）</div>
      </div>
      <div class="muted">主色：#409EFF · 深灰：#303133</div>
    </header>

    <main class="app-main">
      <div class="grid">
        <UploadPanel
          class="card"
          :is-running="isRunning"
          @uploaded="handleUploaded"
          @removed="handleRemoved"
          @start-detect="handleStartDetect"
        />

        <ResultPanel
          class="card"
          :file-name="uploadedFileName"
          :video-url="uploadedVideoUrl"
          :result="result"
        />

        <LogPanel class="card" :logs="logs" />

        <ReportPanel
          class="card"
          v-model:report-type="reportType"
          :status="reportStatus"
          :link="reportLink"
          :can-generate="canGenerateReport"
          :can-download="canDownloadReport"
          @generate="handleGenerateReport"
          @download="handleDownloadReport"
        />
      </div>
    </main>
  </div>
</template>
