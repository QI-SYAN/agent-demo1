<script setup>
// web/src/components/UploadPanel.vue
// 1) 视频上传区：
// - 使用 el-upload，支持拖拽/点击
// - 静态演示：用自定义 http-request 模拟“上传进度条”
// - 上传完成后，向父组件抛出 uploaded 事件（携带 fileName/videoUrl）

import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled, VideoPlay, Delete } from '@element-plus/icons-vue'

const props = defineProps({
  // 是否正在检测（检测期间禁用上传/开始按钮）
  isRunning: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits(['uploaded', 'removed', 'start-detect'])

// 本地选中的文件（静态演示，不会真正上传到服务器）
const fileList = ref([])
const videoUrl = ref('')
const videoLocator = ref('')

const accept = '.mp4,.avi,.mov,video/mp4,video/quicktime,video/x-msvideo'

const hasVideo = computed(() => !!videoUrl.value)

// 选择文件后，生成本地预览 URL
function handleChange(uploadFile, uploadFiles) {
  fileList.value = uploadFiles
  const raw = uploadFile?.raw
  if (!raw) return

  // 限制大小 500MB（静态提示 + 简单校验）
  const maxBytes = 500 * 1024 * 1024
  if (raw.size > maxBytes) {
    ElMessage.error('文件过大：最大支持 500MB')
    fileList.value = []
    videoUrl.value = ''
    return
  }

  // 释放旧的 URL，避免内存泄露
  if (videoUrl.value) {
    URL.revokeObjectURL(videoUrl.value)
  }

  videoUrl.value = URL.createObjectURL(raw)
  videoLocator.value = ''
}

function handleRemove() {
  // 清理预览 URL
  if (videoUrl.value) {
    URL.revokeObjectURL(videoUrl.value)
  }
  videoUrl.value = ''
  videoLocator.value = ''
  fileList.value = []
  emit('removed')
}

// 自定义上传请求：真实上传到后端 /upload，并用 XHR 获取上传进度
// 注意：我们在 vite.config.js 里配置了代理，因此这里用相对路径即可。
function realHttpRequest(options) {
  const xhr = new XMLHttpRequest()
  const form = new FormData()
  form.append('file', options.file)

  xhr.open('POST', '/upload')

  xhr.upload.onprogress = (evt) => {
    if (!evt.lengthComputable) return
    const percent = Math.round((evt.loaded / evt.total) * 100)
    options.onProgress?.({ percent })
  }

  xhr.onerror = () => {
    options.onError?.(new Error('upload failed'))
  }

  xhr.onload = () => {
    if (xhr.status < 200 || xhr.status >= 300) {
      options.onError?.(new Error(`upload failed: ${xhr.status}`))
      return
    }
    try {
      const json = JSON.parse(xhr.responseText)
      options.onSuccess?.(json)
    } catch {
      options.onError?.(new Error('invalid upload response'))
    }
  }

  xhr.send(form)
}

function handleUploadSuccess(response, uploadFile) {
  // 后端返回：{ video_locator, file_name, content_type }
  videoLocator.value = response?.video_locator || ''

  if (!videoLocator.value) {
    ElMessage.warning('上传成功，但未获取到 video_locator（请检查后端返回）')
  }

  emit('uploaded', {
    fileName: response?.file_name || uploadFile?.name || '未命名视频',
    videoUrl: videoUrl.value,
    videoLocator: videoLocator.value,
  })
}

function handleStartDetect() {
  emit('start-detect')
}

// 当检测开始时，禁用上传操作（父组件控制 props.isRunning）
const uploadDisabled = computed(() => props.isRunning)

</script>

<template>
  <el-card shadow="hover" class="panel">
    <template #header>
      <div class="panel__header">
        <div class="panel__title">
          <el-icon><UploadFilled /></el-icon>
          <span>视频上传</span>
        </div>
        <div class="panel__hint">支持 MP4 / AVI / MOV，最大 500MB</div>
      </div>
    </template>

    <el-upload
      class="uploader"
      drag
      action="#"
      :accept="accept"
      :limit="1"
      :file-list="fileList"
      :disabled="uploadDisabled"
      :http-request="realHttpRequest"
      :on-change="handleChange"
      :on-remove="handleRemove"
      :on-success="handleUploadSuccess"
      :auto-upload="true"
    >
      <el-icon class="uploader__icon"><UploadFilled /></el-icon>
      <div class="el-upload__text">
        将视频拖拽到此处，或 <em>点击选择</em>
      </div>
      <template #tip>
        <div class="uploader__tip">上传到本地服务：/upload（需先启动后端）</div>
      </template>
    </el-upload>

    <div v-if="hasVideo" class="preview">
      <div class="preview__meta">
        <div class="preview__file">
          <el-icon><VideoPlay /></el-icon>
          <span class="preview__name">{{ fileList?.[0]?.name }}</span>
        </div>
        <el-button :disabled="uploadDisabled" type="danger" plain :icon="Delete" @click="handleRemove">
          移除
        </el-button>
      </div>

      <!-- 简易缩略预览：直接用 video 展示首帧/可播放（静态演示） -->
      <video class="preview__video" :src="videoUrl" muted controls></video>
    </div>

    <div class="actions">
      <el-button
        type="primary"
        size="large"
        :disabled="!hasVideo || isRunning"
        @click="handleStartDetect"
      >
        开始检测
      </el-button>
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
  color: var(--deep-gray);
}

.panel__hint {
  font-size: 12px;
  color: #909399;
}

.uploader {
  width: 100%;
}

.uploader__icon {
  font-size: 28px;
  color: var(--primary);
}

.uploader__tip {
  font-size: 12px;
  color: #909399;
}

.preview {
  margin-top: 12px;
  padding: 12px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  background: #fff;
}

.preview__meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.preview__file {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.preview__name {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.preview__video {
  width: 100%;
  max-height: 220px;
  border-radius: 8px;
  background: #000;
}

.actions {
  margin-top: 14px;
  display: flex;
  justify-content: flex-end;
}
</style>
