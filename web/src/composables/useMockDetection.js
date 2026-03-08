// web/src/composables/useMockDetection.js
// 这个 composable 负责“静态演示”的检测流程：
// - 维护日志列表
// - 维护是否正在检测
// - 用 setTimeout 模拟检测进度与结果产出
// 注意：这里没有任何真实接口调用，便于后续替换为真实 API。

import { nextTick, ref } from 'vue'

/**
 * @typedef {'INFO' | 'WARNING' | 'ERROR'} LogLevel
 * @typedef {{ time: string; level: LogLevel; message: string }} LogItem
 */

function pad2(n) {
  return String(n).padStart(2, '0')
}

function nowHHmmss() {
  const d = new Date()
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`
}

export function useMockDetection() {
  /** @type {import('vue').Ref<LogItem[]>} */
  const logs = ref([])
  const isRunning = ref(false)

  function resetLogs() {
    logs.value = []
  }

  /**
   * 追加一条日志
   * @param {LogLevel} level
   * @param {string} message
   */
  function pushLog(level, message) {
    logs.value.push({
      time: nowHHmmss(),
      level,
      message,
    })
  }

  /**
   * 模拟一次检测流程。
   * @param {{ onFinish?: () => void }} options
   */
  function startMockDetection(options = {}) {
    if (isRunning.value) return

    isRunning.value = true
    pushLog('INFO', '开始检测：初始化任务...')

    // 这里用固定的“3~5条”日志模拟。
    // 你后续接真实 API 时，只需要把这些 setTimeout 替换成接口返回即可。
    setTimeout(() => pushLog('INFO', '解析视频元数据：分辨率、帧率、编码信息'), 300)
    setTimeout(() => pushLog('INFO', '运行车辆检测与跟踪：提取轨迹与关键帧'), 900)
    setTimeout(() => pushLog('WARNING', '发现疑似轨迹中断：00:12.4 附近'), 1500)
    setTimeout(() => pushLog('INFO', '调用视觉模型复核：对比关键帧差异'), 2200)

    setTimeout(async () => {
      pushLog('INFO', '检测完成：已生成初步结论与证据')
      isRunning.value = false

      // 让 UI 有机会在同一 tick 内完成渲染
      await nextTick()

      if (typeof options.onFinish === 'function') {
        options.onFinish()
      }
    }, 3000)
  }

  return {
    logs,
    isRunning,
    pushLog,
    resetLogs,
    startMockDetection,
  }
}
