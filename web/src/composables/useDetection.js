// web/src/composables/useDetection.js
// 真实检测 composable：
// - 调用后端 FastAPI 的 /detect 接口
// - 维护 logs / isRunning
// - 由于后端暂未提供实时日志流，这里用“前端模拟日志 + 等待接口返回”组合实现

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

export function useDetection() {
  /** @type {import('vue').Ref<LogItem[]>} */
  const logs = ref([])
  const isRunning = ref(false)

  function resetLogs() {
    logs.value = []
  }

  /**
   * @param {LogLevel} level
   * @param {string} message
   */
  function pushLog(level, message) {
    logs.value.push({ time: nowHHmmss(), level, message })
  }

  function appendAuditTrail(auditTrail = []) {
    for (const entry of auditTrail) {
      const text = String(entry || '')
      let level = 'INFO'
      if (/error|失败|异常/i.test(text)) level = 'ERROR'
      else if (/warning|可疑|深查|人工复核|run_deferred/i.test(text)) level = 'WARNING'
      pushLog(level, text)
    }
  }

  function appendPlanningSummary(resp = {}) {
    const planningText = String(resp?.planning_text || '').trim()
    const initialTools = Array.isArray(resp?.initial_tools) ? resp.initial_tools : []
    const deferredTools = Array.isArray(resp?.deferred_tools) ? resp.deferred_tools : []
    const reflectionNotes = Array.isArray(resp?.reflection_notes) ? resp.reflection_notes : []
    const reflectionDecision = resp?.reflection_decision
    const evidence = Array.isArray(resp?.evidence) ? resp.evidence : []

    if (initialTools.length || deferredTools.length) {
      pushLog('INFO', `工具计划：初筛=[${initialTools.join(', ') || '无'}]；深查=[${deferredTools.join(', ') || '无'}]`)
    }

    if (planningText) {
      pushLog('INFO', '规划文本：')
      const lines = planningText
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .slice(0, 12)
      for (const line of lines) {
        pushLog('INFO', line)
      }
      if (planningText.split(/\r?\n/).filter(Boolean).length > 12) {
        pushLog('INFO', '（规划文本较长，前端日志区已截取前 12 行展示）')
      }
    }

    if (reflectionDecision) {
      pushLog('WARNING', `反思决策：${reflectionDecision}`)
    }
    for (const note of reflectionNotes) {
      pushLog('WARNING', `反思说明：${note}`)
    }

    if (evidence.length) {
      pushLog('INFO', `证据摘要：共 ${evidence.length} 条工具结果`)
      for (const item of evidence.slice(0, 4)) {
        const tool = item?.tool || item?.tool_name || 'unknown'
        const status = item?.status || 'ok'
        const detail = item?.detail
        let detailText = ''
        if (typeof detail === 'string') detailText = detail
        else if (detail && typeof detail === 'object') detailText = JSON.stringify(detail)
        pushLog(status === 'error' ? 'ERROR' : 'INFO', `工具 ${tool} (${status})：${detailText}`)
      }
      if (evidence.length > 4) {
        pushLog('INFO', '（证据较多，日志区仅展示前 4 条摘要）')
      }
    }
  }

  /**
   * 启动一次真实检测。
   * @param {{ videoLocator: string; prompt?: string; onFinish?: (resp: any) => void }} options
   */
  async function startDetection(options) {
    if (isRunning.value) return
    if (!options?.videoLocator) {
      pushLog('ERROR', '缺少 videoLocator：请先上传视频并等待上传完成')
      return
    }

    isRunning.value = true
    pushLog('INFO', '开始检测：向后端提交任务...')

    // 在等待后端返回期间，模拟输出几条进度日志（便于 UI 演示）
    const timers = []
    timers.push(setTimeout(() => pushLog('INFO', '解析视频元数据...'), 300))
    timers.push(setTimeout(() => pushLog('INFO', '执行车辆检测/轨迹分析...'), 900))
    timers.push(setTimeout(() => pushLog('INFO', '执行背景残差与视觉复核...'), 1500))

    try {
      const res = await fetch('/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_locator: options.videoLocator,
          prompt: options.prompt || '请重点判断是否存在车辆被删除/消失的篡改迹象，并给出理由。',
        }),
      })

      if (!res.ok) {
        throw new Error(`detect failed: ${res.status}`)
      }

      const json = await res.json()

      pushLog('INFO', '检测完成：已收到后端结果')
      appendAuditTrail(json?.audit_trail || [])
      appendPlanningSummary(json)
      isRunning.value = false
      await nextTick()

      if (typeof options.onFinish === 'function') {
        options.onFinish(json)
      }
    } catch (err) {
      pushLog('ERROR', `检测失败：${String(err?.message || err)}`)
      isRunning.value = false
    } finally {
      timers.forEach((t) => clearTimeout(t))
    }
  }

  return {
    logs,
    isRunning,
    resetLogs,
    pushLog,
    startDetection,
  }
}
