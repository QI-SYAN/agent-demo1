
前端使用
cd /d d:\bishe\LangGraph\video_tampering_agent\web
npm install
npm run dev

后端运行
cd /d d:\bishe\LangGraph\video_tampering_agent
D:\miniconda\envs\Langgraph\python.exe -m pip install -e .
D:\miniconda\envs\Langgraph\python.exe -m uvicorn video_tampering_agent.service:create_app --factory --host 127.0.0.1 --port 8000 --reload


App.vue：负责页面布局（四块怎么摆）、负责状态汇总（当前文件、检测结果、报告状态）
components/UploadPanel.vue：只负责上传 UI + 点击开始
components/ResultPanel.vue：只负责展示结果（类型/置信度/风险/可疑时间段/视频预留）
components/LogPanel.vue：只负责展示日志并自动滚动
components/ReportPanel.vue：只负责报告类型选择、生成/下载按钮与状态展示
composables/useMockDetection.js：只负责“模拟检测流程逻辑”（日志节奏、3 秒后出结果）
composables/useDetection.js：负责调用后端 /detect，并把 audit_trail、规划文本、反思信息优先灌入日志区


典型的 Vite + Vue3 工程渲染链路如下（你可以用这个去对照你的文件）：
index.html
这是浏览器最先加载的文件。它会包含一行类似：
script type="module" src="/src/main.js" 或 /src/main.ts

main.js（或 main.ts）
做三件事：
createApp(App) 创建 Vue 应用
app.use(ElementPlus) 注册 Element Plus
app.mount('#app') 把 App 挂到页面某个 div 上

App.vue
这是“根组件”。你做的四区域布局（上传/结果/日志/报告）通常就放在这里或由它组合子组件。
web/src/components/*.vue
上传区、结果区、日志区、报告区分别封装成组件，在 App.vue 中引用并组合。

web/src/composables/*
放可复用的“逻辑函数”（不是 UI）。你现在的“模拟检测日志 + 3 秒出结果”就适合放在 composable 里。