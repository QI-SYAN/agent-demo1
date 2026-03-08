import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/artifacts': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/upload': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/detect': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
