import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    outDir: '../../../../static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/generate': 'http://localhost:8000',
      '/adapters': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/metrics': 'http://localhost:8000',
      '/runs': 'http://localhost:8000',
      '/chat': 'http://localhost:8000',
    }
  }
})
