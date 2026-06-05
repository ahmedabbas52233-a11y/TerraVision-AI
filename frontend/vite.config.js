import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // Proxy /v1/* to the FastAPI backend during development
      '/v1': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: { vendor: ['react', 'react-dom'], leaflet: ['leaflet'] },
      },
    },
  },
  define: {
    // Inject API base URL at build time via env var; defaults to same origin
    __API_BASE__: JSON.stringify(process.env.VITE_API_URL || ''),
  },
})
