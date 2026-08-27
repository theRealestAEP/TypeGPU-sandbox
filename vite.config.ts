import { execSync } from 'node:child_process';
import { defineConfig } from 'vite';
import typegpu from 'unplugin-typegpu/vite';

let build = 'dev';
try {
  build = execSync('git rev-parse --short HEAD').toString().trim();
} catch {
  // Outside a git checkout the stamp just says dev.
}

export default defineConfig({
  plugins: [typegpu()],
  server: { host: '127.0.0.1', port: 5173 },
  define: { __BUILD__: JSON.stringify(build) },
});
