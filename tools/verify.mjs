// Headless smoke test: load the page in Chromium with WebGPU enabled, wait for
// the solver to run, and report status plus probe counters. Exits non-zero on
// a frame failure, a shader compile error, or a device-lost event.
import { chromium } from 'playwright-core';

const executablePath = process.env.CHROME_PATH;
const browser = await chromium.launch({
  executablePath,
  headless: true,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--use-angle=vulkan',
    '--use-webgpu-adapter=swiftshader',
  ],
});

const page = await browser.newPage();
const failures = [];
page.on('console', (message) => {
  const text = message.text();
  if (message.type() === 'error' || /WGSL|shader|compil/i.test(text)) {
    failures.push(`${message.location()?.url ?? ''}: ${text}`);
  }
});
page.on('pageerror', (error) => failures.push(String(error)));
page.on('requestfailed', (request) => failures.push(`REQUEST FAILED: ${request.url()}`));

await page.goto('http://127.0.0.1:5173/?quality=low&particles=20000', {
  waitUntil: 'domcontentloaded',
});

await page.waitForFunction(() => globalThis.probe !== undefined, null, { timeout: 30000 });
// Water phase: pour for a while and let the solver settle.
await page.keyboard.press('Space');
await page.evaluate(() => globalThis.pump(120));
await page.waitForFunction(
  () => globalThis.counters().solverSteps > 60,
  null,
  { timeout: 30000 },
);

// Smoke phase: switch medium and run the grid, bake and march. The SwiftShader
// device in headless cannot service readbacks between frames, so verification
// here is "dispatched without errors", checked via the failure log below.
const encodesBeforeSmoke = await page.evaluate(() => globalThis.counters().encodes);
await page.keyboard.press('n');
await page.evaluate(() => globalThis.pump(40));
await page.waitForFunction(
  (before) => globalThis.counters().encodes > before,
  encodesBeforeSmoke,
  { timeout: 30000 },
);

const counters = await page.evaluate(() => globalThis.counters());
const status = await page.evaluate(() => {
  const element = document.getElementById('status');
  return element ? element.textContent : '';
});

console.log(JSON.stringify({ counters, status, failures }, null, 2));
await browser.close();
if (failures.length > 0) {
  process.exit(1);
}
