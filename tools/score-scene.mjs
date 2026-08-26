// Scores a candidate photo for the demo: can liquid actually gather in it?
//
// A scene holds water where the gravitational potential has a closed
// depression - somewhere downhill in every direction. Depth alone does not tell
// you that, because "downhill" depends on which way gravity points, and the
// app measures that from the scene itself. So this runs the real pipeline: it
// loads the image through the app's own depth model, reads back the gravity it
// estimated, then floods the potential to find every basin and how deep each
// one can get before it spills.
//
//   node tools/score-scene.mjs <image> [--region x0,x1,y0,y1]
//
// Needs the dev server running (`npm run dev`) and ffmpeg on PATH.
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright-core';

const file = process.argv[2];
if (!file) {
  console.error('usage: node tools/score-scene.mjs <image> [--region x0,x1,y0,y1]');
  process.exit(2);
}
const regionArg = process.argv.indexOf('--region');
const region =
  regionArg > 0 ? process.argv[regionArg + 1].split(',').map(Number) : [0.02, 0.98, 0.02, 0.98];

const N = 150;
const CHROME =
  process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--enable-unsafe-webgpu'],
});
const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
await page.goto('http://127.0.0.1:5173/?quality=high&particles=20000', {
  waitUntil: 'domcontentloaded',
});
await page.waitForFunction(() => globalThis.probe !== undefined, null, { timeout: 90000 });

// Feed the candidate in through the app's own file picker, so it goes down
// exactly the path a dropped image would.
await page.setInputFiles('#picker', file);
await page.waitForTimeout(6000);
await page.evaluate(() => document.getElementById('drain').click());

const gravity = await page.evaluate(async () => {
  const { down } = await globalThis.gravityDir();
  return [down.x, down.y, down.z];
});

// The DEPTH view renders scene depth normalised to 0..1, so its pixels are the
// depth field itself rather than a picture of it.
//
// Captured with a real screenshot and decoded through ffmpeg. Drawing the WebGPU
// canvas into a 2D canvas and reading it back looks like it should work and
// silently yields black - which scored every scene at zero, including ones known
// to hold water.
await page.evaluate(() => document.querySelectorAll('#views button')[1].click());
await page.waitForTimeout(1500);
const work = mkdtempSync(join(tmpdir(), 'scene-score-'));
const shot = join(work, 'depth.png');
const raw = join(work, 'depth.gray');
await page.locator('#stage canvas').screenshot({ path: shot });
await browser.close();

execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', shot, '-vf', `scale=${N}:${N}`,
  '-pix_fmt', 'gray', '-f', 'rawvideo', raw]);
const grey = readFileSync(raw);
rmSync(work, { recursive: true, force: true });
if (grey.length < N * N) {
  console.error(`decoded ${grey.length} bytes, expected ${N * N}`);
  process.exit(1);
}
const depth = Array.from({ length: N * N }, (_, i) => grey[i] / 255);

const [gx, gy, gz] = gravity;
const at = (i, j) => depth[j * N + i];
const U = (i, j) => -(gx * ((i + 0.5) / N) + gy * ((j + 0.5) / N) + gz * at(i, j));

// Priority flood. filled[p] is the highest level p can hold before it finds a
// downhill path off the edge of the frame; filled - U is the water depth there.
const filled = new Float64Array(N * N).fill(Infinity);
const seen = new Uint8Array(N * N);
const heap = [];
const push = (v, i, j) => {
  heap.push([v, i, j]);
  let c = heap.length - 1;
  while (c > 0) {
    const p = (c - 1) >> 1;
    if (heap[p][0] <= heap[c][0]) break;
    [heap[p], heap[c]] = [heap[c], heap[p]];
    c = p;
  }
};
const pop = () => {
  const top = heap[0];
  const last = heap.pop();
  if (heap.length) {
    heap[0] = last;
    let p = 0;
    for (;;) {
      const l = p * 2 + 1;
      const r = l + 1;
      let m = p;
      if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
      if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
      if (m === p) break;
      [heap[p], heap[m]] = [heap[m], heap[p]];
      p = m;
    }
  }
  return top;
};
for (let j = 0; j < N; j++) {
  for (const i of [0, N - 1]) {
    const k = j * N + i;
    if (seen[k]) continue;
    seen[k] = 1;
    filled[k] = U(i, j);
    push(filled[k], i, j);
  }
}
for (let i = 0; i < N; i++) {
  for (const j of [0, N - 1]) {
    const k = j * N + i;
    if (seen[k]) continue;
    seen[k] = 1;
    filled[k] = U(i, j);
    push(filled[k], i, j);
  }
}
while (heap.length) {
  const [lv, i, j] = pop();
  for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    const a = i + di;
    const b = j + dj;
    if (a < 0 || a >= N || b < 0 || b >= N) continue;
    const k = b * N + a;
    if (seen[k]) continue;
    seen[k] = 1;
    filled[k] = Math.max(U(a, b), lv);
    push(filled[k], a, b);
  }
}

const [x0, x1, y0, y1] = region;
const cells = [];
for (let j = 0; j < N; j++) {
  for (let i = 0; i < N; i++) {
    const x = (i + 0.5) / N;
    const y = (j + 0.5) / N;
    if (x > x0 && x < x1 && y > y0 && y < y1) cells.push([i, j]);
  }
}
const wet = cells.filter(([i, j]) => filled[j * N + i] - U(i, j) > 0.002);
const deepest = wet.reduce((m, [i, j]) => Math.max(m, filled[j * N + i] - U(i, j)), 0);
const k = Math.abs(gz) / gy;

const pct = (v) => `${(v * 100).toFixed(1)}%`;
console.log(`\nscene: ${file}`);
console.log(`region: x ${x0}..${x1}, y ${y0}..${y1}\n`);
console.log(`  measured gravity     (${gx.toFixed(3)}, ${gy.toFixed(3)}, ${gz.toFixed(3)})`);
console.log(`  camera pitch         ${((Math.atan2(-gz, gy) * 180) / Math.PI).toFixed(0)} deg below level`);
console.log(`  relief ratio k       ${k.toFixed(2)}   (want 0.9 or more; 1.3 is ideal)`);
console.log(`  area that holds      ${pct(wet.length / cells.length)}`);
console.log(`  deepest water        ${deepest.toFixed(3)}\n`);

// Depth decides this, not area. Measured on real scenes: a bathroom shot from
// 21 degrees holds water across 31% of the tub and still reads as empty, because
// it is only 0.069 deep; a sink shot from 39 holds it across 6% and reads as a
// full basin, because it is 0.174 deep. Area tells you how much of the frame
// gets wet, depth tells you whether it looks like anything.
const verdict =
  deepest > 0.13
    ? 'GOOD - deep enough to read as a filled basin.'
    : k < 0.8
      ? `TOO FLAT AN ANGLE - the camera is only ${((Math.atan2(-gz, gy) * 180) / Math.PI).toFixed(0)} deg below level, so
           the basin is nearly edge-on and there is no downhill into it.
           Find a shot looking further down into the vessel.`
      : deepest > 0.07
        ? 'WEAK - it holds a film, but not enough to read as filled.'
        : 'POOR - liquid will run straight out of this one.';
console.log(`  ${verdict}\n`);

// Row profile, so it is obvious where in the frame the water can and cannot go.
console.log('  where it can hold water, by band:');
for (let band = 0; band < 20; band++) {
  const lo = y0 + ((y1 - y0) * band) / 20;
  const hi = y0 + ((y1 - y0) * (band + 1)) / 20;
  const inBand = cells.filter(([, j]) => {
    const y = (j + 0.5) / N;
    return y >= lo && y < hi;
  });
  if (!inBand.length) continue;
  const w = inBand.filter(([i, j]) => filled[j * N + i] - U(i, j) > 0.002).length;
  const f = w / inBand.length;
  console.log(`    y=${lo.toFixed(2)}  ${pct(f).padStart(6)} |${'#'.repeat(Math.round(f * 40))}`);
}
console.log();
