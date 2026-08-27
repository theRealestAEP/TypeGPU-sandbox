import { d, std, tgpu } from 'typegpu';
import type { StorageFlag, TgpuBuffer, TgpuComputePass, TgpuRoot, TgpuSampler } from 'typegpu';
import type { SingleChannelTexture } from '../gpu/blur.ts';

/**
 * Finds every place the scene could hold liquid, from the depth field alone.
 *
 * A cup, a bowl, a sink and a bathtub are the same thing to a heightfield: a
 * closed depression under gravity. Flooding the gravitational potential from
 * the frame's edges gives, per texel, the highest level it can hold before the
 * liquid finds a path out - and a vessel is a connected pocket where that
 * level sits above the surface. Its rim is the pocket's boundary, its depth is
 * the deepest gap, and its capacity is the integral. No detector and no labels:
 * anything concave enough to hold water is found, including things no object
 * class would name.
 *
 * The flood runs on the CPU over a small downsample. At this size it is well
 * under a millisecond, which makes it fine to re-run continuously on video.
 */

export const VESSEL_RES = 64;

const sampleLayout = tgpu.bindGroupLayout({
  surface: { texture: d.texture2d(d.f32) },
  fieldSampler: { sampler: 'filtering' },
  out: { storage: d.arrayOf(d.f32, VESSEL_RES * VESSEL_RES), access: 'mutable' },
});

const sampleKernel = tgpu.computeFn({
  workgroupSize: [8, 8],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= d.u32(VESSEL_RES) || gid.y >= d.u32(VESSEL_RES)) {
    return;
  }
  const uv = (d.vec2f(d.f32(gid.x), d.f32(gid.y)) + 0.5) / d.f32(VESSEL_RES);
  sampleLayout.$.out[gid.y * d.u32(VESSEL_RES) + gid.x] = std.textureSampleLevel(
    sampleLayout.$.surface,
    sampleLayout.$.fieldSampler,
    uv,
    0,
  ).x;
});

export interface Vessel {
  /** Bounding box of the pocket, in frame fractions. */
  readonly box: { x0: number; y0: number; x1: number; y1: number };
  /** Centre of capacity, in frame fractions. */
  readonly at: { x: number; y: number };
  /** Deepest the liquid can stand before it spills. */
  readonly depth: number;
  /** Integral of holdable depth over the pocket - relative volume. */
  readonly capacity: number;
  /** Fraction of the frame the pocket covers. */
  readonly area: number;
}

/** Ignore pockets shallower than this; the field's noise makes them. */
const MIN_DEPTH = 0.02;
/** Ignore pockets smaller than this fraction of the frame. */
const MIN_AREA = 0.002;
const MAX_VESSELS = 6;

/**
 * Priority-flood from the frame border. Identical in spirit to the offline
 * scorer; kept dependency-free and allocation-light so it can run every few
 * hundred milliseconds without anyone noticing.
 */
export function findVessels(
  depth: ArrayLike<number>,
  down: readonly [number, number, number],
  depthScale: number,
): Vessel[] {
  const n = VESSEL_RES;
  const [dx, dy, dz] = down;
  const potential = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = (i + 0.5) / n;
      const y = (j + 0.5) / n;
      potential[j * n + i] = -(dx * x + dy * y + dz * depth[j * n + i] * depthScale);
    }
  }

  // Binary min-heap over cell indices, keyed by fill level.
  const filled = new Float64Array(n * n).fill(Infinity);
  const seen = new Uint8Array(n * n);
  const heap: number[] = [];
  const less = (a: number, b: number) => filled[a] < filled[b];
  const push = (k: number) => {
    heap.push(k);
    let c = heap.length - 1;
    while (c > 0) {
      const p = (c - 1) >> 1;
      if (less(heap[c], heap[p])) {
        [heap[p], heap[c]] = [heap[c], heap[p]];
        c = p;
      } else {
        break;
      }
    }
  };
  const pop = (): number => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0 && last !== undefined) {
      heap[0] = last;
      let p = 0;
      for (;;) {
        const l = p * 2 + 1;
        const r = l + 1;
        let m = p;
        if (l < heap.length && less(heap[l], heap[m])) m = l;
        if (r < heap.length && less(heap[r], heap[m])) m = r;
        if (m === p) break;
        [heap[p], heap[m]] = [heap[m], heap[p]];
        p = m;
      }
    }
    return top;
  };

  for (let j = 0; j < n; j++) {
    for (const i of [0, n - 1]) {
      const k = j * n + i;
      if (!seen[k]) {
        seen[k] = 1;
        filled[k] = potential[k];
        push(k);
      }
    }
  }
  for (let i = 0; i < n; i++) {
    for (const j of [0, n - 1]) {
      const k = j * n + i;
      if (!seen[k]) {
        seen[k] = 1;
        filled[k] = potential[k];
        push(k);
      }
    }
  }
  while (heap.length > 0) {
    const k = pop();
    const i = k % n;
    const j = (k - i) / n;
    const spread = (a: number, b: number) => {
      if (a < 0 || a >= n || b < 0 || b >= n) return;
      const m = b * n + a;
      if (seen[m]) return;
      seen[m] = 1;
      filled[m] = Math.max(potential[m], filled[k]);
      push(m);
    };
    spread(i + 1, j);
    spread(i - 1, j);
    spread(i, j + 1);
    spread(i, j - 1);
  }

  // Connected pockets of holdable water, by flood fill over a boolean mask.
  const wet = new Uint8Array(n * n);
  for (let k = 0; k < n * n; k++) {
    if (filled[k] - potential[k] > MIN_DEPTH) wet[k] = 1;
  }
  const vessels: Vessel[] = [];
  const stack: number[] = [];
  for (let start = 0; start < n * n; start++) {
    if (wet[start] !== 1) continue;
    let x0 = n;
    let y0 = n;
    let x1 = 0;
    let y1 = 0;
    let capacity = 0;
    let cells = 0;
    let cx = 0;
    let cy = 0;
    let deepest = 0;
    stack.push(start);
    wet[start] = 2;
    while (stack.length > 0) {
      const k = stack.pop();
      if (k === undefined) break;
      const i = k % n;
      const j = (k - i) / n;
      const w = filled[k] - potential[k];
      cells += 1;
      capacity += w;
      cx += i * w;
      cy += j * w;
      if (w > deepest) deepest = w;
      if (i < x0) x0 = i;
      if (j < y0) y0 = j;
      if (i > x1) x1 = i;
      if (j > y1) y1 = j;
      const walk = (a: number, b: number) => {
        if (a < 0 || a >= n || b < 0 || b >= n) return;
        const m = b * n + a;
        if (wet[m] === 1) {
          wet[m] = 2;
          stack.push(m);
        }
      };
      walk(i + 1, j);
      walk(i - 1, j);
      walk(i, j + 1);
      walk(i, j - 1);
    }
    const area = cells / (n * n);
    if (area < MIN_AREA || deepest < MIN_DEPTH * 1.5) continue;
    vessels.push({
      box: { x0: x0 / n, y0: y0 / n, x1: (x1 + 1) / n, y1: (y1 + 1) / n },
      at: { x: cx / Math.max(capacity, 1e-9) / n, y: cy / Math.max(capacity, 1e-9) / n },
      depth: deepest,
      capacity: capacity / (n * n),
      area,
    });
  }
  vessels.sort((a, b) => b.capacity - a.capacity);
  return vessels.slice(0, MAX_VESSELS);
}

export interface VesselProbe {
  readonly buffer: TgpuBuffer<d.WgslArray<d.F32>> & StorageFlag;
  initAsync(): Promise<void>;
  /** Refreshes the downsample the CPU flood reads. */
  encode(pass: TgpuComputePass): void;
  destroy(): void;
}

export function createVesselProbe(
  root: TgpuRoot,
  surface: SingleChannelTexture,
  fieldSampler: TgpuSampler,
): VesselProbe {
  const buffer = root
    .createBuffer(d.arrayOf(d.f32, VESSEL_RES * VESSEL_RES))
    .$usage('storage');
  const bindGroup = root.createBindGroup(sampleLayout, {
    surface: surface.createView(),
    fieldSampler,
    out: buffer,
  });
  const pipeline = root.createComputePipeline({ compute: sampleKernel });
  const groups = Math.ceil(VESSEL_RES / 8);

  return {
    buffer,

    async initAsync() {
      await pipeline.initAsync();
    },

    encode(pass) {
      pipeline.with(pass).with(bindGroup).dispatchWorkgroups(groups, groups);
    },

    destroy() {
      buffer.destroy();
    },
  };
}
