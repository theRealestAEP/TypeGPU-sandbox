import { d, std, tgpu } from 'typegpu';
import type { TgpuComputePass, TgpuRoot } from 'typegpu';
import { parseDepthBundle } from '../vendor/depthart/bundle.ts';
import { DepthInferencePlan } from '../vendor/depthart/depthart.ts';
import { DepthDisparityRangeEstimator } from '../vendor/depthart/disparity-range.ts';
import { FIELD_RES } from '../sim/schemas.ts';
import type { SingleChannelTexture } from '../gpu/blur.ts';
import type { CameraFrame, DepthSource } from './depth-field.ts';
import { fetchModel, modelVariant, type ModelSize } from './model-store.ts';

const NORMALIZE_WORKGROUP = 64;
const RESAMPLE_WORKGROUP = 8;
const RESAMPLE_WORKGROUPS = Math.ceil(FIELD_RES / RESAMPLE_WORKGROUP);

/**
 * How fast the stabilised disparity range chases the current frame. Slow on
 * purpose: the model emits *relative* depth, so the range shifts whenever
 * anything enters the scene, and a fast blend makes the whole world breathe
 * in and out of the liquid plane.
 */
const RANGE_BLEND = 0.12;

/** Temporal filter on the normalised depth: gentle when still, fast when moving. */
const STILL_ALPHA = 0.32;
const MOVING_ALPHA = 0.8;
const MOTION_LOW = 0.02;
const MOTION_HIGH = 0.09;

const NormalizeParams = d.struct({
  outputSize: d.vec2u,
  reset: d.u32,
});

const rangeLayout = tgpu.bindGroupLayout({
  params: { uniform: NormalizeParams },
  frameRange: { storage: d.vec2f, access: 'readonly' },
  stableRange: { storage: d.vec2f, access: 'mutable' },
});

const normalizeLayout = tgpu.bindGroupLayout({
  params: { uniform: NormalizeParams },
  disparity: { storage: d.arrayOf(d.vec4f), access: 'readonly' },
  stableRange: { storage: d.vec2f, access: 'readonly' },
  history: { storage: d.arrayOf(d.f32), access: 'mutable' },
});

const resampleLayout = tgpu.bindGroupLayout({
  params: { uniform: NormalizeParams },
  history: { storage: d.arrayOf(d.f32), access: 'readonly' },
  target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
});

const stabilizeKernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
  const low = rangeLayout.$.frameRange.x;
  const high = std.max(rangeLayout.$.frameRange.y, low + 0.001);

  if (rangeLayout.$.params.reset !== 0) {
    rangeLayout.$.stableRange = d.vec2f(low, high);
    return;
  }

  rangeLayout.$.stableRange = d.vec2f(
    std.mix(rangeLayout.$.stableRange.x, low, RANGE_BLEND),
    std.mix(rangeLayout.$.stableRange.y, high, RANGE_BLEND),
  );
});

const normalizeKernel = tgpu.computeFn({
  workgroupSize: [NORMALIZE_WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  const size = normalizeLayout.$.params.outputSize;
  if (gid.x >= size.x * size.y) {
    return;
  }

  const low = normalizeLayout.$.stableRange.x;
  const span = std.max(normalizeLayout.$.stableRange.y - low, 0.001);
  const disparity = normalizeLayout.$.disparity[gid.x].x;

  let normalized = d.f32(0);
  // The model can emit NaN; this self-comparison is the cheapest way to spot it.
  if (disparity === disparity) {
    normalized = std.saturate((disparity - low) / span);
  }

  let filtered = d.f32(normalized);
  if (normalizeLayout.$.params.reset === 0) {
    const previous = normalizeLayout.$.history[gid.x];
    const motion = std.smoothstep(MOTION_LOW, MOTION_HIGH, std.abs(normalized - previous));
    filtered = std.mix(previous, normalized, std.mix(STILL_ALPHA, MOVING_ALPHA, motion));
  }

  normalizeLayout.$.history[gid.x] = filtered;
});

const historyAt = (coord: d.v2i) => {
  'use gpu';
  const size = d.vec2i(resampleLayout.$.params.outputSize);
  const clamped = std.clamp(coord, d.vec2i(0), size - 1);
  return resampleLayout.$.history[d.u32(clamped.y) * d.u32(size.x) + d.u32(clamped.x)];
};

/** Bilinear resample from the model's own grid onto the fixed simulation field. */
const resampleKernel = tgpu.computeFn({
  workgroupSize: [RESAMPLE_WORKGROUP, RESAMPLE_WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= FIELD_RES || gid.y >= FIELD_RES) {
    return;
  }
  const uv = (d.vec2f(gid.xy) + 0.5) / d.f32(FIELD_RES);
  const source = uv * d.vec2f(resampleLayout.$.params.outputSize) - 0.5;
  const corner = d.vec2i(std.floor(source));
  const blend = source - std.floor(source);

  const top = std.mix(historyAt(corner), historyAt(corner + d.vec2i(1, 0)), blend.x);
  const bottom = std.mix(
    historyAt(corner + d.vec2i(0, 1)),
    historyAt(corner + d.vec2i(1, 1)),
    blend.x,
  );

  // Channel two is coverage; the model has an answer for every texel.
  std.textureStore(
    resampleLayout.$.target,
    gid.xy,
    d.vec4f(std.mix(top, bottom, blend.y), 1, 0, 1),
  );
});

export interface ModelDepthSource extends DepthSource {
  /** Download, compile and attach a bundle. Safe to call again to switch size. */
  load(size: ModelSize, signal: AbortSignal): Promise<void>;
  readonly ready: boolean;
}

export function createModelDepth(root: TgpuRoot, depth: SingleChannelTexture): ModelDepthSource {
  const hasShaderF16 = root.device.features.has('shader-f16');
  const estimator = new DepthDisparityRangeEstimator(root);
  const frameRange = root.createBuffer(d.vec2f, d.vec2f(0, 1)).$usage('storage');
  const stableRange = root.createBuffer(d.vec2f, d.vec2f(0, 1)).$usage('storage');
  const params = root
    .createBuffer(NormalizeParams, { outputSize: d.vec2u(1), reset: 1 })
    .$usage('uniform');

  const target = depth.createView(d.textureStorage2d('rgba16float', 'write-only'));
  const rangeBindGroup = root.createBindGroup(rangeLayout, { params, frameRange, stableRange });

  const stabilize = root.createComputePipeline({ compute: stabilizeKernel });
  const normalize = root.createComputePipeline({ compute: normalizeKernel });
  const resample = root.createComputePipeline({ compute: resampleKernel });

  let plan: DepthInferencePlan | undefined;
  let attached:
    | {
        readonly disparity: ReturnType<typeof createDisparityBuffer>;
        readonly history: ReturnType<typeof createHistoryBuffer>;
        readonly normalizeBindGroup: ReturnType<typeof root.createBindGroup>;
        readonly resampleBindGroup: ReturnType<typeof root.createBindGroup>;
        readonly workgroups: number;
      }
    | undefined;
  let firstFrame = true;

  function createDisparityBuffer(pixels: number, source: GPUBuffer) {
    return root.createBuffer(d.arrayOf(d.vec4f, pixels), source).$usage('storage');
  }

  function createHistoryBuffer(pixels: number) {
    return root.createBuffer(d.arrayOf(d.f32, pixels)).$usage('storage');
  }

  function detach(): void {
    estimator.detach();
    attached?.disparity.destroy();
    attached?.history.destroy();
    attached = undefined;
    plan?.destroy();
    plan = undefined;
  }

  function attach(next: DepthInferencePlan): void {
    detach();
    const [width, height] = next.outputSize;
    const pixels = width * height;

    const disparity = createDisparityBuffer(pixels, next.outputBuffer);
    const history = createHistoryBuffer(pixels);

    attached = {
      disparity,
      history,
      normalizeBindGroup: root.createBindGroup(normalizeLayout, {
        params,
        disparity,
        stableRange,
        history,
      }),
      resampleBindGroup: root.createBindGroup(resampleLayout, { params, history, target }),
      workgroups: Math.ceil(pixels / NORMALIZE_WORKGROUP),
    };

    estimator.attach(disparity, frameRange, pixels);
    params.write({ outputSize: d.vec2u(width, height), reset: 1 });
    plan = next;
    firstFrame = true;
  }

  return {
    minIntervalMs: 100,

    get ready() {
      return plan !== undefined;
    },

    async initAsync() {
      await Promise.all([
        estimator.initAsync(),
        stabilize.initAsync(),
        normalize.initAsync(),
        resample.initAsync(),
      ]);
    },

    async load(size, signal) {
      const variant = modelVariant(size, hasShaderF16);
      if (!variant) {
        throw new Error(`The ${size} model needs shader-f16, which this device lacks.`);
      }
      const bundle = parseDepthBundle(await fetchModel(variant, signal));
      const next = new DepthInferencePlan(root, bundle);
      try {
        await next.initAsync();
      } catch (error) {
        next.destroy();
        throw error;
      }
      attach(next);
    },

    encode(pass: TgpuComputePass, frame: CameraFrame | undefined) {
      if (!plan || !attached || !frame) {
        return;
      }
      params.patch({ reset: firstFrame ? 1 : 0 });

      plan.encodeFrame(pass, frame.texture, {
        uvTransform: frame.uvTransform,
        mirrorX: frame.mirror,
        swapAxes: frame.swapAxes,
      });
      estimator.encode(pass);
      stabilize.with(pass).with(rangeBindGroup).dispatchWorkgroups(1);
      normalize
        .with(pass)
        .with(attached.normalizeBindGroup)
        .dispatchWorkgroups(attached.workgroups);
      resample
        .with(pass)
        .with(attached.resampleBindGroup)
        .dispatchWorkgroups(RESAMPLE_WORKGROUPS, RESAMPLE_WORKGROUPS);

      firstFrame = false;
    },

    destroy() {
      detach();
      estimator.destroy();
      frameRange.destroy();
      stableRange.destroy();
      params.destroy();
    },
  };
}
