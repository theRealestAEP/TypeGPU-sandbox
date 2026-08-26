import { d, std, tgpu } from 'typegpu';
import type {
  SampledFlag,
  StorageFlag,
  TgpuComputePass,
  TgpuRoot,
  TgpuSampler,
  TgpuTexture,
} from 'typegpu';

export type SingleChannelTexture = TgpuTexture<{
  size: [number, number];
  format: 'rgba16float';
}> &
  SampledFlag &
  StorageFlag;

const WORKGROUP = 8;

export const blurLayout = tgpu.bindGroupLayout({
  source: { texture: d.texture2d(d.f32) },
  blurSampler: { sampler: 'filtering' },
  target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
});

const resolutionSlot = tgpu.slot<number>();
const radiusSlot = tgpu.slot<number>();
const tapsSlot = tgpu.slot<number[]>();
const axisSlot = tgpu.slot<d.v2f>();

export function gaussianTaps(radius: number, sigma: number): number[] {
  const taps: number[] = [];
  let total = 0;
  for (let offset = -radius; offset <= radius; offset++) {
    const weight = Math.exp(-(offset * offset) / (2 * sigma * sigma));
    taps.push(weight);
    total += weight;
  }
  return taps.map((weight) => weight / total);
}

const blurKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP, WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  const resolution = resolutionSlot.$;
  if (gid.x >= d.u32(resolution) || gid.y >= d.u32(resolution)) {
    return;
  }
  const centre = (d.vec2f(gid.xy) + 0.5) / d.f32(resolution);
  const step = axisSlot.$ / d.f32(resolution);

  // Two channels, because the fluid surface stores a weighted depth alongside
  // the weight itself and both have to be blurred together before the divide.
  let total = d.vec2f();
  for (const tap of tgpu.unroll(std.range(radiusSlot.$ * 2 + 1))) {
    const weight = tapsSlot.$[tap];
    total =
      total +
      std.textureSampleLevel(
        blurLayout.$.source,
        blurLayout.$.blurSampler,
        centre + step * (tap - radiusSlot.$),
        0,
      ).xy *
        weight;
  }

  std.textureStore(blurLayout.$.target, gid.xy, d.vec4f(total, 0, 1));
});

export interface BlurTargets {
  readonly resolution: number;
  readonly radius: number;
  readonly sigma: number;
  readonly source: SingleChannelTexture;
  /** Holds the horizontal pass; may not alias source or target. */
  readonly scratch: SingleChannelTexture;
  readonly target: SingleChannelTexture;
  readonly blurSampler: TgpuSampler;
}

export interface SeparableBlur {
  initAsync(): Promise<void>;
  encode(pass: TgpuComputePass): void;
}

/** Two-pass Gaussian. Both the wall field and the liquid surface run through it. */
export function createSeparableBlur(root: TgpuRoot, targets: BlurTargets): SeparableBlur {
  const taps = gaussianTaps(targets.radius, targets.sigma);
  const groups = Math.ceil(targets.resolution / WORKGROUP);

  const writeView = (texture: SingleChannelTexture) =>
    texture.createView(d.textureStorage2d('rgba16float', 'write-only'));

  const horizontalBindGroup = root.createBindGroup(blurLayout, {
    source: targets.source.createView(),
    blurSampler: targets.blurSampler,
    target: writeView(targets.scratch),
  });

  const verticalBindGroup = root.createBindGroup(blurLayout, {
    source: targets.scratch.createView(),
    blurSampler: targets.blurSampler,
    target: writeView(targets.target),
  });

  const configured = root
    .with(resolutionSlot, targets.resolution)
    .with(radiusSlot, targets.radius)
    .with(tapsSlot, taps);

  const horizontal = configured
    .with(axisSlot, d.vec2f(1, 0))
    .createComputePipeline({ compute: blurKernel });
  const vertical = configured
    .with(axisSlot, d.vec2f(0, 1))
    .createComputePipeline({ compute: blurKernel });

  return {
    async initAsync() {
      await Promise.all([horizontal.initAsync(), vertical.initAsync()]);
    },
    encode(pass) {
      horizontal.with(pass).with(horizontalBindGroup).dispatchWorkgroups(groups, groups);
      vertical.with(pass).with(verticalBindGroup).dispatchWorkgroups(groups, groups);
    },
  };
}

export function createFieldTexture(root: TgpuRoot, resolution: number): SingleChannelTexture {
  return root
    .createTexture({ size: [resolution, resolution], format: 'rgba16float' })
    .$usage('sampled', 'storage');
}
