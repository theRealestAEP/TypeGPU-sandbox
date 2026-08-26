import { d, std, tgpu } from 'typegpu';
import type { TgpuComputePass, TgpuRoot, TgpuSampler } from 'typegpu';
import { gaussianTaps, type SingleChannelTexture } from './blur.ts';

const WORKGROUP = 8;

/**
 * Narrow-range smoothing for the fluid surface.
 *
 * A plain Gaussian is wrong here: across a silhouette it averages foreground
 * depth with whatever is behind it and invents a ramp that never existed. The
 * obvious repair is to weight each tap by how close its depth is to the centre's
 * and let distant taps drop out - and that fails in the opposite direction, hard
 * enough to be worth spelling out. Once liquid occupies the full depth of the
 * box, two neighbouring pixels routinely see surfaces half the box apart, every
 * tap falls outside the tolerance, and the filter quietly does nothing at all:
 * the reconstructed surface comes back as a field of flat discs, one per
 * particle, with the edges intact. That is what "loose particles" looks like
 * from the inside.
 *
 * So taps are clamped into a narrow window around the centre instead of being
 * rejected by it. Every tap then carries its full spatial weight - which is what
 * actually smooths the per-particle bumps - while the clamp still stops a distant
 * surface from dragging the answer across a silhouette. Uncovered taps clamp to
 * the far edge of the window, so a ragged boundary rounds off toward the
 * background rather than ending in a hard bead.
 *
 * Channel layout is (depth, coverage).
 */
export const surfaceFilterLayout = tgpu.bindGroupLayout({
  source: { texture: d.texture2d(d.f32) },
  filterSampler: { sampler: 'filtering' },
  target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
});

const resolutionSlot = tgpu.slot<number>();
const radiusSlot = tgpu.slot<number>();
const tapsSlot = tgpu.slot<number[]>();
const axisSlot = tgpu.slot<d.v2f>();
/** Depth difference at which a tap stops counting as the same surface. */
const rangeSlot = tgpu.slot<number>();

const sampleAt = (uv: d.v2f) => {
  'use gpu';
  return std.textureSampleLevel(
    surfaceFilterLayout.$.source,
    surfaceFilterLayout.$.filterSampler,
    uv,
    0,
  ).xy;
};

const filterKernel = tgpu.computeFn({
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
  const middle = sampleAt(centre);
  const anchored = middle.y > 0.001;

  // The window the surface is allowed to move within, this pass.
  const near = middle.x + rangeSlot.$;
  const far = middle.x - rangeSlot.$;

  let depthSum = d.f32(0);
  let depthWeight = d.f32(0);
  let coverSum = d.f32(0);
  let coverWeight = d.f32(0);

  for (const tap of tgpu.unroll(std.range(radiusSlot.$ * 2 + 1))) {
    const spatial = tapsSlot.$[tap];
    const value = sampleAt(centre + step * (tap - radiusSlot.$));

    coverSum += value.y * spatial;
    coverWeight += spatial;

    // Larger depth is nearer the camera here, so open air clamps to the far
    // edge and pulls the silhouette back instead of leaving it standing.
    const held = std.select(far, std.clamp(value.x, far, near), value.y > 0.001);
    depthSum += std.select(value.x, held, anchored) * spatial;
    depthWeight += spatial;
  }

  std.textureStore(
    surfaceFilterLayout.$.target,
    gid.xy,
    d.vec4f(
      depthSum / std.max(depthWeight, 0.0001),
      coverSum / std.max(coverWeight, 0.0001),
      0,
      1,
    ),
  );
});

export interface SurfaceFilterTargets {
  readonly resolution: number;
  readonly radius: number;
  readonly sigma: number;
  /** Depth difference at which taps stop being treated as one surface. */
  readonly range: number;
  readonly source: SingleChannelTexture;
  readonly scratch: SingleChannelTexture;
  readonly target: SingleChannelTexture;
  readonly filterSampler: TgpuSampler;
}

export interface SurfaceFilter {
  initAsync(): Promise<void>;
  encode(pass: TgpuComputePass): void;
}

export function createSurfaceFilter(
  root: TgpuRoot,
  targets: SurfaceFilterTargets,
): SurfaceFilter {
  const taps = gaussianTaps(targets.radius, targets.sigma);
  const groups = Math.ceil(targets.resolution / WORKGROUP);

  const writeView = (texture: SingleChannelTexture) =>
    texture.createView(d.textureStorage2d('rgba16float', 'write-only'));

  const horizontalBindGroup = root.createBindGroup(surfaceFilterLayout, {
    source: targets.source.createView(),
    filterSampler: targets.filterSampler,
    target: writeView(targets.scratch),
  });

  const verticalBindGroup = root.createBindGroup(surfaceFilterLayout, {
    source: targets.scratch.createView(),
    filterSampler: targets.filterSampler,
    target: writeView(targets.target),
  });

  const configured = root
    .with(resolutionSlot, targets.resolution)
    .with(radiusSlot, targets.radius)
    .with(tapsSlot, taps)
    .with(rangeSlot, targets.range);

  const horizontal = configured
    .with(axisSlot, d.vec2f(1, 0))
    .createComputePipeline({ compute: filterKernel });
  const vertical = configured
    .with(axisSlot, d.vec2f(0, 1))
    .createComputePipeline({ compute: filterKernel });

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
