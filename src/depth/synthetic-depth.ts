import { d, std, tgpu } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SingleChannelTexture } from '../gpu/blur.ts';
import { FIELD_RES } from '../sim/schemas.ts';
import type { DepthSource } from './depth-field.ts';

const WORKGROUP = 8;
const WORKGROUPS = Math.ceil(FIELD_RES / WORKGROUP);

/**
 * Depths are what a camera would actually measure, 1 nearest. The back of a rim
 * is genuinely further from the camera than the front of it, and that gap is
 * what lets liquid in: the back lip sits behind the pour and the front lip
 * catches it. Flatten a rim to one depth and the ring closes over the opening,
 * so nothing can get in.
 */
const BACKGROUND = 0.08;
const TABLE_BACK = 0.58;
const TABLE_FRONT = 0.92;
const BODY = 0.86;
const RIM_FRONT = 0.82;
const RIM_BACK = 0.44;
const INTERIOR_TOP = 0.4;
const INTERIOR_BOTTOM = 0.24;
const HANDLE = 0.78;

const TABLE_Y = 0.9;
/** Squared rim-ellipse value inside which the interior is visible. */
const RIM_OPENING = 0.72;

const SyntheticParams = d.struct({ time: d.f32 });

const syntheticLayout = tgpu.bindGroupLayout({
  params: { uniform: SyntheticParams },
  target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
});

/**
 * One open vessel: a tapered body under an elliptical rim, with the interior
 * carved out behind it. Returns (hit, depth) so callers can overwrite the table
 * rather than blend with it.
 */
const vesselAt = (
  point: d.v2f,
  centre: d.v2f,
  radii: d.v2f,
  baseY: number,
  taper: number,
) => {
  'use gpu';
  const offset = (point - centre) / radii;
  const rim = std.dot(offset, offset);
  const towardsViewer = std.saturate((offset.y + 1) * 0.5);

  const drop = std.saturate((point.y - centre.y) / (baseY - centre.y));
  const halfWidth = std.mix(radii.x, radii.x * taper, drop);
  const inBody = point.y > centre.y && point.y < baseY && std.abs(point.x - centre.x) < halfWidth;

  if (rim < RIM_OPENING) {
    return d.vec3f(1, std.mix(INTERIOR_TOP, INTERIOR_BOTTOM, towardsViewer), 0);
  }
  if (rim < 1) {
    return d.vec3f(1, std.mix(RIM_BACK, RIM_FRONT, towardsViewer), 0);
  }
  if (inBody) {
    return d.vec3f(1, BODY, 0);
  }
  return d.vec3f();
};

/**
 * Three vessels on a table, all seen from above so their openings face the pour.
 * A mug with a handle, a wide shallow bowl, and a tall narrow glass - different
 * volumes, so they fill at visibly different rates.
 */
const syntheticKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP, WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= FIELD_RES || gid.y >= FIELD_RES) {
    return;
  }
  const point = (d.vec2f(gid.xy) + 0.5) / d.f32(FIELD_RES);
  let depth = d.f32(BACKGROUND);

  if (point.y > TABLE_Y) {
    depth = std.mix(TABLE_BACK, TABLE_FRONT, (point.y - TABLE_Y) / (1 - TABLE_Y));
  }

  // Mug handle, before its body so the body edge covers the join.
  const handleCentre = d.vec2f(0.325, 0.65);
  const handleReach = std.length(point - handleCentre);
  if (handleReach > 0.042 && handleReach < 0.063 && point.x > handleCentre.x - 0.02) {
    depth = HANDLE;
  }

  const mug = vesselAt(point, d.vec2f(0.22, 0.52), d.vec2f(0.105, 0.068), 0.9, 0.86);
  if (mug.x > 0) {
    depth = mug.y;
  }

  const bowl = vesselAt(point, d.vec2f(0.52, 0.68), d.vec2f(0.16, 0.1), 0.9, 0.5);
  if (bowl.x > 0) {
    depth = bowl.y;
  }

  const glass = vesselAt(point, d.vec2f(0.81, 0.36), d.vec2f(0.075, 0.048), 0.9, 0.95);
  if (glass.x > 0) {
    depth = glass.y;
  }

  // Channel two is coverage, which the depth-aware filter reads. A built scene
  // covers every texel.
  std.textureStore(syntheticLayout.$.target, gid.xy, d.vec4f(depth, 1, 0, 1));
});

export function createSyntheticDepth(root: TgpuRoot, depth: SingleChannelTexture): DepthSource {
  const params = root.createBuffer(SyntheticParams, { time: 0 }).$usage('uniform');
  const bindGroup = root.createBindGroup(syntheticLayout, {
    params,
    target: depth.createView(d.textureStorage2d('rgba16float', 'write-only')),
  });
  const pipeline = root.createComputePipeline({ compute: syntheticKernel });
  const start = performance.now();

  return {
    minIntervalMs: 0,

    async initAsync() {
      await pipeline.initAsync();
    },

    encode(pass) {
      params.write({ time: (performance.now() - start) / 1000 });
      pipeline.with(bindGroup).with(pass).dispatchWorkgroups(WORKGROUPS, WORKGROUPS);
    },

    destroy() {
      params.destroy();
    },
  };
}
