import { d, std, tgpu } from 'typegpu';
import type {
  StorageFlag,
  TgpuBuffer,
  TgpuComputePass,
  TgpuRoot,
  TgpuSampler,
  UniformFlag,
} from 'typegpu';
import { createFieldTexture, type SingleChannelTexture } from '../gpu/blur.ts';
import { createSurfaceFilter } from '../gpu/surface-filter.ts';
import { FIELD_RES, FieldParams, MAX_DEPTH_SCALE, SceneState } from './schemas.ts';

/**
 * Smoothing for the collision surface, and it has to be depth-aware.
 *
 * The field's small-scale roughness turns into a landscape of shallow pockets
 * that park about 3% of the water above the water line, so it does need
 * smoothing. A plain Gaussian wide enough to clear the pockets also erases the
 * geometry worth colliding with: at radius 4 the pool photo came back as a soft
 * ramp with no rim, no floor and no walls, which is why liquid slid down it
 * instead of gathering in it. Weighting each tap by how close its depth is to
 * the centre's keeps both - noise inside a surface averages away, a step at the
 * rim does not, because taps across it simply drop out.
 */
const FILTER_RADIUS = 4;
const FILTER_SIGMA = 2.2;
/**
 * Depth gap at which a tap stops counting as the same surface, in normalised
 * depth. Comfortably above the field's noise and far below any real step.
 */
const FILTER_RANGE = 0.035;

const ORIENT_THREADS = 256;
/** Texels each thread walks while estimating the scene's orientation. */
const ORIENT_STRIDE = Math.ceil((FIELD_RES * FIELD_RES) / ORIENT_THREADS);

const orientLayout = tgpu.bindGroupLayout({
  params: { uniform: FieldParams },
  surface: { texture: d.texture2d(d.f32) },
  /** The surface as of the previous update, for measuring global drift. */
  surfacePrev: { texture: d.texture2d(d.f32) },
  fieldSampler: { sampler: 'filtering' },
  scene: { storage: SceneState, access: 'mutable' },
});

const normalSums = tgpu.workgroupVar(d.arrayOf(d.vec3f, ORIENT_THREADS));
const normalCounts = tgpu.workgroupVar(d.arrayOf(d.f32, ORIENT_THREADS));
/** Per-thread advance of the surface since the last update, for the mean. */
const driftSums = tgpu.workgroupVar(d.arrayOf(d.f32, ORIENT_THREADS));
const firstGuess = tgpu.workgroupVar(d.vec3f);

/** Second pass keeps only normals within this much of the first estimate. */
const AGREEMENT = 0.82;

/**
 * How far a surface has to lean toward the top of the image before it counts as
 * ground. This threshold is the whole difference between a working estimate and
 * a useless one. A wall facing the camera, or distant background, has a normal
 * of almost exactly (0, 0, 1) - no vertical lean at all - and a threshold near
 * zero lets depth noise push half of those texels through. They then dominate
 * the average by sheer area, and the answer comes back as "up is toward the
 * camera", which is to say gravity points into the scene and every drop of water
 * slides to the back wall. Measured on the pool photo before this: 93% of gravity
 * pointed into the scene against 38% down the image.
 */
const GROUND_LEAN = 0.35;

/**
 * Fraction of the field that has to read as ground before the measurement is
 * trusted outright. Below it the answer is blended toward a level camera.
 */
const CONFIDENT_AREA = 0.1;

/**
 * Largest z component gravity may have. A depth map cannot tell a floor seen
 * from above from a wall seen head-on - both are flat and both face the
 * camera - so the estimate needs *a* limit. But it was set at 0.72, about 46
 * degrees, and a camera looking down into a pool at a normal shooting angle
 * measures 0.8-0.9: the cap was rotating gravity toward down-the-image by
 * fifteen-plus degrees, and on a sloped basin any standing sideways pull walks
 * the entire pool out over the near rim and down to the bottom of the frame.
 * The wall-mistaken-for-floor case is already excluded upstream, by the ground
 * lean threshold and the agreement refit; those surfaces have no vertical lean
 * and never enter the average. What remains past them is genuinely
 * overhead-looking geometry, so the cap only guards against extremes now.
 */
const MAX_INTO_SCENE = 0.92;

const scaledDepthAt = (uv: d.v2f) => {
  'use gpu';
  return (
    std.textureSampleLevel(orientLayout.$.surface, orientLayout.$.fieldSampler, uv, 0).x *
    orientLayout.$.params.depthScale
  );
};

const scaledPrevDepthAt = (uv: d.v2f) => {
  'use gpu';
  return (
    std.textureSampleLevel(orientLayout.$.surfacePrev, orientLayout.$.fieldSampler, uv, 0).x *
    orientLayout.$.params.depthScale
  );
};

const surfaceNormalAt = (coord: d.v2u) => {
  'use gpu';
  const step = d.f32(1 / FIELD_RES);
  const uv = (d.vec2f(coord) + 0.5) * step;
  const slope = d.vec2f(
    scaledDepthAt(uv + d.vec2f(step, 0)) - scaledDepthAt(uv - d.vec2f(step, 0)),
    scaledDepthAt(uv + d.vec2f(0, step)) - scaledDepthAt(uv - d.vec2f(0, step)),
  ) / (2 * step);
  return std.normalize(d.vec3f(slope * -1, 1));
};

/**
 * Estimates which way is down, so nobody has to guess a camera angle.
 *
 * Liquid rests on surfaces whose normal opposes gravity, so gravity is simply
 * the reverse of the average up-facing surface normal. Only texels whose normal
 * leans toward the top of the image count: those are the surfaces you are
 * looking down onto - a tabletop, a floor - while a wall facing the camera has
 * no vertical lean and is correctly ignored.
 */
/**
 * How much of the previous gravity estimate each new measurement keeps. The
 * refit already rejects disagreeing normals, but on a moving scene - walking
 * cats, a panning camera - the accepted population itself swings frame to
 * frame, and writing the raw estimate through made the floor plane and every
 * resting drop twitch with it. A slow blend keeps the world still while the
 * estimate tracks genuinely new geometry within about a second.
 */
const ORIENTATION_SMOOTHING = 0.9;

const orientKernel = tgpu.computeFn({
  workgroupSize: [ORIENT_THREADS],
  in: { lid: d.builtin.localInvocationIndex },
})(({ lid }) => {
  'use gpu';
  // Pass one: the mean up-facing normal across the whole field, plus each
  // thread's share of how far the surface advanced since the last update.
  let sum = d.vec3f();
  let count = d.f32(0);
  let driftSum = d.f32(0);
  for (const step of std.range(ORIENT_STRIDE)) {
    const index = lid * ORIENT_STRIDE + step;
    if (index < FIELD_RES * FIELD_RES) {
      const normal = surfaceNormalAt(d.vec2u(index % FIELD_RES, index / FIELD_RES));
      driftSum += scaledDepthAt((d.vec2f(index % FIELD_RES, index / FIELD_RES) + 0.5) *
        (1 / d.f32(FIELD_RES))) -
        scaledPrevDepthAt((d.vec2f(index % FIELD_RES, index / FIELD_RES) + 0.5) *
          (1 / d.f32(FIELD_RES)));
      // Leaning toward the top of the image means it faces upward in the world.
      if (normal.y < -GROUND_LEAN) {
        sum = sum + normal;
        count += 1;
      }
    }
  }
  normalSums.$[lid] = d.vec3f(sum);
  normalCounts.$[lid] = count;
  driftSums.$[lid] = driftSum;
  std.workgroupBarrier();

  if (lid === 0) {
    let total = d.vec3f();
    for (const slot of std.range(ORIENT_THREADS)) {
      total = total + normalSums.$[slot];
    }
    let guess = d.vec3f(0, -1, 0);
    if (std.length(total) > 0.001) {
      guess = std.normalize(total);
    }
    firstGuess.$ = d.vec3f(guess);
  }
  std.workgroupBarrier();

  // Pass two: refit using only the surfaces that agree with that estimate. A
  // plain mean is dragged around by whatever else is in frame - a wall, a chair -
  // and even a few degrees of tilt leaves a standing tangential pull that slides
  // the whole pool across a table it should be resting flat on.
  let agreed = d.vec3f();
  let agreedCount = d.f32(0);
  for (const step of std.range(ORIENT_STRIDE)) {
    const index = lid * ORIENT_STRIDE + step;
    if (index < FIELD_RES * FIELD_RES) {
      const normal = surfaceNormalAt(d.vec2u(index % FIELD_RES, index / FIELD_RES));
      if (normal.y < -GROUND_LEAN && std.dot(normal, firstGuess.$) > AGREEMENT) {
        agreed = agreed + normal;
        agreedCount += 1;
      }
    }
  }
  normalSums.$[lid] = d.vec3f(agreed);
  normalCounts.$[lid] = agreedCount;
  std.workgroupBarrier();

  if (lid === 0) {
    let total = d.vec3f();
    let found = d.f32(0);
    for (const slot of std.range(ORIENT_THREADS)) {
      total = total + normalSums.$[slot];
      found += normalCounts.$[slot];
    }
    let up = d.vec3f(firstGuess.$);
    if (found > 0 && std.length(total) > 0.001) {
      up = std.normalize(total);
    }

    // A camera is level far more often than not, so that is the starting
    // assumption, and the measurement only moves it as far as the evidence
    // supports. With no ground in frame - a webcam pointed across a room at a
    // person, say - there is nothing to measure and the level answer stands,
    // which is right, instead of the average of a wall.
    const area = found / d.f32(FIELD_RES * FIELD_RES);
    const confidence = std.saturate(area / d.f32(CONFIDENT_AREA));
    let down = std.normalize(std.mix(d.vec3f(0, 1, 0), up * -1, confidence));

    // Cap the lean into the scene. Beyond this the estimate is more likely to be
    // a wall mistaken for a floor than a genuinely overhead camera.
    if (std.abs(down.z) > MAX_INTO_SCENE) {
      const flat = std.length(down.xy);
      const keep = std.sqrt(1 - MAX_INTO_SCENE * MAX_INTO_SCENE);
      const sign = std.select(d.f32(1), d.f32(-1), down.z < 0);
      if (flat > 0.0001) {
        down = d.vec3f(down.xy * (keep / flat), MAX_INTO_SCENE * sign);
      } else {
        down = d.vec3f(0, keep, MAX_INTO_SCENE * sign);
      }
    }

    // Blend with the last published estimate before writing back. The buffer
    // is both the source and the destination: an EMA whose state lives where
    // everyone downstream reads it. Without the blend the floor plane twitched
    // every time the scene moved; with it, water holds still while the
    // estimate tracks genuinely new geometry within about a second.
    const previous = d.vec3f(orientLayout.$.scene.down);
    down = std.normalize(
      down * (1 - ORIENTATION_SMOOTHING) + previous * ORIENTATION_SMOOTHING,
    );

    // A pinned direction wins outright. No blend: the whole point is that the
    // scene stops arguing about it.
    if (orientLayout.$.params.manual !== 0) {
      down = std.normalize(orientLayout.$.params.manualDown);
    }

    orientLayout.$.scene.down = d.vec3f(down);

    // Mean forward advance of the whole surface this update. Monocular depth
    // renormalises its range whenever the scene's content shifts, so static
    // geometry registers phantom motion; subtracting the field-wide mean in
    // the fluid keeps obstacle contacts honest while real local motion - a
    // descending paw, a hand over the spout - still counts.
    let driftTotal = d.f32(0);
    for (const slot of std.range(ORIENT_THREADS)) {
      driftTotal += driftSums.$[slot];
    }
    orientLayout.$.scene.drift = driftTotal / d.f32(FIELD_RES * FIELD_RES);
  }
});

export interface FieldTuning {
  depthScale: number;
  /** Pin gravity instead of measuring it. */
  manual: boolean;
  /** Degrees the camera looks below level. */
  pitch: number;
  /** Degrees the horizon is rotated in the frame. */
  roll: number;
}

export const defaultFieldTuning: FieldTuning = {
  depthScale: MAX_DEPTH_SCALE,
  manual: false,
  pitch: 0,
  roll: 0,
};

/**
 * Which way is down, for a camera pitched below level and rolled in frame.
 * y runs down the image and z runs toward the viewer, so pitching down tips the
 * vector away from the camera.
 */
export function downFromAngles(pitch: number, roll: number): [number, number, number] {
  const p = (pitch * Math.PI) / 180;
  const r = (roll * Math.PI) / 180;
  return [Math.sin(r) * Math.cos(p), Math.cos(r) * Math.cos(p), -Math.sin(p)];
}

export interface SurfaceField {
  /** Smoothed scene depth, 0 far and 1 near. */
  readonly surface: SingleChannelTexture;
  readonly params: TgpuBuffer<typeof FieldParams> & UniformFlag;
  /** Which way is down and how far the whole surface moved, refreshed with the depth. */
  readonly scene: TgpuBuffer<typeof SceneState> & StorageFlag;
  /** The collision surface as of the previous depth update. */
  readonly surfacePrev: SingleChannelTexture;
  /**
   * The collision surface as it stands right now, walked from `surfacePrev`
   * toward `surface`. Depth arrives at video rate and the solver runs far
   * faster, so this is what the fluid actually collides against.
   */
  readonly live: SingleChannelTexture;
  initAsync(): Promise<void>;
  encode(pass: TgpuComputePass): void;
  /** Re-walks `live` for the current point in the depth interval. Every frame. */
  advance(pass: TgpuComputePass, phase: number): void;
  tune(next: Partial<FieldTuning>): void;
  destroy(): void;
}

export function createSurfaceField(
  root: TgpuRoot,
  depth: SingleChannelTexture,
  fieldSampler: TgpuSampler,
): SurfaceField {
  let tuning: FieldTuning = { ...defaultFieldTuning };

  const params = root
    .createBuffer(FieldParams, {
      depthScale: defaultFieldTuning.depthScale,
      manual: 0,
      manualDown: downFromAngles(0, 0),
    })
    .$usage('uniform');
  const scene = root
    .createBuffer(SceneState, { down: d.vec3f(0, 1, 0), drift: 0 })
    .$usage('storage');
  const scratch = createFieldTexture(root, FIELD_RES);
  const surface = createFieldTexture(root, FIELD_RES);
  const surfacePrev = createFieldTexture(root, FIELD_RES);
  const live = createFieldTexture(root, FIELD_RES);

  const filter = createSurfaceFilter(root, {
    resolution: FIELD_RES,
    radius: FILTER_RADIUS,
    sigma: FILTER_SIGMA,
    range: FILTER_RANGE,
    source: depth,
    scratch,
    target: surface,
    filterSampler: fieldSampler,
  });

  const orientBindGroup = root.createBindGroup(orientLayout, {
    params,
    surface: surface.createView(),
    surfacePrev: surfacePrev.createView(),
    fieldSampler,
    scene,
  });
  const orient = root.createComputePipeline({ compute: orientKernel });

  /** Copies the live surface into surfacePrev, one texel per thread. */
  const holdLayout = tgpu.bindGroupLayout({
    source: { texture: d.texture2d(d.f32) },
    target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
  });
  const holdKernel = tgpu.computeFn({
    workgroupSize: [8, 8],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= d.u32(FIELD_RES) || gid.y >= d.u32(FIELD_RES)) {
      return;
    }
    const value = std.textureLoad(holdLayout.$.source, gid.xy, 0);
    std.textureStore(holdLayout.$.target, gid.xy, value);
  });
  const holdBindGroup = root.createBindGroup(holdLayout, {
    source: surface.createView(),
    target: surfacePrev.createView(d.textureStorage2d('rgba16float', 'write-only')),
  });
  const hold = root.createComputePipeline({ compute: holdKernel });
  const holdGroups = Math.ceil(FIELD_RES / 8);

  /**
   * Walks the collision surface from the previous measurement to the newest one
   * across the depth interval, so the wall travels at the speed it was measured
   * at instead of teleporting on the frame the depth lands.
   *
   * Done once here rather than in the solver: every collision query would
   * otherwise cost two texture samples instead of one, and the solver makes
   * about ten of them per particle per kernel. Measured at 38% of frame rate.
   */
  const walkLayout = tgpu.bindGroupLayout({
    from: { texture: d.texture2d(d.f32) },
    to: { texture: d.texture2d(d.f32) },
    phase: { uniform: d.f32 },
    target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
  });
  const walkKernel = tgpu.computeFn({
    workgroupSize: [8, 8],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= d.u32(FIELD_RES) || gid.y >= d.u32(FIELD_RES)) {
      return;
    }
    const from = std.textureLoad(walkLayout.$.from, gid.xy, 0);
    const to = std.textureLoad(walkLayout.$.to, gid.xy, 0);
    std.textureStore(walkLayout.$.target, gid.xy, std.mix(from, to, walkLayout.$.phase));
  });
  const phase = root.createBuffer(d.f32, 1).$usage('uniform');
  const walkBindGroup = root.createBindGroup(walkLayout, {
    from: surfacePrev.createView(),
    to: surface.createView(),
    phase,
    target: live.createView(d.textureStorage2d('rgba16float', 'write-only')),
  });
  const walk = root.createComputePipeline({ compute: walkKernel });

  return {
    surface,
    params,
    surfacePrev,
    scene,
    live,

    async initAsync() {
      await Promise.all([
        filter.initAsync(),
        orient.initAsync(),
        hold.initAsync(),
        walk.initAsync(),
      ]);
    },

    advance(pass, next) {
      phase.write(Math.min(Math.max(next, 0), 1));
      walk.with(pass).with(walkBindGroup).dispatchWorkgroups(holdGroups, holdGroups);
    },

    encode(pass) {
      hold.with(pass).with(holdBindGroup).dispatchWorkgroups(holdGroups, holdGroups);
      filter.encode(pass);
      orient.with(pass).with(orientBindGroup).dispatchWorkgroups(1);
    },

    tune(next) {
      tuning = { ...tuning, ...next };
      const [x, y, z] = downFromAngles(tuning.pitch, tuning.roll);
      params.write({
        depthScale: tuning.depthScale,
        manual: tuning.manual ? 1 : 0,
        manualDown: [x, y, z],
      });
    },

    destroy() {
      params.destroy();
      scene.destroy();
      scratch.destroy();
      surface.destroy();
      surfacePrev.destroy();
      live.destroy();
      phase.destroy();
    },
  };
}
