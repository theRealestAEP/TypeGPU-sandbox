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
const scatterOff = tgpu.workgroupVar(d.arrayOf(d.vec3f, ORIENT_THREADS));
const seedDown = tgpu.workgroupVar(d.vec3f);
const driftShared = tgpu.workgroupVar(d.f32);
const groundSums = tgpu.workgroupVar(d.arrayOf(d.f32, ORIENT_THREADS));
const groundVecs = tgpu.workgroupVar(d.arrayOf(d.vec3f, ORIENT_THREADS));
const groundShare = tgpu.workgroupVar(d.f32);
const groundTight = tgpu.workgroupVar(d.f32);
/** Per-thread advance of the surface since the last update, for the mean. */
const driftSums = tgpu.workgroupVar(d.arrayOf(d.f32, ORIENT_THREADS));

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

/** Multiplies the symmetric scatter matrix, held as its six unique terms. */
const scatterTimes = (diagonal: d.v3f, offDiagonal: d.v3f, v: d.v3f) => {
  'use gpu';
  return d.vec3f(
    diagonal.x * v.x + offDiagonal.x * v.y + offDiagonal.y * v.z,
    offDiagonal.x * v.x + diagonal.y * v.y + offDiagonal.z * v.z,
    offDiagonal.y * v.x + offDiagonal.z * v.y + diagonal.z * v.z,
  );
};

/**
 * Points an axis away from the camera. An axis is a line, so each has to be
 * pointed before any of them can be compared; you pour into things, so down
 * leads into the scene and never back out of it.
 */
const aimedIntoScene = (a: d.v3f) => {
  'use gpu';
  return std.select(a * -1, a, a.z < 0);
};

/** How closely a normal must agree with the seed to join the refit. */
const AGREEMENT = 0.94;

/**
 * Roll this small snaps to exactly level. Cameras are held level far more often
 * than they are held one degree off it, and a standing half-degree tilt is a
 * standing sideways pull the whole pool slowly obeys.
 */
const LEVEL_SNAP = 0.045;

/** Power iteration. Enough steps to separate the axes of a room. */
const POWER_STEPS = 24;

const dominantAxis = (diagonal: d.v3f, offDiagonal: d.v3f, seed: d.v3f) => {
  'use gpu';
  let v = d.vec3f(seed);
  for (const _step of std.range(POWER_STEPS)) {
    const next = scatterTimes(diagonal, offDiagonal, v);
    const size = std.length(next);
    if (size > 1e-12) {
      v = next / size;
    }
  }
  return v;
};

/**
 * Which way is down, from the shape of the scene's normal distribution.
 *
 * The old estimator averaged every surface that leaned toward the top of the
 * image and called the reverse of that gravity. That works looking down at a
 * table and fails in a room, because a wall facing the camera has a normal of
 * almost exactly (0, 0, 1) and a floor seen at a shallow angle has a normal
 * barely distinguishable from it. Both have a small y. No threshold on y can
 * separate them - measured on the bathroom, loosening the gate from 0.35 to
 * 0.05 pulled the answer from 22 degrees to 26 as the tiles flooded in.
 *
 * What does separate them is structure. Surfaces in a built scene are not
 * scattered; they cluster onto a handful of mutually square directions - walls,
 * floor, counter. Fit that frame instead of averaging, and the question stops
 * being "which texels are ground" and becomes "which of these three axes is
 * vertical", which the image answers unambiguously: gravity is the axis that
 * runs most steeply down the picture.
 *
 * The frame comes from the scatter matrix of the normals. Its dominant
 * eigenvector is whatever surface owns the most area - in a tiled bathroom,
 * the wall. Deflate that out and the next one appears, and the third is their
 * cross product. No thresholds anywhere, and a scene made mostly of floor works
 * by the same path as one made mostly of wall.
 */
const orientKernel = tgpu.computeFn({
  workgroupSize: [ORIENT_THREADS],
  in: { lid: d.builtin.localInvocationIndex },
})(({ lid }) => {
  'use gpu';
  // Every normal contributes to the scatter matrix. Six unique terms, carried
  // as two vectors: the diagonal and the off-diagonal.
  let diagonal = d.vec3f();
  let offDiagonal = d.vec3f();
  let driftSum = d.f32(0);
  let groundSum = d.f32(0);
  let groundVec = d.vec3f();
  for (const step of std.range(ORIENT_STRIDE)) {
    const index = lid * ORIENT_STRIDE + step;
    if (index < FIELD_RES * FIELD_RES) {
      const coord = d.vec2u(index % FIELD_RES, index / FIELD_RES);
      const uv = (d.vec2f(d.f32(index % FIELD_RES), d.f32(index / FIELD_RES)) + 0.5) *
        (1 / d.f32(FIELD_RES));
      driftSum += scaledDepthAt(uv) - scaledPrevDepthAt(uv);

      const normal = surfaceNormalAt(coord);
      // Surfaces you could stand on lean toward the top of the image. How much
      // of the frame they cover is the evidence that gravity is measurable at
      // all - a webcam full of wall and face has none.
      if (normal.y < -0.12) {
        groundSum += 1;
        groundVec = groundVec + normal;
      }
      diagonal = diagonal + normal * normal;
      offDiagonal = offDiagonal +
        d.vec3f(normal.x * normal.y, normal.x * normal.z, normal.y * normal.z);
    }
  }
  normalSums.$[lid] = d.vec3f(diagonal);
  scatterOff.$[lid] = d.vec3f(offDiagonal);
  driftSums.$[lid] = driftSum;
  groundSums.$[lid] = groundSum;
  groundVecs.$[lid] = d.vec3f(groundVec);
  std.workgroupBarrier();

  if (lid === 0) {
    let diag = d.vec3f();
    let off = d.vec3f();
    let driftTotal = d.f32(0);
    for (const slot of std.range(ORIENT_THREADS)) {
      diag = diag + normalSums.$[slot];
      off = off + scatterOff.$[slot];
      driftTotal += driftSums.$[slot];
    }
    driftShared.$ = driftTotal / d.f32(FIELD_RES * FIELD_RES);
    let groundTotal = d.f32(0);
    let groundDir = d.vec3f();
    for (const slot of std.range(ORIENT_THREADS)) {
      groundTotal += groundSums.$[slot];
      groundDir = groundDir + groundVecs.$[slot];
    }
    groundShare.$ = groundTotal / d.f32(FIELD_RES * FIELD_RES);
    // Mean resultant length: 1 when every ground normal agrees, small when
    // they scatter. A floor is coherent; a face is not.
    groundTight.$ = std.length(groundDir) / std.max(groundTotal, 1);

    // The frame: biggest cluster, then the biggest of what is left, then the
    // one square to both.
    const first = dominantAxis(diag, off, d.vec3f(0, 0, 1));
    const power = std.dot(first, scatterTimes(diag, off, first));
    const deflatedDiag = diag - first * first * power;
    const deflatedOff = off -
      d.vec3f(first.x * first.y, first.x * first.z, first.y * first.z) * power;
    const second = std.normalize(
      dominantAxis(deflatedDiag, deflatedOff, d.vec3f(0, 1, 0)) -
        first * std.dot(first, dominantAxis(deflatedDiag, deflatedOff, d.vec3f(0, 1, 0))),
    );
    const third = std.normalize(std.cross(first, second));

    const a1 = aimedIntoScene(first);
    const a2 = aimedIntoScene(second);
    const a3 = aimedIntoScene(third);

    // Of those, gravity is the one running most steeply down the picture.
    let axis = d.vec3f(a1);
    if (a2.y > axis.y) {
      axis = d.vec3f(a2);
    }
    if (a3.y > axis.y) {
      axis = d.vec3f(a3);
    }
    seedDown.$ = std.normalize(std.select(axis * -1, axis, axis.y > 0));
  }
  std.workgroupBarrier();

  // Refit. The frame fit says which cluster is the ground, which is the part no
  // average could work out; averaging that cluster says precisely where it
  // points, which is the part no single fitted axis can, because it inherits
  // every wobble in the surface it was fitted to. On the tub that showed up as
  // roll drifting several degrees off level. Seed from the fit, then average.
  const seed = d.vec3f(seedDown.$);
  let agreed = d.vec3f();
  let agreedCount = d.f32(0);
  for (const step of std.range(ORIENT_STRIDE)) {
    const index = lid * ORIENT_STRIDE + step;
    if (index < FIELD_RES * FIELD_RES) {
      const normal = surfaceNormalAt(d.vec2u(index % FIELD_RES, index / FIELD_RES));
      if (std.dot(normal, seed * -1) > AGREEMENT) {
        agreed = agreed + normal;
        agreedCount += 1;
      }
    }
  }
  normalSums.$[lid] = d.vec3f(agreed);
  scatterOff.$[lid] = d.vec3f(agreedCount, 0, 0);
  std.workgroupBarrier();

  if (lid === 0) {
    let total = d.vec3f();
    let found = d.f32(0);
    for (const slot of std.range(ORIENT_THREADS)) {
      total = total + normalSums.$[slot];
      found += scatterOff.$[slot].x;
    }
    // The two estimates are good at different halves of the answer, so each
    // supplies the half it is good at.
    //
    // Pitch comes from the frame fit, because at a shallow angle the ground and
    // the wall behind it differ by about ten degrees of normal - closer than any
    // cone - and only the orthogonality of the fitted frame tells them apart.
    // Averaging cannot; it just splits the difference and reports the camera as
    // steeper than it is.
    //
    // Roll comes from averaging the ground cluster, because a single fitted axis
    // carries every wobble of the surface it was fitted to, and on the tub that
    // showed up as four degrees of tilt on a level shot.
    const seeded = d.vec3f(seedDown.$);
    let down = d.vec3f(seeded);
    if (found > 32 && std.length(total) > 0.001) {
      const refined = std.normalize(std.normalize(total) * -1);
      const pitch = std.asin(std.clamp(-seeded.z, -1, 1));
      let roll = std.atan2(refined.x, std.max(refined.y, 1e-4));
      if (std.abs(roll) < LEVEL_SNAP) {
        roll = 0;
      }
      const flat = std.cos(pitch);
      down = std.normalize(
        d.vec3f(std.sin(roll) * flat, std.cos(roll) * flat, -std.sin(pitch)),
      );
    }

    // Steepness has to be earned twice over. The frame fit always produces
    // SOME axis, and area alone is not evidence: a webcam frame full of face,
    // shoulders and house plant has plenty of texels leaning up-image, and
    // they once read as the camera pointing 68 degrees into a level room. Real
    // ground is not merely present, it is COHERENT - every floor texel's
    // normal points the same way, while a face's scatter over a hemisphere.
    // The prior is a level camera, and only abundant, agreeing ground moves it.
    const confidence = std.saturate((groundShare.$ - 0.05) / 0.1) *
      std.saturate((groundTight.$ - 0.7) / 0.2);
    down = std.normalize(std.mix(d.vec3f(0, 1, 0), down, confidence));

    // The source's own prior caps the pitch outright. Three rounds of ever
    // smarter statistical gating still read one real webcam as pointing sixty
    // degrees into a level room, because a face plus a desk slice can satisfy
    // any test built from normals alone. A webcam IS near level - that is a
    // fact about how laptops are used, not about this frame - so the camera
    // source refuses steep answers entirely and leaves the rare exception to
    // the manual pin, which still overrides everything below.
    if (-down.z > orientLayout.$.params.maxDownZ) {
      const tilt = std.atan2(down.x, down.y);
      const flat = std.sqrt(1 - orientLayout.$.params.maxDownZ * orientLayout.$.params.maxDownZ);
      down = d.vec3f(
        std.sin(tilt) * flat,
        std.cos(tilt) * flat,
        -orientLayout.$.params.maxDownZ,
      );
    }

    // Cap the lean into the scene. Past this the fit is more likely to have
    // locked onto a wall than onto genuinely overhead geometry.
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

    // Blend with the last published estimate. The buffer is both source and
    // destination: an EMA whose state lives where everyone downstream reads it.
    const previous = d.vec3f(orientLayout.$.scene.down);
    down = std.normalize(
      down * (1 - ORIENTATION_SMOOTHING) + previous * ORIENTATION_SMOOTHING,
    );

    if (orientLayout.$.params.manual !== 0) {
      down = std.normalize(orientLayout.$.params.manualDown);
    }
    orientLayout.$.scene.down = d.vec3f(down);
    orientLayout.$.scene.drift = driftShared.$;
  }
});

export interface FieldTuning {
  depthScale: number;
  /** Steepest pitch, in degrees, the measurement may report. */
  maxPitch: number;
  /** Pin gravity instead of measuring it. */
  manual: boolean;
  /** Degrees the camera looks below level. */
  pitch: number;
  /** Degrees the horizon is rotated in the frame. */
  roll: number;
}

export const defaultFieldTuning: FieldTuning = {
  depthScale: MAX_DEPTH_SCALE,
  maxPitch: 85,
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
  /** The scene with transient occluders forgotten; what stands behind a hand. */
  readonly background: SingleChannelTexture;
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
      maxDownZ: Math.sin((defaultFieldTuning.maxPitch * Math.PI) / 180),
    })
    .$usage('uniform');
  const scene = root
    .createBuffer(SceneState, { down: d.vec3f(0, 1, 0), drift: 0 })
    .$usage('storage');
  const scratch = createFieldTexture(root, FIELD_RES);
  const surface = createFieldTexture(root, FIELD_RES);
  const surfacePrev = createFieldTexture(root, FIELD_RES);
  const live = createFieldTexture(root, FIELD_RES);
  const background = createFieldTexture(root, FIELD_RES);

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
   * The scene as it stands when nothing is passing in front of it.
   *
   * A heightfield only knows the front-most surface, so a hand crossing a sink
   * REPLACES the sink at those texels, and the water beneath instantly loses
   * its floor. The background layer remembers: anything revealed - the live
   * surface at or deeper than what is stored - is adopted at once, while a
   * nearer arrival has to persist for a couple of seconds before it counts as
   * scenery. A passing hand never qualifies; a cup set down does.
   */
  const backLayout = tgpu.bindGroupLayout({
    liveIn: { texture: d.texture2d(d.f32) },
    backIn: { texture: d.texture2d(d.f32) },
    target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
  });
  const backKernel = tgpu.computeFn({
    workgroupSize: [8, 8],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= d.u32(FIELD_RES) || gid.y >= d.u32(FIELD_RES)) {
      return;
    }
    const now = std.textureLoad(backLayout.$.liveIn, gid.xy, 0).x;
    const held = std.textureLoad(backLayout.$.backIn, gid.xy, 0).x;
    let next = std.mix(held, now, 0.03);
    if (now <= held + 0.004 || held < 0.02) {
      next = now;
    }
    std.textureStore(backLayout.$.target, gid.xy, d.vec4f(next, 1, 0, 1));
  });
  // Ping-pong through the free scratch texture: read background, write scratch,
  // then the copy below publishes scratch back.
  const backStep = root.createBindGroup(backLayout, {
    liveIn: surface.createView(),
    backIn: background.createView(),
    target: scratch.createView(d.textureStorage2d('rgba16float', 'write-only')),
  });
  const backPublish = root.createBindGroup(holdLayout, {
    source: scratch.createView(),
    target: background.createView(d.textureStorage2d('rgba16float', 'write-only')),
  });
  const backPipe = root.createComputePipeline({ compute: backKernel });

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
    background,

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
      backPipe.with(pass).with(backStep).dispatchWorkgroups(holdGroups, holdGroups);
      hold.with(pass).with(backPublish).dispatchWorkgroups(holdGroups, holdGroups);
      orient.with(pass).with(orientBindGroup).dispatchWorkgroups(1);
    },

    tune(next) {
      tuning = { ...tuning, ...next };
      const [x, y, z] = downFromAngles(tuning.pitch, tuning.roll);
      params.write({
        depthScale: tuning.depthScale,
        manual: tuning.manual ? 1 : 0,
        manualDown: [x, y, z],
        maxDownZ: Math.sin((tuning.maxPitch * Math.PI) / 180),
      });
    },

    destroy() {
      params.destroy();
      scene.destroy();
      scratch.destroy();
      surface.destroy();
      surfacePrev.destroy();
      live.destroy();
      background.destroy();
      phase.destroy();
    },
  };
}
