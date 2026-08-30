import { common, d, std, tgpu } from 'typegpu';
import type {
  TgpuBuffer,
  TgpuCommandEncoder,
  TgpuRoot,
  TgpuSampler,
  UniformFlag,
} from 'typegpu';
import { createFieldTexture, createSeparableBlur, type SingleChannelTexture } from '../gpu/blur.ts';
import { createSurfaceFilter } from '../gpu/surface-filter.ts';
import type { CameraFrame } from '../depth/depth-field.ts';
import {
  CameraParams,
  FieldParams,
  KERNEL_RADIUS,
  PARTICLE_COUNT,
  Particle,
  REST_SPACING,
  SURFACE_RES,
  Z_MAX,
} from '../sim/schemas.ts';
import type { ParticleBuffer } from '../sim/schemas.ts';

/**
 * Wide enough to merge neighbouring splats into one surface. At a radius near
 * the splat's own size the filter only rounds off each bead and leaves the bumps
 * between them, so a sheet of liquid reads as a heap of separate spheres.
 */
const DEPTH_FILTER_RADIUS = 16;
const DEPTH_FILTER_SIGMA = 8;
/** Taps more than a few kernel radii away in depth belong to another surface. */
const DEPTH_FILTER_RANGE = KERNEL_RADIUS * 3;
const THICKNESS_BLUR_RADIUS = 4;
const THICKNESS_BLUR_SIGMA = 2.2;

const VERTS_PER_SPLAT = 6;

/**
 * Thickness at which liquid starts to read, and where it is fully opaque. Set
 * high on purpose: at the ragged edge of a body of water individual particles
 * poke out of the surface, and anything that renders them at all renders them as
 * beads. Real water pulls that edge into a meniscus; half a million particles at
 * this scale cannot, so the edge is faded out instead.
 */
/**
 * How fast opacity builds with optical depth. Set so a lone airborne droplet -
 * about a fifth of a splat's worth - reads at roughly a fifth opacity: present,
 * refracting, catching a highlight, but mostly showing what is behind it. Which
 * is what a droplet looks like.
 */
const BODY_DENSITY = 1.2;

export const DebugView = {
  LIQUID: 0,
  SCENE_DEPTH: 1,
  FLUID_DEPTH: 2,
  THICKNESS: 3,
  SMOKE: 4,
} as const;

const quadCorners = tgpu.const(d.arrayOf(d.vec2f, VERTS_PER_SPLAT), [
  d.vec2f(-1, -1),
  d.vec2f(1, -1),
  d.vec2f(-1, 1),
  d.vec2f(-1, 1),
  d.vec2f(1, -1),
  d.vec2f(1, 1),
]);

const SplatParams = d.struct({
  radius: d.f32,
  /**
   * Pour channels: per cup, the interior x-span and the mouth's lower edge
   * (xLeft, xRight, yEnd, active). Water falling toward a glass is between
   * the camera and the scene in reality, so inside these regions the
   * occlusion discard stands down - the depth model's halo around a
   * close-held glass was swallowing the whole stream.
   */
  mouths: d.arrayOf(d.vec4f, 3),
});

const LookParams = d.struct({
  /** Non-zero draws the detected-vessel outlines; tied to the tune drawer. */
  cupLines: d.f32,
  /** Pour channels, mirrored from the splat params: xL, xR, yEnd, active. */
  cupMouths: d.arrayOf(d.vec4f, 3),
  tint: d.vec3f,
  surfaceLow: d.f32,
  surfaceHigh: d.f32,
  relief: d.f32,
  refraction: d.f32,
  thickness: d.f32,
  absorption: d.f32,
  /** Light the liquid returns itself, standing in for subsurface scattering. */
  scatter: d.f32,
  specular: d.f32,
  /**
   * Distance from the eye to the image plane. The simulation is orthographic,
   * but shading is not: without a real view ray every pixel sits at the same
   * angle to the surface, Fresnel is constant everywhere, and a flat pool can
   * never go mirror-like toward the far end. Smaller is a wider lens.
   */
  lens: d.f32,
  reflection: d.f32,
  caustics: d.f32,
  foam: d.f32,
  /** Lightning, 0 to 1. Lights the whole scene, not only the water. */
  flash: d.f32,
  /** Where the scene's key light is. Highlights that disagree with the photo
   * are one of the loudest tells that the water was added afterwards. */
  sun: d.vec3f,
  /**
   * A light that lives in the scene rather than outside it.
   *
   * The measured sun is a direction and nothing more - every point in the frame
   * gets the same one. A torch has a place, so its light falls off, sweeps as
   * you move it, and picks out whatever is nearest to it. That is the whole
   * point of injecting a light into a photograph: it is the cue that says the
   * light is in the room with the scene rather than painted over it.
   */
  torch: d.f32,
  torchAt: d.vec3f,
  /**
   * Lamps the user has planted. Position and power in one array, colour and
   * size in the other; power zero means the slot is empty. They light the
   * scene itself as well as the liquid, which is what makes them read as
   * objects in the room rather than stickers on the glass.
   */
  lightsA: d.arrayOf(d.vec4f, 8),
  lightsB: d.arrayOf(d.vec4f, 8),
  /**
   * Planted objects, drawn in the composite: xy image position, z depth, and
   * the kind in w (0 empty, 1 cigarette, 2 torch, 3 campfire). Each prop also
   * occupies one of the leading lamp slots for its glow; propCount says how
   * many, so the orb pass leaves those slots to the drawn bodies.
   */
  // Five slots: four placed props plus the cursor preview while the Objects
  // tool is in hand - the object rides the pointer before it is planted.
  props: d.arrayOf(d.vec4f, 5),
  propCount: d.u32,
  /** Seconds, for the flames and coals that animate in the shader. */
  time: d.f32,
  /** The spout, drawn as a small lit sphere so its place in depth is legible. */
  spout: d.vec4f,
  /** Glowing glasses on a tracked face: lens centres, then colour + power. */
  glassesA: d.vec4f,
  glassesB: d.vec4f,
  /**
   * Detected drinking vessels, as field-space boxes. Inside one, the scene is
   * assumed transparent - a glass - so water behind its front face still
   * renders, which is what lets a held glass visibly fill.
   */
  cups: d.arrayOf(d.vec4f, 3),
  debug: d.u32,
});

/** Normal of the scene surface, for lighting the photo itself. */
const sceneNormalAt = (uv: d.v2f) => {
  'use gpu';
  const step = d.f32(2 / SURFACE_RES);
  const slope = d.vec2f(
    sceneAt(uv + d.vec2f(step, 0)) - sceneAt(uv - d.vec2f(step, 0)),
    sceneAt(uv + d.vec2f(0, step)) - sceneAt(uv - d.vec2f(0, step)),
  ) / (2 * step);
  return std.normalize(d.vec3f(slope * -1, 1));
};

/**
 * What one planted lamp adds to a point with the given normal. Inverse-square
 * over the lamp's own size, so a small lamp is a candle and a large one is a
 * ceiling bulb.
 */
const lampLight = (
  lampAt: d.v3f,
  power: number,
  tint: d.v3f,
  size: number,
  point: d.v3f,
  normal: d.v3f,
) => {
  'use gpu';
  const toLamp = lampAt - point;
  const reach = std.max(std.length(toLamp), 1e-4);
  const fall = power / (1 + (reach * reach) / std.max(size * size, 1e-4));
  return tint * (fall * std.saturate(std.dot(normal, toLamp / reach)));
};

const spin = (v: d.v2f, angle: number) => {
  'use gpu';
  const c = std.cos(angle);
  const s = std.sin(angle);
  return d.vec2f(c * v.x - s * v.y, s * v.x + c * v.y);
};

/**
 * Soft coverage of a rounded bar: p in the bar's own frame, x running from 0
 * to len along the axis, y across. The edge is feathered about a texel so the
 * solids below do not alias.
 */
const barMask = (p: d.v2f, len: number, girth: number) => {
  'use gpu';
  const along = std.clamp(p.x, 0, len);
  const gap = std.length(p - d.vec2f(along, 0)) - girth;
  return 1 - std.smoothstep(-0.0012, 0.0012, gap);
};

/**
 * One tongue of fire, in the flame's own frame: origin at the base, x across
 * in flame-widths, y up in flame-heights. Layered sines stand in for noise -
 * the same trick as the lamp flicker - swaying the tongue harder the further
 * up it goes, which is what rolls a steady cone into a live flame.
 */
const flameTongue = (q: d.v2f, time: number, seed: number) => {
  'use gpu';
  const sway = (std.sin(time * 6.3 + seed + q.y * 5.1) * 0.22 +
    std.sin(time * 11.7 + seed * 2.7 + q.y * 9.3) * 0.11) * q.y;
  const lick = 1 + 0.14 * std.sin(time * 9.1 + seed * 4.2);
  const x = q.x + sway;
  const y = q.y / lick;
  // A teardrop: widest a third of the way up, pinched toward the tip, and
  // nearly closed again at the base where the fuel is.
  const girth = std.max(std.sin(std.saturate(y * 0.86 + 0.1) * Math.PI) * (1 - y * 0.35), 0.02);
  const core = 1 - std.length(d.vec2f(x / girth, (y - 0.42) / 0.62));
  return std.saturate(core * 1.7);
};

/** Black-body-ish ramp: red at the edges, orange body, near-white core. */
const flameGlowColour = (heat: number) => {
  'use gpu';
  const body = std.mix(d.vec3f(0.85, 0.12, 0.02), d.vec3f(1, 0.55, 0.08), std.saturate(heat * 1.4));
  return std.mix(body, d.vec3f(1, 0.93, 0.62), std.saturate(heat * heat * 1.2));
};

/**
 * A cigarette. The coal sits at the planted point, so the standing smoke wisp
 * rises off it; the body runs down-right at an ashtray lean - charred paper
 * behind the coal, white paper, then the tan filter over the last third.
 */
const drawCigarette = (colour: d.v3f, uv: d.v2f, at: d.v3f, time: number, dim: number) => {
  'use gpu';
  const reach = 0.3 + at.z * 0.6;
  const len = 0.085 * reach;
  const girth = len * 0.052;
  // Local frame with x running from the coal toward the filter.
  const local = spin(uv - at.xy, -0.42);
  const body = barMask(local, len, girth);
  const t = std.saturate(local.x / len);

  let wrap = std.mix(
    d.vec3f(0.94, 0.92, 0.88),
    d.vec3f(0.83, 0.58, 0.28),
    std.smoothstep(0.66, 0.7, t),
  );
  // The band where the filter is glued on.
  wrap = std.mix(wrap, d.vec3f(0.58, 0.38, 0.18), std.smoothstep(0.63, 0.65, t) * (1 - std.smoothstep(0.67, 0.69, t)));
  // Char, right behind the coal.
  wrap = std.mix(d.vec3f(0.13, 0.12, 0.11), wrap, std.smoothstep(0.05, 0.13, t));
  // Rounded shading across the rod, so it reads as a rod and not a stripe.
  const across = std.saturate(std.abs(local.y) / girth);
  const shade = 0.72 + 0.28 * std.sqrt(std.max(1 - across * across, 0));
  let out = std.mix(colour, wrap * shade, body * dim);

  // The coal breathes: two sine rates that do not divide, like the lamp flicker.
  const coalSpan = std.length(local);
  const throb = 0.75 + 0.16 * std.sin(time * 9.3) + 0.09 * std.sin(time * 27.1 + 1.7);
  const coal = std.exp(-(coalSpan * coalSpan) / (girth * girth * 2.2)) * throb;
  out = out + d.vec3f(1, 0.3, 0.04) * (coal * 1.5 * dim) + d.vec3f(1, 0.8, 0.45) * (coal * coal * dim);
  return out;
};

/**
 * A torch: a leaning handle below the planted point, a dark wrapped head, and
 * a tongue of fire standing on it. The flame is emission; the handle is a
 * solid that ghosts like the spout when something stands nearer.
 */
const drawTorch = (colour: d.v3f, uv: d.v2f, at: d.v3f, time: number, dim: number) => {
  'use gpu';
  const reach = 0.3 + at.z * 0.6;
  const stickLen = 0.1 * reach;
  const stickGirth = 0.0065 * reach;
  // Handle runs down-screen with a slight lean; x along the handle.
  const local = spin(uv - at.xy, -(Math.PI / 2 - 0.1));
  const stick = barMask(local, stickLen, stickGirth);
  const head = barMask(local, stickLen * 0.26, stickGirth * 2.1);
  const across = std.saturate(std.abs(local.y) / (stickGirth * 2.1));
  const shade = 0.7 + 0.3 * std.sqrt(std.max(1 - across * across, 0));
  const grain = d.vec3f(0.24, 0.16, 0.09) * (0.91 + 0.09 * std.sin(local.x * 500));
  // The wrap: darker, with a diagonal binding line.
  const bind = 0.78 + 0.22 * std.sin((local.x + local.y * 2) * 900);
  let out = std.mix(colour, grain * shade, stick * dim);
  out = std.mix(out, d.vec3f(0.16, 0.1, 0.05) * (bind * shade), head * dim);

  // Two tongues, the inner one hotter and faster, standing on the head.
  const q = d.vec2f((uv.x - at.x) / (0.024 * reach), (at.y - uv.y) / (0.085 * reach));
  const seed = at.x * 43.7 + at.y * 17.3;
  const heat = std.saturate(
    flameTongue(q, time, seed) +
      flameTongue(d.vec2f(q.x * 1.7, q.y * 1.45 - 0.05), time * 1.35, seed + 3.1) * 0.7,
  );
  out = std.mix(out, flameGlowColour(heat), std.saturate(heat * 1.15) * dim);
  // A soft halo, so the fire bleeds a little light past its own edge.
  const haloSpan = std.length(d.vec2f(q.x, (q.y - 0.4) * 0.7));
  out = out + d.vec3f(1, 0.45, 0.1) * (std.exp(-haloSpan * haloSpan * 2.2) * 0.18 * dim);
  return out;
};

/**
 * A campfire: crossed logs on the ground, an ember bed pulsing between them,
 * and a broad fire of three tongues out of phase, so the whole body rolls
 * instead of waving one arm.
 */
const drawCampfire = (colour: d.v3f, uv: d.v2f, at: d.v3f, time: number, dim: number) => {
  'use gpu';
  const reach = 0.3 + at.z * 0.6;
  const logLen = 0.11 * reach;
  const logGirth = 0.009 * reach;
  const bed = at.xy + d.vec2f(0, 0.02 * reach);
  // Two logs crossed; each local frame starts at its own left end.
  const localA = spin(uv - bed, -0.24) + d.vec2f(logLen * 0.5, 0);
  const localB = spin(uv - bed, 0.31) + d.vec2f(logLen * 0.5, 0);
  const logA = barMask(localA, logLen, logGirth);
  const logB = barMask(localB, logLen, logGirth);

  // The ember bed under the logs, breathing slower than any flame.
  const emberSpan = std.length((uv - bed) / d.vec2f(1.6, 1));
  const throb = 0.7 + 0.2 * std.sin(time * 5.7) + 0.1 * std.sin(time * 13.9 + 0.8);
  const ember = std.exp(-(emberSpan * emberSpan) / (logGirth * logGirth * 30)) * throb;

  // Bark, lit from within by its own fire - gently, or the logs read as
  // striped candy rather than wood.
  const barkA = d.vec3f(0.27, 0.17, 0.1) * (0.92 + 0.08 * std.sin(localA.x * 520)) * (0.75 + ember * 0.5);
  const barkB = d.vec3f(0.2, 0.12, 0.07) * (0.92 + 0.08 * std.sin(localB.x * 640)) * (0.75 + ember * 0.5);
  let out = std.mix(colour, barkB, logB * dim);
  out = std.mix(out, barkA, logA * dim);
  out = out + d.vec3f(1, 0.35, 0.06) * (ember * 0.7 * dim);

  // One tall tongue and two short ones tucked close, so the fire reads as a
  // single rolling body rather than a pair of horns.
  const q = d.vec2f((uv.x - at.x) / (0.05 * reach), (at.y - uv.y) / (0.12 * reach));
  const seed = at.x * 31.9 + at.y * 23.1;
  const heat = std.saturate(
    flameTongue(q, time, seed) +
      flameTongue(d.vec2f((q.x + 0.4) * 1.5, q.y * 1.9), time * 1.2, seed + 2.4) * 0.7 +
      flameTongue(d.vec2f((q.x - 0.38) * 1.6, q.y * 2.1 + 0.05), time * 0.9, seed + 5.2) * 0.7,
  );
  out = std.mix(out, flameGlowColour(heat), std.saturate(heat * 1.2) * dim);
  const haloSpan = std.length(d.vec2f(q.x * 0.8, (q.y - 0.3) * 0.6));
  out = out + d.vec3f(1, 0.4, 0.08) * (std.exp(-haloSpan * haloSpan * 1.8) * 0.22 * dim);
  return out;
};

const splatLayout = tgpu.bindGroupLayout({
  splat: { uniform: SplatParams },
  field: { uniform: FieldParams },
  scene: { texture: d.texture2d(d.f32) },
  linear: { sampler: 'filtering' },
});

/**
 * Particles reach the splat shader as instance attributes rather than a storage
 * buffer read. Chrome's compatibility mode - what older Android devices get -
 * allows zero storage buffers in the vertex stage, and this is the only place
 * that would have needed one.
 */
const particleLayout = tgpu.vertexLayout(d.arrayOf(Particle), 'instance');

const compositeLayout = tgpu.bindGroupLayout({
  field: { uniform: FieldParams },
  look: { uniform: LookParams },
  fluid: { texture: d.texture2d(d.f32) },
  /** Unfiltered surface, so the composite can see how broken up it is. */
  rawFluid: { texture: d.texture2d(d.f32) },
  /** The scene with passing occluders forgotten; where lamps actually hang. */
  sceneBack: { texture: d.texture2d(d.f32) },
  thickness: { texture: d.texture2d(d.f32) },
  scene: { texture: d.texture2d(d.f32) },
  /** Ray-marched smoke, premultiplied colour in rgb and coverage in a. */
  smoke: { texture: d.texture2d(d.f32) },
  linear: { sampler: 'filtering' },
});

const frameLayout = tgpu.bindGroupLayout({
  frame: { externalTexture: d.textureExternal() },
  camera: { uniform: CameraParams },
});

/** Comptime switch: with no camera attached the page still needs something behind the liquid. */
const cameraSlot = tgpu.slot<boolean>();

const splatVertex = tgpu.vertexFn({
  in: { pos: d.vec3f, vel: d.vec3f, vertex: d.builtin.vertexIndex },
  out: { position: d.builtin.position, offset: d.vec2f, centre: d.f32, spot: d.vec2f },
})(({ pos, vel, vertex }) => {
  'use gpu';
  const corner = quadCorners.$[vertex];
  const centre = d.vec3f(pos);
  const radius = splatLayout.$.splat.radius;

  // Fast water stretches along its motion. A stream one particle wide - a
  // pour in flight, a runnel down the tub's front - rendered as round beads
  // is a dotted line: the beads never merge, coverage stays under the surface
  // gate, and the water is invisible exactly where it moves. Elongating each
  // splat into its neighbour turns the dots into one glassy streak. Resting
  // water is untouched; the stretch only wakes above walking pace.
  const speed = std.length(vel.xy);
  const stretch = std.min(std.max(speed - 0.15, 0) * 0.08, radius * 5);
  let along = d.vec2f(0, 1);
  if (speed > 1e-4) {
    along = vel.xy / speed;
  }
  const across = d.vec2f(-along.y, along.x);

  // Image space is the unit square with y down; clip space is [-1, 1] with y up,
  // so a world radius spans twice as much clip space.
  const point = centre.xy +
    along * (corner.x * (radius + stretch)) +
    across * (corner.y * radius);
  return {
    position: d.vec4f(point.x * 2 - 1, 1 - point.y * 2, 0.5, 1),
    offset: d.vec2f(corner),
    centre: centre.z,
    spot: d.vec2f(point),
  };
});

/**
 * Scene depth at a splat's own texel, for occlusion at the source. The splat
 * passes are depth-blind toward the SCENE: thickness is additive in 2D, so
 * water deliberately spouted behind a person still piled its optical mass
 * onto their face, and one stray front droplet made the whole hidden column
 * render. Water deeper than the scene by more than a jitter allowance simply
 * does not splat - it is behind something, and the something is opaque.
 */
const splatOccluderAt = (uv: d.v2f) => {
  'use gpu';
  return (
    std.textureSample(splatLayout.$.scene, splatLayout.$.linear, uv).x *
    splatLayout.$.field.depthScale
  );
};

const inPourChannel = (uv: d.v2f) => {
  'use gpu';
  let inside = false;
  for (const slot of std.range(3)) {
    const mouth = splatLayout.$.splat.mouths[slot];
    if (mouth.w > 0.5 && uv.x > mouth.x && uv.x < mouth.y && uv.y < mouth.z) {
      inside = true;
    }
  }
  return inside;
};

/**
 * Nearest-surface pass. Each particle is a sphere; the depth test keeps whichever
 * is closest to the camera, and the colour target carries that depth alongside a
 * coverage flag so the blur afterwards can average only over covered texels.
 */
const splatDepthFragment = tgpu.fragmentFn({
  in: { offset: d.vec2f, centre: d.f32, spot: d.vec2f },
  out: { surface: d.vec4f, depth: d.builtin.fragDepth },
})(({ offset, centre, spot }) => {
  'use gpu';
  const occluder = splatOccluderAt(spot);
  const hiddenHere = centre < occluder - d.f32(KERNEL_RADIUS) * 2 && !inPourChannel(spot);
  const radial = std.dot(offset, offset);
  if (radial > 1 || hiddenHere) {
    std.discard();
  }
  const bulge = std.sqrt(std.max(1 - radial, 0)) * splatLayout.$.splat.radius;
  const nearness = centre + bulge;
  return {
    surface: d.vec4f(nearness, 1, 0, 1),
    // Larger nearness is closer, so invert it into a normal depth test.
    depth: std.saturate(1 - nearness / d.f32(Z_MAX)),
  };
});

const splatThicknessFragment = tgpu.fragmentFn({
  in: { offset: d.vec2f, centre: d.f32, spot: d.vec2f },
  out: d.vec4f,
})(({ offset, centre, spot }) => {
  'use gpu';
  const occluder = splatOccluderAt(spot);
  const hiddenHere = centre < occluder - d.f32(KERNEL_RADIUS) * 2 && !inPourChannel(spot);
  const radial = std.dot(offset, offset);
  if (radial > 1 || hiddenHere) {
    std.discard();
  }
  const falloff = 1 - radial;
  return d.vec4f(falloff * falloff * falloff, 0, 0, 1);
});

/**
 * Temporal accumulation of the filtered surface.
 *
 * The solver leaves a residual jitter of well under a millimetre per step. The
 * body of the water does not care - thickness is additive, so the BODY view is
 * glass smooth - but the surface pass keeps whichever particle is nearest per
 * texel, and jitter re-decides that election every frame. The reconstructed
 * surface twitches, and every term derived from it - normals, glints, caustics,
 * foam - twitches with it. That is the boiling, and it is a rendering artefact:
 * the liquid underneath is still.
 *
 * The fix is the standard one for screen-space fluids: blend each frame's
 * filtered surface toward the last presented one, hard when the surface is
 * nearly unchanged (settled water), not at all when it moves a real distance
 * (a splash must not ghost). The blend is per-texel, so a pour landing in a
 * still pool ripples where it lands and stays calm elsewhere.
 */
const TEMPORAL_BLEND = 0.88;

const temporalLayout = tgpu.bindGroupLayout({
  current: { texture: d.texture2d(d.f32) },
  history: { texture: d.texture2d(d.f32) },
  target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
});

const temporalKernel = tgpu.computeFn({
  workgroupSize: [8, 8],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= d.u32(SURFACE_RES) || gid.y >= d.u32(SURFACE_RES)) {
    return;
  }
  const current = std.textureLoad(temporalLayout.$.current, gid.xy, 0);
  const previous = std.textureLoad(temporalLayout.$.history, gid.xy, 0);

  // The whole texel blends, coverage included, and there is deliberately no
  // gate of any kind. Two gates were tried and each one released exactly where
  // it was needed most. A motion gate let go whenever jitter flipped which
  // particle was nearest, because that flip steps the surface by a whole bead
  // height and per-frame motion cannot tell it from a wave. A coverage gate
  // then pinned the body but left the waterline boiling, because the free
  // surface is precisely where particles pop in and out of the splat election
  // and coverage itself flickers. Blending coverage means an arriving or
  // leaving edge fades over a few frames instead of cutting - masked by its
  // own low alpha, and honestly closer to how a meniscus moves anyway.
  std.textureStore(
    temporalLayout.$.target,
    gid.xy,
    std.mix(current, previous, d.f32(TEMPORAL_BLEND)),
  );
});

/** Publishes the blended surface back to both the live and history textures. */
const publishLayout = tgpu.bindGroupLayout({
  source: { texture: d.texture2d(d.f32) },
  live: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
  keep: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
});

const publishKernel = tgpu.computeFn({
  workgroupSize: [8, 8],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= d.u32(SURFACE_RES) || gid.y >= d.u32(SURFACE_RES)) {
    return;
  }
  const value = std.textureLoad(publishLayout.$.source, gid.xy, 0);
  std.textureStore(publishLayout.$.live, gid.xy, value);
  std.textureStore(publishLayout.$.keep, gid.xy, value);
});

/** Filtered surface nearness. Channel layout is (depth, coverage). */
const fluidAt = (uv: d.v2f) => {
  'use gpu';
  return std.textureSample(compositeLayout.$.fluid, compositeLayout.$.linear, uv).x;
};

const coverageAt = (uv: d.v2f) => {
  'use gpu';
  return std.textureSample(compositeLayout.$.fluid, compositeLayout.$.linear, uv).y;
};

const sceneBackAt = (uv: d.v2f) => {
  'use gpu';
  return (
    std.textureSample(compositeLayout.$.sceneBack, compositeLayout.$.linear, uv).x *
    compositeLayout.$.field.depthScale
  );
};

const sceneAt = (uv: d.v2f) => {
  'use gpu';
  return (
    std.textureSample(compositeLayout.$.scene, compositeLayout.$.linear, uv).x *
    compositeLayout.$.field.depthScale
  );
};

/**
 * Field UV to camera-texture UV, matching `DepthFramePreprocessor` step for
 * step: mirror, centred square crop, then the orientation transform.
 */
const cameraUv = (uv: d.v2f) => {
  'use gpu';
  const mirrored = d.vec2f(std.select(uv.x, 1 - uv.x, frameLayout.$.camera.mirror !== 0), uv.y);

  let sourceSize = d.vec2f(std.textureDimensions(frameLayout.$.frame));
  if (frameLayout.$.camera.swapAxes !== 0) {
    sourceSize = d.vec2f(sourceSize.yx);
  }

  // The crop keeps the shorter side whole and trims the longer one evenly.
  const scale = std.min(sourceSize.x, sourceSize.y) / sourceSize;
  const cropped = (d.vec2f(1) - scale) * 0.5 + mirrored * scale;

  return frameLayout.$.camera.uvTransform * (cropped - d.vec2f(0.5)) + d.vec2f(0.5);
};

const backdropAt = (uv: d.v2f) => {
  'use gpu';
  if (cameraSlot.$) {
    return std.textureSampleBaseClampToEdge(
      frameLayout.$.frame,
      compositeLayout.$.linear,
      cameraUv(std.saturate(uv)),
    ).rgb;
  }
  // No camera: shade the scene depth so the collision geometry stays visible.
  const nearness = sceneAt(uv) / std.max(compositeLayout.$.field.depthScale, 0.0001);
  return std.mix(d.vec3f(0.05, 0.06, 0.08), d.vec3f(0.36, 0.39, 0.44), nearness);
};

/** Water reflects about 2% straight on. The rest of the curve is Schlick's. */
const WATER_F0 = 0.02;
/** Air into water. */
const WATER_IOR = 1 / 1.33;

const rawFluidAt = (uv: d.v2f) => {
  'use gpu';
  return std.textureSample(compositeLayout.$.rawFluid, compositeLayout.$.linear, uv).xy;
};

/**
 * Texture sampling in a fragment shader must sit in uniform control flow, so
 * there are no early returns on per-pixel conditions here: everything is
 * computed, then selected. The debug branches are safe because they test a
 * uniform.
 */
const compositeFragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
  'use gpu';
  const look = compositeLayout.$.look;

  const coverage = coverageAt(uv);
  const fluidZ = fluidAt(uv);
  const sceneZ = sceneAt(uv);
  const thick = std.textureSample(compositeLayout.$.thickness, compositeLayout.$.linear, uv).x;

  // Surface normal straight off the reconstructed depth. Two texels apart: one
  // is inside the filter's own footprint and reads as noise rather than slope.
  const step = d.f32(2 / SURFACE_RES);
  const right = fluidAt(uv + d.vec2f(step, 0));
  const left = fluidAt(uv - d.vec2f(step, 0));
  const below = fluidAt(uv + d.vec2f(0, step));
  const above = fluidAt(uv - d.vec2f(0, step));
  const slope = d.vec2f(right - left, below - above) / (2 * step);
  const normal = std.normalize(d.vec3f(slope * -look.relief, 1));

  // Highlights get their own, wider normal. A specular lobe raised to the
  // 200th power turns sub-pixel normal wobble - invisible in refraction - into
  // visible sparkle flicker; measured on a settled tub, specular and caustics
  // were HALF the remaining frame-to-frame churn. Sampling the slope over a
  // wider stencil for the shiny terms only keeps the crisp refraction detail
  // and calms the glints.
  const wideStep = d.f32(5 / SURFACE_RES);
  const wideSlope = d.vec2f(
    fluidAt(uv + d.vec2f(wideStep, 0)) - fluidAt(uv - d.vec2f(wideStep, 0)),
    fluidAt(uv + d.vec2f(0, wideStep)) - fluidAt(uv - d.vec2f(0, wideStep)),
  ) / (2 * wideStep);
  const glossNormal = std.normalize(d.vec3f(wideSlope * -look.relief, 1));

  // A real view ray, even though the simulation is orthographic.
  const toEye = std.normalize(d.vec3f((d.vec2f(0.5) - uv), look.lens));
  const incident = toEye * -1;
  const facing = std.saturate(std.dot(normal, toEye));

  const alpha = std.smoothstep(look.surfaceLow, look.surfaceHigh, coverage);
  // Real occlusion: scene geometry standing nearer than the liquid hides it.
  let hidden = std.smoothstep(0, 0.015, sceneZ - fluidZ);
  // Except inside a detected glass, which is transparent to its own contents.
  for (const slot of std.range(3)) {
    const cup = look.cups[slot];
    if (
      cup.z > cup.x &&
      uv.x > cup.x && uv.x < cup.z && uv.y > cup.y && uv.y < cup.w
    ) {
      hidden *= 0.15;
    }
    // And in the pour channel above it: the stream between camera and glass
    // is in front of everything in reality, whatever the depth halo says.
    const mouth = look.cupMouths[slot];
    if (mouth.w > 0.5 && uv.x > mouth.x && uv.x < mouth.y && uv.y < mouth.z) {
      hidden *= 0.05;
    }
  }

  // Optical depth: how much liquid lies along the ray. Deliberately unbounded.
  // Clamping this to one was throwing away every bit of depth the liquid had -
  // the splats saturate at about four deep, so a pool fifty particles deep came
  // out the same flat value as a puddle, and nothing in the picture said which
  // was which.
  const optical = thick * look.thickness;
  // Bounded companion, for the terms below that are ratios rather than
  // distances and have no meaning past one.
  const thickness = std.saturate(optical);

  // How opaque the liquid is has to follow how much of it lies along the ray,
  // not merely whether any of it does. Coverage is one wherever a splat lands,
  // so a single airborne droplet covers its texels completely and comes out as
  // an opaque bead of glass - which is most of why spray read as loose
  // particles rather than as water. Thickness is additive across particles, so
  // it separates a droplet from a body of liquid.
  // Opacity follows the same exponential the absorption does. A droplet is
  // thin, so it is faint and mostly shows what is behind it, bent - which is
  // what a droplet is. The old hard cull put the floor at roughly a particle
  // and a half of overlap, so spray, the leading edge of a pour and the whole
  // arc of a splash were not faint, they were absent, and water only appeared
  // once it had already landed and pooled.
  const body = 1 - std.exp(-optical * BODY_DENSITY);
  const visible = alpha * body * (1 - hidden);
  let plain = backdropAt(uv);
  // Planted lamps fall on the photo itself. This is most of the light demo:
  // the room brightens where the lamp is, dims with distance, and shades with
  // the scene's own measured geometry.
  const scenePoint = d.vec3f(uv, sceneZ);
  const sceneNormal = sceneNormalAt(uv);
  let sceneGlow = d.vec3f();
  for (const slot of std.range(8)) {
    const a = look.lightsA[slot];
    if (a.w > 0.001) {
      const b = look.lightsB[slot];
      // The lamp sits at the depth it was placed at (the scroll wheel), with
      // the remembered scene as a floor - never behind the wall it was clicked
      // on, and never remounted onto a hand passing in front. When the live
      // surface stands nearer than the lamp, the lamp is behind it, so its
      // light dims instead of shining through.
      const hungZ = std.max(a.z, sceneBackAt(a.xy) + 0.06);
      const hung = d.vec3f(a.xy, hungZ);
      let open = d.f32(1);
      if (sceneAt(a.xy) > hungZ + 0.06) {
        open = 0.12;
      }
      sceneGlow = sceneGlow +
        lampLight(hung, a.w * open, b.xyz, b.w, scenePoint, sceneNormal);
    }
  }
  if (look.torch > 0.001) {
    sceneGlow = sceneGlow +
      lampLight(look.torchAt, look.torch, d.vec3f(1, 0.6, 0.26), 0.14, scenePoint, sceneNormal);
  }
  // Light-up glasses cast onto the face and the room like two small lamps.
  if (look.glassesB.w > 0.001) {
    const lensL = look.glassesA.xy;
    const lensR = look.glassesA.zw;
    sceneGlow = sceneGlow +
      lampLight(d.vec3f(lensL, sceneAt(lensL) + 0.05), look.glassesB.w * 0.7, look.glassesB.xyz, 0.09, scenePoint, sceneNormal) +
      lampLight(d.vec3f(lensR, sceneAt(lensR) + 0.05), look.glassesB.w * 0.7, look.glassesB.xyz, 0.09, scenePoint, sceneNormal);
  }
  // Compressed so a bright lamp saturates toward its own colour instead of
  // clipping the tile to paper white.
  const glowTone = sceneGlow / (d.vec3f(1) + sceneGlow * 0.55);
  plain = plain * (1 + glowTone * 1.4) + glowTone * 0.05;

  // Refraction through a real interface rather than a nudge along the normal.
  // Sampled per channel with slightly different indices - dispersion. Real
  // water does this; it is why edges of ripples carry a faint colour fringe,
  // and its absence is one of the quiet tells of fake water.
  let bend = std.refract(incident, normal, WATER_IOR);
  if (std.dot(bend, bend) < 0.0001) {
    bend = d.vec3f(incident);
  }
  const shift = look.refraction * thickness;
  const behindR = backdropAt(uv + bend.xy * (shift * 1.045));
  const behindG = backdropAt(uv + bend.xy * shift);
  const behindB = backdropAt(uv + bend.xy * (shift * 0.955));
  const behind = d.vec3f(behindR.r, behindG.g, behindB.b);

  // Caustics: a curved surface focuses light into bright ripple lines on
  // whatever lies beneath. Curvature is the Laplacian of the surface depth,
  // measured on a deliberately wide stencil - at the two-texel spacing used for
  // normals it just picks out the bump of each individual particle and lays a
  // filigree over the whole pool.
  const wide = d.f32(11 / SURFACE_RES);
  // The dead zone keeps residual surface noise from twinkling the caustic
  // lines on and off; real ripples curve far harder than the floor here.
  const curvature = std.max(
    std.abs(
      fluidAt(uv + d.vec2f(wide, 0)) +
        fluidAt(uv - d.vec2f(wide, 0)) +
        fluidAt(uv + d.vec2f(0, wide)) +
        fluidAt(uv - d.vec2f(0, wide)) -
        fluidZ * 4,
    ) - 0.012,
    0,
  );
  // Gain divided so the term stops living at saturation, where any noise
  // crossing the threshold rerolls the whole pattern every frame.
  const focus = 1 + std.saturate(curvature * (look.caustics * 70)) * thickness;

  // Beer-Lambert: the further light travels through the liquid, the more of
  // everything but the tint colour is absorbed.
  // Beer-Lambert on the unbounded depth, so a metre of water absorbs like a
  // metre of water. This is where the sense of depth actually comes from.
  const transmitted =
    behind * std.exp((look.tint - d.vec3f(1)) * (look.absorption * optical)) * focus;
  // Without this the liquid is invisible against a dark scene: transmission
  // alone can only ever darken what is already behind it.
  const scattered = look.tint * (look.scatter * thickness);

  // Reflection, approximated by walking the scene image along the bounce.
  const bounce = std.reflect(incident, glossNormal);
  const mirrored = backdropAt(uv + bounce.xy * look.reflection);

  // Schlick. This ratio - clear straight on, mirror at a glance - is most of
  // what makes a surface read as water rather than tinted glass.
  const fresnel = WATER_F0 + (1 - WATER_F0) * std.pow(1 - facing, 5);

  // Layered specular: a tight highlight off micro-detail plus a broad sheen off
  // the overall curvature reads as a wet surface; one lobe alone reads as
  // plastic or as nothing at all.
  const glint = std.pow(std.saturate(std.dot(bounce, look.sun)), 220) * look.specular;
  const sheen = std.pow(std.saturate(std.dot(glossNormal, std.normalize(look.sun + toEye))), 48) *
    look.specular * 0.25;
  const sparkle = std.pow(
    std.saturate(std.dot(bounce, look.sun)),
    900,
  ) * look.specular * 0.8;

  // Foam where the surface is broken up: the filtered surface is smooth, the raw
  // one is not, and the gap between them is the churn. Only where the raw buffer
  // actually holds a sample - an uncovered texel reads zero depth, and taking
  // that difference at face value laces the whole surface with white. Gated on
  // thin water too, because whitewater is aerated spray, not the body of a pool.
  // Whitewater is aerated water at a broken surface, not a lone drop in the air.
  // The gap between raw and filtered depth is largest of all at an isolated
  // droplet, so without a coverage gate the foam term picks out exactly the
  // particles that should be least visible and paints them white.
  const raw = rawFluidAt(uv);
  let churn = d.f32(0);
  if (raw.y > 0.5) {
    // Aeration lives in a band of depth. Too thin and there is no surface to
    // break - a lone droplet is clear glass, not foam - and too thick and you
    // are looking into the bulk, which is where the old upper edge already cut
    // it off. The lower edge is the new half: the raw-to-filtered gap this term
    // keys on is largest of all at an isolated droplet, so without it the foam
    // picked out exactly the particles that should read as glass and painted
    // them white. Invisible before, because those droplets were culled anyway.
    const aerated =
      std.smoothstep(0.2, 0.6, optical) * (1 - std.smoothstep(0.8, 1.8, optical));
    // Below the floor, a raw-vs-filtered gap is solver jitter, not aeration.
    // The residual jitter re-rolls the raw buffer every frame, and multiplied
    // by the foam gain it painted flickering white speckle across settled
    // water - the boiling. A genuinely broken surface gaps by several kernel
    // radii, so the floor costs real whitewater nothing.
    const gap = std.max(std.abs(raw.x - fluidZ) - d.f32(KERNEL_RADIUS) * 1.2, 0);
    churn =
      std.saturate(gap * look.foam) *
      aerated *
      std.smoothstep(0.35, 0.8, coverage);
  }
  // Point light. Distance is measured in the scene's own space, so a torch held
  // near the surface pools brightly and one across the room barely reaches.
  const toTorch = look.torchAt - d.vec3f(uv, fluidZ);
  const reach = std.max(std.length(toTorch), 1e-4);
  const torchDir = toTorch / reach;
  const TORCH_TINT = d.vec3f(1, 0.6, 0.26);
  const torchFall = look.torch / (1 + reach * reach * 26);
  const torchLit = TORCH_TINT * torchFall *
    (std.saturate(std.dot(normal, torchDir)) * 0.9 +
      std.pow(std.saturate(std.dot(bounce, torchDir)), 90) * 0.6);

  // Planted lamps land on the water too, at the same hung positions.
  let lampLit = d.vec3f();
  const waterPoint = d.vec3f(uv, fluidZ);
  for (const slot of std.range(8)) {
    const a = look.lightsA[slot];
    if (a.w > 0.001) {
      const b = look.lightsB[slot];
      const hungZ = std.max(a.z, sceneBackAt(a.xy) + 0.06);
      let open = d.f32(1);
      if (sceneAt(a.xy) > hungZ + 0.06) {
        open = 0.12;
      }
      lampLit = lampLit +
        lampLight(d.vec3f(a.xy, hungZ), a.w * open, b.xyz, b.w, waterPoint, normal);
    }
  }

  const water = std.mix(transmitted + scattered, mirrored, fresnel);
  const lit = std.mix(water, d.vec3f(0.92, 0.95, 0.97), churn) +
    d.vec3f(glint + sheen + sparkle) + torchLit + lampLit * 0.6;

  // Depth-graded light. The scene is lit from the camera's side, so water far
  // back sits further from every source and under more atmosphere: dimmer,
  // cooler, lower contrast. Uniform shading across depth is what made deep and
  // near water read identically flat - "mono-lit".
  //
  // `fluidZ` is nearness, not distance - larger is closer. Grading on it
  // directly put the haze on the near water and left the far water at full
  // brightness, which is the effect backwards.
  const distance = std.saturate(1 - fluidZ / d.f32(Z_MAX));
  const aerial = std.exp(-distance * 1.4);
  const graded = lit * (0.35 + 0.65 * aerial) *
    std.mix(d.vec3f(0.85, 0.92, 1.05), d.vec3f(1), aerial);

  let colour = std.mix(plain, graded, visible);

  // Floating objects are solid, so they replace what is behind them rather than
  // bending it. They share the liquid's depth buffer, which is what makes a duck
  // half under the surface look half under the surface: the same nearest-wins
  // test decides water-in-front from object-in-front, per texel.

  // A whisper of an outline on every detected vessel, so whether the detector
  // fired is never a mystery - but only while the tune drawer is open, the
  // same rule as the gravity arrow: diagnostics live with the controls.
  for (const slot of std.range(3)) {
    const cup = look.cups[slot];
    if (look.cupLines > 0.5 && cup.z > cup.x) {
      const inX = std.min(uv.x - cup.x, cup.z - uv.x);
      const inY = std.min(uv.y - cup.y, cup.w - uv.y);
      const inset = std.min(inX, inY);
      if (inset > 0 && inset < 0.004) {
        colour = std.mix(colour, d.vec3f(0.55, 0.85, 1.0), 0.35);
      }
    }
  }

  // The glasses themselves: two glowing rings and a bridge, drawn on the face
  // before the smoke composites, so what stands in front still covers them.
  if (look.glassesB.w > 0.001) {
    const lensL = look.glassesA.xy;
    const lensR = look.glassesA.zw;
    const span = std.max(std.length(lensR - lensL), 1e-4);
    const lensSize = std.clamp(span * 0.32, 0.012, 0.06);
    const ringL = std.abs(std.length(uv - lensL) - lensSize);
    const ringR = std.abs(std.length(uv - lensR) - lensSize);
    const ring = std.min(ringL, ringR);
    const axis = (lensR - lensL) / span;
    const along = std.clamp(std.dot(uv - lensL, axis), 0, span);
    const bridge = std.length(uv - (lensL + axis * along));
    const glowLine = std.min(ring, std.max(bridge - lensSize * 0.15, 0));
    const shine = std.exp(-std.max(glowLine, 0) / 0.004) * look.glassesB.w;
    colour = colour + look.glassesB.xyz * shine;
  }

  // Every placed lamp gets a body: a small emissive orb at the depth it hangs,
  // occluded by whatever stands nearer, so its place in the room is readable
  // even before its glow lands anywhere.
  for (const slot of std.range(8)) {
    const a = look.lightsA[slot];
    // The leading slots belong to planted props, which get a drawn body below
    // instead of the generic orb.
    if (a.w > 0.001 && slot >= look.propCount) {
      const orbAt = d.vec2f(a.x, a.y);
      const orbSpan = std.length(uv - orbAt);
      // Generous pre-test with the largest size an orb can reach, so the
      // texture samples below only run near a lamp at all.
      if (orbSpan < 0.022 * 3.4) {
        const b = look.lightsB[slot];
        // Explicit-level samples: this branch is per-pixel, and implicit
        // derivatives are not allowed in non-uniform control flow.
        const backHere = std.textureSampleLevel(
          compositeLayout.$.sceneBack, compositeLayout.$.linear, orbAt, 0,
        ).x * compositeLayout.$.field.depthScale;
        const liveHere = std.textureSampleLevel(
          compositeLayout.$.scene, compositeLayout.$.linear, orbAt, 0,
        ).x * compositeLayout.$.field.depthScale;
        const orbZ = std.max(a.z, backHere + 0.06);
        // Size follows depth, the one perspective cue an orthographic composite
        // can still give: a lamp by the camera is a bulb, one across the room a
        // distant point.
        const orbSize = d.f32(0.0045) + orbZ * 0.011;
        // A lamp is not a painted ball - it glows. The body is emissive:
        // a hot near-white core inside a soft additive halo of the lamp's
        // own colour, so the source reads as the brightest thing in its
        // neighbourhood instead of a dark bead the light pretends to come
        // from.
        const core = std.exp(-(orbSpan * orbSpan) / (orbSize * orbSize * 0.35));
        const halo = std.exp(-(orbSpan * orbSpan) / (orbSize * orbSize * 3.5));
        let strength = std.min(a.w, 1.2);
        if (liveHere > orbZ + 0.05) {
          strength *= 0.15;
        }
        colour = colour +
          b.xyz * (halo * 0.7 * strength) +
          std.mix(b.xyz, d.vec3f(1), 0.75) * (core * 1.25 * strength);
      }
    }
  }

  // Planted objects, drawn as the things they are. Each carries its own lamp
  // in the slots the orb loop skips above, so the glow is already in the
  // scene; this pass adds the body the glow comes from. Same occlusion rule
  // as the orbs: something standing nearer ghosts the prop.
  for (const slot of std.range(5)) {
    const prop = look.props[slot];
    if (prop.w > 0.5) {
      const reach = 0.3 + prop.z * 0.6;
      if (std.length(uv - prop.xy) < 0.2 * reach) {
        // Explicit-level sample: this branch is per-pixel.
        const liveHere = std.textureSampleLevel(
          compositeLayout.$.scene, compositeLayout.$.linear, prop.xy, 0,
        ).x * compositeLayout.$.field.depthScale;
        let dim = d.f32(1);
        if (liveHere > prop.z + 0.05) {
          dim = 0.15;
        }
        const at = d.vec3f(prop.xyz);
        if (prop.w < 1.5) {
          colour = drawCigarette(colour, uv, at, look.time, dim);
        } else if (prop.w < 2.5) {
          colour = drawTorch(colour, uv, at, look.time, dim);
        } else {
          colour = drawCampfire(colour, uv, at, look.time, dim);
        }
      }
    }
  }

  // The spout, drawn as a small solid sphere at its true position and depth.
  // A flat ring said where the spout was in the image; a sphere that shades
  // with the measured light, shrinks with distance, and slides behind whatever
  // stands nearer says where it is in the room, which is the part that was
  // impossible to read. When it is hidden it stays as a faint ghost, so it can
  // be found without un-hiding it.
  let spoutAt = look.spout.xyz;
  // The light tool's marker sits where the lamp actually hangs - the same
  // depth rule as the glow - not at the raw spout depth, which floats in
  // front of the whole scene and lies about where the light is.
  if (std.abs(look.spout.w - 0.8) < 0.01) {
    spoutAt = d.vec3f(
      spoutAt.xy,
      std.max(spoutAt.z, sceneBackAt(spoutAt.xy) + 0.06),
    );
  }
  const spoutOffset = uv - spoutAt.xy;
  const spoutSize = d.f32(0.011) + spoutAt.z * 0.012;
  const spoutRadial = std.length(spoutOffset) / spoutSize;
  if (spoutRadial < 1) {
    const bulge = std.sqrt(std.max(1 - spoutRadial * spoutRadial, 0));
    const front = spoutAt.z + bulge * spoutSize;
    const ballNormal = std.normalize(d.vec3f(spoutOffset / spoutSize, bulge));
    const lit = std.saturate(std.dot(ballNormal, look.sun)) * 0.55 + 0.3;
    const ballGlint = std.pow(
      std.saturate(std.dot(std.reflect(incident, ballNormal), look.sun)),
      80,
    ) * 0.5;
    const ball = d.vec3f(1, 0.72, 0.34) * lit + d.vec3f(ballGlint);
    // Behind the scene, or under water: a ghost, not a bead.
    let solid = d.f32(0.85) * look.spout.w;
    if (sceneZ > front) {
      solid = 0.14;
    } else if (fluidZ > front && coverage > 0.4) {
      solid *= 0.45;
    }
    // Soft edge, one texel wide.
    const rim = 1 - std.smoothstep(0.86, 1, spoutRadial);
    colour = std.mix(colour, ball, solid * rim);
  }

  // Smoke sits between the scene and the eye and is already premultiplied, so
  // laying it over is one multiply-add. Its march stopped at the scene surface,
  // which is what keeps a plume behind a mug rather than painted across it.
  const haze = std.textureSample(compositeLayout.$.smoke, compositeLayout.$.linear, uv);
  colour = haze.rgb + colour * (1 - haze.a);

  // Lightning falls on the scene as well as the liquid, so it goes on last.
  colour = colour * (1 + look.flash * 0.7) + d.vec3f(look.flash * 0.05);

  if (look.debug === DebugView.SCENE_DEPTH) {
    colour = d.vec3f(sceneZ / std.max(compositeLayout.$.field.depthScale, 0.0001));
  } else if (look.debug === DebugView.FLUID_DEPTH) {
    colour = d.vec3f(fluidZ / d.f32(Z_MAX)) * std.step(0.02, coverage);
  } else if (look.debug === DebugView.THICKNESS) {
    // Optical depth, not the clamped copy - the point of this view is to show
    // how deep the liquid gets, which a clamped value cannot.
    colour = d.vec3f(optical * 0.25);
  } else if (look.debug === DebugView.SMOKE) {
    colour = d.vec3f(haze.a);
  }

  return d.vec4f(colour, 1);
});

export interface LiquidLook {
  tint: [number, number, number];
  surfaceLow: number;
  surfaceHigh: number;
  relief: number;
  refraction: number;
  thickness: number;
  absorption: number;
  scatter: number;
  specular: number;
  /** Brightness of the placed torch. Zero is off. */
  torch: number;
  torchAt: readonly [number, number, number];
  glassesA: readonly [number, number, number, number];
  glassesB: readonly [number, number, number, number];
  cups: readonly (readonly [number, number, number, number])[];
  /** Non-zero draws the detected-vessel outlines; tied to the tune drawer. */
  cupLines: number;
  /** Pour channels above cup mouths: xLeft, xRight, yEnd, active. */
  cupMouths: readonly (readonly [number, number, number, number])[];
  /** Planted lamps: xyz position + power, then rgb tint + size. */
  lightsA: readonly (readonly [number, number, number, number])[];
  lightsB: readonly (readonly [number, number, number, number])[];
  /** Planted objects for the composite: xy, depth, kind (0 empty). */
  props: readonly (readonly [number, number, number, number])[];
  /** Spout marker: image xy, depth z, and ring emphasis in w. */
  spout: readonly [number, number, number, number];
  lens: number;
  reflection: number;
  caustics: number;
  foam: number;
  flash: number;
  /** Key light bearing, in degrees clockwise from straight up the image. */
  sunBearing: number;
  /** Key light height, in degrees above the image plane. */
  sunHeight: number;
  splatRadius: number;
  debug: number;
}

export const defaultLook: LiquidLook = {
  tint: [0.74, 0.89, 0.96],
  surfaceLow: 0.1,
  surfaceHigh: 0.62,
  relief: 0.25,
  refraction: 0.09,
  thickness: 0.5,
  absorption: 1.1,
  scatter: 0.1,
  specular: 1.1,
  torch: 0,
  torchAt: [0.5, 0.4, Z_MAX * 0.8],
  glassesA: [0.4, 0.4, 0.6, 0.4],
  glassesB: [1, 1, 1, 0],
  cups: Array.from({ length: 3 }, () => [0, 0, 0, 0] as const),
  cupLines: 0,
  cupMouths: Array.from({ length: 3 }, () => [0, 0, 0, 0] as const),
  lightsA: Array.from({ length: 8 }, () => [0, 0, 0, 0] as const),
  lightsB: Array.from({ length: 8 }, () => [0, 0, 0, 0.1] as const),
  props: Array.from({ length: 4 }, () => [0, 0, 0, 0] as const),
  spout: [0.5, 0.1, Z_MAX * 0.92, 0.5],
  lens: 1.15,
  reflection: 0.12,
  caustics: 0.5,
  foam: 6,
  flash: 0,
  sunBearing: -30,
  sunHeight: 44,
  /**
   * Wider than the rest spacing on purpose. A sheet of liquid one particle deep
   * is the common case away from a full basin, and at a radius near the spacing
   * such a sheet renders as separate beads with gaps between them - which is
   * most of why thin water read as scattered glass rather than as water.
   */
  splatRadius: REST_SPACING * 1.2,
  debug: DebugView.LIQUID,
};

/** Two angles are easier to aim at a photograph than three components. */
function sunVector(bearing: number, height: number): [number, number, number] {
  const yaw = (bearing * Math.PI) / 180;
  const pitch = (height * Math.PI) / 180;
  const flat = Math.cos(pitch);
  // y is down the image, so a light above the subject has a negative y.
  return [Math.sin(yaw) * flat, -Math.cos(yaw) * flat, Math.sin(pitch)];
}

export interface LiquidInputs {
  readonly context: ReturnType<TgpuRoot['configureContext']>;
  readonly particles: ParticleBuffer;
  readonly scene: SingleChannelTexture;
  readonly sceneBack: SingleChannelTexture;
  readonly smoke: SingleChannelTexture;
  readonly fieldParams: TgpuBuffer<typeof FieldParams> & UniformFlag;
  readonly linear: TgpuSampler;
}

export interface LiquidRenderer {
  initAsync(): Promise<void>;
  /** Splat and blur the fluid surface; call inside the frame's command encoder. */
  encodeSurface(encoder: TgpuCommandEncoder, hasWater: boolean): void;
  /** Draw the composite to the canvas. Pass the camera frame when there is one. */
  encodeComposite(encoder: TgpuCommandEncoder, frame: CameraFrame | undefined): void;
  look(next: Partial<LiquidLook>): void;
  /** Per-frame lightning level; patched on its own to avoid a full buffer write. */
  setFlash(value: number): void;
  destroy(): void;
}

export function createLiquidRenderer(root: TgpuRoot, inputs: LiquidInputs): LiquidRenderer {
  let look: LiquidLook = { ...defaultLook };
  /** Whether the water textures hold anything worth clearing. */
  let surfaceDirty = true;
  /** Latest shader clock, patched each frame and echoed by full look writes. */
  let timeNow = 0;

  const splatParams = root
    .createBuffer(SplatParams, {
      radius: look.splatRadius,
      mouths: [0, 1, 2].map((): [number, number, number, number] => [0, 0, 0, 0]),
    })
    .$usage('uniform');
  const lookParams = root.createBuffer(LookParams).$usage('uniform');
  const cameraParams = root.createBuffer(CameraParams).$usage('uniform');

  const renderable = (): SingleChannelTexture =>
    root
      .createTexture({ size: [SURFACE_RES, SURFACE_RES], format: 'rgba16float' })
      .$usage('sampled', 'storage', 'render');

  const rawDepth = renderable();
  const rawThickness = renderable();
  const scratch = createFieldTexture(root, SURFACE_RES);
  const fluid = createFieldTexture(root, SURFACE_RES);
  const history = createFieldTexture(root, SURFACE_RES);
  const thickness = createFieldTexture(root, SURFACE_RES);
  const thickHistory = createFieldTexture(root, SURFACE_RES);

  const depthBuffer = root
    .createTexture({ size: [SURFACE_RES, SURFACE_RES], format: 'depth24plus' })
    .$usage('render');

  const rawDepthView = rawDepth.createView('render');
  const rawThicknessView = rawThickness.createView('render');
  const depthBufferView = depthBuffer.createView('render');

  const splatBindGroup = root.createBindGroup(splatLayout, {
    splat: splatParams,
    field: inputs.fieldParams,
    scene: inputs.scene.createView(),
    linear: inputs.linear,
  });

  const compositeBindGroup = root.createBindGroup(compositeLayout, {
    field: inputs.fieldParams,
    look: lookParams,
    fluid: fluid.createView(),
    rawFluid: rawDepth.createView(),
    sceneBack: inputs.sceneBack.createView(),
    thickness: thickness.createView(),
    scene: inputs.scene.createView(),
    smoke: inputs.smoke.createView(),
    linear: inputs.linear,
  });

  const depthPipeline = root.createRenderPipeline({
    attribs: { pos: particleLayout.attrib.pos, vel: particleLayout.attrib.vel },
    vertex: splatVertex,
    fragment: splatDepthFragment,
    targets: { surface: { format: 'rgba16float' } },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
  });

  const thicknessPipeline = root.createRenderPipeline({
    attribs: { pos: particleLayout.attrib.pos, vel: particleLayout.attrib.vel },
    vertex: splatVertex,
    fragment: splatThicknessFragment,
    targets: {
      format: 'rgba16float',
      blend: {
        color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      },
    },
  });

  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  const withCamera = root.with(cameraSlot, true).createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: compositeFragment,
    targets: { format: canvasFormat },
  });
  const withoutCamera = root.with(cameraSlot, false).createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: compositeFragment,
    targets: { format: canvasFormat },
  });

  const depthFilter = createSurfaceFilter(root, {
    resolution: SURFACE_RES,
    radius: DEPTH_FILTER_RADIUS,
    sigma: DEPTH_FILTER_SIGMA,
    range: DEPTH_FILTER_RANGE,
    source: rawDepth,
    scratch,
    target: fluid,
    filterSampler: inputs.linear,
  });

  const temporalBindGroup = root.createBindGroup(temporalLayout, {
    current: fluid.createView(),
    history: history.createView(),
    target: scratch.createView(d.textureStorage2d('rgba16float', 'write-only')),
  });
  const temporal = root.createComputePipeline({ compute: temporalKernel });
  const publishBindGroup = root.createBindGroup(publishLayout, {
    source: scratch.createView(),
    live: fluid.createView(d.textureStorage2d('rgba16float', 'write-only')),
    keep: history.createView(d.textureStorage2d('rgba16float', 'write-only')),
  });
  const publish = root.createComputePipeline({ compute: publishKernel });
  const surfaceGroups = Math.ceil(SURFACE_RES / 8);

  // Thickness gets the same settling. It never elects a nearest particle, but
  // it is the sum of every soft splat over the texel, and at the waterline
  // that sum breathes with the jitter. Thickness drives opacity, absorption
  // and the refraction shift, so its breathing was most of what survived the
  // surface fix - frozen surface, still 0.6 grey levels of flicker.
  const thickTemporalBindGroup = root.createBindGroup(temporalLayout, {
    current: thickness.createView(),
    history: thickHistory.createView(),
    target: scratch.createView(d.textureStorage2d('rgba16float', 'write-only')),
  });
  const thickPublishBindGroup = root.createBindGroup(publishLayout, {
    source: scratch.createView(),
    live: thickness.createView(d.textureStorage2d('rgba16float', 'write-only')),
    keep: thickHistory.createView(d.textureStorage2d('rgba16float', 'write-only')),
  });

  const thicknessBlur = createSeparableBlur(root, {
    resolution: SURFACE_RES,
    radius: THICKNESS_BLUR_RADIUS,
    sigma: THICKNESS_BLUR_SIGMA,
    source: rawThickness,
    scratch,
    target: thickness,
    blurSampler: inputs.linear,
  });

  /** Copies a readonly 4-tuple into the mutable tuple the buffer write wants. */
  function four(v: readonly [number, number, number, number]): [number, number, number, number] {
    return [v[0], v[1], v[2], v[3]];
  }

  function writeLook(): void {
    splatParams.write({
      radius: look.splatRadius,
      mouths: [0, 1, 2].map((i): [number, number, number, number] => {
        const mouth = look.cupMouths[i];
        return mouth ? [mouth[0], mouth[1], mouth[2], mouth[3]] : [0, 0, 0, 0];
      }),
    });
    lookParams.write({
      tint: look.tint,
      surfaceLow: look.surfaceLow,
      surfaceHigh: Math.max(look.surfaceHigh, look.surfaceLow + 0.01),
      relief: look.relief,
      refraction: look.refraction,
      thickness: look.thickness,
      absorption: look.absorption,
      scatter: look.scatter,
      specular: look.specular,
      torch: look.torch,
      torchAt: look.torchAt,
      lightsA: look.lightsA.map(four),
      lightsB: look.lightsB.map(four),
      props: look.props.map(four),
      // Only the four placed slots: the preview row is not a lamp, and
      // counting it skipped a real light's orb.
      propCount: look.props.slice(0, 4).filter((p) => p[3] > 0.5).length,
      time: timeNow,
      spout: four(look.spout),
      glassesA: four(look.glassesA),
      glassesB: four(look.glassesB),
      cups: look.cups.map(four),
      cupLines: look.cupLines,
      cupMouths: look.cupMouths.map(four),
      lens: look.lens,
      reflection: look.reflection,
      caustics: look.caustics,
      foam: look.foam,
      flash: look.flash,
      sun: sunVector(look.sunBearing, look.sunHeight),
      debug: look.debug,
    });
  }

  writeLook();

  return {
    async initAsync() {
      await Promise.all([
        depthPipeline.initAsync(),
        thicknessPipeline.initAsync(),
        withCamera.initAsync(),
        withoutCamera.initAsync(),
        depthFilter.initAsync(),
        temporal.initAsync(),
        publish.initAsync(),
        thicknessBlur.initAsync(),
      ]);
    },

    encodeSurface(encoder, hasWater) {
      // An empty scene pays nothing: no splats, no blurs, no temporal walk.
      // Two full-particle render passes plus four compute dispatches ran
      // every frame for water that did not exist - most of the idle cost.
      if (!hasWater) {
        if (surfaceDirty) {
          fluid.clear();
          history.clear();
          thickness.clear();
          thickHistory.clear();
          rawDepth.clear();
          rawThickness.clear();
          surfaceDirty = false;
        }
        return;
      }
      surfaceDirty = true;
      const depthPass = encoder.beginRenderPass({
        colorAttachments: { view: rawDepthView, loadOp: 'clear', storeOp: 'store' },
        depthStencilAttachment: {
          view: depthBufferView,
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });
      depthPipeline
        .with(depthPass)
        .with(splatBindGroup)
        .with(particleLayout, inputs.particles)
        .draw(VERTS_PER_SPLAT, PARTICLE_COUNT);
      depthPass.end();

      const thicknessPass = encoder.beginRenderPass({
        colorAttachments: { view: rawThicknessView, loadOp: 'clear', storeOp: 'store' },
      });
      thicknessPipeline
        .with(thicknessPass)
        .with(splatBindGroup)
        .with(particleLayout, inputs.particles)
        .draw(VERTS_PER_SPLAT, PARTICLE_COUNT);
      thicknessPass.end();

      // Sequential: both blurs share one scratch texture.
      const blurPass = encoder.beginComputePass();
      depthFilter.encode(blurPass);
      // Settle the filtered surface against last frame's before anyone reads
      // it. The scratch texture is free again once the filter has run.
      temporal.with(blurPass).with(temporalBindGroup)
        .dispatchWorkgroups(surfaceGroups, surfaceGroups);
      publish.with(blurPass).with(publishBindGroup)
        .dispatchWorkgroups(surfaceGroups, surfaceGroups);
      thicknessBlur.encode(blurPass);
      temporal.with(blurPass).with(thickTemporalBindGroup)
        .dispatchWorkgroups(surfaceGroups, surfaceGroups);
      publish.with(blurPass).with(thickPublishBindGroup)
        .dispatchWorkgroups(surfaceGroups, surfaceGroups);
      blurPass.end();
    },

    encodeComposite(encoder, frame) {
      // The flames and coals animate on this clock; a patch, not a full write,
      // for the same reason as the lightning flash.
      timeNow = performance.now() / 1000;
      lookParams.patch({ time: timeNow });
      if (frame) {
        cameraParams.write({
          uvTransform: frame.uvTransform,
          mirror: frame.mirror ? 1 : 0,
          swapAxes: frame.swapAxes ? 1 : 0,
        });
      }

      const pass = encoder.beginRenderPass({ colorAttachments: { view: inputs.context } });
      if (frame) {
        withCamera
          .with(pass)
          .with(compositeBindGroup)
          .with(root.createBindGroup(frameLayout, { frame: frame.texture, camera: cameraParams }))
          .draw(3);
      } else {
        withoutCamera.with(pass).with(compositeBindGroup).draw(3);
      }
      pass.end();
    },

    look(next) {
      look = { ...look, ...next };
      writeLook();
    },

    setFlash(value) {
      look = { ...look, flash: value };
      lookParams.patch({ flash: value });
    },

    destroy() {
      splatParams.destroy();
      lookParams.destroy();
      cameraParams.destroy();
      rawDepth.destroy();
      rawThickness.destroy();
      scratch.destroy();
      fluid.destroy();
      thickness.destroy();
      history.destroy();
      thickHistory.destroy();
      depthBuffer.destroy();
    },
  };
}
