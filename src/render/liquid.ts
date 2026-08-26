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

const SplatParams = d.struct({ radius: d.f32 });

const LookParams = d.struct({
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
  debug: d.u32,
});

const splatLayout = tgpu.bindGroupLayout({
  splat: { uniform: SplatParams },
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
  in: { pos: d.vec3f, vertex: d.builtin.vertexIndex },
  out: { position: d.builtin.position, offset: d.vec2f, centre: d.f32 },
})(({ pos, vertex }) => {
  'use gpu';
  const corner = quadCorners.$[vertex];
  const centre = d.vec3f(pos);
  const radius = splatLayout.$.splat.radius;

  // Image space is the unit square with y down; clip space is [-1, 1] with y up,
  // so a world radius spans twice as much clip space.
  const point = centre.xy + corner * radius;
  return {
    position: d.vec4f(point.x * 2 - 1, 1 - point.y * 2, 0.5, 1),
    offset: d.vec2f(corner),
    centre: centre.z,
  };
});

/**
 * Nearest-surface pass. Each particle is a sphere; the depth test keeps whichever
 * is closest to the camera, and the colour target carries that depth alongside a
 * coverage flag so the blur afterwards can average only over covered texels.
 */
const splatDepthFragment = tgpu.fragmentFn({
  in: { offset: d.vec2f, centre: d.f32 },
  out: { surface: d.vec4f, depth: d.builtin.fragDepth },
})(({ offset, centre }) => {
  'use gpu';
  const radial = std.dot(offset, offset);
  if (radial > 1) {
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
  in: { offset: d.vec2f, centre: d.f32 },
  out: d.vec4f,
})(({ offset }) => {
  'use gpu';
  const radial = std.dot(offset, offset);
  if (radial > 1) {
    std.discard();
  }
  const falloff = 1 - radial;
  return d.vec4f(falloff * falloff * falloff, 0, 0, 1);
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

  // A real view ray, even though the simulation is orthographic.
  const toEye = std.normalize(d.vec3f((d.vec2f(0.5) - uv), look.lens));
  const incident = toEye * -1;
  const facing = std.saturate(std.dot(normal, toEye));

  const alpha = std.smoothstep(look.surfaceLow, look.surfaceHigh, coverage);
  // Real occlusion: scene geometry standing nearer than the liquid hides it.
  const hidden = std.smoothstep(0, 0.015, sceneZ - fluidZ);

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
  const plain = backdropAt(uv);

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
  const wide = d.f32(7 / SURFACE_RES);
  const curvature = std.abs(
    fluidAt(uv + d.vec2f(wide, 0)) +
      fluidAt(uv - d.vec2f(wide, 0)) +
      fluidAt(uv + d.vec2f(0, wide)) +
      fluidAt(uv - d.vec2f(0, wide)) -
      fluidZ * 4,
  );
  const focus = 1 + std.saturate(curvature * (look.caustics * 200)) * thickness;

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
  const bounce = std.reflect(incident, normal);
  const mirrored = backdropAt(uv + bounce.xy * look.reflection);

  // Schlick. This ratio - clear straight on, mirror at a glance - is most of
  // what makes a surface read as water rather than tinted glass.
  const fresnel = WATER_F0 + (1 - WATER_F0) * std.pow(1 - facing, 5);

  // Layered specular: a tight highlight off micro-detail plus a broad sheen off
  // the overall curvature reads as a wet surface; one lobe alone reads as
  // plastic or as nothing at all.
  const glint = std.pow(std.saturate(std.dot(bounce, look.sun)), 220) * look.specular;
  const sheen = std.pow(std.saturate(std.dot(normal, std.normalize(look.sun + toEye))), 48) *
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
    churn =
      std.saturate(std.abs(raw.x - fluidZ) * look.foam) *
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

  const water = std.mix(transmitted + scattered, mirrored, fresnel);
  const lit = std.mix(water, d.vec3f(0.92, 0.95, 0.97), churn) +
    d.vec3f(glint + sheen + sparkle) + torchLit;

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
  relief: 0.4,
  refraction: 0.09,
  thickness: 0.22,
  absorption: 2.2,
  scatter: 0.1,
  specular: 1.1,
  torch: 0,
  torchAt: [0.5, 0.4, Z_MAX * 0.8],
  lens: 1.15,
  reflection: 0.12,
  caustics: 0.8,
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
  splatRadius: REST_SPACING * 1.7,
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
  readonly smoke: SingleChannelTexture;
  readonly fieldParams: TgpuBuffer<typeof FieldParams> & UniformFlag;
  readonly linear: TgpuSampler;
}

export interface LiquidRenderer {
  initAsync(): Promise<void>;
  /** Splat and blur the fluid surface; call inside the frame's command encoder. */
  encodeSurface(encoder: TgpuCommandEncoder): void;
  /** Draw the composite to the canvas. Pass the camera frame when there is one. */
  encodeComposite(encoder: TgpuCommandEncoder, frame: CameraFrame | undefined): void;
  look(next: Partial<LiquidLook>): void;
  /** Per-frame lightning level; patched on its own to avoid a full buffer write. */
  setFlash(value: number): void;
  destroy(): void;
}

export function createLiquidRenderer(root: TgpuRoot, inputs: LiquidInputs): LiquidRenderer {
  let look: LiquidLook = { ...defaultLook };

  const splatParams = root.createBuffer(SplatParams, { radius: look.splatRadius }).$usage('uniform');
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
  const thickness = createFieldTexture(root, SURFACE_RES);

  const depthBuffer = root
    .createTexture({ size: [SURFACE_RES, SURFACE_RES], format: 'depth24plus' })
    .$usage('render');

  const rawDepthView = rawDepth.createView('render');
  const rawThicknessView = rawThickness.createView('render');
  const depthBufferView = depthBuffer.createView('render');

  const splatBindGroup = root.createBindGroup(splatLayout, { splat: splatParams });

  const compositeBindGroup = root.createBindGroup(compositeLayout, {
    field: inputs.fieldParams,
    look: lookParams,
    fluid: fluid.createView(),
    rawFluid: rawDepth.createView(),
    thickness: thickness.createView(),
    scene: inputs.scene.createView(),
    smoke: inputs.smoke.createView(),
    linear: inputs.linear,
  });

  const depthPipeline = root.createRenderPipeline({
    attribs: { pos: particleLayout.attrib.pos },
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
    attribs: { pos: particleLayout.attrib.pos },
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

  const thicknessBlur = createSeparableBlur(root, {
    resolution: SURFACE_RES,
    radius: THICKNESS_BLUR_RADIUS,
    sigma: THICKNESS_BLUR_SIGMA,
    source: rawThickness,
    scratch,
    target: thickness,
    blurSampler: inputs.linear,
  });

  function writeLook(): void {
    splatParams.write({ radius: look.splatRadius });
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
        thicknessBlur.initAsync(),
      ]);
    },

    encodeSurface(encoder) {
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
      thicknessBlur.encode(blurPass);
      blurPass.end();
    },

    encodeComposite(encoder, frame) {
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
      depthBuffer.destroy();
    },
  };
}
