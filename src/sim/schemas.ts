import { d } from 'typegpu';
import type { StorageFlag, TgpuBuffer, VertexFlag } from 'typegpu';

/**
 * The simulation is genuinely three-dimensional. x and y are the image plane
 * (x right, y *down*, matching WebGPU's top-left texture origin) and z is
 * distance toward the camera: z = 0 is as far as the scene goes, larger z is
 * nearer. That matches the depth field, where 1 is nearest.
 *
 * A depth map only shows the front-most surface, so the rule is simple and it
 * is the whole collision model: a particle behind the visible surface is inside
 * something, and a particle in front of it is in open air.
 */
/**
 * Phones get a smaller simulation. This is decided once, at module load, before
 * any pipeline exists, so the sizes are still compile-time constants baked into
 * the generated WGSL - there is no runtime resizing anywhere.
 *
 * Override with ?quality=low or ?quality=high to check either path on any device.
 */
function chooseQuality(): 'low' | 'high' {
  const requested = new URLSearchParams(globalThis.location?.search ?? '').get('quality');
  if (requested === 'low' || requested === 'high') {
    return requested;
  }
  return globalThis.matchMedia?.('(pointer: coarse)').matches ? 'low' : 'high';
}

export const QUALITY = chooseQuality();

export const GRID_X = QUALITY === 'low' ? 48 : 80;
export const GRID_Y = GRID_X;

/**
 * The reconstruction is isotropic: one unit of normalised scene depth is one
 * unit of image width.
 *
 * This is the only defensible neutral convention, because a monocular depth
 * map cannot tell you its own scale - the model returns relative depth, and
 * turning that into a z extent needs an assumption from somewhere. Isotropy is
 * the assumption that adds nothing: it says the depth axis is measured in the
 * same units as the other two, and no more.
 *
 * The previous 0.45 was an assumption too, and a strong one - "every scene is
 * less than half as deep as it is wide". Squashing z that far tilts every real
 * horizontal surface into a ramp. Measured on the pool photo: the floor came
 * out sloped end to end, so no water level existed anywhere in it and the basin
 * could only hold a wedge against its near rim - 11.6% of the pool's area at
 * any level, with the far two thirds unable to hold a drop. At isotropy the
 * same floor holds water across about two thirds of its area.
 *
 * The depth-scale control runs up to this, and the play volume is sized from
 * it, rather than the volume being an independent number that silently
 * truncates deep scenes.
 */
export const MAX_DEPTH_SCALE = 1;

/**
 * Room in front of the nearest scene surface, for water to stand in and for the
 * spout to pour from. Without it a scene that uses the full depth range leaves
 * nowhere legal for a drop to be.
 */
const FRONT_HEADROOM = 0.25;

export const GRID_Z = Math.round(GRID_X * (MAX_DEPTH_SCALE + FRONT_HEADROOM));
export const GRID_CELLS = GRID_X * GRID_Y * GRID_Z;

/** Cubic cells, one kernel radius across. */
export const CELL_SIZE = 1 / GRID_X;
export const KERNEL_RADIUS = CELL_SIZE;

/** Front of the simulation volume; nothing may travel nearer than this. */
export const Z_MAX = GRID_Z * CELL_SIZE;

/**
 * Depth change at one texel between consecutive depth updates past which the
 * change is a topology flip - an occluder arriving or leaving - rather than a
 * surface in motion. It matches the collision slab's thickness (surfaceShell,
 * twelve kernel radii): a surface cannot move further than its own slab in
 * one depth interval, so a bigger jump means the texel changed owners, and
 * walking a wall between the two owners only sweeps phantom motion through
 * the water column. Set lower it misfires on the monocular depth model's
 * global renormalisation - a hand entering the frame shifts the measured
 * depth of the whole sink, and snapping that jump teleports the floor under
 * the settled pool. In normalised depth units; multiply by depthScale before
 * comparing scaled samples.
 */
export const TOPOLOGY_SNAP = KERNEL_RADIUS * 12;

/**
 * Rest spacing. At 2.2 particles per kernel radius a particle sees roughly 45
 * neighbours, the usual sweet spot in 3D.
 */
export const REST_SPACING = KERNEL_RADIUS / 2.2;

/**
 * Overridable so the cost of a bigger pour can be measured rather than guessed.
 */
function chooseParticles(fallback: number): number {
  const asked = Number(new URLSearchParams(globalThis.location?.search ?? '').get('particles'));
  return Number.isFinite(asked) && asked >= 1000 ? Math.round(asked) : fallback;
}

/**
 * Enough to actually fill a basin rather than puddle in one, but capped where
 * the solver still holds its 120Hz step rate. At 400,000 the sim ran at half
 * speed on the reference machine - water in visible slow motion, which reads
 * as lag - and every containment fix made it worse by keeping more particles
 * alive. 180,000 holds rate and still fills the tub; ?particles=N overrides.
 */
export const PARTICLE_COUNT = chooseParticles(QUALITY === 'low' ? 40000 : 140000);

/** Side length of the depth field, in texels. */
export const FIELD_RES = QUALITY === 'low' ? 192 : 256;

/** Fluid surface resolution. Higher than the depth field: this one is looked at. */
export const SURFACE_RES = QUALITY === 'low' ? 320 : 512;

/**
 * Smoke runs on its own grid, coarser than the fluid's. The fluid grid exists to
 * find neighbours within one kernel radius, so its spacing is set by the particle
 * size; smoke is a volume integral, and its detail comes from the advection
 * rather than from the cell count. Cells stay cubic and cover the same box.
 */
export const SMOKE_X = QUALITY === 'low' ? 40 : 64;
export const SMOKE_Y = SMOKE_X;
export const SMOKE_Z = Math.round(SMOKE_X * Z_MAX);
export const SMOKE_CELLS = SMOKE_X * SMOKE_Y * SMOKE_Z;
export const SMOKE_CELL = 1 / SMOKE_X;

/** Side of the ray-marched smoke image, upscaled during the composite. */
export const SMOKE_RES = QUALITY === 'low' ? 288 : 448;

/**
 * Jacobi sweeps per pressure solve. Keep it even: the two pressure buffers swap
 * every sweep, so an even count always leaves the answer in the first one and
 * the projection pass can bind it without tracking which is which.
 */
export const PRESSURE_ITERATIONS = QUALITY === 'low' ? 12 : 20;

export const SMOKE_WORKGROUP = 4;

export const WORKGROUP_SIZE = 64;

export const PARTICLE_WORKGROUPS = Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE);

/** Solver iterations per step. Three is enough to look incompressible. */
export const SOLVER_ITERATIONS = 3;

export const Particle = d.struct({
  pos: d.vec3f,
  /** Position at the start of the step; velocity is derived from the delta. */
  prev: d.vec3f,
  vel: d.vec3f,
});

export const ParticleArray = d.arrayOf(Particle, PARTICLE_COUNT);
export type ParticleBuffer = TgpuBuffer<typeof ParticleArray> & StorageFlag & VertexFlag;

export const SimParams = d.struct({
  /** Magnitude; the direction is read from the orientation buffer. */
  gravity: d.f32,
  emitter: d.vec3f,
  dt: d.f32,
  /** Seconds between the last two surface updates, for obstacle motion. */
  obstacleDt: d.f32,
  kernelRadius: d.f32,
  restDensity: d.f32,
  /** Constraint force mixing, the epsilon in the lambda denominator. */
  relaxation: d.f32,
  /** Artificial pressure strength; reads as surface tension. */
  cohesion: d.f32,
  /** XSPH velocity smoothing. */
  viscosity: d.f32,
  emitSpeed: d.f32,
  emitSpread: d.f32,
  /** Velocity retained per step while in contact with a surface. */
  surfaceFriction: d.f32,
  /** Largest push-out a single contact may apply. Caps depth-flicker blowups. */
  pushLimit: d.f32,
  /**
   * How thick the scene's visible surface is treated as being. A depth map says
   * nothing about what is behind a surface; assuming "solid forever" makes every
   * object an infinitely deep wall, so liquid can never pass behind a cup. A
   * finite slab lets it round the back and reappear.
   */
  surfaceShell: d.f32,
  /** Scene depth 0..1 maps onto this much z. Deeper scene, deeper cavities. */
  depthScale: d.f32,
  poly6: d.f32,
  spiky: d.f32,
  /** Poly6 evaluated at 0.2 * kernelRadius, the cohesion reference. */
  cohesionRef: d.f32,
  /** Particles allowed to re-enter at the spout per step. */
  emitRate: d.u32,
  /** Non-zero scatters new particles across the whole frame instead of a spout. */
  storm: d.u32,
  /** Sideways gust applied to rain as it enters, so a storm blows rather than drips. */
  wind: d.f32,
  /**
   * Share of the emit window that comes from the spout rather than the sky,
   * while a storm is running. Rain used to claim every slot, so holding the
   * pour during a storm did nothing at all - the tap was dead exactly when you
   * most wanted to aim it.
   */
  spoutShare: d.f32,
  /**
   * Normalised scene depth a raindrop needs beneath it to be worth spawning.
   * Distant background is still a camera-facing surface, so rain lands on it and
   * stacks up there instead of reaching the foreground.
   */
  rainReach: d.f32,
  /**
   * Non-zero recycles settled water from the bottom of the pool back to the
   * spout. Without it the particle budget is a hard ceiling: once every drop is
   * in the scene the spout silently stops, which reads as the pour randomly
   * refusing to work.
   */
  recycle: d.u32,
  /** How deep below the surface counts as settled, for recycling. */
  recycleBand: d.f32,
  frame: d.u32,
  /**
   * Detected drinking vessels, as field-space boxes; empty slots have x1 <= x0.
   * The carve gives a glass an interior but takes its transparent near wall
   * with it, and a heightfield has no way to put it back - one depth per
   * texel. These boxes let the solver re-add that wall analytically.
   */
  cups: d.arrayOf(d.vec4f, 3),
  /** Per cup, the vessel's front depth (raw 0..1); w is unused. */
  cupFronts: d.vec4f,
  /** Per cup, how deep its carved interior goes (raw 0..1); w is unused. */
  cupCarves: d.vec4f,
  /** How many cup slots are live. Zero lets every cup loop skip outright. */
  cupCount: d.u32,
  /** Per cup, how fast its box is moving (units per second; z, w unused). */
  cupShifts: d.arrayOf(d.vec4f, 3),
  /**
   * Burning props' flame regions: xy plant point, z depth, and the flame's
   * half-width in w - zero for an empty or already-doused slot. The census
   * counts water passing through each region, and the CPU douses on it.
   */
  props: d.arrayOf(d.vec4f, 4),
  /**
   * First particle index that can possibly be alive. Emission sweeps indices
   * downward from the top of the array, so everything ever woken lives in a
   * contiguous suffix - and every particle kernel can skip the prefix
   * entirely. A 20-second pour touches 30k slots, not 140k.
   */
  liveBase: d.u32,
  /** Rotating emission cursor within the woken suffix, advanced on the CPU. */
  emitCursor: d.u32,
});

export const FieldParams = d.struct({
  depthScale: d.f32,
  /**
   * Non-zero pins gravity to `manualDown` instead of measuring it.
   *
   * The measurement is sound - it reads the camera's pitch off the scene's own
   * surfaces - but a correct answer is not always the answer you want. A tub
   * shot from twenty degrees above level really is nearly edge-on, and no
   * estimator can make liquid gather in it. Being able to say which way down is
   * turns "this photo does not work" into "tilt it until it does".
   */
  manual: d.u32,
  manualDown: d.vec3f,
  /** sin of the steepest pitch the measurement may report for this source. */
  maxDownZ: d.f32,
});

/**
 * What the scene as a whole is doing, measured from the depth field each update.
 * One buffer rather than two because storage bindings are a scarce resource -
 * eight per shader stage - and these are written by the same kernel and read by
 * the same passes.
 */
export const SceneState = d.struct({
  /** Unit "down" in camera axes. */
  down: d.vec3f,
  /** Mean forward advance of the whole surface since the last depth update. */
  drift: d.f32,
  /** Fraction of the frame whose normals lean up-image - candidate ground. */
  groundShare: d.f32,
  /** Mean resultant length of those normals - how much they agree. */
  groundTight: d.f32,
  /** Up-image steepness of their mean - real floors lean, wall gradients face. */
  groundLean: d.f32,
});

/** How a camera frame is cropped and turned to line up with the field. */
export const CameraParams = d.struct({
  uvTransform: d.mat2x2f,
  mirror: d.u32,
  swapAxes: d.u32,
});

/** The light in the picture, measured from the picture each depth update. */
export const LightState = d.struct({
  /** Unit direction toward the key light, in camera axes. */
  sun: d.vec3f,
  /** Its colour, normalised to unit brightness. */
  colour: d.vec3f,
  /** Average colour of the whole frame, which is the fill the scene sits in. */
  ambient: d.vec3f,
  /** How dominant the key is, 0 to 1. Flat scenes score low and are not trusted. */
  strength: d.f32,
});

/** 3D poly6 normalisation: 315 / (64 pi h^9). */
export function poly6Coefficient(radius: number): number {
  return 315 / (64 * Math.PI * radius ** 9);
}

/** 3D spiky gradient normalisation: -45 / (pi h^6). */
export function spikyCoefficient(radius: number): number {
  return -45 / (Math.PI * radius ** 6);
}

export function poly6(radius: number, distance: number): number {
  const offset = radius * radius - distance * distance;
  return offset <= 0 ? 0 : poly6Coefficient(radius) * offset ** 3;
}

/**
 * Density a particle measures inside a relaxed cubic lattice. Deriving it beats
 * hand-tuning: get it wrong and the fluid either clumps or quietly explodes.
 */
export function latticeRestDensity(radius: number, spacing: number): number {
  const reach = Math.ceil(radius / spacing);
  let density = 0;
  for (let z = -reach; z <= reach; z++) {
    for (let y = -reach; y <= reach; y++) {
      for (let x = -reach; x <= reach; x++) {
        density += poly6(radius, Math.hypot(x, y, z) * spacing);
      }
    }
  }
  return density;
}

export const REST_DENSITY = latticeRestDensity(KERNEL_RADIUS, REST_SPACING);

/** Velocity plus the two scalars it carries. One struct, one advection pass. */
export const SmokeCell = d.struct({
  vel: d.vec3f,
  density: d.f32,
  /** Drives buoyancy. Smoke that has cooled stops climbing and starts drifting. */
  heat: d.f32,
});

export const SmokeCellArray = d.arrayOf(SmokeCell, SMOKE_CELLS);
export const PressureArray = d.arrayOf(d.f32, SMOKE_CELLS);
export const CurlArray = d.arrayOf(d.vec3f, SMOKE_CELLS);

export const SmokeParams = d.struct({
  dt: d.f32,
  emitter: d.vec3f,
  emitRadius: d.f32,
  /** Density added per second at the source. */
  emitRate: d.f32,
  /** Heat added per second at the source. */
  emitHeat: d.f32,
  /**
   * Standing sources planted in the scene - a cigarette on a counter, a torch
   * on a wall - each xyz position plus emission rate in w. Zero rate is an
   * empty slot. Their footprint and heat ride in `propTraits`: radius in x,
   * heat in y.
   */
  props: d.arrayOf(d.vec4f, 4),
  propTraits: d.arrayOf(d.vec4f, 4),
  /** Lift per unit heat, along the measured up. */
  buoyancy: d.f32,
  /** Heat lost per second, as a rate in an exponential decay. */
  cooling: d.f32,
  /** Smoke lost per second, likewise. */
  dissipation: d.f32,
  /** Sideways push. The storm feeds its gust in here. */
  wind: d.f32,
  /** How hard vorticity confinement pushes back along the surviving swirl. */
  swirl: d.f32,
  /** Velocity lost per second. Still air; also what bounds the confinement. */
  drag: d.f32,
  depthScale: d.f32,
  /** Direction toward the key light; the bake pass marches along it. */
  sun: d.vec3f,
  /** Extinction per unit density per unit distance. */
  opacity: d.f32,
  /** Colour in shadow. */
  shade: d.vec3f,
  /** Colour in full light. */
  tint: d.vec3f,
  frame: d.u32,
});
