import { d, tgpu } from 'typegpu';

/** Injected by vite: the git commit this page was built from. */
declare const __BUILD__: string;
import { ROTATIONS, createCameraSession, detectRotation, rotationGeometry } from './camera.ts';
import type { Rotation } from './camera.ts';
import type { CameraFrame, DepthSource } from './depth/depth-field.ts';
import { carveDepth, createCarver } from './depth/carve.ts';
import { createSceneLight } from './depth/lighting.ts';
import type { Gestures, Tracked } from './track/gestures.ts';
import { type Vessel, createVesselProbe, floodVessels } from './depth/vessels.ts';
import { createModelDepth } from './depth/model-depth.ts';
import { MODEL_SIZES, RECOMMENDED_MODEL, type ModelSize } from './depth/model-store.ts';
import { createSyntheticDepth } from './depth/synthetic-depth.ts';
import { createFieldTexture } from './gpu/blur.ts';
import { createHud } from './hud.ts';
import type { TuneGroup } from './hud.ts';
import { createLiquidRenderer, defaultLook } from './render/liquid.ts';
import { createFluid, defaultTuning } from './sim/fluid.ts';
import {
  FIELD_RES,
  MAX_DEPTH_SCALE,
  PARTICLE_COUNT,
  REST_SPACING,
  Z_MAX,
} from './sim/schemas.ts';
import { createSmoke, defaultSmokeTuning } from './sim/smoke.ts';
import { createSurfaceField } from './sim/surface-field.ts';

/**
 * Half a display frame. See FIXED_DT in sim/fluid.ts for why the step is this
 * short: the residual churn in a settled pool is proportional to how much
 * velocity gravity adds per step, so halving the step halves the churn.
 */
const SOLVER_STEP = 1 / 120;
/**
 * Two solver steps per rendered frame at 60 Hz, and any time left over is
 * discarded.
 *
 * Letting the loop catch up sounds right and is a trap: a frame that runs long
 * asks for more steps, which makes the next frame longer, which asks for more
 * again. Under load the simulation now runs slower than the wall clock rather
 * than the whole page stuttering, which is the better of the two.
 */
// Three, not two: at 50-70fps a two-step budget cannot reach the 120Hz the
// solver is tuned for, and the accumulator clip below discards the shortfall
// as lost time - the water ran in chronic slow motion and read as lag.
const MAX_STEPS_PER_FRAME = 3;
const MAX_CANVAS_SIDE = 1024;
const HINT_MS = 7000;

const SCENES = [
  { id: 'vessels', label: 'Vessels', key: 'Q' },
  { id: 'photo', label: 'Photo', key: 'W' },
  { id: 'clip', label: 'Clip', key: 'E' },
  { id: 'camera', label: 'Camera', key: 'R' },
] as const;
type SceneId = (typeof SCENES)[number]['id'];

/**
 * The spout carries one thing at a time. Water and smoke sharing a source and a
 * button was the confusing part: there was no way to turn the water off, and
 * both could run at once for no reason anyone could see.
 */
const MEDIA = [
  { id: 'water', label: 'Water', key: 'M' },
  { id: 'smoke', label: 'Smoke', key: 'N' },
  { id: 'light', label: 'Light', key: 'L' },
  { id: 'props', label: 'Objects', key: 'B' },
] as const;

/**
 * Things that can be planted in the scene and keep going on their own. Each is
 * a standing smoke source plus, where it burns, a lamp - a cigarette is a wisp
 * and a dim ember, a torch is a plume and a flame.
 */
const PROP_KINDS = [
  // `lift` raises the smoke source to the flame's tip: emitted at the planted
  // point, the plume drew straight over the drawn fire and hid it.
  {
    id: 'cigarette',
    label: 'Cigarette',
    smoke: { rate: 0.5, radius: 0.035, heat: 2.5, lift: 0.012 },
    lamp: { power: 0.35, size: 0.05, tint: [1, 0.45, 0.18] as const, flicker: 0.25 },
  },
  {
    id: 'torch',
    label: 'Torch',
    smoke: { rate: 1.4, radius: 0.075, heat: 6, lift: 0.07 },
    lamp: { power: 2.4, size: 0.16, tint: [1, 0.58, 0.2] as const, flicker: 1 },
  },
  {
    id: 'campfire',
    label: 'Fire',
    smoke: { rate: 2.2, radius: 0.09, heat: 7.5, lift: 0.14 },
    lamp: { power: 3.4, size: 0.24, tint: [1, 0.5, 0.16] as const, flicker: 1 },
  },
] as const;
type MediumId = (typeof MEDIA)[number]['id'];

const VIEWS = [
  { id: 'liquid', label: 'Liquid', key: '1' },
  { id: 'scene', label: 'Depth', key: '2' },
  { id: 'fluid', label: 'Surface', key: '3' },
  { id: 'thickness', label: 'Body', key: '4' },
  { id: 'smoke', label: 'Smoke', key: '5' },
] as const;

const PHOTO_URL = '/bathroom.jpg';
const CLIP_URL = '/dishes.webm';

/**
 * Named setups. Each one is a scene plus the handful of settings that make it
 * read, so the interesting states are one click instead of a hotkey and four
 * sliders.
 */
interface Scenario {
  readonly id: string;
  readonly label: string;
  readonly note: string;
  readonly scene: SceneId;
  readonly storm: boolean;
  readonly medium: MediumId;
  /** Water released per step while pouring. */
  readonly flow: number;
  /** Smoke released per second while pouring. */
  readonly smoke: number;
  readonly spout: readonly [number, number];
  /** Spout depth. Near the camera keeps water in front of foreground subjects. */
  readonly spoutZ?: number;
  /** A still of its own; scenarios without one share the default photo. */
  readonly photo?: string;
  /** A clip of its own; scenarios without one share the default clip. */
  readonly clip?: string;
  readonly look: Partial<typeof defaultLook>;
}

const SCENARIOS: readonly Scenario[] = [
  {
    id: 'fill',
    label: 'Fill the tub',
    note: 'photo - steady pour',
    scene: 'photo',
    storm: false,
    medium: 'water',
    flow: 55,
    smoke: 0,
    // Under the tap, so the pour comes out of the spout in the picture. The
    // depth matters as much as the aim: at the tub's true, nearly level pitch,
    // gravity barely carries a drop backward, so a spout left at the default
    // near-camera depth rains in front of the tub and everything lands on the
    // bathroom floor. What looked like the tub leaking was the tub never being
    // hit. Spawning just above the basin mouth fills it 8.5x better.
    spout: [0.5, 0.36],
    spoutZ: 0.45,
    look: { caustics: 0.4, foam: 6, scatter: 0.1 },
  },
  {
    id: 'steam',
    label: 'Stove and hood',
    note: 'photo - steam pools under the hood',
    scene: 'photo',
    photo: '/kitchen.jpg',
    storm: false,
    medium: 'smoke',
    flow: 45,
    smoke: 4,
    // On the hob, and DEEP at it: left at the near-camera default depth the
    // plume rose in front of the hood and drew over it. At the hob's own depth
    // it mushrooms against the underside, which is the whole demo.
    spout: [0.55, 0.87],
    spoutZ: 0.4,
    look: { caustics: 0.6, foam: 8, scatter: 0.18 },
  },
  {
    id: 'live',
    label: 'Live camera',
    note: 'your webcam',
    scene: 'camera',
    storm: false,
    medium: 'water',
    flow: 40,
    smoke: 0,
    spout: [0.5, 0.12],
    look: { caustics: 0.6, foam: 8, scatter: 0.12 },
  },
];

function requireElements() {
  const canvas = document.querySelector('canvas');
  const video = document.querySelector('video');
  const picker = document.getElementById('picker');
  if (
    !(canvas instanceof HTMLCanvasElement) ||
    !(video instanceof HTMLVideoElement) ||
    !(picker instanceof HTMLInputElement)
  ) {
    throw new Error('The page is missing its canvas, video or file input.');
  }
  return { canvas, video, picker };
}

const TUNE_GROUPS: readonly TuneGroup[] = [
  {
    // Filters, not a tool: nothing pours or places, so this rides on the scene
    // instead of the tool row, appearing for every tool while the camera runs.
    title: 'face filters',
    scene: 'camera',
    fields: [
      { key: 'faceSweat', label: 'sweat', min: 0, max: 12, step: 1, value: 0, format: (v) => v.toFixed(0) },
      { key: 'faceTears', label: 'tears', min: 0, max: 12, step: 1, value: 0, format: (v) => v.toFixed(0) },
      { key: 'faceGlasses', label: 'party glasses', min: 0, max: 2, step: 0.05, value: 0 },
    ],
  },
  {
    title: 'world',
    fields: [
      // Min 0.75, measured: a shallower world starves every vessel of the
      // depth its cavity needs - retention on the benchmark glass falls
      // from 4,911 (1.0) to 2,750 (0.8), 580 (0.7), 2 (0.14). Below the
      // knee the slider is a self-destruct, not a control.
      { key: 'depthScale', label: 'scene depth', min: 0.75, max: MAX_DEPTH_SCALE, step: 0.01, value: defaultTuning.depthScale },
      { key: 'gravity', label: 'gravity', min: 0, max: 5, step: 0.05, value: defaultTuning.gravity },
      { key: 'gravityPitch', label: 'down: pitch', min: -20, max: 85, step: 1, value: 0, format: (v) => v.toFixed(0) + '\u00b0' },
      { key: 'gravityRoll', label: 'down: roll', min: -45, max: 45, step: 1, value: 0, format: (v) => v.toFixed(0) + '\u00b0' },
      // Max 0.18, not more: past the carve's reach the shell dissolves every
      // wall in the scene - burial can never exceed it, so the weir waves
      // water through glass sides and tub rims alike. Measured: one particle
      // of 26k retained at 0.3.
      { key: 'surfaceShell', label: 'object thickness', min: 0.01, max: 0.18, step: 0.005, value: defaultTuning.surfaceShell },
      { key: 'rainReach', label: 'play depth', min: 0, max: 0.8, step: 0.01, value: defaultTuning.rainReach },
    ],
  },
  {
    title: 'pour',
    tool: 'water',
    fields: [
      { key: 'emitRate', label: 'flow', min: 1, max: 300, step: 1, value: defaultTuning.emitRate, format: (v) => v.toFixed(0) },

      { key: 'emitSpeed', label: 'speed', min: 0, max: 2, step: 0.01, value: defaultTuning.emitSpeed },
      { key: 'emitSpread', label: 'spread', min: 0.005, max: 0.2, step: 0.005, value: defaultTuning.emitSpread },
      { key: 'emitterZ', label: 'spout depth (scroll)', min: 0.05, max: Z_MAX - 0.05, step: 0.005, value: defaultTuning.emitterZ },
    ],
  },
  {
    title: 'physics',
    tool: 'water',
    fields: [
      { key: 'viscosity', label: 'viscosity', min: 0, max: 0.4, step: 0.005, value: defaultTuning.viscosity },
      { key: 'cohesion', label: 'cohesion', min: 0, max: 8e-5, step: 1e-6, value: defaultTuning.cohesion, format: (v) => v.toExponential(1) },
      { key: 'relaxation', label: 'relaxation', min: 1, max: 2000, step: 1, value: defaultTuning.relaxation, format: (v) => v.toFixed(0) },
      { key: 'surfaceFriction', label: 'friction', min: 0.5, max: 1, step: 0.005, value: defaultTuning.surfaceFriction },
    ],
  },
  {
    title: 'smoke',
    tool: 'smoke',
    fields: [
      { key: 'smokeRate', label: 'smoke', min: 0, max: 8, step: 0.1, value: 3 },
      { key: 'emitHeat', label: 'heat', min: 0, max: 8, step: 0.1, value: defaultSmokeTuning.emitHeat },
      { key: 'buoyancy', label: 'lift', min: 0, max: 3, step: 0.02, value: defaultSmokeTuning.buoyancy },
      { key: 'cooling', label: 'cooling', min: 0, max: 2, step: 0.02, value: defaultSmokeTuning.cooling },
      { key: 'dissipation', label: 'fade', min: 0, max: 1.5, step: 0.01, value: defaultSmokeTuning.dissipation },
      { key: 'swirl', label: 'swirl', min: 0, max: 40, step: 0.5, value: defaultSmokeTuning.swirl, format: (v) => v.toFixed(1) },
      { key: 'drag', label: 'still air', min: 0, max: 3, step: 0.05, value: defaultSmokeTuning.drag },
      { key: 'emitRadius', label: 'source size', min: 0.01, max: 0.2, step: 0.005, value: defaultSmokeTuning.emitRadius },
      { key: 'opacity', label: 'thickness', min: 4, max: 90, step: 1, value: defaultSmokeTuning.opacity, format: (v) => v.toFixed(0) },
    ],
  },
  {
    title: 'lamp',
    tool: 'light',
    fields: [
      { key: 'lightPower', label: 'brightness', min: 0.2, max: 6, step: 0.05, value: 1.4 },
      { key: 'lightColor', label: 'colour', min: 0, max: 1, step: 1, value: 0, color: '#ffa847' },
      { key: 'lightSize', label: 'size', min: 0.04, max: 0.5, step: 0.01, value: 0.16 },
      { key: 'torch', label: 'held light (follows spout)', min: 0, max: 4, step: 0.05, value: defaultLook.torch },
    ],
  },
  {
    title: 'look',
    tool: 'water',
    fields: [
      { key: 'splatRadius', label: 'bead size', min: REST_SPACING * 0.6, max: REST_SPACING * 3, step: REST_SPACING / 40, value: defaultLook.splatRadius, format: (v) => (v / REST_SPACING).toFixed(2) + '\u00d7' },
      { key: 'surfaceLow', label: 'surface', min: 0.02, max: 1, step: 0.01, value: defaultLook.surfaceLow },
      { key: 'sunBearing', label: 'light bearing', min: -180, max: 180, step: 1, value: defaultLook.sunBearing, format: (v) => v.toFixed(0) + '\u00b0' },
      { key: 'sunHeight', label: 'light height', min: 0, max: 89, step: 1, value: defaultLook.sunHeight, format: (v) => v.toFixed(0) + '\u00b0' },
      { key: 'lens', label: 'lens (wide - long)', min: 0.5, max: 3, step: 0.05, value: defaultLook.lens },
      { key: 'reflection', label: 'reflection', min: 0, max: 0.5, step: 0.005, value: defaultLook.reflection },
      { key: 'refraction', label: 'refraction', min: 0, max: 0.3, step: 0.002, value: defaultLook.refraction },
      { key: 'caustics', label: 'caustics', min: 0, max: 4, step: 0.05, value: defaultLook.caustics },
      { key: 'foam', label: 'foam', min: 0, max: 90, step: 1, value: defaultLook.foam, format: (v) => v.toFixed(0) },
      { key: 'relief', label: 'relief', min: 0, max: 2, step: 0.02, value: defaultLook.relief },
      { key: 'thickness', label: 'thickness', min: 0.01, max: 0.6, step: 0.005, value: defaultLook.thickness },
      { key: 'scatter', label: 'scatter', min: 0, max: 2, step: 0.02, value: defaultLook.scatter },
      { key: 'absorption', label: 'absorption', min: 0, max: 5, step: 0.05, value: defaultLook.absorption },
    ],
  },
];

const SMOKE_KEYS = new Set([
  'emitHeat', 'buoyancy', 'cooling', 'dissipation', 'emitRadius', 'opacity', 'swirl', 'drag',
]);

const LOOK_KEYS = new Set([
  'surfaceLow', 'refraction', 'relief', 'thickness', 'scatter', 'absorption',
  'lens', 'reflection', 'caustics', 'foam', 'splatRadius', 'sunBearing', 'sunHeight',
]);

async function main(): Promise<void> {
  const { canvas, video, picker } = requireElements();
  if (!navigator.gpu) {
    throw new Error('This browser has no WebGPU. Try Chrome 113+, Edge, or Safari 18+.');
  }

  const root = await tgpu.init({ device: { optionalFeatures: ['shader-f16', 'timestamp-query'] } });
  const context = root.configureContext({ canvas, alphaMode: 'opaque' });

  const linear = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  const depth = createFieldTexture(root, FIELD_RES);
  const synthetic = createSyntheticDepth(root, depth);
  const model = createModelDepth(root, depth);
  const field = createSurfaceField(root, depth, linear);
  const sceneLight = createSceneLight(root, field.surface, linear);
  const vesselProbe = createVesselProbe(root, field.surface, linear);
  const carver = createCarver(root, depth);
  const fluid = createFluid(root, {
    surface: field.surface,
    surfacePrev: field.surfacePrev,
    surfaceLive: field.live,
    surfaceBack: field.background,
    scene: field.scene,
    fieldSampler: linear,
    obstacleDt: () => obstacleDt,
  });
  const smoke = createSmoke(root, {
    surface: field.surface,
    scene: field.scene,
    // The smoke shades with the light measured from the picture, so a plume in
    // a warm room is a warm plume.
    light: sceneLight.state,
    fieldSampler: linear,
  });
  const renderer = createLiquidRenderer(root, {
    context,
    particles: fluid.particles,
    scene: field.surface,
    sceneBack: field.background,
    smoke: smoke.image,
    fieldParams: field.params,
    linear,
  });

  const camera = createCameraSession(video);
  const abort = new AbortController();

  let sceneId: SceneId = 'vessels';
  let modelSize: ModelSize = RECOMMENDED_MODEL;
  /** undefined means follow the automatic guess; a value pins it. */
  let rotationOverride: Rotation | undefined;
  let mirrorOverride: boolean | undefined;
  let loadedSize: ModelSize | undefined;
  /** The still currently in use: the bundled one until a file replaces it. */
  let photo: ImageBitmap | undefined;
  /** Which bundled still the photo scene shows; scenarios can swap it. */
  let photoUrl = PHOTO_URL;
  const photoCache = new Map<string, ImageBitmap>();
  let dropSerial = 0;
  /** A dropped video clip, which drives depth per frame like the camera does. */
  let clipUrl: string | undefined;
  /** The clip currently in use, bundled or dropped. */
  let clipSource = CLIP_URL;
  /** Held between building the frame and submitting, then closed. */
  let photoFrame: VideoFrame | undefined;
  /** Latched by the button or space; the pointer pours while held regardless. */
  let latched = false;
  let holding = false;
  /** Rain needs no aiming, so it pours on its own. */
  let storming = false;
  /** What the spout is carrying. One at a time. */
  let medium: MediumId = 'water';
  /** One blanking pass owed to the compositor after a scene reset. */
  let smokeStale = false;
  /** Smoke keeps simulating once any has been emitted, until the reset. */
  let smokeAlive = false;
  // NOT seeded from defaultSmokeTuning.emitRate: that is the emitter's at-rest
  // state (zero, correctly), while this is how hard the smoke TOOL pours when
  // held. Seeding from the struct made the smoke tool silently emit nothing
  // until some scenario happened to overwrite the rate.
  let smokeRate = 3;
  let gust = 0;
  /** Face and hand tracking, loaded the first time a video scene opens. */
  let gestures: Gestures | undefined;
  let gesturesLoading = false;
  let tracked: Tracked = { vessels: [], brow: [], eyes: [], lenses: [], effort: 0 };
  /** Fingertip control of the active tool; the toggle beside the readouts. */
  // Off by default: the hand landmarker costs GPU every frame whether or not
  // a hand is in view, and the fingertip cursor is a novelty. The chip turns
  // it on, which rebuilds the tracker with the model included.
  let handControl = false;
  /** Light-up glasses: brightness slider; the colour cycles on its own. */
  let glassesGlow = 0;
  let glassesOn = false;
  /** Pinching index and thumb pours, hands-free. */
  let fingerPour = false;
  /** Water from the face itself: sliders in the face group. */
  // Parked for now at the user's request; flip to bring sweat, tears and the
  // party glasses back. The tracking itself stays on - cups and the fingertip
  // cursor need it.
  const FACE_FILTERS = false;
  let sweatRate = 0;
  let tearRate = 0;
  let faceOwned = false;
  let springStep = 0;
  let recycling = false;
  let solverSteps = 0;
  let encodes = 0;
  let flash = 0;
  let nextStrike = 0;
  let reStrikeAt = 0;
  let spoutX = defaultTuning.emitterX;
  let spoutY = defaultTuning.emitterY;
  let spoutZ = defaultTuning.emitterZ;
  let lastDepthAt = 0;
  /** Seconds between the last two depth updates; drives obstacle-motion contact. */
  let obstacleDt = 1 / 30;
  /**
   * While a scene is loading, the video element and the depth texture still
   * hold the OLD scene, and every depth update the estimator ran against them
   * re-converged gravity toward a scene the user had already left. Freezing
   * the depth pipeline until the new source is actually delivering frames is
   * what makes a scene change start clean.
   */
  let sceneLoading = false;
  let accumulator = 0;
  let previousTime = performance.now();
  let emitRate = defaultTuning.emitRate;
  /** Torch brightness before the flicker is applied. */
  let torchStrength = defaultLook.torch;
  /** Lights the user has planted in the scene. Position, then colour+size. */
  const placedLights: { at: [number, number, number]; tint: [number, number, number]; size: number; power: number }[] = [];
  let lightPower = 1.4;
  let lightRgb: [number, number, number] = [1, 0.66, 0.28];
  let lightSize = 0.16;
  /** Where the pointer last was over the plate, for the lamp that follows it. */
  let hoverX = 0.5;
  let hoverY = 0.5;
  /** Planted objects: standing smoke sources with their own light. */
  const placedProps: {
    kind: (typeof PROP_KINDS)[number];
    at: [number, number, number];
  }[] = [];
  let propKind: (typeof PROP_KINDS)[number] = PROP_KINDS[1];

  function pushProps(): void {
    smoke.tune({
      props: placedProps.map((prop) => ({
        // The drawn flame scales with depth; its smoke has to start where the
        // flame ends, or the plume paints over the fire.
        at: [
          prop.at[0],
          prop.at[1] - prop.kind.smoke.lift * (0.3 + prop.at[2] * 0.6),
          prop.at[2],
        ],
        rate: prop.kind.smoke.rate,
        radius: prop.kind.smoke.radius,
        heat: prop.kind.smoke.heat,
      })),
    });
    if (placedProps.length > 0) {
      smokeAlive = true;
    }
    // The composite draws each prop's body - rod, torch, logs and flame - so
    // it needs the kind as well as the place.
    renderer.look({
      props: [0, 1, 2, 3].map((i): [number, number, number, number] => {
        const prop = placedProps[i];
        return prop
          ? [prop.at[0], prop.at[1], prop.at[2], PROP_KINDS.indexOf(prop.kind) + 1]
          : [0, 0, 0, 0];
      }),
    });
    pushLights();
  }

  function placeProp(): void {
    if (placedProps.length >= 4) {
      placedProps.shift();
    }
    placedProps.push({ kind: propKind, at: [hoverX, hoverY, spoutZ] });
    pushProps();
  }

  function lightTint(): [number, number, number] {
    return lightRgb;
  }

  /**
   * Empties everything the user has added: water, smoke, planted lamps. Reset
   * does it on demand, and a change of picture does it automatically - liquid
   * that settled into the old scene's basins is standing in mid-air in the new
   * one, and a lamp hung on a wall that no longer exists is worse.
   */
  function resetScene(): void {
    fluid.drain();
    inScene = 0;
    recycling = false;
    fluid.tune({ recycle: false });
    smokeAlive = false;
    smokeStale = true;
    placedLights.length = 0;
    placedProps.length = 0;
    pushProps();
    // Gravity starts over too: unpinned, and the estimate's memory snapped to
    // level so the new scene measures fresh instead of inheriting the last
    // scene's tilt - the webcam was opening with the city's steep pitch.
    gravityManual = false;
    field.tune({ manual: false });
    field.scene.write({ down: [0, 1, 0], drift: 0, groundShare: 0, groundTight: 0, groundLean: 0 });
    // An active pour re-arms its medium; without this a scenario that resets
    // and then pours smoke had its smoke switched off by its own reset.
    applyFlow();
  }

  /**
   * Seven placed lamps plus one live slot: while the light tool is in hand, the
   * last slot is a full lamp that rides the cursor, so what you see hovering is
   * exactly what a click will leave behind - same shading path, no preview
   * approximation.
   */
  function pushLights(): void {
    const a: [number, number, number, number][] = [];
    const b: [number, number, number, number][] = [];
    const now = performance.now();
    for (const prop of placedProps) {
      const f = prop.kind.lamp.flicker;
      const wave = 1 - f * 0.22 + f * (0.13 * Math.sin(now / 67) + 0.09 * Math.sin(now / 29 + 2.1));
      a.push([prop.at[0], prop.at[1], prop.at[2], prop.kind.lamp.power * wave]);
      b.push([prop.kind.lamp.tint[0], prop.kind.lamp.tint[1], prop.kind.lamp.tint[2], prop.kind.lamp.size]);
    }
    for (let i = 0; i < 7 - placedProps.length; i++) {
      const l = placedLights[i];
      a.push(l ? [l.at[0], l.at[1], l.at[2], l.power] : [0, 0, 0, 0]);
      b.push(l ? [l.tint[0], l.tint[1], l.tint[2], l.size] : [0, 0, 0, 0.1]);
    }
    if (medium === 'light') {
      const tint = lightTint();
      a.push([hoverX, hoverY, spoutZ, lightPower]);
      b.push([tint[0], tint[1], tint[2], lightSize]);
    } else {
      a.push([0, 0, 0, 0]);
      b.push([0, 0, 0, 0.1]);
    }
    renderer.look({ lightsA: a, lightsB: b });
  }

  function placeLight(): void {
    // Seven is plenty; past that, the oldest one moves.
    if (placedLights.length >= 7) {
      placedLights.shift();
    }
    placedLights.push({
      at: [hoverX, hoverY, spoutZ],
      tint: lightTint(),
      size: lightSize,
      power: lightPower,
    });
    pushLights();
  }
  /**
   * Water in the scene, counted by the solver and read back a few times a
   * second. Adding up what the spout released instead was open-loop: nothing
   * ever told it about water that drained off the frame, so the meter sat at
   * two thirds full over a pool with nothing in it.
   */
  let inScene = 0;
  let lastWetAt = 0;

  // GPU pass attribution, when the device allows timing. Slots: 0/1 depth
  // pipeline (model, carve, field, probes), 2/3 solver. Read a few times a
  // second; gpuMs holds the latest sample for perf() and the readout.
  const canTime = root.enabledFeatures.has('timestamp-query');
  const gpuQuery = canTime ? root.createQuerySet('timestamp', 4) : undefined;
  const gpuMs = { depth: 0, sim: 0 };
  let lastGpuReadAt = 0;
  const depthTimestamps = gpuQuery
    ? { querySet: gpuQuery, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
    : undefined;
  const simTimestamps = gpuQuery
    ? { querySet: gpuQuery, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 }
    : undefined;
  /** One census readback in flight at a time. */
  let censusPending = false;
  let lastCensusAt = 0;
  /** Same for the measured light, which changes far too slowly to need more. */
  let lightPending = false;
  let lastLightAt = 0;
  /** Gravity, once pinned by hand, stops being measured. */
  let gravityManual = false;
  let gravityPitch = 0;
  let gravityRoll = 0;
  let gravityPending = false;
  let lastGravityAt = 0;
  /** The vessel probe re-floods a downsample of the scene while tuning. */
  let vesselPending = false;
  /** Latest probe cells, reused to read a held vessel's front depth. */
  let probeCells: number[] | undefined;
  let lastVesselAt = 0;
  let depthScaleNow = defaultTuning.depthScale;

  const hud = createHud(
    {
      scenarios: SCENARIOS.map(({ id, label, note }) => ({ id, label, note })),
      views: VIEWS,
      media: MEDIA,
      model: { label: 'size', options: MODEL_SIZES, value: RECOMMENDED_MODEL },
      rotation: {
        label: 'rotation',
        options: ['auto', ...ROTATIONS.map((r) => String(r))],
        value: 'auto',
      },
      mirror: { label: 'mirror', options: ['auto', 'on', 'off'], value: 'auto' },
      groups: TUNE_GROUPS,
    },
    {
      onView: (index) => renderer.look({ debug: index }),
      onRotation: (value) => {
        // The dropdown is built from ROTATIONS, so any non-auto value is one of them.
        rotationOverride = ROTATIONS.find((candidate) => String(candidate) === value);
      },
      onMirror: (value) => {
        mirrorOverride = value === 'auto' ? undefined : value === 'on';
      },
      onModel: (id) => {
        const size = MODEL_SIZES.find((candidate) => candidate === id);
        if (size) {
          void selectModel(size);
        }
      },
      onPour: (on) => {
        latched = on;
        applyFlow();
      },
      onStorm: (on) => {
        storming = on;
        fluid.tune({ storm: on, wind: 0 });
        smoke.tune({ wind: 0 });
        if (!on) {
          renderer.setFlash(0);
          flash = 0;
        }
        applyFlow();
        hud.dismissHint();
      },
      onMedium: (id) => {
        const match = MEDIA.find((candidate) => candidate.id === id);
        if (match) {
          setMedium(match.id);
        }
        hud.dismissHint();
      },
      onOpen: () => picker.click(),
      onScenario: (id) => {
        const scenario = SCENARIOS.find((candidate) => candidate.id === id);
        if (!scenario) {
          return;
        }
        spoutX = scenario.spout[0];
        spoutY = scenario.spout[1];
        spoutZ = scenario.spoutZ ?? defaultTuning.emitterZ;
        emitRate = scenario.flow;
        storming = scenario.storm;
        hud.setStorm(scenario.storm);
        fluid.tune({
          storm: scenario.storm,
          emitterX: spoutX,
          emitterY: spoutY,
          emitterZ: spoutZ,
          wind: 0,
        });
        smoke.tune({ emitterX: spoutX, emitterY: spoutY, emitterZ: spoutZ, wind: 0 });
        renderer.look(scenario.look);
        if (scenario.smoke > 0) {
          smokeRate = scenario.smoke;
        }
        setMedium(scenario.medium);
        hud.setMedium(scenario.medium);
        // A named setup should show its thing without anyone hunting for Hold.
        latched = true;
        hud.setPouring(true);
        applyFlow();
        syncSpout();
        hud.dismissHint();
        gravityManual = false;
        field.tune({ manual: false });
        const nextPhoto = scenario.photo ?? PHOTO_URL;
        const photoChanged = scenario.scene === 'photo' && nextPhoto !== photoUrl;
        photoUrl = nextPhoto;
        const nextClip = scenario.clip ?? CLIP_URL;
        const clipChanged = scenario.scene === 'clip' && nextClip !== clipSource;
        clipSource = nextClip;
        if (sceneId !== scenario.scene || photoChanged || clipChanged) {
          resetScene();
          void selectScene(scenario.scene);
        }
      },
      onDrain: () => {
        resetScene();
        hud.resetTune();
      },
      onTune: (key, value) => {
        if (key === 'sunBearing' || key === 'sunHeight') {
          // One key light. Smoke lit from somewhere else reads as a sticker.
          renderer.look({ [key]: value });
          smoke.tune({ [key]: value });
          return;
        }
        if (LOOK_KEYS.has(key)) {
          renderer.look({ [key]: value });
          return;
        }
        // Touching either angle pins gravity. The scene stops measuring it, and
        // stays pinned until a different scene or scenario is loaded.
        if (key === 'gravityPitch' || key === 'gravityRoll') {
          gravityManual = true;
          if (key === 'gravityPitch') {
            gravityPitch = value;
          } else {
            gravityRoll = value;
          }
          field.tune({ manual: true, pitch: gravityPitch, roll: gravityRoll });
          showGravity();
          hud.setStatus('info', `Down is pinned at ${gravityPitch.toFixed(0)}\u00b0 pitch. Pick a scene to measure it again.`);
          return;
        }
        if (key === 'surfaceShell') {
          surfaceShellNow = value;
        }
        if (key === 'depthScale') {
          depthScaleNow = value;
          // The solver, the smoke and the compositor must agree on how deep the
          // scene is.
          fluid.tune({ depthScale: value });
          field.tune({ depthScale: value });
          smoke.tune({ depthScale: value });
          return;
        }
        if (key === 'lightPower' || key === 'lightColor' || key === 'lightSize') {
          if (key === 'lightPower') {
            lightPower = value;
          } else if (key === 'lightColor') {
            // The colour well delivers packed 0xRRGGBB.
            lightRgb = [
              ((value >> 16) & 255) / 255,
              ((value >> 8) & 255) / 255,
              (value & 255) / 255,
            ];
          } else {
            lightSize = value;
          }
          pushLights();
          return;
        }
        if (key === 'torch') {
          torchStrength = value;
          renderer.look({ torch: value, torchAt: [spoutX, spoutY, spoutZ] });
          return;
        }
        if (key === 'faceSweat' || key === 'faceTears' || key === 'faceGlasses') {
          if (key === 'faceSweat') {
            sweatRate = value;
          } else if (key === 'faceTears') {
            tearRate = value;
          } else {
            glassesGlow = value;
          }
          return;
        }
        if (key === 'emitRate') {
          emitRate = value;
          applyFlow();
          return;
        }
        if (key === 'smokeRate') {
          smokeRate = value;
          applyFlow();
          return;
        }
        if (SMOKE_KEYS.has(key)) {
          smoke.tune({ [key]: value });
          return;
        }
        if (key === 'emitterZ') {
          setSpoutDepth(value);
          return;
        }
        fluid.tune({ [key]: value });
      },
    },
  );

  hud.setStatus('info', 'Compiling pipelines…');
  await Promise.all([
    synthetic.initAsync(),
    model.initAsync(),
    field.initAsync(),
    fluid.initAsync(),
    smoke.initAsync(),
    sceneLight.initAsync(),
    vesselProbe.initAsync(),
    carver.initAsync(),
    renderer.initAsync(),
  ]);

  // --- aiming ---
  function pouringNow(): boolean {
    return storming || latched || holding || fingerPour;
  }

  /** One source, one medium. The other one is off, whatever the button says. */
  function applyFlow(): void {
    const live = pouringNow();
    // Aiming the spout during a storm is a thing people try immediately, so keep
    // half the window for it rather than letting the rain have all of it.
    const aiming = latched || holding;
    fluid.tune({
      emitRate: live && medium === 'water' ? emitRate : 0,
      spoutShare: storming ? (aiming ? 0.5 : 0) : 1,
    });
    smoke.tune({ emitRate: live && medium === 'smoke' ? smokeRate : 0 });
    if (live && medium === 'smoke') {
      smokeAlive = true;
    }
  }

  /**
   * A tool, not a mode. Switching tools changes what the spout emits and which
   * settings the drawer shows; whatever is already in the scene stays and keeps
   * simulating. Water under a plume of steam is the whole point.
   */
  function setMedium(next: MediumId): void {
    if (next === medium) {
      return;
    }
    medium = next;
    document.body.dataset.tool = medium;
    hud.setToolFilter(medium);
    // Adds or removes the cursor lamp as the light tool comes and goes, and
    // the spout marker as the props tool does.
    pushLights();
    syncSpout();
    applyFlow();
  }

  /**
   * Flame, not a bulb. A steady point light reads as a rendered sphere; the
   * unsteadiness is most of what says fire.
   */
  function driveTorch(now: number): void {
    // Burning props flicker; refreshing the light array each frame is what
    // makes an ember breathe instead of glowing like an LED.
    if (placedProps.some((prop) => prop.kind.lamp.flicker > 0)) {
      pushLights();
    }
    if (torchStrength <= 0) {
      return;
    }
    const flicker =
      0.82 +
      0.11 * Math.sin(now / 71) +
      0.07 * Math.sin(now / 31 + 1.7) +
      0.05 * Math.sin(now / 13 + 4.1);
    renderer.look({ torch: torchStrength * flicker, torchAt: [spoutX, spoutY, spoutZ] });
  }

  /**
   * Draws the measured down on the picture: the arrow is the part of it that
   * lies in the image, and the part running into the scene is written out,
   * since it has no length on screen to show.
   */
  const gravityArrow = document.getElementById('gravityArrow');
  const gravityNote = document.getElementById('gravityNote');
  function showGravity(): void {
    if (!gravityArrow || !gravityNote) {
      return;
    }
    const flat = Math.cos((gravityPitch * Math.PI) / 180);
    gravityArrow.style.setProperty('--roll', `${gravityRoll.toFixed(1)}deg`);
    gravityArrow.style.setProperty('--len', `${(flat * 6).toFixed(2)}rem`);
    const lean = gravityManual ? 'pinned' : 'measured';
    gravityNote.textContent =
      `down · ${gravityPitch.toFixed(0)}\u00b0 into scene · ${gravityRoll.toFixed(0)}\u00b0 roll · ${lean}` +
      ` · ${frameMsEma.toFixed(1)}ms (peak ${frameMsPeak.toFixed(0)})` +
      (gpuMs.depth + gpuMs.sim > 0
        ? ` · gpu depth ${gpuMs.depth.toFixed(1)} sim ${gpuMs.sim.toFixed(1)}`
        : '');
  }

  // Hand control: shown only where there is a hand to track, toggled by
  // button or the H key.
  const handToggle = document.getElementById('handToggle');
  // "Face filters" is a doorway, not a tool: it opens the drawer where the
  // camera-scene filters live.
  const faceFilters = document.getElementById('faceFilters');
  if (!FACE_FILTERS && faceFilters) {
    faceFilters.hidden = true;
  }
  faceFilters?.addEventListener('click', () => {
    const drawer = document.getElementById('tune');
    if (drawer?.dataset.open !== 'true') {
      document.getElementById('tuneToggle')?.click();
    }
  });
  function setHandControl(on: boolean): void {
    handControl = on;
    handToggle?.setAttribute('aria-pressed', String(on));
  }
  handToggle?.addEventListener('click', () => setHandControl(!handControl));
  addEventListener('keydown', (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
      return;
    }
    if (event.key.toLowerCase() === 'h') {
      setHandControl(!handControl);
    }
  });

  // The Objects tool anchors the far end of the bar: it plants things rather
  // than changing what the spout carries, so it stands apart from the media
  // cluster it is built with. The button keeps its listeners across the move.
  const propsButton = document.querySelector('#media button[data-id="props"]');
  if (propsButton) {
    document.querySelector('.bar')?.append(propsButton);
  }

  // The Objects selector: a panel in the right rail, one entry per kind, shown
  // while the Objects tool is in hand. Each entry carries a small drawn swatch
  // so the choice is a picture rather than a word in a dropdown.
  const propPanel = document.getElementById('propPanel');
  if (propPanel) {
    const choices = new Map<string, HTMLButtonElement>();
    for (const kind of PROP_KINDS) {
      const choice = document.createElement('button');
      choice.type = 'button';
      choice.className = 'prop-choice';
      choice.setAttribute('aria-pressed', String(kind === propKind));
      const swatch = document.createElement('span');
      swatch.className = `prop-swatch prop-swatch-${kind.id}`;
      choice.append(swatch, kind.label);
      choice.addEventListener('click', () => {
        propKind = kind;
        for (const [id, other] of choices) {
          other.setAttribute('aria-pressed', String(id === kind.id));
        }
      });
      propPanel.append(choice);
      choices.set(kind.id, choice);
    }
  }

  const vesselHost = document.getElementById('vessels');
  function drawVessels(list: Vessel[]): void {
    if (!vesselHost) {
      return;
    }
    while (vesselHost.children.length > list.length) {
      vesselHost.lastChild?.remove();
    }
    while (vesselHost.children.length < list.length) {
      const el = document.createElement('div');
      el.className = 'vessel';
      el.append(document.createElement('span'));
      vesselHost.append(el);
    }
    list.forEach((vessel, index) => {
      const el = vesselHost.children[index];
      if (!(el instanceof HTMLElement)) {
        return;
      }
      el.style.left = `${vessel.box.x0 * 100}%`;
      el.style.top = `${vessel.box.y0 * 100}%`;
      el.style.width = `${(vessel.box.x1 - vessel.box.x0) * 100}%`;
      el.style.height = `${(vessel.box.y1 - vessel.box.y0) * 100}%`;
      const label = el.firstChild;
      if (label instanceof HTMLElement) {
        label.textContent = `holds ${vessel.depth.toFixed(2)}`;
      }
    });
  }

  function syncSpout(): void {
    // The props tool plants at the click, and a pour marker at the same spot
    // reads as part of the object - so it stands down while props are in hand.
    const emphasis = medium === 'props' ? 0 : pouringNow() ? 1 : 0.55;
    renderer.look({ spout: [spoutX, spoutY, spoutZ, emphasis] });
  }

  function moveSpout(clientX: number, clientY: number): void {
    const bounds = canvas.getBoundingClientRect();
    spoutX = Math.min(Math.max((clientX - bounds.left) / bounds.width, 0.03), 0.97);
    spoutY = Math.min(Math.max((clientY - bounds.top) / bounds.height, 0.02), 0.9);
    fluid.tune({ emitterX: spoutX, emitterY: spoutY });
    smoke.tune({ emitterX: spoutX, emitterY: spoutY });
    syncSpout();
  }

  function setSpoutDepth(next: number): void {
    spoutZ = Math.min(Math.max(next, 0.05), Z_MAX - 0.05);
    fluid.tune({ emitterZ: spoutZ });
    smoke.tune({ emitterZ: spoutZ });
    if (medium === 'light') {
      // The scroll wheel is the lamp's depth control; the glow and the marker
      // both follow it immediately.
      pushLights();
      renderer.look({ spout: [hoverX, hoverY, spoutZ, 0.8] });
      return;
    }
    syncSpout();
  }

  canvas.addEventListener('pointerdown', (event) => {
    holding = true;
    canvas.setPointerCapture(event.pointerId);
    hud.dismissHint();
    moveSpout(event.clientX, event.clientY);
    // The light tool plants a lamp where you touch; the spout follows so the
    // preview and the scroll-for-depth control still make sense.
    if (medium === 'light') {
      placeLight();
    }
    if (medium === 'props') {
      placeProp();
    }
    applyFlow();
  });
  canvas.addEventListener('pointermove', (event) => {
    const bounds = canvas.getBoundingClientRect();
    hoverX = Math.min(Math.max((event.clientX - bounds.left) / bounds.width, 0.02), 0.98);
    hoverY = Math.min(Math.max((event.clientY - bounds.top) / bounds.height, 0.02), 0.98);
    if (medium === 'light') {
      // The lamp in hand follows the cursor without any button held, and the
      // marker ball rides along so the glow has a visible source.
      pushLights();
      renderer.look({ spout: [hoverX, hoverY, spoutZ, 0.8] });
    }
    if (holding) {
      moveSpout(event.clientX, event.clientY);
    }
  });
  function releasePointer(event: PointerEvent): void {
    holding = false;
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    applyFlow();
    syncSpout();
  }
  canvas.addEventListener('pointerup', releasePointer);
  canvas.addEventListener('pointercancel', releasePointer);

  // Scroll moves the spout through depth: toward the camera, or back into the scene.
  canvas.addEventListener(
    'wheel',
    (event) => {
      event.preventDefault();
      hud.dismissHint();
      setSpoutDepth(spoutZ - event.deltaY * 0.0004);
    },
    { passive: false },
  );

  addEventListener('resize', syncSpout);

  // Bring in any image or video as the scene, by drop or by file picker. A clip
  // runs through the same path the camera does, so depth tracks it frame by frame.
  const stage = canvas.parentElement;

  function stopClip(): void {
    video.pause();
    video.removeAttribute('loop');
    video.removeAttribute('src');
    video.load();
  }

  /** Point the video element at a URL and wait until it has frames. */
  async function playClip(url: string): Promise<void> {
    video.srcObject = null;
    video.src = url;
    video.loop = true;
    await new Promise<void>((resolve, reject) => {
      if (video.readyState >= 2) {
        resolve();
        return;
      }
      video.addEventListener('loadeddata', () => resolve(), { once: true });
      video.addEventListener('error', () => reject(new Error('unreadable video')), { once: true });
    });
    // Chrome will refuse to start a video-only element that is off screen. The
    // frames are already there, so a refusal is not a failure to load.
    void video.play().catch(() => undefined);
  }

  async function useDroppedFile(file: File): Promise<void> {
    try {
      hud.setStatus('info', `Reading ${file.name}…`);
      // A dropped file is a new scene, and a new scene measures its own
      // gravity. The pin used to survive drops, so one touched slider poisoned
      // every image loaded after it.
      gravityManual = false;
      field.tune({ manual: false });

      if (file.type.startsWith('video/')) {
        if (clipUrl) {
          URL.revokeObjectURL(clipUrl);
        }
        clipUrl = URL.createObjectURL(file);
        clipSource = clipUrl;
        await selectScene('clip');
      } else {
        // The dropped still joins the cache like any other, under its own key.
        // Closing the old bitmap here was the bug the media agent caught: the
        // "old" bitmap was also a cache entry, and the next scenario handed the
        // detached corpse to VideoFrame on every frame from then on.
        const bitmap = await createImageBitmap(file);
        photoUrl = `drop:${file.name}:${dropSerial++}`;
        photoCache.set(photoUrl, bitmap);
        if (photoCache.size > 6) {
          for (const [key, cached] of photoCache) {
            if (key !== photoUrl) {
              photoCache.delete(key);
              cached.close();
              break;
            }
          }
        }
        await selectScene('photo');
      }
      hud.setStatus('info', `Depth from ${file.name}. Hold to pour, scroll for depth.`);
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      hud.setStatus('error', `Could not read that file: ${reason}`);
    }
  }

  // A clip paused by backgrounding picks up again when the tab comes back.
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && sceneId === 'clip' && video.paused) {
      void video.play().catch(() => undefined);
    }
  });

  picker.addEventListener('change', () => {
    const chosen = picker.files?.[0];
    if (chosen) {
      void useDroppedFile(chosen);
    }
    picker.value = '';
  });

  if (stage) {
    stage.addEventListener('dragover', (event) => {
      event.preventDefault();
      stage.dataset.drop = 'true';
    });
    stage.addEventListener('dragleave', () => {
      stage.dataset.drop = 'false';
    });
    stage.addEventListener('drop', (event) => {
      event.preventDefault();
      stage.dataset.drop = 'false';
      const file = event.dataTransfer?.files[0];
      if (file && (file.type.startsWith('image/') || file.type.startsWith('video/'))) {
        void useDroppedFile(file);
      }
    });
  }
  setTimeout(() => hud.dismissHint(), HINT_MS);

  // --- sources ---
  function activeSource(): DepthSource {
    return sceneId !== 'vessels' && model.ready ? model : synthetic;
  }

  /**
   * A video with no decoded frame yet makes importExternalTexture throw, and no
   * property on the element predicts it: readyState and videoWidth both read
   * ready during the gap after a source switch. Asking and catching is the only
   * honest test, and a frame or two without a backdrop is not worth stopping for.
   */
  function importFrame(source: HTMLVideoElement | VideoFrame): GPUExternalTexture | undefined {
    try {
      return root.device.importExternalTexture({ source });
    } catch {
      return undefined;
    }
  }

  function currentFrame(): CameraFrame | undefined {
    if (sceneId === 'clip' && video.readyState >= 2) {
      const texture = importFrame(video);
      return texture
        ? { texture, uvTransform: d.mat2x2f.identity(), swapAxes: false, mirror: false }
        : undefined;
    }
    if (sceneId === 'photo' && photo) {
      photoFrame = new VideoFrame(photo, { timestamp: performance.now() * 1000 });
      const texture = importFrame(photoFrame);
      return texture
        ? { texture, uvTransform: d.mat2x2f.identity(), swapAxes: false, mirror: false }
        : undefined;
    }
    const source = camera.source();
    if (!source) {
      return undefined;
    }
    const texture = importFrame(source);
    if (!texture) {
      return undefined;
    }
    const geometry = rotationGeometry(rotationOverride ?? detectRotation(camera.track()));
    return {
      texture,
      uvTransform: geometry.uvTransform,
      swapAxes: geometry.swapAxes,
      // Mirrored once, here. Every stage downstream shares this one space.
      mirror: mirrorOverride ?? camera.facing === 'user',
    };
  }

  async function ensureModel(): Promise<void> {
    if (loadedSize === modelSize) {
      return;
    }
    hud.setStatus('info', `Downloading the ${modelSize} depth model…`);
    await model.load(modelSize, abort.signal);
    loadedSize = modelSize;
  }

  async function selectModel(size: ModelSize): Promise<void> {
    modelSize = size;
    if (sceneId === 'vessels') {
      return;
    }
    try {
      await ensureModel();
      hud.setStatus('info', `Running the ${size} depth model.`);
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      hud.setStatus('error', `Could not load the ${size} model: ${reason}`);
    }
  }

  async function selectScene(next: SceneId): Promise<void> {
    const previous = sceneId;
    sceneId = next;
    sceneLoading = true;
    document.body.dataset.scene = next;
    hud.setSceneFilter(next);
    // A webcam is near level; a photo or clip can be shot from anywhere.
    // Five degrees, down from eighteen: a front camera is level - the user
    // has said so every time it drifted - and a close-held glass plus the
    // hand under it forms exactly the coherent up-image-leaning cluster that
    // passes every statistical gate. The pin still overrides for the rare
    // ceiling-mounted webcam.
    field.tune({ maxPitch: next === 'camera' ? 5 : 85 });
    if (handToggle) {
      handToggle.hidden = next !== 'camera' && next !== 'clip';
    }
    if (faceFilters) {
      faceFilters.hidden = !FACE_FILTERS || next !== 'camera';
    }
    try {
      if (next === 'vessels') {
        camera.stop();
        stopClip();
        hud.setStatus('info', 'A mug, a bowl and a glass. Hold to pour, scroll for depth.');
        return;
      }
      if (next === 'photo') {
        camera.stop();
        stopClip();
        hud.setStatus('info', 'Loading the photo…');
        photo = photoCache.get(photoUrl);
        if (!photo) {
          photo = await createImageBitmap(await (await fetch(photoUrl)).blob());
          photoCache.set(photoUrl, photo);
        }
        await ensureModel();
        hud.setStatus('info', 'Real depth from a photo. Hold to pour, scroll for depth.');
        return;
      }
      if (next === 'clip') {
        camera.stop();
        hud.setStatus('info', 'Loading the clip…');
        await playClip(clipSource);
        await ensureModel();
        hud.setStatus('info', 'Depth from video, frame by frame. Hold to pour.');
        return;
      }
      hud.setStatus('info', 'Waiting for the camera…');
      stopClip();
      await camera.start();
      await ensureModel();
      hud.setStatus('info', 'Live camera depth. Hold to pour, scroll for depth.');
    } catch (error) {
      sceneId = previous === next ? 'vessels' : previous;
      camera.stop();
      const reason = error instanceof Error ? error.message : String(error);
      hud.setStatus('error', `Could not switch: ${reason}`);
    } finally {
      sceneLoading = false;
    }
  }

  function syncCanvasSize(): void {
    const ratio = Math.min(globalThis.devicePixelRatio || 1, 2);
    const side = Math.min(MAX_CANVAS_SIDE, Math.max(1, Math.round(canvas.clientWidth * ratio)));
    if (canvas.width !== side || canvas.height !== side) {
      canvas.width = side;
      canvas.height = side;
    }
  }

  function step(now: number): void {
    const elapsed = Math.min((now - previousTime) / 1000, 0.05);
    previousTime = now;
    accumulator += elapsed;

    syncCanvasSize();
    const cameraFrame = currentFrame();
    const source = activeSource();
    const encoder = root['~unstable'].createCommandEncoder();

    if (!sceneLoading && now - lastDepthAt >= source.minIntervalMs) {
      if (lastDepthAt > 0) {
        obstacleDt = Math.min(Math.max((now - lastDepthAt) / 1000, 1 / 240), 1 / 10);
      }
      const depthPass = encoder.beginComputePass(
        depthTimestamps ? { timestampWrites: depthTimestamps } : undefined,
      );
      source.encode(depthPass, cameraFrame);
      // Detected vessels get their interiors before anything downstream reads
      // the depth: from here on, the glass IS hollow.
      carver.encode(depthPass);
      field.encode(depthPass);
      // Measured from the same frame the depth came from, so a still, a clip
      // and the webcam all go down one path.
      sceneLight.encode(depthPass, cameraFrame);
      vesselProbe.encode(depthPass);
      depthPass.end();
      lastDepthAt = now;
    }

    let steps = 0;
    while (accumulator >= SOLVER_STEP && steps < MAX_STEPS_PER_FRAME) {
      accumulator -= SOLVER_STEP;
      steps++;
    }
    // Drop the debt rather than owing it to the next frame.
    // Keep up to two steps of debt so one long frame borrows rather than
    // burns; only sustained overload sheds time.
    accumulator = Math.min(accumulator, SOLVER_STEP * 2);
    // Water in flight, water settled, or water about to exist - otherwise the
    // whole solver stack sleeps. It used to grind every dormant particle
    // through three substeps and both splat passes for an empty scene, which
    // was most of "10-20fps with no particles".
    const hasWater = inScene > 0 || pouringNow() || now - lastWetAt < 1500;
    if (hasWater) {
      lastWetAt = pouringNow() || inScene > 0 ? now : lastWetAt;
    }
    if (steps > 0 && hasWater) {
      const simPass = encoder.beginComputePass(
        simTimestamps ? { timestampWrites: simTimestamps } : undefined,
      );
      // Walk the collision surface to where it stands at this instant. Depth
      // lands at video rate; without this the wall jumps a whole frame of an
      // obstacle's motion at once and the solver shoves water out of it far
      // faster than the obstacle ever moved.
      field.advance(
        simPass,
        lastDepthAt > 0 ? (now - lastDepthAt) / (obstacleDt * 1000) : 1,
      );
      for (let index = 0; index < steps; index++) {
        fluid.encode(simPass);
        solverSteps++;
      }
      simPass.end();
      // The spout draws from the dormant pool. Once that pool is too thin to
      // cover a frame's worth of emission the pour would silently stop, so
      // start circulating settled water instead. Latched, because the census
      // wobbles either side of the threshold once recycling is running and
      // toggling on every wobble buys nothing.
      if (
        pouringNow() &&
        medium === 'water' &&
        !recycling &&
        (PARTICLE_COUNT - inScene < emitRate * MAX_STEPS_PER_FRAME ||
          // The solver's cost is proportional to alive water - measured 22ms
          // of a 24ms frame at a 100k pool. When pacing slips, stop growing
          // the pool and circulate instead: the machine sets its own budget.
          (frameMsEma > 24 && inScene > 20000))
      ) {
        recycling = true;
        fluid.tune({ recycle: true });
      }
    }

    // Recount the water a few times a second. The readback is four bytes and
    // one map, but it lands a frame or two late, so a fresh request only goes
    // out once the last one is back.
    // Whatever tool is in hand - water in the scene is water in the scene, and
    // a meter that freezes when you put the tap down reads as broken.
    if (!censusPending && now - lastCensusAt > 200) {
      censusPending = true;
      lastCensusAt = now;
      void fluid.population
        .read()
        .then(([count]) => {
          inScene = count;
          hud.setFill(inScene / PARTICLE_COUNT);
        })
        .catch(() => {
          // A read in flight when the device is torn down aborts; expected.
        })
        .finally(() => {
          censusPending = false;
        });
    }

    // Follow the light in the picture. Read back rather than bound into every
    // shader: it is three vectors that move slowly, and a readback costs one
    // map instead of a binding in six pipelines.
    if (!lightPending && now - lastLightAt > 400) {
      lightPending = true;
      lastLightAt = now;
      void sceneLight.state
        .read()
        .then((light) => {
          const [x, y, z] = [light.sun.x, light.sun.y, light.sun.z];
          const height = (Math.asin(Math.max(Math.min(z, 1), -1)) * 180) / Math.PI;
          const bearing = (Math.atan2(x, -y) * 180) / Math.PI;
          renderer.look({ sunBearing: bearing, sunHeight: height });
          smoke.tune({ sunBearing: bearing, sunHeight: height });
          hud.setTune('sunBearing', bearing);
          hud.setTune('sunHeight', height);
        })
        .catch(() => {
          // A read in flight when the device is torn down aborts; expected.
        })
        .finally(() => {
          lightPending = false;
        });
    }

    // While it is measuring, show the measurement on the controls, so what the
    // scene believes is always visible and one drag away from being overridden.
    if (!gravityManual && !gravityPending && now - lastGravityAt > 500) {
      gravityPending = true;
      lastGravityAt = now;
      void field.scene
        .read()
        .then(({ down }) => {
          gravityPitch = (Math.asin(Math.max(Math.min(-down.z, 1), -1)) * 180) / Math.PI;
          gravityRoll = (Math.atan2(down.x, down.y) * 180) / Math.PI;
          hud.setTune('gravityPitch', gravityPitch);
          hud.setTune('gravityRoll', gravityRoll);
          showGravity();
        })
        .catch(() => {
          // A read in flight when the device is torn down aborts; expected.
        })
        .finally(() => {
          gravityPending = false;
        });
    }

    // Sample the GPU pass timings a few times a second. One readback covers
    // both slots; skip while a previous map is in flight.
    if (gpuQuery?.available && now - lastGpuReadAt > 500) {
      lastGpuReadAt = now;
      gpuQuery.resolve();
      void gpuQuery
        .read()
        .then((stamps) => {
          gpuMs.depth = Number(stamps[1] - stamps[0]) / 1e6;
          gpuMs.sim = Number(stamps[3] - stamps[2]) / 1e6;
        })
        .catch(() => {
          // Device teardown mid-read; expected.
        });
    }

    driveTorch(now);
    driveTracking(now);
    driveCups();

    // The flood runs all the time now, not only while the drawer is open: its
    // spill surface feeds the solver's calm gate, which is how the sim knows
    // that water piled above a basin's rim has nowhere to rest. The outline
    // overlay still only draws with the drawer.
    if (!vesselPending && now - lastVesselAt > 600) {
      vesselPending = true;
      lastVesselAt = now;
      void vesselProbe.buffer
        .read()
        .then((cells) => {
          probeCells = cells;
          const pitch = (gravityPitch * Math.PI) / 180;
          const roll = (gravityRoll * Math.PI) / 180;
          const flood = floodVessels(
            cells,
            [Math.sin(roll) * Math.cos(pitch), Math.cos(roll) * Math.cos(pitch), -Math.sin(pitch)],
            depthScaleNow,
          );
          fluid.setSpill(Float32Array.from(flood.filled));
          if (document.getElementById('tune')?.dataset.open === 'true') {
            drawVessels(flood.vessels);
          } else {
            drawVessels([]);
          }
        })
        .catch(() => {
          // A read in flight when the device is torn down aborts; expected.
        })
        .finally(() => {
          vesselPending = false;
        });
    }

    if (storming) {
      driveStorm(now);
    }

    // Smoke steps once a frame. Semi-Lagrangian advection is stable at any step
    // size, so it takes the frame time rather than the solver's fixed tick.
    // Once smoke exists it keeps living whatever tool is in hand; only a scene
    // reset clears it.
    if (smokeAlive || smokeStale) {
      const smokePass = encoder.beginComputePass();
      // Not either-or: after a scene change with the pour held, the old plume
      // must be wiped and the new one stepped in the same frame.
      if (smokeStale) {
        smoke.clear(smokePass);
        smokeStale = false;
      }
      if (smokeAlive) {
        driveSmoke(now);
        smoke.encode(smokePass, Math.min(elapsed, 1 / 30));
        smoke.encodeRender(smokePass);
      }
      smokePass.end();
    }

    renderer.encodeSurface(encoder, hasWater);
    renderer.encodeComposite(encoder, cameraFrame);
    encoder.submit();
    encodes++;

    photoFrame?.close();
    photoFrame = undefined;
  }

  /**
   * A source that never moves makes a column that never bends. Real smoke comes
   * off a source that breathes and drifts, and the plume inherits that: the
   * wobble is what rolls the column into puffs. Two slow rates that do not
   * divide into each other keep it from finding a rhythm.
   */
  function driveSmoke(now: number): void {
    const time = now / 1000;
    smoke.tune({
      emitterX: spoutX + Math.sin(time * 0.83) * 0.018 + Math.sin(time * 2.31) * 0.007,
      emitterY: spoutY + Math.sin(time * 1.27) * 0.012,
      emitRate:
        (pouringNow() && medium === 'smoke' ? smokeRate : 0) *
        (0.8 + 0.3 * Math.sin(time * 1.11) + 0.15 * Math.sin(time * 2.7)),
    });
  }

  /**
   * Gusts and lightning. Both are timed on the CPU: the wind is a damped random
   * walk fed to the rain as it enters, and a strike is a spike that decays over
   * a few frames, with a second smaller one part of the time - real lightning
   * flickers rather than fading once.
   */
  function driveStorm(now: number): void {
    gust = Math.max(-0.5, Math.min(0.5, gust * 0.985 + (Math.random() - 0.5) * 0.05));
    fluid.tune({ wind: gust });
    if (smokeAlive) {
      smoke.tune({ wind: gust });
    }

    if (now >= nextStrike) {
      flash = 1;
      nextStrike = now + 3000 + Math.random() * 7000;
      reStrikeAt = Math.random() < 0.45 ? now + 90 + Math.random() * 120 : 0;
    } else if (reStrikeAt !== 0 && now >= reStrikeAt) {
      flash = Math.max(flash, 0.7);
      reStrikeAt = 0;
    }

    flash *= 0.84;
    renderer.setFlash(flash < 0.004 ? 0 : flash);
  }

  /** Row-major forward maps matching rotationGeometry's matrices. */
  const ROTATION_FORWARD = {
    0: [1, 0, 0, 1],
    90: [0, 1, -1, 0],
    180: [-1, 0, 0, -1],
    270: [0, -1, 1, 0],
  } satisfies Record<number, readonly [number, number, number, number]>;

  let gesturesBuiltFor = '';
  function ensureGestures(): void {
    const wanted = `${FACE_FILTERS}:${handControl}`;
    if (gesturesLoading || (gestures && gesturesBuiltFor === wanted)) {
      return;
    }
    const stale = gestures;
    gestures = undefined;
    stale?.destroy();
    gesturesLoading = true;
    gesturesBuiltFor = wanted;
    void import('./track/gestures.ts')
      .then(async (mod) => {
        gestures = await mod.createGestures({ face: FACE_FILTERS, hands: handControl });
      })
      .catch((error) => {
        const reason = error instanceof Error ? error.message : String(error);
        hud.setStatus('info', `Face and hand tracking unavailable: ${reason}`);
      })
      .finally(() => {
        gesturesLoading = false;
      });
  }

  /**
   * Face and hand effects, run at video rate.
   *
   * The fingertip is a cursor: whatever tool is in hand follows it, and a
   * pinch is the pour button. Sweat and tears are water sources pinned to the
   * face itself - one emitter cycling around the springs at 120Hz reads as all
   * of them running at once - and the auto-aim spawn clamp lands each drop ON
   * the face, so it rolls down the real geometry.
   */
  /** Cups flow to the renderer and the carver whatever scene is up. */
  let latestCups: {
    box: readonly [number, number, number, number];
    front: number;
    carve: number;
  }[] = [];
  let surfaceShellNow = defaultTuning.surfaceShell;
  let lastCupsAt = performance.now();
  let shiftEma: [number, number] = [0, 0];
  function driveCups(): void {
    // Detected cups reach the renderer, and the auto-aim spawn does the rest:
    // pour over a held glass and the water spawns just in front of it. The
    // collision carve that gives the glass a real interior lands once the
    // boundary work in flight settles; until then the visual transparency and
    // the spawn aim are already live.
    const cupBoxes: [number, number, number, number][] = [];
    for (const vessel of tracked.vessels.slice(0, 3)) {
      cupBoxes.push([vessel.x0, vessel.y0, vessel.x1, vessel.y1]);
    }
    while (cupBoxes.length < 3) {
      cupBoxes.push([0, 0, 0, 0]);
    }
    renderer.look({ cups: cupBoxes });

    // The carve needs each vessel's front depth; the probe's downsample is the
    // cheapest honest source. The estimate has to be stable above all else:
    // every flicker re-digs the cavity at a new depth, and that topology quake
    // ejects whatever water had settled in it. Measured flapping between the
    // glass (0.79) and the hand holding it (1.0) when a nearest-of-five rule
    // sampled around the box centre. And it must survive a glass the depth
    // model cannot see at all: against a dark room the walls read as VOID,
    // so a wall-margin median collapsed to nothing and every cup mechanism
    // keyed off nonsense. A high percentile over a grid across the whole box
    // finds the nearest real material - the hand holding the glass, the rim
    // highlights, whatever the model did register - and ignores the void by
    // construction. An eased hold on top, against per-frame depth noise.
    const cups = tracked.vessels.slice(0, 3).map((vessel) => {
      let front = 0.5;
      if (probeCells) {
        const at = (x: number, y: number) => {
          const i = Math.min(Math.max(Math.round(x * 64), 0), 63);
          const j = Math.min(Math.max(Math.round(y * 64), 0), 63);
          return probeCells?.[j * 64 + i] ?? 0;
        };
        const w = vessel.x1 - vessel.x0;
        const h = vessel.y1 - vessel.y0;
        const samples: number[] = [];
        for (const tx of [0.1, 0.3, 0.5, 0.7, 0.9]) {
          for (const ty of [0.15, 0.35, 0.55, 0.75, 0.92]) {
            samples.push(at(vessel.x0 + w * tx, vessel.y0 + h * ty));
          }
        }
        samples.sort((a, b) => a - b);
        front = samples[20];
      }
      const prior = latestCups.find(
        (old) =>
          Math.abs(old.box[0] - vessel.x0) < 0.15 &&
          Math.abs(old.box[1] - vessel.y0) < 0.15,
      );
      if (prior) {
        front = prior.front + (front - prior.front) * 0.1;
      }
      // The detector reliably cuts off the top of a clear glass - a
      // transparent rim against a pale wall has almost no features - and an
      // uncarved rim is a solid lid: pours splashed off it and ran down the
      // outside. But a fixed extension overshoots when the detector was
      // right, and then the mouth logic waits for water in the middle of the
      // wall. So the box grows upward only while the probe still reads glass
      // there: the vessel's own front, or the cavity our carve already dug.
      let y0 = vessel.y0;
      if (probeCells) {
        const h = vessel.y1 - vessel.y0;
        const carve = carveDepth(vessel.x1 - vessel.x0, surfaceShellNow);
        const cols = [0.35, 0.5, 0.65].map((t) =>
          Math.min(Math.max(Math.round((vessel.x0 + (vessel.x1 - vessel.x0) * t) * 64), 0), 63),
        );
        let j = Math.min(Math.max(Math.round(vessel.y0 * 64), 0), 63);
        const jMin = Math.max(Math.round((vessel.y0 - h * 0.35) * 64), 0);
        while (j - 1 >= jMin) {
          const row = cols
            .map((i) => probeCells?.[(j - 1) * 64 + i] ?? 0)
            .sort((a, b) => a - b);
          // The carved acceptance must be tight: the cavity floor sits at
          // front - carve exactly, while background walls land anywhere. At
          // 0.14 the tolerance matched a door 0.14 away and the extension
          // climbed it to the frame top, capturing pours in mid-air.
          const glassy =
            Math.abs(row[1] - front) < 0.14 ||
            Math.abs(row[1] - (front - carve)) < 0.1;
          if (!glassy) {
            break;
          }
          j -= 1;
        }
        y0 = j / 64;
      }
      // Eased against the previous frame's answer: the walk re-reads a noisy
      // probe every frame, and each wobble of the box top moves the mouth
      // band and re-digs the carve - at rest, the interior visibly lost and
      // regained its depth. The prior's own top is the memory.
      const remembered = latestCups.find(
        (held) =>
          Math.abs(held.box[0] - vessel.x0) < 0.15 &&
          Math.abs(held.box[3] - vessel.y1) < 0.15,
      );
      if (remembered) {
        y0 = remembered.box[1] + (y0 - remembered.box[1]) * 0.12;
      }
      // The box's own velocity, so the solver can carry the water inside a
      // moving glass. Eased boxes move smoothly; the remaining rate is real
      // motion, clamped so one detection jump cannot fling the pool.
      const dtCups = Math.max((performance.now() - lastCupsAt) / 1000, 1 / 240);
      let shift: [number, number] = [0, 0];
      if (remembered) {
        // Deadband before dividing by frame time: the eased box wobbles a
        // fraction of a texel every frame, and jitter over 8ms became +-1.5
        // units/s of advection thrashing the pool against its own walls -
        // "splashes from every wall" on a perfectly static photo. Real hand
        // motion sustains past the deadband; noise does not.
        const rate = (d: number) => {
          if (Math.abs(d) < 0.0035) {
            return 0;
          }
          return Math.min(Math.max(d / dtCups, -1.5), 1.5);
        };
        const dx = ((vessel.x0 + vessel.x1) - (remembered.box[0] + remembered.box[2])) / 2;
        const dy = ((y0 + vessel.y1) - (remembered.box[1] + remembered.box[3])) / 2;
        shift = [
          shiftEma[0] + (rate(dx) - shiftEma[0]) * 0.3,
          shiftEma[1] + (rate(dy) - shiftEma[1]) * 0.3,
        ];
        shiftEma = shift;
      }
      return {
        box: [vessel.x0, y0, vessel.x1, vessel.y1] as const,
        front,
        carve: carveDepth(vessel.x1 - vessel.x0, surfaceShellNow),
        shift,
      };
    });
    lastCupsAt = performance.now();
    carver.set(cups);
    // The solver gets the same cups, for the near-wall containment.
    fluid.tune({ cups });
    latestCups = cups;
    // The outline is a diagnostic; it lives with the controls, like the
    // gravity arrow. The pour channels ride along: the render must know where
    // a falling stream outranks the depth model's halo.
    renderer.look({
      cupLines: document.getElementById('tune')?.dataset.open === 'true' ? 1 : 0,
      cupMouths: [0, 1, 2].map((i): [number, number, number, number] => {
        const cup = cups[i];
        if (!cup) {
          return [0, 0, 0, 0];
        }
        const w = cup.box[2] - cup.box[0];
        return [
          cup.box[0] + w * 0.14,
          cup.box[2] - w * 0.14,
          cup.box[1] + (cup.box[3] - cup.box[1]) * 0.25,
          1,
        ];
      }),
    });

  }

  /** The dropped photo drawn once for the detector; keyed by its cache id. */
  let stillCanvas: HTMLCanvasElement | undefined;
  let stillKey = '';

  function driveTracking(now: number): void {
    if (sceneId === 'photo') {
      // A glass in a dropped photo deserves an interior as much as one held
      // to the camera. Only the vessel detector runs - there is no face to
      // track on a still that never changes - but it must KEEP running: the
      // persistence rule needs its three consecutive rounds, and the carve
      // needs the box refreshed if a new photo lands.
      const bitmap = photoCache.get(photoUrl);
      if (!bitmap) {
        return;
      }
      ensureGestures();
      if (!gestures) {
        return;
      }
      if (stillKey !== photoUrl || !stillCanvas) {
        stillCanvas = document.createElement('canvas');
        stillCanvas.width = bitmap.width;
        stillCanvas.height = bitmap.height;
        stillCanvas.getContext('2d')?.drawImage(bitmap, 0, 0);
        stillKey = photoUrl;
      }
      tracked = gestures.readStill(
        stillCanvas,
        { mirror: false, swapAxes: false, uv: [1, 0, 0, 1] },
        now,
      );
      return;
    }
    if (sceneId !== 'camera' && sceneId !== 'clip') {
      if (fingerPour) {
        fingerPour = false;
        applyFlow();
      }
      return;
    }
    ensureGestures();
    if (!gestures) {
      return;
    }
    const rotation = sceneId === 'camera' ? (rotationOverride ?? detectRotation(camera.track())) : 0;
    tracked = gestures.read(
      video,
      {
        mirror: sceneId === 'camera' ? (mirrorOverride ?? camera.facing === 'user') : false,
        swapAxes: rotation === 90 || rotation === 270,
        // SAFETY: rotation comes from rotationGeometry's domain - one of the
        // four quarter turns - and the fallback covers anything else.
        uv: ROTATION_FORWARD[rotation as 0 | 90 | 180 | 270] ?? ROTATION_FORWARD[0],
      },
      now,
    );

    const tip = handControl ? tracked.tip : undefined;
    if (tip && !holding) {
      hoverX = Math.min(Math.max(tip.x, 0.02), 0.98);
      hoverY = Math.min(Math.max(tip.y, 0.02), 0.98);
      if (medium === 'light') {
        pushLights();
        renderer.look({ spout: [hoverX, hoverY, spoutZ, 0.8] });
      } else {
        spoutX = hoverX;
        spoutY = hoverY;
        fluid.tune({ emitterX: spoutX, emitterY: spoutY });
        smoke.tune({ emitterX: spoutX, emitterY: spoutY });
        syncSpout();
      }
    }
    const thumb = handControl ? tracked.thumb : undefined;
    const pinched =
      tip !== undefined &&
      thumb !== undefined &&
      Math.hypot(tip.x - thumb.x, tip.y - thumb.y) < 0.055;
    if (pinched !== fingerPour) {
      fingerPour = pinched;
      applyFlow();
    }

    // Light-up glasses ride the irises, colour wheeling on its own.
    if (FACE_FILTERS && glassesGlow > 0 && tracked.lenses.length === 2) {
      const hue = (now / 25) % 360;
      const h = hue / 60;
      const x = 1 - Math.abs((h % 2) - 1);
      const sector = Math.floor(h) % 6;
      const wheel: [number, number, number][] = [
        [1, x, 0], [x, 1, 0], [0, 1, x], [0, x, 1], [x, 0, 1], [1, 0, x],
      ];
      const [r, g, b] = wheel[sector];
      renderer.look({
        glassesA: [tracked.lenses[0].x, tracked.lenses[0].y, tracked.lenses[1].x, tracked.lenses[1].y],
        glassesB: [r * 0.7 + 0.3, g * 0.7 + 0.3, b * 0.7 + 0.3, glassesGlow],
      });
      glassesOn = true;
    } else if (glassesOn) {
      glassesOn = false;
      renderer.look({ glassesB: [1, 1, 1, 0] });
    }

    // The face's own water. It borrows the one fluid emitter whenever the user
    // is not actively pouring, cycling it around the springs.
    const springs: { x: number; y: number; rate: number }[] = [];
    if (FACE_FILTERS && sweatRate > 0) {
      const rate = sweatRate * (0.4 + 1.6 * tracked.effort);
      for (const point of tracked.brow) {
        springs.push({ x: point.x, y: point.y, rate });
      }
    }
    if (FACE_FILTERS && tearRate > 0) {
      for (const point of tracked.eyes) {
        springs.push({ x: point.x, y: point.y, rate: tearRate });
      }
    }
    const userPouring = storming || latched || holding || fingerPour;
    if (springs.length > 0 && !userPouring) {
      springStep = (springStep + 1) % springs.length;
      const spring = springs[springStep];
      faceOwned = true;
      fluid.tune({
        emitterX: Math.min(Math.max(spring.x, 0.02), 0.98),
        emitterY: Math.min(Math.max(spring.y, 0.02), 0.98),
        emitRate: Math.round(spring.rate),
        emitSpread: 0.008,
        emitSpeed: 0.05,
      });
    } else if (faceOwned) {
      faceOwned = false;
      fluid.tune({
        emitterX: spoutX,
        emitterY: spoutY,
        emitSpread: defaultTuning.emitSpread,
        emitSpeed: defaultTuning.emitSpeed,
      });
      applyFlow();
    }
  }

  /** Frame pacing, for the lag question "is it me or the sim". EMA tracks
   * the typical frame, the decaying peak keeps the worst recent hitch
   * visible for a few seconds. Shown with the gravity probe while the
   * drawer is open; globalThis.perf() in dev. */
  let frameMsEma = 16.7;
  let frameMsPeak = 16.7;
  let lastFrameAt = 0;

  function frame(now: number): void {
    if (lastFrameAt > 0) {
      const ms = now - lastFrameAt;
      frameMsEma += (ms - frameMsEma) * 0.05;
      frameMsPeak = Math.max(frameMsPeak * 0.995, ms);
    }
    lastFrameAt = now;
    try {
      step(now);
    } catch (error) {
      // Keep going. Nearly every failure here is a source that is briefly not
      // ready - a clip mid-switch, a camera track restarting - and stopping the
      // loop turns a hiccup into a page that never comes back.
      const reason = error instanceof Error ? error.message : String(error);
      hud.setStatus('error', `Frame failed: ${reason}`);
    }
    requestAnimationFrame(frame);
  }

  if (import.meta.env.DEV) {
    // A hidden tab never gets requestAnimationFrame, so expose a manual pump
    // for driving the simulation from a headless check.
    Object.assign(globalThis, {
      probe: () => fluid.particles.read(),
      tracked: () => tracked,
      fakeVessel: (x0: number, y0: number, x1: number, y1: number) => {
        tracked = {
          ...tracked,
          vessels: [{ x0, y0, x1, y1, label: 'cup' }],
        };
      },
      look: (next: Parameters<typeof renderer.look>[0]) => renderer.look(next),
      gravityDir: () => field.scene.read(),
      storm: () => ({ flash, gust }),
      smokeGrid: () => smoke.readCells(),
      counters: () => ({ solverSteps, encodes }),
      cupState: () => latestCups,
      fluidWindow: () => fluid.window(),
      perf: () => ({ frameMsEma, frameMsPeak, gpuDepthMs: gpuMs.depth, gpuSimMs: gpuMs.sim, solverSteps, encodes }),
      probeCells: () => probeCells,
      gestureState: () => (gestures ? 'loaded' : gesturesLoading ? 'loading' : 'none'),
      tuneFluid: (next: Parameters<typeof fluid.tune>[0]) => fluid.tune(next),
      pump: (count: number) => {
        for (let index = 0; index < count; index++) {
          step(previousTime + SOLVER_STEP * 1000);
        }
      },
    });
  }

  void root.device.lost.then((info) => {
    hud.setStatus('error', `GPU device lost: ${info.message || info.reason}`);
  });

  // Open on the bathroom rather than the built vessels. selectScene is what
  // actually fetches the photo and warms the depth model; setting the id alone
  // leaves the page black.
  void selectScene('photo');

  applyFlow();
  syncSpout();
  document.body.dataset.tool = medium;
  hud.setToolFilter(medium);
  hud.setFill(0);
  hud.setStatus(
    'info',
    `${PARTICLE_COUNT.toLocaleString()} particles · build ${__BUILD__}. Hold to pour.`,
  );
  requestAnimationFrame(frame);
}

main().catch((error) => {
  const status = document.getElementById('status');
  if (status) {
    status.dataset.tone = 'error';
    status.textContent = error instanceof Error ? error.message : String(error);
  }
});
