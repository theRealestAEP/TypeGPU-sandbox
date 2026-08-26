import { d, std, tgpu } from 'typegpu';
import type { StorageFlag, TgpuBuffer, TgpuComputePass, TgpuRoot, TgpuSampler } from 'typegpu';
import type { SingleChannelTexture } from '../gpu/blur.ts';
import { CameraParams, LightState } from '../sim/schemas.ts';
import type { CameraFrame } from './depth-field.ts';

/**
 * Where the light in the picture is coming from, measured from the picture.
 *
 * The scene is lit however it was lit when it was shot, and until now the app
 * guessed with two sliders. Anything the liquid reflects, and every object
 * floating in it, was lit from somewhere unrelated to the room around it - which
 * is the surest way to make added things look added.
 *
 * The estimate is a regression, not a heuristic. A surface's brightness rises
 * with how squarely it faces the light, so across the frame luminance and
 * surface normal covary, and the direction of that covariance is the direction
 * of the light:
 *
 *     L  =  normalise( sum (Y - mean Y) * (n - mean n) )
 *
 * Both terms are mean-centred, which makes this exactly the slope of the
 * least-squares fit of luminance against normal, with the ambient level falling
 * out as the intercept. It needs no assumption about albedo beyond it being
 * uncorrelated with orientation, which over a whole frame is fair.
 *
 * Normals come from the depth field, so this works the same for a still, a clip
 * or the webcam - all three land in the same texture.
 */

const THREADS = 256;
/** Samples across the frame. 64 squared is plenty for a single direction. */
const LIGHT_RES = 64;
const STRIDE = Math.ceil((LIGHT_RES * LIGHT_RES) / THREADS);

/**
 * How much of the previous estimate each new one keeps.
 *
 * A still frame does not need this, but a clip and a webcam very much do: the
 * frame is noisy, people walk through it, and exposure hunts. Writing the raw
 * estimate through makes every highlight in the scene swim. Slow enough to hold
 * still, quick enough to follow someone carrying a lamp across a room.
 */
const SMOOTHING = 0.93;

/** Below this much agreement the frame has no dominant light worth following. */
const MIN_STRENGTH = 0.004;

const lightLayout = tgpu.bindGroupLayout({
  frame: { externalTexture: d.textureExternal() },
  camera: { uniform: CameraParams },
  surface: { texture: d.texture2d(d.f32) },
  fieldSampler: { sampler: 'filtering' },
  light: { storage: LightState, access: 'mutable' },
});

/**
 * Three scratch arrays, used twice. Workgroup storage is capped at 16 KB and a
 * vec3f occupies 16 bytes there, so five arrays of 256 did not fit. The second
 * pass reuses the first pass's arrays, which is safe because thread zero has
 * already folded them into the scalars below and every thread has passed the
 * barrier since.
 */
const lumaSums = tgpu.workgroupVar(d.arrayOf(d.f32, THREADS));
const normalSums = tgpu.workgroupVar(d.arrayOf(d.vec3f, THREADS));
const colourSums = tgpu.workgroupVar(d.arrayOf(d.vec3f, THREADS));
const meanLuma = tgpu.workgroupVar(d.f32);
const meanNormal = tgpu.workgroupVar(d.vec3f);

/** Rec. 709, because the eye weighs green far above blue. */
const luminanceOf = (colour: d.v3f) => {
  'use gpu';
  return std.dot(colour, d.vec3f(0.2126, 0.7152, 0.0722));
};

/**
 * Field uv to camera-texture uv. Must match the depth preprocessor step for
 * step - mirror, centred square crop, orientation - or the luminance at a texel
 * belongs to a different part of the scene than the normal at that texel, and
 * the whole correlation is against the wrong thing.
 */
const frameUvOf = (uv: d.v2f) => {
  'use gpu';
  const camera = lightLayout.$.camera;
  const mirrored = d.vec2f(std.select(uv.x, 1 - uv.x, camera.mirror !== 0), uv.y);
  let sourceSize = d.vec2f(std.textureDimensions(lightLayout.$.frame));
  if (camera.swapAxes !== 0) {
    sourceSize = d.vec2f(sourceSize.yx);
  }
  const scale = std.min(sourceSize.x, sourceSize.y) / sourceSize;
  const cropped = (d.vec2f(1) - scale) * 0.5 + mirrored * scale;
  return camera.uvTransform * (cropped - d.vec2f(0.5)) + d.vec2f(0.5);
};

const normalAt = (uv: d.v2f) => {
  'use gpu';
  const step = d.f32(1.5 / LIGHT_RES);
  const slope = d.vec2f(
    std.textureSampleLevel(lightLayout.$.surface, lightLayout.$.fieldSampler, uv + d.vec2f(step, 0), 0).x -
      std.textureSampleLevel(lightLayout.$.surface, lightLayout.$.fieldSampler, uv - d.vec2f(step, 0), 0).x,
    std.textureSampleLevel(lightLayout.$.surface, lightLayout.$.fieldSampler, uv + d.vec2f(0, step), 0).x -
      std.textureSampleLevel(lightLayout.$.surface, lightLayout.$.fieldSampler, uv - d.vec2f(0, step), 0).x,
  ) / (2 * step);
  return std.normalize(d.vec3f(slope * -1, 1));
};

const colourAt = (uv: d.v2f) => {
  'use gpu';
  return std.textureSampleBaseClampToEdge(
    lightLayout.$.frame,
    lightLayout.$.fieldSampler,
    frameUvOf(std.saturate(uv)),
  ).rgb;
};

const lightKernel = tgpu.computeFn({
  workgroupSize: [THREADS],
  in: { lid: d.builtin.localInvocationIndex },
})(({ lid }) => {
  'use gpu';
  // Pass one: the averages the regression is centred on.
  let luma = d.f32(0);
  let normal = d.vec3f();
  let colour = d.vec3f();
  for (const step of std.range(STRIDE)) {
    const index = lid * STRIDE + step;
    if (index < LIGHT_RES * LIGHT_RES) {
      const uv = (d.vec2f(d.f32(index % LIGHT_RES), d.f32(index / LIGHT_RES)) + 0.5) / d.f32(LIGHT_RES);
      const sample = colourAt(uv);
      luma += luminanceOf(sample);
      colour = colour + sample;
      normal = normal + normalAt(uv);
    }
  }
  lumaSums.$[lid] = luma;
  normalSums.$[lid] = d.vec3f(normal);
  colourSums.$[lid] = d.vec3f(colour);
  std.workgroupBarrier();

  if (lid === 0) {
    let lumaTotal = d.f32(0);
    let normalTotal = d.vec3f();
    let colourTotal = d.vec3f();
    for (const slot of std.range(THREADS)) {
      lumaTotal += lumaSums.$[slot];
      normalTotal = normalTotal + normalSums.$[slot];
      colourTotal = colourTotal + colourSums.$[slot];
    }
    const count = d.f32(LIGHT_RES * LIGHT_RES);
    meanLuma.$ = lumaTotal / count;
    meanNormal.$ = normalTotal / count;
    lightLayout.$.light.ambient = colourTotal / count;
  }
  std.workgroupBarrier();

  // Pass two: how brightness and orientation move together, and what colour the
  // brighter-than-average part of the frame is.
  const centreLuma = meanLuma.$;
  const centreNormal = d.vec3f(meanNormal.$);
  let cross = d.vec3f();
  let key = d.vec3f();
  for (const step of std.range(STRIDE)) {
    const index = lid * STRIDE + step;
    if (index < LIGHT_RES * LIGHT_RES) {
      const uv = (d.vec2f(d.f32(index % LIGHT_RES), d.f32(index / LIGHT_RES)) + 0.5) / d.f32(LIGHT_RES);
      const sample = colourAt(uv);
      const lift = luminanceOf(sample) - centreLuma;
      cross = cross + (normalAt(uv) - centreNormal) * lift;
      key = key + sample * std.max(lift, 0);
    }
  }
  normalSums.$[lid] = d.vec3f(cross);
  colourSums.$[lid] = d.vec3f(key);
  std.workgroupBarrier();

  if (lid === 0) {
    let crossTotal = d.vec3f();
    let keyTotal = d.vec3f();
    for (const slot of std.range(THREADS)) {
      crossTotal = crossTotal + normalSums.$[slot];
      keyTotal = keyTotal + colourSums.$[slot];
    }
    crossTotal = crossTotal / d.f32(LIGHT_RES * LIGHT_RES);

    // With no dominant light - a flat wall, an evenly lit room - the covariance
    // collapses and its direction is noise. Hold the previous answer rather
    // than chase it.
    const strength = std.length(crossTotal);
    let sun = d.vec3f(lightLayout.$.light.sun);
    if (strength > MIN_STRENGTH) {
      sun = std.normalize(crossTotal);
    }

    // Normalised to unit brightness: this says what colour the light is, and
    // the exposure of the shot is not ours to reproduce.
    let colour = d.vec3f(1);
    const keyLuma = luminanceOf(keyTotal);
    if (keyLuma > 1e-5) {
      colour = keyTotal / keyLuma;
    }

    const previous = d.vec3f(lightLayout.$.light.sun);
    const blended = std.normalize(sun * (1 - SMOOTHING) + previous * SMOOTHING);
    lightLayout.$.light.sun = d.vec3f(blended);
    lightLayout.$.light.colour = std.mix(
      colour,
      d.vec3f(lightLayout.$.light.colour),
      SMOOTHING,
    );
    lightLayout.$.light.strength = std.mix(
      std.saturate(strength / (MIN_STRENGTH * 8)),
      lightLayout.$.light.strength,
      SMOOTHING,
    );
  }
});

export interface SceneLight {
  readonly state: TgpuBuffer<typeof LightState> & StorageFlag;
  initAsync(): Promise<void>;
  /** Re-measures from the current frame. Skipped when there is no frame to read. */
  encode(pass: TgpuComputePass, frame: CameraFrame | undefined): void;
  destroy(): void;
}

export function createSceneLight(
  root: TgpuRoot,
  surface: SingleChannelTexture,
  fieldSampler: TgpuSampler,
): SceneLight {
  const state = root
    .createBuffer(LightState, {
      // Over the viewer's left shoulder: the safe default a lighting artist
      // reaches for, and where the sliders used to sit.
      sun: d.vec3f(-0.4, -0.55, 0.73),
      colour: d.vec3f(1, 1, 1),
      ambient: d.vec3f(0.5, 0.5, 0.5),
      strength: 0,
    })
    .$usage('storage');
  const camera = root.createBuffer(CameraParams).$usage('uniform');
  const pipeline = root.createComputePipeline({ compute: lightKernel });

  return {
    state,

    async initAsync() {
      await pipeline.initAsync();
    },

    encode(pass, frame) {
      if (!frame) {
        return;
      }
      camera.write({
        uvTransform: frame.uvTransform,
        mirror: frame.mirror ? 1 : 0,
        swapAxes: frame.swapAxes ? 1 : 0,
      });
      const bindGroup = root.createBindGroup(lightLayout, {
        frame: frame.texture,
        camera,
        surface: surface.createView(),
        fieldSampler,
        light: state,
      });
      pipeline.with(pass).with(bindGroup).dispatchWorkgroups(1);
    },

    destroy() {
      state.destroy();
      camera.destroy();
    },
  };
}
