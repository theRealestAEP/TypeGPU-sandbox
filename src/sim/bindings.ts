import { d, tgpu } from 'typegpu';
import { Particle, SceneState, SimParams } from './schemas.ts';

/** Everything the position-based-fluid passes touch. */
export const simLayout = tgpu.bindGroupLayout({
  params: { uniform: SimParams },
  particles: { storage: d.arrayOf(Particle), access: 'mutable' },
  /**
   * Solver scratch, one per particle: the position correction in xyz and the
   * constraint multiplier in w. They ride together because storage bindings are
   * capped at eight per stage and these two are written and read by the same
   * three kernels, in the same order, every iteration. Packing them costs
   * nothing - a lone vec3f already occupies sixteen bytes.
   */
  deltas: { storage: d.arrayOf(d.vec4f), access: 'mutable' },
  cellStart: { storage: d.arrayOf(d.u32), access: 'readonly' },
  sortedIndex: { storage: d.arrayOf(d.u32), access: 'readonly' },
  /** Newest measured surface. Only the obstacle-speed term reads this. */
  surface: { texture: d.texture2d(d.f32) },
  /** The surface as of the previous depth update, for obstacle motion. */
  surfacePrev: { texture: d.texture2d(d.f32) },
  /** Where the surface stands right now. Everything collides against this. */
  surfaceLive: { texture: d.texture2d(d.f32) },
  /** Which way is down, and how far the whole surface drifted this update. */
  scene: { storage: SceneState, access: 'readonly' },
  /** Particles in the scene, recounted each step. Drives the fill meter. */
  population: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
  fieldSampler: { sampler: 'filtering' },
});

/** Counting sort that rebuilds the neighbour grid each step. */
export const hashLayout = tgpu.bindGroupLayout({
  particles: { storage: d.arrayOf(Particle), access: 'readonly' },
  cellCount: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
  cellCursor: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
  cellStart: { storage: d.arrayOf(d.u32), access: 'mutable' },
  blockSums: { storage: d.arrayOf(d.u32), access: 'mutable' },
  sortedIndex: { storage: d.arrayOf(d.u32), access: 'mutable' },
});
