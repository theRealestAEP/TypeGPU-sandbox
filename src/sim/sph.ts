import { d, std } from 'typegpu';
import { GRID_X, GRID_Y, GRID_Z, CELL_SIZE } from './schemas.ts';

/** 3D poly6 smoothing kernel. `coefficient` is 315 / (64 pi radius^9). */
export const poly6Weight = (distanceSq: number, radius: number, coefficient: number) => {
  'use gpu';
  const offset = radius * radius - distanceSq;
  if (offset <= 0) {
    return d.f32(0);
  }
  return coefficient * offset * offset * offset;
};

/**
 * Gradient of the 3D spiky kernel with respect to the first particle.
 * `offset` is p_i - p_j and `coefficient` is -45 / (pi radius^6).
 */
export const spikyGradient = (offset: d.v3f, radius: number, coefficient: number) => {
  'use gpu';
  const distance = std.length(offset);
  if (distance <= 0.000001 || distance >= radius) {
    return d.vec3f();
  }
  const falloff = radius - distance;
  return offset * ((coefficient * falloff * falloff) / distance);
};

/** Grid coordinate of a position, unclamped. */
export const cellCoordOf = (position: d.v3f) => {
  'use gpu';
  return d.vec3i(std.floor(position / d.f32(CELL_SIZE)));
};

export const cellIndexOf = (coord: d.v3i) => {
  'use gpu';
  return (d.u32(coord.z) * d.u32(GRID_Y) + d.u32(coord.y)) * d.u32(GRID_X) + d.u32(coord.x);
};

export const clampCell = (coord: d.v3i) => {
  'use gpu';
  return std.clamp(coord, d.vec3i(0), d.vec3i(GRID_X - 1, GRID_Y - 1, GRID_Z - 1));
};

export const cellInBounds = (coord: d.v3i) => {
  'use gpu';
  return (
    coord.x >= 0 &&
    coord.y >= 0 &&
    coord.z >= 0 &&
    coord.x < GRID_X &&
    coord.y < GRID_Y &&
    coord.z < GRID_Z
  );
};
