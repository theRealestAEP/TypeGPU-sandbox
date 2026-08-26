import { d, std, tgpu } from 'typegpu';
import { simLayout } from './bindings.ts';
import { cellCoordOf, cellInBounds, cellIndexOf, clampCell } from './sph.ts';
import {
  BODY_COUNT,
  BODY_PARTICLES,
  BODY_START,
  BODY_STRIDE,
  PARTICLE_COUNT,
  Particle,
  REST_SPACING,
  WORKGROUP_SIZE,
  Z_MAX,
} from './schemas.ts';

/**
 * Floating objects, as rigid clusters of particles.
 *
 * The shapes are signed distance functions sampled on the CPU at load. Nothing
 * is downloaded and nothing is a mesh, which suits this renderer: the compositor
 * is screen-space and the solver is particle-based, so a model would have to be
 * voxelised into exactly this anyway. Sampling the field directly skips that and
 * gives an honest volume, which is the number buoyancy is actually made of.
 */

const smin = (a: number, b: number, k: number) => {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
};

const sphere = (x: number, y: number, z: number, r: number) =>
  Math.hypot(x, y, z) - r;

/** Exact enough for sampling: the usual scaled-length approximation. */
const ellipsoid = (
  x: number,
  y: number,
  z: number,
  rx: number,
  ry: number,
  rz: number,
) => {
  const k0 = Math.hypot(x / rx, y / ry, z / rz);
  const k1 = Math.hypot(x / (rx * rx), y / (ry * ry), z / (rz * rz));
  return k1 === 0 ? -Math.min(rx, ry, rz) : (k0 * (k0 - 1)) / k1;
};

const roundBox = (
  x: number,
  y: number,
  z: number,
  bx: number,
  by: number,
  bz: number,
  r: number,
) => {
  const dx = Math.abs(x) - bx;
  const dy = Math.abs(y) - by;
  const dz = Math.abs(z) - bz;
  const outside = Math.hypot(Math.max(dx, 0), Math.max(dy, 0), Math.max(dz, 0));
  return outside + Math.min(Math.max(dx, Math.max(dy, dz)), 0) - r;
};

export interface Floater {
  readonly id: string;
  readonly label: string;
  /** Signed distance in a local frame with +y up, roughly unit scale. */
  readonly sdf: (x: number, y: number, z: number) => number;
  /** Half-extent of the local frame, for the sampling lattice. */
  readonly reach: number;
  /** Relative to the liquid. Under 1 floats, and the number is how deep it sits. */
  readonly density: number;
  readonly tint: readonly [number, number, number];
}

/**
 * A duck: body, head smooth-unioned on so the join reads as a neck, a beak, and
 * a tail. Cork density, so it rides high and bobs when the pour hits it.
 */
const duck: Floater = {
  id: 'duck',
  label: 'Rubber duck',
  reach: 1.6,
  density: 0.32,
  tint: [1.0, 0.78, 0.13],
  sdf(x, y, z) {
    const body = ellipsoid(x, y, z, 1.0, 0.74, 0.82);
    const head = sphere(x - 0.72, y - 0.86, z, 0.5);
    const beak = ellipsoid(x - 1.24, y - 0.78, z, 0.34, 0.15, 0.19);
    const tail = ellipsoid(x + 0.98, y - 0.36, z, 0.42, 0.3, 0.26);
    return smin(smin(smin(body, head, 0.45), tail, 0.25), beak, 0.06);
  },
};

const ball: Floater = {
  id: 'ball',
  label: 'Beach ball',
  reach: 1.2,
  density: 0.18,
  tint: [0.94, 0.32, 0.36],
  sdf: (x, y, z) => sphere(x, y, z, 1),
};

const soap: Floater = {
  id: 'soap',
  label: 'Bar of soap',
  reach: 1.3,
  density: 0.82,
  tint: [0.72, 0.86, 0.78],
  sdf: (x, y, z) => roundBox(x, y, z, 0.72, 0.3, 0.44, 0.22),
};

const boat: Floater = {
  id: 'boat',
  label: 'Toy boat',
  reach: 1.5,
  density: 0.28,
  tint: [0.36, 0.62, 0.9],
  sdf(x, y, z) {
    // A hull is the bottom half of an ellipsoid with the top sliced off, which
    // is also why it sits the way up it does without being told to.
    const hull = ellipsoid(x, y, z, 1.0, 0.62, 0.5);
    const cut = -y - 0.05;
    const cabin = roundBox(x + 0.18, y - 0.42, z, 0.26, 0.24, 0.24, 0.08);
    return smin(Math.max(hull, cut), cabin, 0.12);
  },
};

export const FLOATERS: readonly Floater[] = [duck, ball, boat, soap];

/**
 * Fills a floater with points at roughly the liquid's own spacing, then trims to
 * exactly the cluster size. Matching the spacing matters: the density constraint
 * is what stops water entering the object, and it can only do that if the
 * object's particles sit as close together as the water's do.
 */
/**
 * Fills an object with particles at exactly the liquid's own spacing.
 *
 * The spacing is the part that cannot be negotiated. The density constraint is
 * the only reason water does not pass through a duck, and it can only do that
 * if the duck's particles sit as far apart as the water's do. Pack them tighter
 * and the object reads as a pressurised lump that shoves liquid away from it;
 * pack them looser and it leaks.
 *
 * So the size follows from the budget rather than the other way round: measure
 * what fraction of its box the form occupies, then solve for the size at which
 * that many rest-spaced points lands on the cluster size. An earlier version
 * fixed the size and shrank the lattice until it had enough points, which for a
 * small object meant sampling many times finer than the liquid, and for a short
 * one meant padding the cluster with duplicate points sitting exactly on top of
 * each other - an infinite density, right inside the solver.
 */
export function sampleFloater(floater: Floater) {
  const probe = 40;
  let inside = 0;
  for (let i = 0; i < probe; i++) {
    for (let j = 0; j < probe; j++) {
      for (let k = 0; k < probe; k++) {
        const x = ((i + 0.5) / probe) * 2 * floater.reach - floater.reach;
        const y = ((j + 0.5) / probe) * 2 * floater.reach - floater.reach;
        const z = ((k + 0.5) / probe) * 2 * floater.reach - floater.reach;
        if (floater.sdf(x, y, z) < 0) inside++;
      }
    }
  }
  const fill = Math.max(inside / probe ** 3, 1e-4);
  const size = REST_SPACING * Math.cbrt(BODY_PARTICLES / fill);

  const scale = size / (2 * floater.reach);
  const step = REST_SPACING / scale;
  const points: number[][] = [];
  for (let x = -floater.reach; x <= floater.reach; x += step) {
    for (let y = -floater.reach; y <= floater.reach; y += step) {
      for (let z = -floater.reach; z <= floater.reach; z += step) {
        if (floater.sdf(x, y, z) < 0) points.push([x, y, z]);
      }
    }
  }

  // Outermost first, so any trimming thins the core rather than the skin that
  // has to hold the water out.
  const distance = (p: number[]) => floater.sdf(p[0], p[1], p[2]);
  points.sort((a, b) => distance(b) - distance(a));

  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (const p of points) {
    cx += p[0];
    cy += p[1];
    cz += p[2];
  }
  cx /= Math.max(points.length, 1);
  cy /= Math.max(points.length, 1);
  cz /= Math.max(points.length, 1);

  // Any shortfall is spread out on a lattice rather than duplicated on top of a
  // real particle, which would read as infinite density.
  const out = new Float32Array(BODY_PARTICLES * 4);
  for (let i = 0; i < BODY_PARTICLES; i++) {
    if (i < points.length) {
      out[i * 4] = (points[i][0] - cx) * scale;
      out[i * 4 + 1] = (points[i][1] - cy) * scale;
      out[i * 4 + 2] = (points[i][2] - cz) * scale;
    } else {
      out[i * 4] = (i % 8) * REST_SPACING;
      out[i * 4 + 1] = Math.floor(i / 8) * REST_SPACING;
    }
  }
  return { points: out, size };
}

// --- floater matching -------------------------------------------------------

const quatMul = (a: d.v4f, b: d.v4f) => {
  'use gpu';
  return d.vec4f(
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  );
};

const quatRotate = (q: d.v4f, v: d.v3f) => {
  'use gpu';
  const t = std.cross(q.xyz, v) * 2;
  return v + t * q.w + std.cross(q.xyz, t);
};

const centreSums = tgpu.workgroupVar(d.arrayOf(d.vec3f, WORKGROUP_SIZE));
const wetSums = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));
const covA = tgpu.workgroupVar(d.arrayOf(d.vec3f, WORKGROUP_SIZE));
const covB = tgpu.workgroupVar(d.arrayOf(d.vec3f, WORKGROUP_SIZE));
const covC = tgpu.workgroupVar(d.arrayOf(d.vec3f, WORKGROUP_SIZE));
const bodyCentre = tgpu.workgroupVar(d.vec3f);
const bodySpin = tgpu.workgroupVar(d.vec4f);

/** Newton steps for the rotation extraction. It converges fast from last frame's answer. */
const SPIN_STEPS = 12;

/**
 * Pulls each object's cluster back into a rigid arrangement.
 *
 * The solver has just treated those particles as liquid, so they have drifted
 * out of floater. Shape matching finds the single rotation and translation that
 * best explains where they ended up, then puts them exactly there. What the
 * liquid pushed on survives as motion of the whole object; what it did to the
 * object's floater does not survive at all, which is the definition of rigid.
 *
 * The rotation comes from Muller's iterative extraction rather than a full
 * polar decomposition: it is a handful of quaternion steps, it starts from last
 * frame's answer so it usually converges in two or three, and it cannot flip to
 * a reflection the way an SVD-free 3x3 solve can.
 */
export const rigidKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP_SIZE],
  in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationIndex },
})(({ wid, lid }) => {
  'use gpu';
  const body = wid.x;
  const params = simLayout.$.params;
  // Read once, act on it at the end. A `return` here would sit in front of the
  // barriers below, and a barrier that not every thread reaches is undefined.
  // WGSL rejects it outright, because `live` comes from a read-write buffer the
  // compiler cannot prove holds the same value across the workgroup.
  const live = simLayout.$.bodies[body].live !== 0;

  // Pass one: where the cluster is now, and how much of it is in liquid.
  let sum = d.vec3f();
  let wet = d.f32(0);
  for (const step of std.range(BODY_STRIDE)) {
    const slot = lid * BODY_STRIDE + step;
    if (slot < BODY_PARTICLES) {
      const index = BODY_START + body * BODY_PARTICLES + slot;
      const position = d.vec3f(simLayout.$.particles[index].pos);
      sum = sum + position;

      // Liquid within reach means this part of the object is submerged. Only
      // liquid counts - the object's own particles are always neighbours.
      let touching = d.f32(0);
      const home = clampCell(cellCoordOf(position));
      for (const stepZ of std.range(-1, 2)) {
        for (const stepY of std.range(-1, 2)) {
          for (const stepX of std.range(-1, 2)) {
            const coord = home + d.vec3i(stepX, stepY, stepZ);
            if (cellInBounds(coord)) {
              const cell = cellIndexOf(coord);
              const end = simLayout.$.cellStart[cell + 1];
              for (let s = simLayout.$.cellStart[cell]; s < end; s++) {
                const other = simLayout.$.sortedIndex[s];
                if (other < PARTICLE_COUNT) {
                  const offset = position - simLayout.$.particles[other].pos;
                  if (std.dot(offset, offset) < params.kernelRadius * params.kernelRadius) {
                    touching += 1;
                  }
                }
              }
            }
          }
        }
      }
      // Against a full neighbourhood, not a token one. A particle sitting in
      // liquid at rest density sees roughly forty-five of them, so counting six
      // as "submerged" made buoyancy bang-bang: full lift the instant an object
      // grazed the surface, full weight the instant it left.
      wet += std.saturate(touching / 28);
    }
  }
  centreSums.$[lid] = d.vec3f(sum);
  wetSums.$[lid] = wet;
  std.workgroupBarrier();

  if (lid === 0) {
    let total = d.vec3f();
    let wetTotal = d.f32(0);
    for (const slot of std.range(WORKGROUP_SIZE)) {
      total = total + centreSums.$[slot];
      wetTotal += wetSums.$[slot];
    }
    // Keep objects in the picture. A body that drifts past the drain line reads
    // as dormant to every other stage and freezes there for good, which is a
    // silent way to lose a duck.
    const loose = total / d.f32(BODY_PARTICLES);
    const centre = d.vec3f(
      std.clamp(loose.x, 0.04, 0.96),
      std.clamp(loose.y, 0.04, 0.94),
      std.clamp(loose.z, 0.05, d.f32(Z_MAX) - 0.05),
    );
    bodyCentre.$ = d.vec3f(centre);
    simLayout.$.bodies[body].centre = d.vec3f(centre);
    simLayout.$.bodies[body].wet = wetTotal / d.f32(BODY_PARTICLES);
  }
  std.workgroupBarrier();

  // Pass two: how the cluster is oriented, as the covariance of where each
  // particle is against where it belongs.
  const centre = d.vec3f(bodyCentre.$);
  let a = d.vec3f();
  let b = d.vec3f();
  let c = d.vec3f();
  for (const step of std.range(BODY_STRIDE)) {
    const slot = lid * BODY_STRIDE + step;
    if (slot < BODY_PARTICLES) {
      const index = BODY_START + body * BODY_PARTICLES + slot;
      const offset = simLayout.$.particles[index].pos - centre;
      const rest = simLayout.$.bodyRest[body * BODY_PARTICLES + slot].xyz;
      a = a + offset * rest.x;
      b = b + offset * rest.y;
      c = c + offset * rest.z;
    }
  }
  covA.$[lid] = d.vec3f(a);
  covB.$[lid] = d.vec3f(b);
  covC.$[lid] = d.vec3f(c);
  std.workgroupBarrier();

  if (lid === 0) {
    let a0 = d.vec3f();
    let a1 = d.vec3f();
    let a2 = d.vec3f();
    for (const slot of std.range(WORKGROUP_SIZE)) {
      a0 = a0 + covA.$[slot];
      a1 = a1 + covB.$[slot];
      a2 = a2 + covC.$[slot];
    }

    let spin = d.vec4f(simLayout.$.bodies[body].spin);
    if (std.dot(spin, spin) < 0.5) {
      spin = d.vec4f(0, 0, 0, 1);
    }
    for (const _step of std.range(SPIN_STEPS)) {
      const r0 = quatRotate(spin, d.vec3f(1, 0, 0));
      const r1 = quatRotate(spin, d.vec3f(0, 1, 0));
      const r2 = quatRotate(spin, d.vec3f(0, 0, 1));
      const denominator =
        std.abs(std.dot(r0, a0) + std.dot(r1, a1) + std.dot(r2, a2)) + 1e-9;
      const omega =
        (std.cross(r0, a0) + std.cross(r1, a1) + std.cross(r2, a2)) / denominator;
      const angle = std.length(omega);
      if (angle > 1e-9) {
        const axis = omega / angle;
        const half = angle * 0.5;
        spin = std.normalize(quatMul(d.vec4f(axis * std.sin(half), std.cos(half)), spin));
      }
    }
    simLayout.$.bodies[body].spin = d.vec4f(spin);
    bodySpin.$ = d.vec4f(spin);
  }
  std.workgroupBarrier();

  // Pass three: put every particle exactly where the rigid arrangement says.
  // No barrier past this point, so an object may be skipped here safely.
  if (!live) {
    return;
  }
  const spin = d.vec4f(bodySpin.$);
  for (const step of std.range(BODY_STRIDE)) {
    const slot = lid * BODY_STRIDE + step;
    if (slot < BODY_PARTICLES) {
      const index = BODY_START + body * BODY_PARTICLES + slot;
      const rest = simLayout.$.bodyRest[body * BODY_PARTICLES + slot].xyz;
      const particle = simLayout.$.particles[index];
      simLayout.$.particles[index] = Particle({
        pos: centre + quatRotate(spin, rest),
        prev: d.vec3f(particle.prev),
        vel: d.vec3f(particle.vel),
      });
    }
  }
});

export const BODY_WORKGROUPS = BODY_COUNT;
