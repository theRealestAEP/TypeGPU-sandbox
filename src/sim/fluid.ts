import { d, std, tgpu } from 'typegpu';
import type { StorageFlag, TgpuBuffer, TgpuComputePass, TgpuRoot, TgpuSampler } from 'typegpu';
import type { SingleChannelTexture } from '../gpu/blur.ts';
import { simLayout } from './bindings.ts';
import { createHashGrid } from './hash-grid.ts';
import {
  cellCoordOf,
  cellInBounds,
  cellIndexOf,
  clampCell,
  poly6Weight,
  spikyGradient,
} from './sph.ts';
import {
  FIELD_RES,
  KERNEL_RADIUS,
  MAX_DEPTH_SCALE,
  PARTICLE_COUNT,
  PARTICLE_WORKGROUPS,
  Particle,
  ParticleArray,
  REST_DENSITY,
  REST_SPACING,
  SOLVER_ITERATIONS,
  SceneState,
  SimParams,
  WORKGROUP_SIZE,
  Z_MAX,
  poly6,
  poly6Coefficient,
  spikyCoefficient,
} from './schemas.ts';
import type { ParticleBuffer } from './schemas.ts';

/** Keeps particles a hair inside the side walls and the front plane. */
const EDGE_MARGIN = 0.004;

/** Below the frame by this much counts as drained. */
const DRAIN_LINE = 1.1;

/**
 * Where drained particles wait. Anything past DRAIN_LINE is dormant: it is left
 * out of the neighbour grid and every solver stage returns immediately for it,
 * so the pool costs one idle thread and nothing else.
 */
const DORMANT_Y = 8;

/** Rain always enters at the top of the frame, wherever the spout happens to be. */
const RAIN_TOP = 0.02;


/**
 * Nothing starts in frame. The old block of water dropped into the scene at load
 * was placed at fixed coordinates with no idea what was behind it, so part of it
 * spawned inside the scene's own geometry and stuck there. At 80,000 particles
 * that read as a stray blob; at half a million it is a sheet of glass hanging
 * across the top of the picture, which is what "the whole scene looks submerged"
 * was. Water now only ever enters through the spout, in open air, in front of
 * whatever the camera can see.
 */
export const START_FRACTION = 0;

/** Floats per particle: three vec3f, each padded to 16 bytes. */
const FLOATS_PER_PARTICLE = 12;

const isDormant = (index: number) => {
  'use gpu';
  return simLayout.$.particles[index].pos.y > DRAIN_LINE;
};

const randomUnit = (state: number) => {
  'use gpu';
  let bits = d.u32(state) * 747796405 + 2891336453;
  bits = ((bits >>> ((bits >>> 28) + 4)) ^ bits) * 277803737;
  return d.f32((bits >>> 22) ^ bits) / 4294967295;
};

/**
 * Depth of the visible scene surface at an image position, in world z.
 * `previous` reads the surface as it was at the last depth update, which is how
 * the surface's own motion is measured: a cat's paw descending between updates
 * is a velocity the water has to be shoved with, not just a new wall.
 */
const surfaceNowAt = (image: d.v2f) => {
  'use gpu';
  return (
    std.textureSampleLevel(simLayout.$.surface, simLayout.$.fieldSampler, image, 0).x *
    simLayout.$.params.depthScale
  );
};

const surfacePrevAt = (image: d.v2f) => {
  'use gpu';
  return (
    std.textureSampleLevel(simLayout.$.surfacePrev, simLayout.$.fieldSampler, image, 0).x *
    simLayout.$.params.depthScale
  );
};

/**
 * Where the collision surface is at this instant.
 *
 * The two measured surfaces are one depth interval apart; the solver takes
 * several steps inside that interval, so it walks between them rather than
 * standing on the newest one. Without this the wall teleports the whole of an
 * obstacle's motion on the frame the depth lands, and the push-out - which runs
 * at solver rate, not video rate - clears that penetration about four times
 * faster than the obstacle created it. Every cat that moved threw the pool.
 */
const surfaceAt = (image: d.v2f) => {
  'use gpu';
  return (
    std.textureSampleLevel(simLayout.$.surfaceLive, simLayout.$.fieldSampler, image, 0).x *
    simLayout.$.params.depthScale
  );
};

const surfaceSlope = (image: d.v2f) => {
  'use gpu';
  const step = d.f32(1.5 / FIELD_RES);
  return d.vec2f(
    surfaceAt(image + d.vec2f(step, 0)) - surfaceAt(image - d.vec2f(step, 0)),
    surfaceAt(image + d.vec2f(0, step)) - surfaceAt(image - d.vec2f(0, step)),
  ) / (2 * step);
};

/**
 * The entire collision model. A depth map shows only the front-most surface, so
 * anything behind it is inside something and anything in front of it is open
 * air. A particle that ends up behind gets pushed back out along the surface
 * normal, by a capped amount: depth estimation flickers, and an uncapped push
 * would fling particles across the scene.
 */
const surfaceNormal = (image: d.v2f) => {
  'use gpu';
  return std.normalize(d.vec3f(surfaceSlope(image) * -1, 1));
};

const resolveSurface = (from: d.v3f, position: d.v3f) => {
  'use gpu';
  const params = simLayout.$.params;
  let resolved = d.vec3f(position);

  // Depth cliffs are walls, and this is where they get enforced. Wherever the
  // scene stands a wall's height in front of where the particle came from - a
  // pool's near rim, a kerb, anything seen edge-on - crossing it in image space
  // is refused outright by undoing the image-space part of the move.
  //
  // Without this the deep-burial rescue further down does the opposite of what
  // a wall should: it lifts the particle to the top of the rim and sets it
  // down on the far side, one step per contact, and the pool leaks its whole
  // contents over the brim. A particle already inside something is left alone -
  // it is being pressed in by a pour or a paw, not walking through.
  if (
    position.z - surfaceAt(position.xy) < -params.surfaceShell &&
    from.z - surfaceAt(from.xy) > -params.surfaceShell
  ) {
    resolved = d.vec3f(from.xy, position.z);
  }

  const surface = surfaceAt(resolved.xy);
  const gap = resolved.z - surface;

  // Distant background is still a camera-facing surface, so without the play
  // volume test every far wall behaves as floor: rain lands on the fence behind
  // a pool and stacks up there instead of reaching the water. Beyond the play
  // volume, nothing collides and the drop falls through and drains.
  const playZ = params.rainReach * params.depthScale;

  /**
   * One-way, always out the front. The old two-sided escape let particles that
   * were shoved deeper than one shell - by the pour's impact, by compression at
   * the base of the pool, by a paw sweeping over them - pop out behind the
   * scene and vanish from view while the fill meter kept counting them. That
   * leak was most of why the pool never filled and water piled outside it. A
   * particle that far behind is instead parked just in front of the surface
   * until the solver's own pressure walks it out.
   */
  // The scene is a slab of finite thickness, not a wall that goes on forever.
  // Past the back of it is open air behind the object, which is a real place
  // liquid can be: cover the spout with your hand and the water belongs behind
  // your hand, hidden by it, not on top of it.
  //
  // This is what `surfaceShell` has always claimed to mean, and the rescue
  // below was doing the exact opposite - it snapped anything deeper than the
  // shell to the *front*, so nothing could ever be behind anything. Being past
  // the slab only counts if the particle was already past it when the step
  // started; arriving there within one step means it was pressed through, by a
  // pour's impact or a paw, and that still gets parked out front.
  const buried = -gap;
  const behind = buried > params.surfaceShell;
  const stayedBehind = behind && from.z - surfaceAt(from.xy) < -params.surfaceShell;

  if (surface > playZ && gap < 0 && !stayedBehind) {
    const normal = surfaceNormal(resolved.xy);
    // Push out by exactly how far in the particle is, and no further.
    //
    // De-penetration is already the whole of how a moving obstacle drives
    // liquid: a paw coming down puts water inside the surface by however far it
    // advanced, and shoving it back out by that much is the push. It is also
    // self-limiting - once the particle is clear, the gap is positive and this
    // does nothing.
    //
    // Adding the surface's own advance on top of it is not. That advance spans
    // one depth update, but this runs once per solver iteration - three times a
    // step, 120 steps a second, against roughly 30 depth updates a second - so
    // a paw's motion was applied about twelve times over, and every cat that
    // moved threw the whole pool. Measured on the clip: 14% of the body above
    // 0.3 units/s and a 99th percentile at the velocity cap, against 0.3% on a
    // still photo. The obstacle's speed still caps how fast water may leave the
    // contact, in finalizeKernel; it just no longer injects anything.
    const escape = std.min(buried, params.pushLimit);
    resolved = resolved + normal * escape;

    // Buried far deeper than one shell - under a pour's impact point or a paw -
    // a capped escape crawls out over many frames, dragging neighbours with it
    // and churning everything nearby. Snap to just in front of the surface in
    // one move instead: invisible next to the impact that buried it, and the
    // solver stops paying for the crawl.
    if (behind) {
      resolved = d.vec3f(resolved.xy, surface + params.kernelRadius * 0.5);
    }
  }

  // No level-plane dam here: the heightfield is the floor and the basin's own
  // depth is what gathers water, so the pool level is emergent.

  // Box walls stay as a backstop so nothing escapes the neighbour grid, and the
  // top edge needs one as much as the sides: a particle that leaves the frame
  // there bins into a clamped edge cell and turns the neighbour search
  // quadratic. The bottom edge is the one place that must stay open. A wall
  // there parks every drop that misses the basin in a single edge cell just out
  // of view - the same quadratic cost, and the particles are never seen again
  // either. Past DRAIN_LINE they drain instead, and the spout gets them back.
  return d.vec3f(
    std.clamp(resolved.x, EDGE_MARGIN, 1 - EDGE_MARGIN),
    std.max(resolved.y, EDGE_MARGIN),
    std.clamp(resolved.z, EDGE_MARGIN, Z_MAX - EDGE_MARGIN),
  );
};

const predictKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP_SIZE],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= PARTICLE_COUNT) {
    return;
  }
  const params = simLayout.$.params;
  let position = d.vec3f(simLayout.$.particles[gid.x].pos);
  let velocity = d.vec3f(simLayout.$.particles[gid.x].vel);


  // At capacity, retire the deepest settled water so the spout never runs dry.
  // Those drops sit at the bottom of the pool under everything else, so removing
  // them is invisible, and the emit window rate-limits it to exactly the inflow
  // - the level holds while the water circulates.
  if (params.recycle !== 0) {
    const ticket = (gid.x + params.frame * params.emitRate) % PARTICLE_COUNT;
    if (
      ticket < params.emitRate &&
      position.z < params.recycleBand &&
      std.length(velocity) < 0.06
    ) {
      position = d.vec3f(position.x, DORMANT_Y + 1, position.z);
    }
  }

  // Past the back of the play volume counts as gone, same as off the bottom.
  if (position.z < params.rainReach * params.depthScale * 0.5) {
    position = d.vec3f(position.x, DORMANT_Y + 1, position.z);
  }

  if (position.y > DRAIN_LINE) {
    // Drained. A rotating window of indices is eligible each step, which is
    // what rate-limits the spout; without it every drained particle returns at
    // once and the pour piles up on itself.
    const ticket = (gid.x + params.frame * params.emitRate) % PARTICLE_COUNT;
    if (ticket >= params.emitRate) {
      const parked = d.vec3f(params.emitter.x, DORMANT_Y, params.emitter.z);
      simLayout.$.particles[gid.x] = Particle({
        pos: d.vec3f(parked),
        prev: d.vec3f(parked),
        vel: d.vec3f(),
      });
      return;
    }
    const across = randomUnit(gid.x * 3 + params.frame) - 0.5;
    const along = randomUnit(gid.x * 3 + 1 + params.frame);
    const through = randomUnit(gid.x * 3 + 2 + params.frame) - 0.5;

    // A storm fills most of the window from the sky, but never all of it: what
    // is left is the spout, so the tap still works in the rain.
    if (params.storm !== 0 && d.f32(ticket) >= d.f32(params.emitRate) * params.spoutShare) {
      // Rain: spread across the whole frame and falling along measured gravity,
      // rather than a stream from one spout.
      position = d.vec3f(
        0.05 + (across + 0.5) * 0.9,
        RAIN_TOP,
        0.05 + (through + 0.5) * (Z_MAX - 0.1),
      );
      // Gusts act on the drops as they enter. Pushing every particle sideways
      // would just tilt the pool surface and hold it tilted; rain is what the
      // wind actually gets hold of.
      velocity = d.vec3f(params.wind, 0, -params.emitSpeed + params.wind * 0.35);
    } else {
      position =
        params.emitter +
        d.vec3f(across, along, through) * params.emitSpread;
      velocity = d.vec3f(across * 3 * params.emitSpread, params.emitSpeed, through * 3 * params.emitSpread);
    }

    // The spout pours from the depth it was aimed at, full stop. It used to be
    // shoved in front of whatever covered that texel, so putting a hand over
    // the source made the water land on top of the hand. That was a patch over
    // the collision treating everything behind a surface as inside it; with the
    // slab finite (see resolveSurface) the water simply falls behind the hand
    // and the compositor hides it there, which is what the eye expects.
  }

  // Down as the scene measures it. Gravity along -z alone reads as the safe
  // choice - a depth map is a heightfield, and -z is the axis it is a function
  // of - but it is not, because the contact push runs along the surface normal
  // and the image-space part of that push then has nothing to balance it. On a
  // photo where depth grows toward the bottom of the frame it leaves a standing
  // pull up the image, and the whole body creeps off the top edge. Measured
  // gravity is what the normal force exists to balance; the wall test in
  // resolveSurface is what stops the tangential part running the pool over the
  // near rim.
  velocity = velocity + simLayout.$.scene.down * (params.gravity * params.dt);

  // Never step further than a kernel radius, or neighbours are found too late.
  const reach = std.length(velocity) * params.dt;
  const limit = params.kernelRadius * 0.9;
  if (reach > limit) {
    velocity = velocity * (limit / reach);
  }

  simLayout.$.particles[gid.x] = Particle({
    pos: position + velocity * params.dt,
    prev: d.vec3f(position),
    vel: d.vec3f(velocity),
  });
});

const densityKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP_SIZE],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= PARTICLE_COUNT || isDormant(gid.x)) {
    return;
  }
  const params = simLayout.$.params;
  const position = d.vec3f(simLayout.$.particles[gid.x].pos);
  const radius = params.kernelRadius;

  let density = d.f32(0);
  let gradientSum = d.vec3f();
  let gradientEnergy = d.f32(0);

  const home = clampCell(cellCoordOf(position));
  for (const stepZ of std.range(-1, 2)) {
    for (const stepY of std.range(-1, 2)) {
      for (const stepX of std.range(-1, 2)) {
        const coord = home + d.vec3i(stepX, stepY, stepZ);
        if (cellInBounds(coord)) {
          const cell = cellIndexOf(coord);
          const end = simLayout.$.cellStart[cell + 1];
          for (let slot = simLayout.$.cellStart[cell]; slot < end; slot++) {
            const other = simLayout.$.sortedIndex[slot];
            const offset = position - simLayout.$.particles[other].pos;
            const distanceSq = std.dot(offset, offset);
            if (distanceSq < radius * radius) {
              density += poly6Weight(distanceSq, radius, params.poly6);
              const gradient = spikyGradient(offset, radius, params.spiky) / params.restDensity;
              gradientSum = gradientSum + gradient;
              gradientEnergy += std.dot(gradient, gradient);
            }
          }
        }
      }
    }
  }

  // Only resist compression. A particle at a free surface always measures a
  // density deficit, and letting that pull produces the tensile instability:
  // the surface sucks inward, overshoots, and the whole body heats up into a gas.
  const constraint = std.max(density / params.restDensity - 1, 0);
  const stiffness = gradientEnergy + std.dot(gradientSum, gradientSum) + params.relaxation;
  simLayout.$.deltas[gid.x] = d.vec4f(
    simLayout.$.deltas[gid.x].xyz,
    -constraint / stiffness,
  );
});

const deltaKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP_SIZE],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= PARTICLE_COUNT || isDormant(gid.x)) {
    return;
  }
  const params = simLayout.$.params;
  const position = d.vec3f(simLayout.$.particles[gid.x].pos);
  const lambda = simLayout.$.deltas[gid.x].w;
  const radius = params.kernelRadius;

  let correction = d.vec3f();
  const home = clampCell(cellCoordOf(position));
  for (const stepZ of std.range(-1, 2)) {
    for (const stepY of std.range(-1, 2)) {
      for (const stepX of std.range(-1, 2)) {
        const coord = home + d.vec3i(stepX, stepY, stepZ);
        if (cellInBounds(coord)) {
          const cell = cellIndexOf(coord);
          const end = simLayout.$.cellStart[cell + 1];
          for (let slot = simLayout.$.cellStart[cell]; slot < end; slot++) {
            const other = simLayout.$.sortedIndex[slot];
            const offset = position - simLayout.$.particles[other].pos;
            const distanceSq = std.dot(offset, offset);
            if (distanceSq < radius * radius) {
              // Artificial pressure: a small constant repulsion that stops the
              // clumping a pure density constraint produces, and reads as
              // surface tension.
              const weight = poly6Weight(distanceSq, radius, params.poly6);
              const repulsion = -params.cohesion * std.pow(weight / params.cohesionRef, 4);
              const scale = lambda + simLayout.$.deltas[other].w + repulsion;
              correction = correction + spikyGradient(offset, radius, params.spiky) * scale;
            }
          }
        }
      }
    }
  }

  simLayout.$.deltas[gid.x] = d.vec4f(
    correction / params.restDensity,
    simLayout.$.deltas[gid.x].w,
  );
});

const applyKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP_SIZE],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= PARTICLE_COUNT || isDormant(gid.x)) {
    return;
  }
  const particle = simLayout.$.particles[gid.x];
  simLayout.$.particles[gid.x] = Particle({
    // Tested against where the step started, not against the last solver
    // iteration. The ballistic step in `predict` is not surface-resolved, so a
    // fast drop can already be through a rim by the time the solver first sees
    // it; `prev` is the last position known to be on the legal side of every
    // wall.
    pos: resolveSurface(particle.prev, particle.pos + simLayout.$.deltas[gid.x].xyz),
    prev: d.vec3f(particle.prev),
    vel: d.vec3f(particle.vel),
  });
});

/** Zeroes the census before the step that refills it. One thread is enough. */
const censusResetKernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
  std.atomicStore(simLayout.$.population[0], 0);
});

const finalizeKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP_SIZE],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= PARTICLE_COUNT || isDormant(gid.x)) {
    return;
  }
  // Every stage above this one already skips dormant particles, so reaching
  // here is the definition of "in the scene". Counting it costs one atomic on
  // a thread that is running anyway.
  std.atomicAdd(simLayout.$.population[0], 1);
  const params = simLayout.$.params;
  const particle = simLayout.$.particles[gid.x];
  let velocity = (particle.pos - particle.prev) / params.dt;

  // A large solver correction reads back as a large velocity. Cap it at the same
  // step the predictor allows, so one bad frame cannot inject energy that takes
  // seconds to dissipate.
  const speed = std.length(velocity);
  const limit = (params.kernelRadius * 0.9) / params.dt;
  if (speed > limit) {
    velocity = velocity * (limit / speed);
  }

  // Contact. The band has to be thin: reaching a full kernel radius in front of
  // the surface kept every resting layer permanently "in contact", so the whole
  // pool was friction-damped every step and the surface normal scrambled
  // velocities that had nothing to do with the wall - much of the random
  // twitching. Only drops actually about to touch are treated as contact.
  //
  // Tested where the step started, not where it ended. The push-out clears
  // penetration at solver rate - three iterations, 120 steps a second - while
  // the surface only advances at video rate, so a shoved particle can be well
  // clear of a band this thin by the time finalize looks. Judging on the end
  // position let exactly the particles that were shoved hardest skip the very
  // clamp meant to hold them, and they kept the full push as speed. That is
  // what made a moving cat throw the pool.
  const contact = particle.prev.z - surfaceAt(particle.prev.xy);
  const gap = particle.pos.z - surfaceAt(particle.pos.xy);
  if (
    std.min(contact, gap) < params.kernelRadius * 0.35 &&
    std.max(contact, gap) > -params.surfaceShell
  ) {
    const normal = surfaceNormal(particle.pos.xy);
    // How fast the surface here is advancing, phantom motion already removed.
    const sweepRate =
      std.min(
        std.max(
          surfaceNowAt(particle.pos.xy) - surfacePrevAt(particle.pos.xy) - simLayout.$.scene.drift,
          0,
        ) / std.max(params.obstacleDt, 0.001),
        (params.kernelRadius * 0.9) / params.dt,
      );
    const outward = std.dot(velocity, normal);
    // Water may leave a contact no faster than the surface itself is advancing,
    // plus a small allowance for genuine splash. The push-out acts positionally
    // along this same normal, so every solver correction lands next step as
    // outward velocity; uncapped, that loop compounds frame on frame and the
    // pool boils. Removing both the into-surface press and the excess above the
    // obstacle's own speed keeps the shove without the pump.
    const allowed = sweepRate + params.kernelRadius * 2;
    const removed = std.min(outward, 0) + std.max(outward - allowed, 0);
    velocity = velocity - normal * removed;

    // Friction only belongs on something liquid could actually rest on. A
    // surface whose normal opposes gravity is a floor and holds water still; one
    // that stands square to gravity is a wall and cannot hold it up at all.
    // Damping both the same glued water to the tiles, so instead of running back
    // down into the tub it plastered up the wall and stayed there.
    const floorness = std.saturate(std.dot(normal, simLayout.$.scene.down * -1));
    velocity = velocity * std.mix(1, params.surfaceFriction, floorness);
  }

  simLayout.$.particles[gid.x] = Particle({
    pos: d.vec3f(particle.pos),
    prev: d.vec3f(particle.prev),
    vel: d.vec3f(velocity),
  });
});

const viscosityKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP_SIZE],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= PARTICLE_COUNT || isDormant(gid.x)) {
    return;
  }
  const params = simLayout.$.params;
  const position = d.vec3f(simLayout.$.particles[gid.x].pos);
  const velocity = d.vec3f(simLayout.$.particles[gid.x].vel);
  const radius = params.kernelRadius;

  let smoothed = d.vec3f();
  const home = clampCell(cellCoordOf(position));
  for (const stepZ of std.range(-1, 2)) {
    for (const stepY of std.range(-1, 2)) {
      for (const stepX of std.range(-1, 2)) {
        const coord = home + d.vec3i(stepX, stepY, stepZ);
        if (cellInBounds(coord)) {
          const cell = cellIndexOf(coord);
          const end = simLayout.$.cellStart[cell + 1];
          for (let slot = simLayout.$.cellStart[cell]; slot < end; slot++) {
            const other = simLayout.$.sortedIndex[slot];
            const offset = position - simLayout.$.particles[other].pos;
            const distanceSq = std.dot(offset, offset);
            if (distanceSq < radius * radius) {
              const difference = simLayout.$.particles[other].vel - velocity;
              smoothed = smoothed + difference * poly6Weight(distanceSq, radius, params.poly6);
            }
          }
        }
      }
    }
  }

  // The multiplier has done its job by now, so this pass owns the whole slot.
  simLayout.$.deltas[gid.x] = d.vec4f(
    smoothed * (params.viscosity / params.restDensity),
    0,
  );
});

const relaxKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP_SIZE],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= PARTICLE_COUNT || isDormant(gid.x)) {
    return;
  }
  const particle = simLayout.$.particles[gid.x];
  simLayout.$.particles[gid.x] = Particle({
    pos: d.vec3f(particle.pos),
    prev: d.vec3f(particle.prev),
    vel: particle.vel + simLayout.$.deltas[gid.x].xyz,
  });
});

/**
 * Fixed solver step. PBF is tuned around a constant dt; the caller decides how
 * many.
 *
 * Half a display frame rather than a whole one, because the step length is what
 * sets how much a settled pool simmers. The pressure solve never fully cancels
 * gravity in the iterations it is given, so each step leaves a little of the
 * velocity gravity just added, and the leftover is what churns. Measured: a
 * pool left alone for 150 s held a median speed of 1.7x gravity's per-step
 * velocity and never decayed further, and it scaled with gravity exactly -
 * turning gravity off dropped it to nothing.
 *
 * Shortening the step beats solving harder, and by a wide margin. At matched
 * cost, half the step is 2.2x calmer than twice the iterations; half the step
 * with three iterations is calmer than a full step with sixteen, at half the
 * price. Substeps buy convergence more cheaply than iterations do, which is the
 * usual result for position-based solvers.
 */
const FIXED_DT = 1 / 120;

export interface FluidTuning {
  /** Magnitude only. The direction is measured from the scene each depth update. */
  gravity: number;
  viscosity: number;
  cohesion: number;
  surfaceFriction: number;
  relaxation: number;
  depthScale: number;
  /** How thick scene objects are treated as being, so liquid can pass behind. */
  surfaceShell: number;
  /** Rain across the whole frame instead of a single spout. */
  storm: boolean;
  /** Sideways gust on falling rain. */
  wind: number;
  /** Share of the emit window the spout keeps while a storm runs. */
  spoutShare: number;
  /** How near the scene must be for rain to fall there at all. */
  rainReach: number;
  /** Recycle settled water so the pour has no ceiling. */
  recycle: boolean;
  emitSpeed: number;
  emitSpread: number;
  emitRate: number;
  emitterX: number;
  emitterY: number;
  emitterZ: number;
}

export const defaultTuning: FluidTuning = {
  gravity: 1.6,
  viscosity: 0.05,
  cohesion: 1.5e-5,
  surfaceFriction: 0.96,
  relaxation: 100,
  depthScale: MAX_DEPTH_SCALE,
  surfaceShell: KERNEL_RADIUS * 6,
  storm: false,
  wind: 0,
  spoutShare: 0,
  rainReach: 0,
  recycle: false,
  emitSpeed: 0.5,
  emitSpread: 0.025,
  emitRate: 18,
  emitterX: 0.5,
  emitterY: 0.06,
  emitterZ: Z_MAX * 0.92,
};

export interface Fluid {
  readonly particles: ParticleBuffer;
  readonly grid: ReturnType<typeof createHashGrid>;
  /** Particles currently in the scene, as of the last encoded step. */
  readonly population: TgpuBuffer<d.WgslArray<d.Atomic<d.U32>>> & StorageFlag;
  initAsync(): Promise<void>;
  encode(pass: TgpuComputePass): void;
  tune(next: Partial<FluidTuning>): void;
  reset(): void;
  /** Send every particle back to the dormant pool, emptying the scene. */
  drain(): void;
  destroy(): void;
}

/**
 * A relaxed block of liquid up near the top of the frame, so the first thing you
 * see is a body falling rather than an empty screen. Starting it inside the
 * volume matters: strays outside would all bin into the same edge cells and turn
 * the neighbour search quadratic.
 */
function initialParticles(
  emitter: readonly [number, number, number],
  startFraction: number,
): ArrayBuffer {
  const active = Math.round(PARTICLE_COUNT * startFraction);
  const columns = Math.max(1, Math.floor(0.4 / REST_SPACING));
  const layers = Math.max(1, Math.floor((Z_MAX * 0.35) / REST_SPACING));
  const perSlab = columns * layers;

  const bytes = new ArrayBuffer(PARTICLE_COUNT * FLOATS_PER_PARTICLE * 4);
  const data = new Float32Array(bytes);

  for (let index = 0; index < PARTICLE_COUNT; index++) {
    const base = index * FLOATS_PER_PARTICLE;
    let x = emitter[0];
    let y = DORMANT_Y;
    let z = emitter[2];

    if (index < active) {
      const withinSlab = index % perSlab;
      x = 0.3 + (withinSlab % columns) * REST_SPACING;
      z = Z_MAX * 0.42 + Math.floor(withinSlab / columns) * REST_SPACING;
      y = 0.02 + Math.floor(index / perSlab) * REST_SPACING;
    }

    data[base] = x;
    data[base + 1] = y;
    data[base + 2] = z;
    data[base + 4] = x;
    data[base + 5] = y;
    data[base + 6] = z;
  }
  return bytes;
}

export interface FluidInputs {
  readonly surface: SingleChannelTexture;
  readonly surfacePrev: SingleChannelTexture;
  readonly surfaceLive: SingleChannelTexture;
  readonly scene: TgpuBuffer<typeof SceneState> & StorageFlag;
  readonly fieldSampler: TgpuSampler;
  /** Seconds since the previous depth update, for obstacle motion. */
  obstacleDt: () => number;
}

export function createFluid(root: TgpuRoot, inputs: FluidInputs): Fluid {
  let tuning: FluidTuning = { ...defaultTuning };
  let frame = 0;
  let lastObstacleDt = 1 / 30;

  const particles = root.createBuffer(ParticleArray).$usage('storage', 'vertex');
  const deltas = root.createBuffer(d.arrayOf(d.vec4f, PARTICLE_COUNT)).$usage('storage');
  const params = root.createBuffer(SimParams).$usage('uniform');
  const population = root.createBuffer(d.arrayOf(d.atomic(d.u32), 1)).$usage('storage');
  const grid = createHashGrid(root, particles);

  const bindGroup = root.createBindGroup(simLayout, {
    params,
    particles,
    deltas,
    cellStart: grid.cellStart,
    sortedIndex: grid.sortedIndex,
    population,
    surface: inputs.surface.createView(),
    surfacePrev: inputs.surfacePrev.createView(),
    surfaceLive: inputs.surfaceLive.createView(),
    scene: inputs.scene,
    fieldSampler: inputs.fieldSampler,
  });

  const censusReset = root.createComputePipeline({ compute: censusResetKernel });
  const predict = root.createComputePipeline({ compute: predictKernel });
  const density = root.createComputePipeline({ compute: densityKernel });
  const delta = root.createComputePipeline({ compute: deltaKernel });
  const apply = root.createComputePipeline({ compute: applyKernel });
  const finalize = root.createComputePipeline({ compute: finalizeKernel });
  const viscosity = root.createComputePipeline({ compute: viscosityKernel });
  const relax = root.createComputePipeline({ compute: relaxKernel });

  const stages = [
    censusReset,
    predict,
    density,
    delta,
    apply,
    finalize,
    viscosity,
    relax,
  ];

  function emitterPosition(): [number, number, number] {
    return [tuning.emitterX, tuning.emitterY, tuning.emitterZ];
  }

  function writeParams(): void {
    params.write({
      gravity: tuning.gravity,
      emitter: emitterPosition(),
      dt: FIXED_DT,
      obstacleDt: lastObstacleDt,
      kernelRadius: KERNEL_RADIUS,
      restDensity: REST_DENSITY,
      relaxation: tuning.relaxation,
      cohesion: tuning.cohesion,
      viscosity: tuning.viscosity,
      emitSpeed: tuning.emitSpeed,
      emitSpread: tuning.emitSpread,
      surfaceFriction: tuning.surfaceFriction,
      pushLimit: KERNEL_RADIUS * 0.3,
      surfaceShell: tuning.surfaceShell,
      depthScale: tuning.depthScale,
      poly6: poly6Coefficient(KERNEL_RADIUS),
      spiky: spikyCoefficient(KERNEL_RADIUS),
      cohesionRef: poly6(KERNEL_RADIUS, 0.2 * KERNEL_RADIUS),
      emitRate: Math.max(0, Math.round(tuning.emitRate)),
      storm: tuning.storm ? 1 : 0,
      wind: tuning.wind,
      spoutShare: tuning.spoutShare,
      rainReach: tuning.rainReach,
      recycle: tuning.recycle ? 1 : 0,
      recycleBand: KERNEL_RADIUS * 5,
      frame,
    });
  }

  particles.write(initialParticles(emitterPosition(), START_FRACTION));
  writeParams();

  return {
    particles,
    grid,
    population,

    async initAsync() {
      await Promise.all([grid.initAsync(), ...stages.map((stage) => stage.initAsync())]);
    },

    encode(pass) {
      frame = (frame + 1) >>> 0;
      lastObstacleDt = Math.min(Math.max(inputs.obstacleDt(), 1 / 240), 1 / 10);
      writeParams();

      censusReset.with(pass).with(bindGroup).dispatchWorkgroups(1);
      predict.with(pass).with(bindGroup).dispatchWorkgroups(PARTICLE_WORKGROUPS);
      grid.encode(pass);

      for (let iteration = 0; iteration < SOLVER_ITERATIONS; iteration++) {
        density.with(pass).with(bindGroup).dispatchWorkgroups(PARTICLE_WORKGROUPS);
        delta.with(pass).with(bindGroup).dispatchWorkgroups(PARTICLE_WORKGROUPS);
        apply.with(pass).with(bindGroup).dispatchWorkgroups(PARTICLE_WORKGROUPS);
      }

      finalize.with(pass).with(bindGroup).dispatchWorkgroups(PARTICLE_WORKGROUPS);
      viscosity.with(pass).with(bindGroup).dispatchWorkgroups(PARTICLE_WORKGROUPS);
      relax.with(pass).with(bindGroup).dispatchWorkgroups(PARTICLE_WORKGROUPS);
    },

    tune(next) {
      tuning = { ...tuning, ...next };
      writeParams();
    },

    reset() {
      particles.write(initialParticles(emitterPosition(), START_FRACTION));
    },

    drain() {
      particles.write(initialParticles(emitterPosition(), 0));
    },

    destroy() {
      grid.destroy();
      particles.destroy();
      deltas.destroy();
      params.destroy();
      population.destroy();
    },
  };
}
