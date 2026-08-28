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
  SOLVER_ITERATIONS,
  SceneState,
  SimParams,
  TOPOLOGY_SNAP,
  WORKGROUP_SIZE,
  Z_MAX,
  poly6,
  poly6Coefficient,
  spikyCoefficient,
} from './schemas.ts';
import { BASE, WALL } from '../depth/carve.ts';
import { VESSEL_RES } from '../depth/vessels.ts';
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

const surfaceBackAt = (image: d.v2f) => {
  'use gpu';
  return (
    std.textureSampleLevel(simLayout.$.surfaceBack, simLayout.$.fieldSampler, image, 0).x *
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
/**
 * The slab thickness in WORLD units. surfaceShell is authored in raw scene
 * units, and using it unscaled broke at low depth scale: at 0.14 the shell
 * exceeded the entire world's depth, every surface was "behind" every other,
 * and cups dissolved completely - zero retention, measured. The scene's own
 * depth scale is part of what "thick" means, exactly as TOPOLOGY_SNAP already
 * acknowledged.
 */
const shellNow = () => {
  'use gpu';
  return simLayout.$.params.surfaceShell * simLayout.$.params.depthScale;
};

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

const resolveSurface = (index: number, from: d.v3f, position: d.v3f) => {
  'use gpu';
  const params = simLayout.$.params;
  let resolved = d.vec3f(position);

  // The mouth catches what falls over it. Aiming exactly at the opening is a
  // probe's precision, not a person's: the natural spout sits a little above
  // the glass, and from there the stream lands on the rim texels and runs
  // down the outside - 4 particles held from a pour 12% above the mouth,
  // against thousands from one exactly at it. So any water moving down the
  // image through a cup's mouth band is pulled onto the cavity plane, the
  // way a real opening swallows a stream. The move is a relocation, not a
  // kick: it goes through `boundary`, so finalize strips it from velocity.
  let channel = false;
  for (const slot of std.range(3)) {
    const cup = params.cups[slot];
    if (cup.z <= cup.x) {
      continue;
    }
    const cupWidth = cup.z - cup.x;
    const inSpan =
      position.x > cup.x + cupWidth * d.f32(WALL) &&
      position.x < cup.z - cupWidth * d.f32(WALL);
    // The pour channel: everything in a cup's column above or at its mouth
    // falls free of the scene. A close-held glass stands against a blank
    // wall, and a blank wall's depth is a hallucinated ledge - the stream
    // landed on it and mushroomed in mid-air, metres above the cup it was
    // aimed at. Inside the channel the scene does not exist; the mouth
    // capture below is what catches the water.
    if (inSpan && position.y < cup.y + (cup.w - cup.y) * 0.25) {
      channel = true;
    }
    if (
      inSpan &&
      position.y > from.y &&
      position.y > cup.y &&
      position.y < cup.y + (cup.w - cup.y) * 0.25 &&
      // Only where the mouth truly is: the texel must read as the cavity the
      // carve dug - front minus carve, tightly. A merely-deep texel is the
      // wall behind the glass, and capturing over it hung the stream on an
      // invisible plane in mid-air.
      std.abs(
        surfaceAt(resolved.xy) -
          (params.cupFronts[slot] - params.cupCarves[slot]) * params.depthScale,
      ) < 0.12 * params.depthScale
    ) {
      const plane =
        (params.cupFronts[slot] - params.cupCarves[slot]) * params.depthScale +
        params.kernelRadius * 4;
      // Both sides of the plane on purpose. The spout above a glass lands its
      // stream on whatever the scene holds there - often the far wall, well
      // BEHIND the cavity - and from there the water passed the glass and
      // drained without ever being seen: three quarters of the pour, gone.
      if (std.abs(resolved.z - plane) > params.kernelRadius * 2) {
        const snapped = d.vec3f(resolved.xy, plane);
        simLayout.$.boundary[index] = simLayout.$.boundary[index] + (snapped - resolved);
        resolved = d.vec3f(snapped);
      }
    }

    // The glass has a bottom. The base band is uncarved, and at level pitch
    // the weir prices a down-image exit across it at nothing, so the pool
    // crept out under the glass - the leak was steady and visible. Water
    // inside the cavity may not cross the base line; water running down the
    // OUTSIDE front face is nearer than the wall plane and passes untouched.
    const cupWall = params.cupFronts[slot] * params.depthScale - params.kernelRadius * 0.5;
    const baseLine = cup.w - (cup.w - cup.y) * d.f32(BASE);
    // The floor spans the WHOLE box, wall margins included. Water in the
    // margin strips is inside the glass wall's own footprint, and letting it
    // fall through there was the remaining bottom leak - the pool held in
    // the middle while streams slid down just inside the box edges and out
    // over the hand.
    if (
      position.x > cup.x &&
      position.x < cup.z &&
      resolved.z < cupWall &&
      from.y < baseLine &&
      resolved.y >= baseLine
    ) {
      resolved = d.vec3f(resolved.x, baseLine - 0.001, resolved.z);
    }

    // And sides. The carved ramps were the lateral walls, but their tapered
    // perimeter always holds a band shallower than the collision shell, and
    // the pool bled out through that ring - slowly at the default shell,
    // totally with the slider up. A cup's containment must not depend on the
    // shell at all: one-way clamps at the interior span's edges finish the
    // box. Mouth, base, front, sides - each judged on where the water came
    // from, so outside water is never trapped in.
    const spanL = cup.x + cupWidth * d.f32(WALL);
    const spanR = cup.z - cupWidth * d.f32(WALL);
    if (
      resolved.z < cupWall &&
      resolved.y > cup.y &&
      resolved.y < baseLine &&
      from.x > spanL &&
      from.x < spanR
    ) {
      if (resolved.x <= spanL) {
        resolved = d.vec3f(spanL + 0.001, resolved.yz);
      } else if (resolved.x >= spanR) {
        resolved = d.vec3f(spanR - 0.001, resolved.yz);
      }
    }
  }

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
  if (channel) {
    return d.vec3f(
      std.clamp(resolved.x, EDGE_MARGIN, 1 - EDGE_MARGIN),
      std.max(resolved.y, EDGE_MARGIN),
      std.clamp(resolved.z, EDGE_MARGIN, Z_MAX - EDGE_MARGIN),
    );
  }

  // Water ENTERING a cup's mouth is exempt from the wall rule. The rim texels
  // read as a barrier, and the weir's crest-landing put a stream arriving
  // along the cavity plane ON TOP of the rim - one hop flipped a whole pour
  // from inside the glass to running down its outside. The exemption is
  // strictly one-way: only particles moving down the image, into the glass.
  // Water pressed UP toward the mouth by the pool still meets the weir, which
  // refuses the uphill crossing - that refusal is what keeps the glass shut.
  let atMouth = false;
  if (position.y > from.y) {
    for (const slot of std.range(3)) {
      const cup = params.cups[slot];
      if (cup.z <= cup.x) {
        continue;
      }
      const width = cup.z - cup.x;
      const inSpanHere =
        position.x > cup.x + width * d.f32(WALL) &&
        position.x < cup.z - width * d.f32(WALL);
      if (
        inSpanHere &&
        position.y > cup.y - 0.05 &&
        position.y < cup.y + (cup.w - cup.y) * 0.25
      ) {
        atMouth = true;
      }
      // The same stand-down at the base. The weir crest-landed the pool's
      // deep back edge onto the glass's OUTER front - teleported past the
      // wall plane, where the box clamps no longer own it - and the cup bled
      // from the bottom at exactly the rate the pour filled it. Inside water
      // meeting the base is the y-clamp's job, not the weir's.
      const wall = params.cupFronts[slot] * params.depthScale - params.kernelRadius * 0.5;
      if (
        inSpanHere &&
        from.z < wall &&
        position.y > cup.w - (cup.w - cup.y) * d.f32(BASE) - 0.05
      ) {
        atMouth = true;
      }
    }
  }
  const wallSurf = surfaceAt(resolved.xy);
  if (
    !atMouth &&
    resolved.z - wallSurf < -shellNow() &&
    from.z - surfaceAt(from.xy) > -shellNow()
  ) {
    // The weir rule. A flat refusal here dammed the front rim forever: crossing
    // it demands z-clearance, and with gravity nearly level in z that clearance
    // costs twenty-odd times the true potential barrier - so a brim-full tub
    // could never overflow toward the camera, and the only open exit was up
    // the back wall (water there rides in FRONT of a deeper surface and is
    // never refused). Real water passes a barrier when its level reaches the
    // crest, so that is the test: elevation along measured gravity, not
    // clearance in z. Over the crest, the particle lands ON the barrier and
    // cascades down the far side, which is what overflowing looks like.
    // The crest test must be about where the particle would LAND, not the
    // texel it is stepping onto. Tested against the flank, every wall becomes a
    // staircase: each hop lands on top of the flank at raised z, which at a
    // steep pitch RAISES the particle's elevation and funds the next hop -
    // water walked up the sink's rim one texel at a time and the whole pool
    // drained over the counter. Spilling means going downhill: the landing may
    // never sit meaningfully above where the particle already was.
    const downNow = simLayout.$.scene.down;
    const landZ = wallSurf + params.kernelRadius * 1.5;
    const landing = -(downNow.x * position.x + downNow.y * position.y + downNow.z * landZ);
    const mine = -std.dot(downNow, from);
    if (landing <= mine + 0.005) {
      resolved = d.vec3f(position.xy, landZ);
    } else {
      resolved = d.vec3f(from.xy, position.z);
    }
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
  const behind = buried > shellNow();
  // A texel a transient occluder covers reads far in front of the remembered
  // scene. The water beneath never walked into anything - the surface flipped
  // in front of it when the occluder arrived - so the occluder is visual-only
  // for it: it belongs to the space behind the live surface and collides
  // against the remembered background instead. The margin on the from test
  // matters at the top: water genuinely caught ON an occluder - the pour
  // landing on a hand - jitters a little either side of the surface as it
  // rests, and without the two-radii allowance every dip routed it to the
  // background and it dripped through the hand. Water without an occluder
  // over it belongs behind only if it was already past the slab when the step
  // started; arriving there within one step means it was pressed through, by
  // a pour's impact, and still gets parked out front.
  const occluded = surface - surfaceBackAt(resolved.xy) > shellNow();
  const belongsBehind =
    (occluded && from.z - surface < -params.kernelRadius * 2) ||
    (behind && from.z - surfaceAt(from.xy) < -shellNow());

  if (surface > playZ && gap < 0 && !belongsBehind) {
    const normal = surfaceNormal(resolved.xy);
    // Push out by exactly how far in the particle is, and no further.
    //
    // De-penetration is already the whole of how a moving obstacle drives
    // liquid: a paw coming down puts water inside the surface by however far it
    // advanced, and shoving it back out by that much is the push. It is also
    // self-limiting - once the particle is clear, the gap is positive and this
    // does nothing. The push normally DOES read back as velocity in finalize:
    // that is how resting water tracks a surface that jitters under it.
    // Subtracting it everywhere was tried and made settled pools simmer
    // harder - lifted with no velocity, every downward flicker of the floor
    // became a small free fall and a slap.
    //
    // The exception is a texel whose depth jumped past the topology threshold
    // this update: that texel changed owners - a hand arrived in front of the
    // sink - and the walk sweeping the wall between the two owners is not
    // motion water should inherit. Those pushes are recorded in `boundary`
    // and finalize subtracts them, so a passing occluder shoves nothing.
    const escape = std.min(buried, params.pushLimit);
    const pushed = normal * escape;
    resolved = resolved + pushed;
    const jump = std.abs(surfaceNowAt(resolved.xy) - surfacePrevAt(resolved.xy));
    if (jump > TOPOLOGY_SNAP * params.depthScale) {
      simLayout.$.boundary[index] = simLayout.$.boundary[index] + pushed;
    }

    // Buried far deeper than one shell - under a pour's impact point or a paw -
    // a capped escape crawls out over many frames, dragging neighbours with it
    // and churning everything nearby. Snap to just in front of the surface in
    // one move instead: invisible next to the impact that buried it, and the
    // solver stops paying for the crawl.
    //
    // The snap is a teleport, and a teleport must not become kinetic energy:
    // it is recorded in `boundary` and finalize subtracts it before deriving
    // velocity, so the rescue relocates the particle without throwing it.
    if (behind) {
      const snapped = d.vec3f(resolved.xy, surface + params.kernelRadius * 0.5);
      simLayout.$.boundary[index] = simLayout.$.boundary[index] + (snapped - resolved);
      resolved = d.vec3f(snapped);
    }
  }

  // Behind a passing occluder is not open air. A hand crossing the sink
  // REPLACES the sink in the heightfield, so water under it lost its floor and
  // fell through - the whole pool scattering whenever a hand passed over. The
  // background layer remembers what the occluder covered, and water behind the
  // live surface supports against the scene that is still really there.
  if (belongsBehind) {
    const back = surfaceBackAt(resolved.xy);
    const backGap = resolved.z - back;
    // The background is a finite slab exactly like the live surface. Without
    // the lower bound it was an infinite wall: a person sitting still is
    // adopted as scenery within seconds, and water deliberately spouted
    // behind them - deeper than the whole slab - was walked forward step by
    // step until it surfaced on their face. Deeper than one shell behind the
    // remembered scene is open air: the water falls freely, hidden by what
    // stands in front of it, until something at its own depth catches it.
    if (back > playZ && backGap < 0 && backGap > -shellNow()) {
      resolved = d.vec3f(
        resolved.xy,
        resolved.z + std.min(-backGap, params.pushLimit),
      );
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
  // A fresh step owes no boundary repair yet.
  simLayout.$.boundary[gid.x] = d.vec3f();
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

    // The pour lands on the world under the spout. Left at its raw depth, a
    // near-camera spout rains a sheet in front of everything in the frame - on
    // a webcam that reads as water thrown at the viewer, and on the tub it was
    // the pour missing the basin entirely. Spawning just in front of whatever
    // the scene holds at that spot pours ONTO things: a basin fills, a person
    // gets rained on, a hand held over the spout catches the stream. Scrolling
    // the spout deeper still works - the clamp only stops spawning nearer than
    // the scene, never deeper into it.
    // Except over a cup's pour channel. There the scene under the cursor is
    // whatever depth the model hallucinated for the wall around the glass,
    // and snapping the stream onto it started every drop buried in the halo.
    // Channel water spawns at its raw near-camera depth, falls in clear air
    // in front of everything, and the mouth capture takes it from there.
    let overCup = false;
    for (const slot of std.range(3)) {
      const cup = params.cups[slot];
      if (cup.z <= cup.x) {
        continue;
      }
      const cupWidth = cup.z - cup.x;
      if (
        position.x > cup.x + cupWidth * d.f32(WALL) &&
        position.x < cup.z - cupWidth * d.f32(WALL) &&
        position.y < cup.y + (cup.w - cup.y) * 0.25
      ) {
        overCup = true;
      }
    }
    const under = surfaceAt(position.xy) + params.kernelRadius * 4;
    if (!overCup && position.z > under) {
      position = d.vec3f(position.xy, under);
    }

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
  const calmHere = std.saturate(simLayout.$.deltas[gid.x].w);
  velocity = velocity +
    simLayout.$.scene.down *
      (params.gravity * params.dt * (1 - CALM_GRAVITY_RELIEF * calmHere));

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
    pos: resolveSurface(gid.x, particle.prev, particle.pos + simLayout.$.deltas[gid.x].xyz),
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
  let position = d.vec3f(particle.pos);

  // The glass's near wall, restored. The carve digs a cavity by pushing the
  // surface back, which also deletes the vessel's transparent front face - and
  // under a level camera gravity has no into-scene component, so pool pressure
  // squeezes settled water out toward the viewer, where it free-falls off the
  // glass. Measured: 70% of a full glass drained within four seconds of the
  // pour stopping. So across each cup's interior span - the same margins the
  // carve keeps - water that was behind the vessel's front plane stays behind
  // it: a one-way wall, judged on the previous position so water already in
  // front of the glass is left alone. The span is geometric, not read from the
  // carved surface: the bowl's perimeter ramps taper to nothing, and gating
  // the wall on carved depth left that whole ring open - the pool's rim
  // squeezed out through it and the glass drained anyway.
  for (const slot of std.range(3)) {
    const cup = params.cups[slot];
    if (cup.z <= cup.x) {
      continue;
    }
    const width = cup.z - cup.x;
    const height = cup.w - cup.y;
    const wall = params.cupFronts[slot] * params.depthScale - params.kernelRadius * 0.5;
    if (
      position.x > cup.x + width * d.f32(WALL) &&
      position.x < cup.z - width * d.f32(WALL) &&
      position.y > cup.y &&
      position.y < cup.w - height * d.f32(BASE) &&
      position.z > wall &&
      particle.prev.z < wall
    ) {
      position = d.vec3f(position.xy, wall);
    }
  }

  // `boundary` holds the step's zero-impulse repairs - deep-burial snaps, and
  // push-outs at texels whose depth jumped past the topology threshold.
  // Subtracting them here means those relocations never read back as kinetic
  // energy. Ordinary push-outs on stable texels are NOT in the buffer - their
  // read-back is how resting water tracks a surface that jitters under it.
  let velocity = (position - particle.prev - simLayout.$.boundary[gid.x]) / params.dt;

  // A large solver correction reads back as a large velocity. Cap it at the same
  // step the predictor allows, so one bad frame cannot inject energy that takes
  // seconds to dissipate.
  const speed = std.length(velocity);
  const limit = (params.kernelRadius * 0.9) / params.dt;
  if (speed > limit) {
    velocity = velocity * (limit / speed);
  }

  // A glass is a calm place. A pour that free-falls in arrives as a
  // firehose, shoots down the interior, overruns the floor ramp and exits
  // over the base - at low inflow no pool ever forms, and an above-the-glass
  // pour held 119 particles against thousands from a gentle one. Capping
  // speed across the whole interior is the rim shattering the stream and the
  // vessel confining the churn: pools form at any inflow, and standing water
  // in a glass reads as still, which it should.
  let channel = false;
  for (const slot of std.range(3)) {
    const cup = params.cups[slot];
    if (cup.z <= cup.x) {
      continue;
    }
    const cupWidth = cup.z - cup.x;
    const cupHeight = cup.w - cup.y;
    const cupWall = params.cupFronts[slot] * params.depthScale - params.kernelRadius * 0.5;
    const inSpan =
      position.x > cup.x + cupWidth * d.f32(WALL) &&
      position.x < cup.z - cupWidth * d.f32(WALL);
    // The pour channel again: the same span resolveSurface exempts from the
    // scene must be exempt from the contact clamp here, or the falling
    // stream still decelerates against the hallucinated ledge and mushrooms.
    if (inSpan && position.y < cup.y + cupHeight * 0.25) {
      channel = true;
    }
    if (
      inSpan &&
      position.y < cup.w - cupHeight * d.f32(BASE) &&
      position.z < cupWall
    ) {
      // Applies through the pour channel too, not just below the rim: the
      // untamed stream splashed off the mouth and a third of it ran down the
      // OUTSIDE, which read as the glass leaking.
      // Directional on purpose. A scalar cap clogged the pour: the stream
      // fell in faster than the cap let it leave, the column jammed back up
      // to the spout, and spawn pressure blasted a radial firework there.
      // Falling WITH gravity stays free - the base line catches the overrun
      // now - while sideways and upward motion, the splash directions, stay
      // tamed.
      const downDir = simLayout.$.scene.down;
      const along = std.dot(velocity, downDir);
      const sideways = velocity - downDir * along;
      const drift = std.length(sideways);
      let tamed = d.vec3f(velocity);
      if (drift > 0.3) {
        tamed = downDir * along + sideways * (0.3 / drift);
      }
      if (along < -0.3) {
        tamed = tamed + downDir * (-0.3 - along);
      }
      velocity = d.vec3f(tamed);
    }
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
  const gap = position.z - surfaceAt(position.xy);
  if (
    !channel &&
    std.min(contact, gap) < params.kernelRadius * 0.35 &&
    std.max(contact, gap) > -shellNow()
  ) {
    const normal = surfaceNormal(position.xy);
    // How fast the surface here is advancing, phantom motion already removed.
    const advance = surfaceNowAt(position.xy) - surfacePrevAt(position.xy);
    let sweepRate = std.min(
      std.max(advance - simLayout.$.scene.drift, 0) / std.max(params.obstacleDt, 0.001),
      (params.kernelRadius * 0.9) / params.dt,
    );
    // A jump past the topology threshold is a texel changing owners - a hand
    // arriving in front of the sink - not a surface in motion. No obstacle
    // velocity may be derived from it.
    if (std.abs(advance) > TOPOLOGY_SNAP * params.depthScale) {
      sweepRate = 0;
    }
    const outward = std.dot(velocity, normal);
    // Water may leave a contact no faster than the surface itself is
    // advancing, plus a small allowance for genuine splash. The push-out acts
    // positionally along this same normal, so every solver correction lands
    // next step as outward velocity; uncapped, that loop compounds frame on
    // frame and the pool boils. Removing both the into-surface press and the
    // excess above the obstacle's own speed keeps the shove without the pump.
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
    pos: d.vec3f(position),
    prev: d.vec3f(particle.prev),
    vel: d.vec3f(velocity),
  });
});

/**
 * Below this neighbourhood speed, water is settling rather than flowing, and
 * the smoothing is allowed to climb toward CALM_VISCOSITY.
 */
const CALM_SPEED = 0.25;

/**
 * XSPH coefficient for still water. The free surface of a PBF pool never
 * settles on its own: a particle there measures only density deficit, which
 * the compression-only constraint ignores, so pressure from below flicks it
 * up, gravity drops it back, and it pops like that forever - the simmer that
 * no amount of render smoothing could hide. Keying the smoothing on the
 * neighbourhood's speed rather than the particle's own is what kills exactly
 * the popcorn: a popping particle sits among slow neighbours and gets dragged
 * hard to their mean, while a splash sits among fast ones and keeps its life.
 */
const CALM_VISCOSITY = 0.45;

/**
 * Fraction of velocity removed per step in fully calm water. Smoothing alone
 * halves the popcorn while the pool settles but cannot finish the job - it
 * drags particles toward a neighbourhood mean that is itself oscillating, so
 * the simmer converges to a floor instead of to zero. This term removes energy
 * outright, and only where the neighbourhood is already nearly still: a splash
 * scores calm = 0 and loses nothing.
 */
const SLEEP_DAMP = 0.2;

/**
 * How much of gravity calm water is spared. The simmer has a floor that no
 * damping can reach, because its source refills every substep: gravity adds
 * g*dt, the solver cancels it imperfectly, and the leftover is re-injected
 * fresh - the measured floor sits at about 1.5x g*dt regardless of how hard
 * the surroundings are damped. But water whose neighbourhood is still IS in
 * hydrostatic balance; that is what still means. So calm water takes only the
 * sliver of gravity the balance argument does not cover, and the injection
 * shrinks by the same factor as the floor. A disturbance raises neighbourhood
 * speed, calm collapses, and full gravity returns within a step.
 */
const CALM_GRAVITY_RELIEF = 0.9;

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
  let speedSum = d.f32(0);
  let weightSum = d.f32(0);
  let outward = d.vec3f();
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
              const weight = poly6Weight(distanceSq, radius, params.poly6);
              const difference = simLayout.$.particles[other].vel - velocity;
              smoothed = smoothed + difference * weight;
              speedSum += std.length(simLayout.$.particles[other].vel) * weight;
              weightSum += weight;
              outward = outward + offset * weight;
            }
          }
        }
      }
    }
  }

  // How fast this particle's surroundings are moving, not how fast it is. The
  // distinction is the whole trick - see CALM_VISCOSITY.
  const around = speedSum / std.max(weightSum, 1e-6);
  // Calm needs support as well as stillness. The hydrostatic argument only
  // holds for water carried by water at rest density; a sparse pile of beads is
  // still because it just landed, and sparing it gravity froze it mid-heap.
  // Sparse means little weight in the kernel, so it keeps falling and merges.
  const support = std.saturate(weightSum / (params.restDensity * 0.6));

  // And it needs the local free surface to be LEVEL. "Still means balanced" is
  // only true of a pool whose surface is square to gravity; a heap at rest is
  // out of equilibrium and needs its gravity to slump. Sparing heaps their
  // gravity froze them - the tub's inflow stacked into a standing pillar that
  // climbed the back wall instead of slumping forward. The neighbour-offset
  // sum points out of the liquid at a free surface, so its tilt against "up"
  // says whether this is pool or pile.
  const upNow = simLayout.$.scene.down * -1;
  let levelness = d.f32(1);
  const asymmetry = std.length(outward) / std.max(weightSum * params.kernelRadius, 1e-6);
  if (asymmetry > 0.05) {
    const o = outward / std.max(std.length(outward), 1e-9);
    const tilt = std.length(o - upNow * std.dot(o, upNow));
    levelness = 1 - std.saturate((tilt - 0.6) / 0.25);

    // The water's edge against a wall reads as a heap - its outward vector
    // points sideways - so it could NEVER calm, and the contact line simmered
    // forever: gravity pressed it at the rim, the crossing was refused, and
    // the cycle sustained its own wakefulness. A heap flank has open air
    // beside it; this water is RESTING on a floor. Resting contact on a
    // floor-like surface restores its right to sleep. A pillar against a
    // vertical wall gets nothing back - its footing has no floorness - and
    // still slumps.
    const seat = position.z - surfaceAt(position.xy);
    if (seat < params.kernelRadius) {
      const footing = std.saturate(std.dot(surfaceNormal(position.xy), upNow));
      levelness = std.max(levelness, footing * footing);
    }
  }
  // Blended two-thirds toward yesterday's answer. Without the memory, a
  // particle at the waterline flickered between calm and awake with every
  // wobble of its tilt estimate, and each waking re-armed gravity for a step -
  // the last visible simmer lived exactly there.
  // And it must be somewhere water can actually stand. The flood's spill
  // surface says, per column, the highest potential a pool can reach before
  // it finds an exit; above that line "calm" is a lie - the tub's mound
  // climbed the side walls as sleeping jello because nothing told it that
  // height was uncontainable. Water above spill level keeps its gravity and
  // flows away, however still its neighbourhood momentarily is.
  const texelX = d.u32(std.clamp(position.x * d.f32(VESSEL_RES), 0, d.f32(VESSEL_RES - 1)));
  const texelY = d.u32(std.clamp(position.y * d.f32(VESSEL_RES), 0, d.f32(VESSEL_RES - 1)));
  const cell = texelY * d.u32(VESSEL_RES) + texelX;
  const spillLevel = simLayout.$.spill[cell >> 2][cell & 3];
  const standing = -std.dot(simLayout.$.scene.down, position);
  const restable = 1 - std.saturate((standing - spillLevel) / 0.02);
  const instant = (1 - std.saturate(around / CALM_SPEED)) * support * levelness * restable;
  const calm = std.mix(instant, simLayout.$.calms[gid.x], 0.65);
  simLayout.$.calms[gid.x] = calm;
  const coefficient = params.viscosity + calm * (CALM_VISCOSITY - params.viscosity);

  // The multiplier has done its job by now, so this pass owns the whole slot.
  // The sleep term rides along: it is just a negative multiple of the
  // particle's own velocity, so the relax pass applies both in one add.
  // Calm rides in w so next substep's predict can read it - deltas are indexed
  // by particle id, which is stable across the grid's re-sorts.
  simLayout.$.deltas[gid.x] = d.vec4f(
    smoothed * (coefficient / params.restDensity) - velocity * (calm * calm * SLEEP_DAMP),
    calm,
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
  /** Detected vessels: field-space box, front depth, interior depth. */
  cups: readonly {
    box: readonly [number, number, number, number];
    front: number;
    carve: number;
  }[];
}

export const defaultTuning: FluidTuning = {
  gravity: 1.6,
  viscosity: 0.05,
  cohesion: 1.5e-5,
  surfaceFriction: 0.96,
  relaxation: 100,
  depthScale: MAX_DEPTH_SCALE,
  // Twelve radii, not six. Water jittering at the measured surface flips
  // between "in contact" and "buried, snap out the front" when the shell is
  // thin, and every snap is a teleport that finalize reads back as velocity -
  // observed directly: thickening the shell calms the simmer.
  surfaceShell: KERNEL_RADIUS * 12,
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
  cups: [],
};

export interface Fluid {
  readonly particles: ParticleBuffer;
  readonly grid: ReturnType<typeof createHashGrid>;
  /** Particles currently in the scene, as of the last encoded step. */
  readonly population: TgpuBuffer<d.WgslArray<d.Atomic<d.U32>>> & StorageFlag;
  initAsync(): Promise<void>;
  encode(pass: TgpuComputePass): void;
  tune(next: Partial<FluidTuning>): void;
  /** Uploads the flood's spill-level surface; see the calm gate that reads it. */
  setSpill(filled: Float32Array): void;
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
/**
 * Every particle parked in the dormant pool, so the scene opens empty.
 *
 * There used to be a block of liquid placed here at load. It was dropped in at
 * fixed coordinates with no idea what was behind it, so part of it spawned
 * inside the scene's own geometry and stuck there - at half a million particles
 * that is a sheet of glass hanging across the picture. Liquid now only ever
 * enters through the spout, in open air, in front of whatever the camera sees.
 */
function initialParticles(emitter: readonly [number, number, number]): ArrayBuffer {
  const bytes = new ArrayBuffer(PARTICLE_COUNT * FLOATS_PER_PARTICLE * 4);
  const data = new Float32Array(bytes);

  for (let index = 0; index < PARTICLE_COUNT; index++) {
    const base = index * FLOATS_PER_PARTICLE;
    const x = emitter[0];
    const y = DORMANT_Y;
    const z = emitter[2];

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
  readonly surfaceBack: SingleChannelTexture;
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
  const calms = root.createBuffer(d.arrayOf(d.f32, PARTICLE_COUNT)).$usage('storage');
  const boundary = root.createBuffer(d.arrayOf(d.vec3f, PARTICLE_COUNT)).$usage('storage');
  const grid = createHashGrid(root, particles);

  // Spill-level surface, uploaded from the CPU flood a few times a second.
  // Starts effectively infinite: until the first flood lands, every height is
  // "containable" and the calm system behaves exactly as before.
  const spill = root
    .createBuffer(d.arrayOf(d.vec4f, (VESSEL_RES * VESSEL_RES) / 4))
    .$usage('uniform');
  spill.write(new Float32Array(VESSEL_RES * VESSEL_RES).fill(1e6));

  const bindGroup = root.createBindGroup(simLayout, {
    params,
    particles,
    deltas,
    cellStart: grid.cellStart,
    sortedIndex: grid.sortedIndex,
    population,
    calms,
    boundary,
    surface: inputs.surface.createView(),
    surfacePrev: inputs.surfacePrev.createView(),
    surfaceLive: inputs.surfaceLive.createView(),
    surfaceBack: inputs.surfaceBack.createView(),
    scene: inputs.scene,
    spill,
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

  function cupCarve(cup: FluidTuning['cups'][number] | undefined): number {
    return cup ? cup.carve : 0;
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
      cups: [0, 1, 2].map((i): [number, number, number, number] => {
        const cup = tuning.cups[i];
        return cup ? [cup.box[0], cup.box[1], cup.box[2], cup.box[3]] : [0, 0, 0, 0];
      }),
      cupFronts: [
        tuning.cups[0]?.front ?? 0,
        tuning.cups[1]?.front ?? 0,
        tuning.cups[2]?.front ?? 0,
        0,
      ],
      // The same self-similarity rule the carver uses; the solver needs it to
      // know where a cup's interior plane sits.
      cupCarves: [
        cupCarve(tuning.cups[0]),
        cupCarve(tuning.cups[1]),
        cupCarve(tuning.cups[2]),
        0,
      ],
    });
  }

  particles.write(initialParticles(emitterPosition()));
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

    drain() {
      particles.write(initialParticles(emitterPosition()));
    },

    setSpill(filled) {
      spill.write(filled);
    },

    destroy() {
      grid.destroy();
      particles.destroy();
      deltas.destroy();
      params.destroy();
      population.destroy();
      calms.destroy();
      boundary.destroy();
      spill.destroy();
    },
  };
}
