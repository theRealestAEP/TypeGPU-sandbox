import { d, std, tgpu } from 'typegpu';
import type {
  StorageFlag,
  TgpuBuffer,
  TgpuComputePass,
  TgpuRoot,
  TgpuSampler,
} from 'typegpu';
import type { SingleChannelTexture } from '../gpu/blur.ts';
import {
  MAX_DEPTH_SCALE,
  SceneState,
  PRESSURE_ITERATIONS,
  CurlArray,
  PressureArray,
  SMOKE_CELL,
  SMOKE_RES,
  SMOKE_WORKGROUP,
  SMOKE_X,
  SMOKE_Y,
  SMOKE_Z,
  SmokeCell,
  SmokeCellArray,
  SmokeParams,
  Z_MAX,
} from './schemas.ts';

/**
 * Eulerian smoke on a fixed grid, sharing the scene the liquid collides with.
 *
 * Particles are the right tool for water because water has a surface worth
 * finding. Smoke has none: what you see is the light that survived a path
 * through it, which is an integral over a volume. So smoke gets a grid, and the
 * standard four moves - carry the field along its own velocity, add buoyancy,
 * work out the pressure that makes the flow divergence-free, subtract it.
 *
 * The scene enters exactly once, through the same rule the liquid uses: a cell
 * behind the visible depth surface is inside something. Those cells are solid,
 * so the pressure solve makes the plume climb a wall and roll around a mug
 * without anything ever being told that a wall or a mug is there.
 */

/** Steps the bake pass walks toward the light while measuring shadow. */
const SHADOW_STEPS = 8;
const SHADOW_STRIDE = SMOKE_CELL * 2;

/** Gust units to grid units. The storm hands over roughly +/-0.5. */
const WIND_SCALE = 0.35;

/**
 * Ceiling on cell density, so a parked source cannot run away. Set low: past the
 * point where a cell is already opaque, more density only widens the plateau of
 * fully dark smoke, and the plateau has a hard edge where the cap bites.
 */
const DENSITY_CAP = 2.5;

/**
 * Light that reaches smoke the sun cannot see, bounced off everything else.
 * Without it a thick plume goes to the shadow colour and reads as a hole.
 */
const AMBIENT = 0.18;
/**
 * Sky light from straight above, occluded by the plume itself. This is what
 * puts a dark underside under a lit top: the bake marches down as well as
 * toward the sun, and the mix of the two occlusions is the shading that makes
 * a plume read as a volume instead of a cutout.
 */
const SKY_STEPS = 6;
const SKY_STRIDE = SMOKE_CELL * 2.5;

/**
 * Ceiling on speed, in cells per fixed tick. Vorticity confinement returns
 * energy in proportion to the swirl it finds, so a fast eddy earns a bigger
 * push and gets faster still. Left alone that loop runs away, and the open
 * boundary gives it nothing to push against. The cap costs nothing at ordinary
 * speeds and also keeps the back-trace inside its own neighbourhood.
 */
const MAX_SPEED = SMOKE_CELL * 2 * 60;

const RENDER_WORKGROUP = 8;

export const smokeLayout = tgpu.bindGroupLayout({
  params: { uniform: SmokeParams },
  cellsIn: { storage: d.arrayOf(SmokeCell), access: 'readonly' },
  cellsOut: { storage: d.arrayOf(SmokeCell), access: 'mutable' },
  pressureIn: { storage: d.arrayOf(d.f32), access: 'readonly' },
  pressureOut: { storage: d.arrayOf(d.f32), access: 'mutable' },
  divergence: { storage: d.arrayOf(d.f32), access: 'mutable' },
  /** Curl of the advected velocity, for vorticity confinement. */
  curl: { storage: d.arrayOf(d.vec3f), access: 'mutable' },
  /** Unit "down" in camera axes, measured from the scene. Smoke climbs the reverse. */
  sceneState: { storage: SceneState, access: 'readonly' },
  surface: { texture: d.texture2d(d.f32) },
  fieldSampler: { sampler: 'filtering' },
});

const bakeLayout = tgpu.bindGroupLayout({
  params: { uniform: SmokeParams },
  cells: { storage: d.arrayOf(SmokeCell), access: 'readonly' },
  /** Unit "down" in camera axes; the sky march runs the reverse. */
  sceneState: { storage: SceneState, access: 'readonly' },
  volume: { storageTexture: d.textureStorage3d('rgba16float', 'write-only' as const) },
});

const renderLayout = tgpu.bindGroupLayout({
  params: { uniform: SmokeParams },
  volume: { texture: d.texture3d(d.f32) },
  scene: { texture: d.texture2d(d.f32) },
  /** Unit "down" in camera axes, for the height gradient in the march. */
  sceneState: { storage: SceneState, access: 'readonly' },
  volumeSampler: { sampler: 'filtering' },
  image: { storageTexture: d.textureStorage2d('rgba16float', 'write-only' as const) },
});

const cellIndexAt = (coord: d.v3i) => {
  'use gpu';
  const held = std.clamp(coord, d.vec3i(), d.vec3i(SMOKE_X - 1, SMOKE_Y - 1, SMOKE_Z - 1));
  return (d.u32(held.z) * d.u32(SMOKE_Y) + d.u32(held.y)) * d.u32(SMOKE_X) + d.u32(held.x);
};

/** Centre of a cell, in the same world box the liquid uses. */
const centreOf = (coord: d.v3u) => {
  'use gpu';
  return (d.vec3f(coord) + 0.5) * d.f32(SMOKE_CELL);
};

/** Outside the grid. Kept apart from solid: the two boundaries behave differently. */
const isOutside = (coord: d.v3i) => {
  'use gpu';
  return (
    coord.x < 0 ||
    coord.y < 0 ||
    coord.z < 0 ||
    coord.x >= SMOKE_X ||
    coord.y >= SMOKE_Y ||
    coord.z >= SMOKE_Z
  );
};

/**
 * The whole scene collision model, same as the liquid's: behind the visible
 * depth surface is inside something.
 */
const isSolid = (coord: d.v3i) => {
  'use gpu';
  if (isOutside(coord)) {
    return false;
  }
  const centre = centreOf(d.vec3u(coord));
  const surface =
    std.textureSampleLevel(smokeLayout.$.surface, smokeLayout.$.fieldSampler, centre.xy, 0).x *
    smokeLayout.$.params.depthScale;
  return centre.z < surface;
};

/** Trilinear read of the whole cell, for the semi-Lagrangian back-trace. */
const sampleCell = (position: d.v3f) => {
  'use gpu';
  const grid = position / d.f32(SMOKE_CELL) - 0.5;
  const base = std.floor(grid);
  const blend = grid - base;
  const corner = d.vec3i(base);

  let vel = d.vec3f();
  let density = d.f32(0);
  let heat = d.f32(0);
  for (const stepZ of tgpu.unroll([0, 1])) {
    for (const stepY of tgpu.unroll([0, 1])) {
      for (const stepX of tgpu.unroll([0, 1])) {
        const weight =
          std.mix(1 - blend.x, blend.x, d.f32(stepX)) *
          std.mix(1 - blend.y, blend.y, d.f32(stepY)) *
          std.mix(1 - blend.z, blend.z, d.f32(stepZ));
        const cell =
          smokeLayout.$.cellsIn[cellIndexAt(corner + d.vec3i(stepX, stepY, stepZ))];
        vel = vel + cell.vel * weight;
        density += cell.density * weight;
        heat += cell.heat * weight;
      }
    }
  }
  return SmokeCell({ vel, density, heat });
};

const randomUnit = (state: number) => {
  'use gpu';
  let bits = d.u32(state) * 747796405 + 2891336453;
  bits = ((bits >>> ((bits >>> 28) + 4)) ^ bits) * 277803737;
  return d.f32((bits >>> 22) ^ bits) / 4294967295;
};

/**
 * Carry the field along its own velocity, then add what the step puts in:
 * buoyancy, wind, and the source. One pass, because velocity and the two
 * scalars all ride the same back-trace.
 */
const advectKernel = tgpu.computeFn({
  workgroupSize: [SMOKE_WORKGROUP, SMOKE_WORKGROUP, SMOKE_WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  const coord = d.vec3i(gid);
  if (gid.x >= d.u32(SMOKE_X) || gid.y >= d.u32(SMOKE_Y) || gid.z >= d.u32(SMOKE_Z)) {
    return;
  }
  const index = cellIndexAt(coord);
  const params = smokeLayout.$.params;

  if (isSolid(coord)) {
    smokeLayout.$.cellsOut[index] = SmokeCell({ vel: d.vec3f(), density: 0, heat: 0 });
    return;
  }

  const centre = centreOf(gid);
  const here = smokeLayout.$.cellsIn[index];
  const back = centre - here.vel * params.dt;
  const carried = sampleCell(back);

  // What flows in from beyond the grid is clean air. Without this the trace
  // clamps to the edge and re-reads the cell it just left, so smoke leaving the
  // frame parks in the last row instead of going.
  const inside =
    back.x > 0 && back.y > 0 && back.z > 0 && back.x < 1 && back.y < 1 && back.z < Z_MAX;
  const arriving = std.select(d.vec2f(), d.vec2f(carried.density, carried.heat), inside);

  let vel = d.vec3f(carried.vel);
  let density = arriving.x * std.exp(-params.dissipation * params.dt);
  let heat = arriving.y * std.exp(-params.cooling * params.dt);

  // Up is the reverse of the gravity the scene was measured to have, so a plume
  // climbs the same way the water falls without anyone naming a camera angle.
  const up = smokeLayout.$.sceneState.down * -1;
  vel = vel + up * (params.buoyancy * heat * params.dt);
  vel = vel + d.vec3f(params.wind * WIND_SCALE * params.dt, 0, 0);
  // Still air. Confinement returns energy in proportion to the swirl it finds,
  // so without something taking energy back out the two never balance and the
  // whole field winds up.
  vel = vel * std.exp(-params.drag * params.dt);

  // The source. A perfectly symmetric jet stays a column for far too long, so
  // the inflow gets a small random sideways kick to break it up.
  const reach = std.length(centre - params.emitter);
  const strength = 1 - std.smoothstep(0, params.emitRadius, reach);
  if (strength > 0) {
    const seed = index * 2654435761 + params.frame;
    const jitter = d.vec3f(
      randomUnit(seed) - 0.5,
      randomUnit(seed + 1) - 0.5,
      randomUnit(seed + 2) - 0.5,
    );
    density += params.emitRate * strength * params.dt;
    heat += params.emitHeat * strength * params.dt;
    vel = vel + jitter * (strength * params.emitHeat * params.dt * 0.05);
  }

  // Both are amounts of something, so neither can be less than nothing. The
  // trace undershoots by a fraction of a percent where the field is steep, and
  // a negative density flips the sign of the extinction in the ray march: the
  // plume stops blocking light and starts subtracting it, which paints a
  // hard-edged black hole in the middle of the smoke.
  smokeLayout.$.cellsOut[index] = SmokeCell({
    vel,
    density: std.clamp(density, 0, DENSITY_CAP),
    heat: std.max(heat, 0),
  });
});

/**
 * A wall stops the flow; the edge of the grid does not. The grid is a window
 * onto a room, so its faces are open: flow leaves freely and fresh air comes
 * back in. Sealing them instead turns the volume into a jar, where a plume rises
 * a few cells, meets its own return flow and stalls - which is what it did.
 */
const neighbourVelocity = (coord: d.v3i, own: d.v3f) => {
  'use gpu';
  if (isOutside(coord)) {
    return d.vec3f(own);
  }
  if (isSolid(coord)) {
    return d.vec3f();
  }
  return d.vec3f(smokeLayout.$.cellsOut[cellIndexAt(coord)].vel);
};

/**
 * Curl, then a nudge back toward it. Semi-Lagrangian advection is stable at any
 * step size, and it pays for that by smearing every eddy smaller than a cell -
 * which is most of what makes smoke read as smoke rather than as a grey column.
 * Vorticity confinement measures the swirl that survived and pushes back along
 * it, returning roughly the energy the interpolation took out.
 */
const curlKernel = tgpu.computeFn({
  workgroupSize: [SMOKE_WORKGROUP, SMOKE_WORKGROUP, SMOKE_WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= d.u32(SMOKE_X) || gid.y >= d.u32(SMOKE_Y) || gid.z >= d.u32(SMOKE_Z)) {
    return;
  }
  const coord = d.vec3i(gid);
  const own = smokeLayout.$.cellsOut[cellIndexAt(coord)].vel;
  const right = neighbourVelocity(coord + d.vec3i(1, 0, 0), own);
  const left = neighbourVelocity(coord - d.vec3i(1, 0, 0), own);
  const below = neighbourVelocity(coord + d.vec3i(0, 1, 0), own);
  const above = neighbourVelocity(coord - d.vec3i(0, 1, 0), own);
  const near = neighbourVelocity(coord + d.vec3i(0, 0, 1), own);
  const far = neighbourVelocity(coord - d.vec3i(0, 0, 1), own);

  smokeLayout.$.curl[cellIndexAt(coord)] =
    d.vec3f(
      below.z - above.z - (near.y - far.y),
      near.x - far.x - (right.z - left.z),
      right.y - left.y - (below.x - above.x),
    ) * (0.5 / d.f32(SMOKE_CELL));
});

const curlStrength = (coord: d.v3i) => {
  'use gpu';
  return std.length(smokeLayout.$.curl[cellIndexAt(coord)]);
};

const confineKernel = tgpu.computeFn({
  workgroupSize: [SMOKE_WORKGROUP, SMOKE_WORKGROUP, SMOKE_WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= d.u32(SMOKE_X) || gid.y >= d.u32(SMOKE_Y) || gid.z >= d.u32(SMOKE_Z)) {
    return;
  }
  const coord = d.vec3i(gid);
  if (isSolid(coord)) {
    return;
  }
  const index = cellIndexAt(coord);
  const params = smokeLayout.$.params;

  // Uphill in swirl strength: the direction pointing at the nearest vortex core.
  const gradient = d.vec3f(
    curlStrength(coord + d.vec3i(1, 0, 0)) - curlStrength(coord - d.vec3i(1, 0, 0)),
    curlStrength(coord + d.vec3i(0, 1, 0)) - curlStrength(coord - d.vec3i(0, 1, 0)),
    curlStrength(coord + d.vec3i(0, 0, 1)) - curlStrength(coord - d.vec3i(0, 0, 1)),
  ) * (0.5 / d.f32(SMOKE_CELL));

  const along = gradient / (std.length(gradient) + 0.00001);
  const twist = std.cross(along, smokeLayout.$.curl[index]);
  const cell = smokeLayout.$.cellsOut[index];

  // Only inside the plume. Confinement exists to give back what interpolation
  // took from the smoke, and letting it stir the clear air as well turns the
  // whole box into a slowly winding vortex that eventually saturates.
  const inPlume = std.min(cell.density, 1);
  smokeLayout.$.cellsOut[index] = SmokeCell({
    vel: cell.vel + twist * (params.swirl * inPlume * d.f32(SMOKE_CELL) * params.dt),
    density: cell.density,
    heat: cell.heat,
  });
});

/**
 * Divergence of the advected field, in Stam's scaling: the half-step central
 * difference times the cell size, negated, so the Jacobi sweep below is a plain
 * average plus this term.
 */
const divergenceKernel = tgpu.computeFn({
  workgroupSize: [SMOKE_WORKGROUP, SMOKE_WORKGROUP, SMOKE_WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= d.u32(SMOKE_X) || gid.y >= d.u32(SMOKE_Y) || gid.z >= d.u32(SMOKE_Z)) {
    return;
  }
  const coord = d.vec3i(gid);
  const index = cellIndexAt(coord);
  const own = smokeLayout.$.cellsOut[index].vel;

  const right = neighbourVelocity(coord + d.vec3i(1, 0, 0), own).x;
  const left = neighbourVelocity(coord - d.vec3i(1, 0, 0), own).x;
  const below = neighbourVelocity(coord + d.vec3i(0, 1, 0), own).y;
  const above = neighbourVelocity(coord - d.vec3i(0, 1, 0), own).y;
  const near = neighbourVelocity(coord + d.vec3i(0, 0, 1), own).z;
  const far = neighbourVelocity(coord - d.vec3i(0, 0, 1), own).z;

  smokeLayout.$.divergence[index] =
    -0.5 * d.f32(SMOKE_CELL) * (right - left + (below - above) + (near - far));
  // The sweep starts from rest every frame, so the first buffer clears here.
  smokeLayout.$.pressureOut[index] = 0;
});

/**
 * A wall mirrors the pressure, so there is no gradient pushing into it. Open air
 * beyond the grid sits at zero, which lets the solve move mass out of the box.
 */
const neighbourPressure = (coord: d.v3i, own: number) => {
  'use gpu';
  if (isOutside(coord)) {
    return d.f32(0);
  }
  if (isSolid(coord)) {
    return d.f32(own);
  }
  return smokeLayout.$.pressureIn[cellIndexAt(coord)];
};

const jacobiKernel = tgpu.computeFn({
  workgroupSize: [SMOKE_WORKGROUP, SMOKE_WORKGROUP, SMOKE_WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= d.u32(SMOKE_X) || gid.y >= d.u32(SMOKE_Y) || gid.z >= d.u32(SMOKE_Z)) {
    return;
  }
  const coord = d.vec3i(gid);
  const index = cellIndexAt(coord);
  const own = smokeLayout.$.pressureIn[index];

  const total =
    neighbourPressure(coord + d.vec3i(1, 0, 0), own) +
    neighbourPressure(coord - d.vec3i(1, 0, 0), own) +
    neighbourPressure(coord + d.vec3i(0, 1, 0), own) +
    neighbourPressure(coord - d.vec3i(0, 1, 0), own) +
    neighbourPressure(coord + d.vec3i(0, 0, 1), own) +
    neighbourPressure(coord - d.vec3i(0, 0, 1), own);

  smokeLayout.$.pressureOut[index] = (smokeLayout.$.divergence[index] + total) / 6;
});

const projectKernel = tgpu.computeFn({
  workgroupSize: [SMOKE_WORKGROUP, SMOKE_WORKGROUP, SMOKE_WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= d.u32(SMOKE_X) || gid.y >= d.u32(SMOKE_Y) || gid.z >= d.u32(SMOKE_Z)) {
    return;
  }
  const coord = d.vec3i(gid);
  const index = cellIndexAt(coord);
  if (isSolid(coord)) {
    return;
  }
  const cell = smokeLayout.$.cellsOut[index];
  const own = smokeLayout.$.pressureIn[index];

  const gradient = d.vec3f(
    neighbourPressure(coord + d.vec3i(1, 0, 0), own) -
      neighbourPressure(coord - d.vec3i(1, 0, 0), own),
    neighbourPressure(coord + d.vec3i(0, 1, 0), own) -
      neighbourPressure(coord - d.vec3i(0, 1, 0), own),
    neighbourPressure(coord + d.vec3i(0, 0, 1), own) -
      neighbourPressure(coord - d.vec3i(0, 0, 1), own),
  ) * (0.5 / d.f32(SMOKE_CELL));

  const settled = cell.vel - gradient;
  const speed = std.length(settled);
  smokeLayout.$.cellsOut[index] = SmokeCell({
    vel: settled * (std.min(speed, d.f32(MAX_SPEED)) / std.max(speed, 0.000001)),
    density: cell.density,
    heat: cell.heat,
  });
});

const clearKernel = tgpu.computeFn({
  workgroupSize: [SMOKE_WORKGROUP, SMOKE_WORKGROUP, SMOKE_WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= d.u32(SMOKE_X) || gid.y >= d.u32(SMOKE_Y) || gid.z >= d.u32(SMOKE_Z)) {
    return;
  }
  smokeLayout.$.cellsOut[cellIndexAt(d.vec3i(gid))] = SmokeCell({
    vel: d.vec3f(),
    density: 0,
    heat: 0,
  });
});

/**
 * Density and how much light reaches it, written into a volume texture.
 *
 * Shadow belongs here rather than in the ray march. There are far fewer cells
 * than there are screen pixels, and every pixel that looks at a cell wants the
 * same answer, so measuring it once per cell costs a fraction of measuring it
 * again at every step of every ray.
 */
const bakeKernel = tgpu.computeFn({
  workgroupSize: [SMOKE_WORKGROUP, SMOKE_WORKGROUP, SMOKE_WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= d.u32(SMOKE_X) || gid.y >= d.u32(SMOKE_Y) || gid.z >= d.u32(SMOKE_Z)) {
    return;
  }
  const params = bakeLayout.$.params;
  const density = bakeLayout.$.cells[cellIndexAt(d.vec3i(gid))].density;

  let towards = (d.vec3f(gid) + 0.5) * d.f32(SMOKE_CELL);
  let light = d.f32(1);
  for (const _step of std.range(SHADOW_STEPS)) {
    towards = towards + params.sun * d.f32(SHADOW_STRIDE);
    const cell = d.vec3i(std.floor(towards / d.f32(SMOKE_CELL)));
    const inside =
      cell.x >= 0 &&
      cell.y >= 0 &&
      cell.z >= 0 &&
      cell.x < SMOKE_X &&
      cell.y < SMOKE_Y &&
      cell.z < SMOKE_Z;
    if (inside) {
      const blocker = bakeLayout.$.cells[cellIndexAt(cell)].density;
      light *= std.exp(-blocker * params.opacity * d.f32(SHADOW_STRIDE));
    }
  }

  // Sky occlusion: how much plume sits between this cell and the sky. The march
  // runs along the measured up - the reverse of gravity - so it is correct for
  // any camera tilt, not just a level one. This is the term that gives the
  // plume a dark underside under a lit top: the strongest single cue that smoke
  // is a volume rather than a billboard.
  const skyward = bakeLayout.$.sceneState.down * -1;
  let sky = d.f32(1);
  let overhead = (d.vec3f(gid) + 0.5) * d.f32(SMOKE_CELL);
  for (const _step of std.range(SKY_STEPS)) {
    overhead = overhead + skyward * d.f32(SKY_STRIDE);
    const cell = d.vec3i(std.floor(overhead / d.f32(SMOKE_CELL)));
    const inGrid =
      cell.y < SMOKE_Y && cell.x >= 0 && cell.x < SMOKE_X && cell.z >= 0 && cell.z < SMOKE_Z;
    if (inGrid) {
      const blocker = bakeLayout.$.cells[cellIndexAt(cell)].density;
      sky *= std.exp(-blocker * params.opacity * d.f32(SKY_STRIDE) * 0.5);
    }
  }

  // Multiple scattering lifts the darkest interior; the sky term shapes what
  // remains. Written as (density, light) for the march.
  const lit = std.mix(d.f32(AMBIENT), 1, light);
  const shaded = lit * (0.35 + 0.65 * sky);
  std.textureStore(
    bakeLayout.$.volume,
    gid,
    d.vec4f(density, shaded, 0, 1),
  );
});

/**
 * Front-to-back march through the volume.
 *
 * The simulation is orthographic, so every pixel's ray is a straight line along
 * z at a fixed image position - the march is a column integral, and the scene's
 * own depth is where it stops, which is what puts smoke behind a mug rather than
 * over it.
 */
const renderKernel = tgpu.computeFn({
  workgroupSize: [RENDER_WORKGROUP, RENDER_WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= d.u32(SMOKE_RES) || gid.y >= d.u32(SMOKE_RES)) {
    return;
  }
  const params = renderLayout.$.params;
  const uv = (d.vec2f(gid.xy) + 0.5) / d.f32(SMOKE_RES);
  const stop =
    std.textureSampleLevel(renderLayout.$.scene, renderLayout.$.volumeSampler, uv, 0).x *
    params.depthScale;

  // The measured up, flattened to a unit vector: the march has no view of the
  // scene, but smoke still fades toward the sky the way a thick plume does.
  const up = std.normalize(renderLayout.$.sceneState.down * -1);
  // A pseudo view ray, so the phase term has a direction to work with. The
  // simulation is orthographic, but smoke scatters as if seen through a lens.
  const toEye = std.normalize(d.vec3f((d.vec2f(0.5) - uv) * 0.35, 1));

  let colour = d.vec3f();
  let transmittance = d.f32(1);
  for (const step of std.range(SMOKE_Z)) {
    const z = d.f32(Z_MAX) - (d.f32(step) + 0.5) * d.f32(SMOKE_CELL);
    const sample = std.textureSampleLevel(
      renderLayout.$.volume,
      renderLayout.$.volumeSampler,
      d.vec3f(uv, z / d.f32(Z_MAX)),
      0,
    );
    if (z > stop) {
      const opacity = 1 - std.exp(-sample.x * params.opacity * d.f32(SMOKE_CELL));
      // Two-lobe phase: back-scattering (sun behind the plume, toward the eye)
      // brightens it, forward-scattering keeps the rest from going flat.
      const sunDot = std.dot(toEye, params.sun);
      const phase = 0.75 + 0.35 * std.saturate(-sunDot) + 0.2 * sunDot * sunDot;
      // Height falloff toward the measured sky: thin smoke high up lets the
      // ambient through, dense base holds its shade.
      const heightFade = std.saturate(std.dot(d.vec3f(uv, z / d.f32(Z_MAX)), up) * 1.5 + 0.5);
      const lit = std.mix(
        params.shade,
        params.tint,
        sample.y,
      ) * std.mix(d.vec3f(0.75, 0.8, 0.9), d.vec3f(1), heightFade);
      colour = colour + lit * (opacity * transmittance * phase);
      transmittance *= 1 - opacity;
    }
  }

  // Premultiplied, so the composite is one multiply-add.
  std.textureStore(renderLayout.$.image, gid.xy, d.vec4f(colour, 1 - transmittance));
});

export interface SmokeTuning {
  emitterX: number;
  emitterY: number;
  emitterZ: number;
  emitRadius: number;
  emitRate: number;
  emitHeat: number;
  buoyancy: number;
  cooling: number;
  dissipation: number;
  wind: number;
  swirl: number;
  drag: number;
  depthScale: number;
  opacity: number;
  sunBearing: number;
  sunHeight: number;
}

export const defaultSmokeTuning: SmokeTuning = {
  emitterX: 0.5,
  emitterY: 0.72,
  emitterZ: Z_MAX * 0.55,
  // A source only a few cells across can only make a ribbon. Widening it is what
  // gives the plume room to roll over on itself and show any structure at all.
  emitRadius: 0.11,
  emitRate: 0,
  emitHeat: 7,
  buoyancy: 2.4,
  cooling: 0.55,
  dissipation: 0.15,
  wind: 0,
  swirl: 20,
  drag: 0.1,
  depthScale: MAX_DEPTH_SCALE,
  opacity: 26,
  sunBearing: -30,
  sunHeight: 44,
};

const SHADE: [number, number, number] = [0.2, 0.21, 0.24];
const TINT: [number, number, number] = [0.9, 0.91, 0.93];

export interface SmokeInputs {
  readonly surface: SingleChannelTexture;
  readonly scene: TgpuBuffer<typeof SceneState> & StorageFlag;
  readonly fieldSampler: TgpuSampler;
}

export interface Smoke {
  /** Premultiplied smoke image the composite lays over the scene. */
  readonly image: SingleChannelTexture;
  initAsync(): Promise<void>;
  /** One solver step plus the shadow bake. */
  encode(pass: TgpuComputePass, dt: number): void;
  /** Ray march into `image`. Separate, so it can run at render rate. */
  encodeRender(pass: TgpuComputePass): void;
  /** Empty the volume and the image. */
  clear(pass: TgpuComputePass): void;
  tune(next: Partial<SmokeTuning>): void;
  /** The live grid, for checking the solver from a headless run. */
  readCells(): Promise<readonly { vel: d.v3f; density: number; heat: number }[]>;
  destroy(): void;
}

/** Bearing and height to a direction, matching the liquid's key light. */
function sunVector(bearing: number, height: number): [number, number, number] {
  const yaw = (bearing * Math.PI) / 180;
  const pitch = (height * Math.PI) / 180;
  const flat = Math.cos(pitch);
  return [Math.sin(yaw) * flat, -Math.cos(yaw) * flat, Math.sin(pitch)];
}

export function createSmoke(root: TgpuRoot, inputs: SmokeInputs): Smoke {
  let tuning: SmokeTuning = { ...defaultSmokeTuning };
  let frame = 0;

  const params = root.createBuffer(SmokeParams).$usage('uniform');
  const cells = [
    root.createBuffer(SmokeCellArray).$usage('storage'),
    root.createBuffer(SmokeCellArray).$usage('storage'),
  ] as const;
  const pressure = [
    root.createBuffer(PressureArray).$usage('storage'),
    root.createBuffer(PressureArray).$usage('storage'),
  ] as const;
  const divergence = root.createBuffer(PressureArray).$usage('storage');
  const curl = root.createBuffer(CurlArray).$usage('storage');

  const volume = root
    .createTexture({
      size: [SMOKE_X, SMOKE_Y, SMOKE_Z],
      format: 'rgba16float',
      dimension: '3d',
    })
    .$usage('sampled', 'storage');

  const image = root
    .createTexture({ size: [SMOKE_RES, SMOKE_RES], format: 'rgba16float' })
    .$usage('sampled', 'storage');

  /**
   * The cell buffers swap once a step; the pressure buffers swap every Jacobi
   * sweep. The two are independent, so each cell phase gets both pressure
   * arrangements: `pass0` reads pressure 0 and writes pressure 1, `pass1` the
   * other way round.
   */
  const solveGroup = (cellPhase: number, pressurePhase: number) =>
    root.createBindGroup(smokeLayout, {
      params,
      cellsIn: cells[cellPhase],
      cellsOut: cells[1 - cellPhase],
      pressureIn: pressure[pressurePhase],
      pressureOut: pressure[1 - pressurePhase],
      divergence,
      curl,
      sceneState: inputs.scene,
      surface: inputs.surface.createView(),
      fieldSampler: inputs.fieldSampler,
    });

  const phases = [
    { pass0: solveGroup(0, 0), pass1: solveGroup(0, 1) },
    { pass0: solveGroup(1, 0), pass1: solveGroup(1, 1) },
  ] as const;

  const volumeWrite = volume.createView(d.textureStorage3d('rgba16float', 'write-only'));
  // The bake reads what advection just wrote, which is the other cell buffer.
  const bakeGroups = [
    root.createBindGroup(bakeLayout, {
      params,
      cells: cells[1],
      sceneState: inputs.scene,
      volume: volumeWrite,
    }),
    root.createBindGroup(bakeLayout, {
      params,
      cells: cells[0],
      sceneState: inputs.scene,
      volume: volumeWrite,
    }),
  ] as const;

  const renderGroup = root.createBindGroup(renderLayout, {
    params,
    volume: volume.createView(d.texture3d(d.f32)),
    scene: inputs.surface.createView(),
    sceneState: inputs.scene,
    volumeSampler: inputs.fieldSampler,
    image: image.createView(d.textureStorage2d('rgba16float', 'write-only' as const)),
  });

  const advect = root.createComputePipeline({ compute: advectKernel });
  const swirl = root.createComputePipeline({ compute: curlKernel });
  const confine = root.createComputePipeline({ compute: confineKernel });
  const divide = root.createComputePipeline({ compute: divergenceKernel });
  const jacobi = root.createComputePipeline({ compute: jacobiKernel });
  const project = root.createComputePipeline({ compute: projectKernel });
  const clearCells = root.createComputePipeline({ compute: clearKernel });
  const bake = root.createComputePipeline({ compute: bakeKernel });
  const march = root.createComputePipeline({ compute: renderKernel });

  const gridGroups: [number, number, number] = [
    Math.ceil(SMOKE_X / SMOKE_WORKGROUP),
    Math.ceil(SMOKE_Y / SMOKE_WORKGROUP),
    Math.ceil(SMOKE_Z / SMOKE_WORKGROUP),
  ];
  const imageGroups = Math.ceil(SMOKE_RES / RENDER_WORKGROUP);

  let phase: 0 | 1 = 0;

  function write(dt: number): void {
    const sun = sunVector(tuning.sunBearing, tuning.sunHeight);
    params.write({
      dt,
      emitter: [tuning.emitterX, tuning.emitterY, tuning.emitterZ],
      emitRadius: tuning.emitRadius,
      emitRate: tuning.emitRate,
      emitHeat: tuning.emitHeat,
      buoyancy: tuning.buoyancy,
      cooling: tuning.cooling,
      dissipation: tuning.dissipation,
      wind: tuning.wind,
      swirl: tuning.swirl,
      drag: tuning.drag,
      depthScale: tuning.depthScale,
      sun,
      opacity: tuning.opacity,
      shade: SHADE,
      tint: TINT,
      frame,
    });
  }

  write(1 / 60);

  return {
    image,

    async initAsync() {
      await Promise.all([
        advect.initAsync(),
        swirl.initAsync(),
        confine.initAsync(),
        divide.initAsync(),
        jacobi.initAsync(),
        project.initAsync(),
        clearCells.initAsync(),
        bake.initAsync(),
        march.initAsync(),
      ]);
    },

    encode(pass, dt) {
      frame++;
      write(dt);
      const { pass0, pass1 } = phases[phase];

      advect.with(pass).with(pass0).dispatchWorkgroups(...gridGroups);
      swirl.with(pass).with(pass0).dispatchWorkgroups(...gridGroups);
      confine.with(pass).with(pass0).dispatchWorkgroups(...gridGroups);
      // Run on pass1, whose output side is pressure 0 - the buffer the first
      // sweep reads. The solve starts from rest every frame.
      divide.with(pass).with(pass1).dispatchWorkgroups(...gridGroups);
      for (let sweep = 0; sweep < PRESSURE_ITERATIONS; sweep++) {
        jacobi
          .with(pass)
          .with(sweep % 2 === 0 ? pass0 : pass1)
          .dispatchWorkgroups(...gridGroups);
      }
      // An even sweep count leaves the answer back in pressure 0, which is what
      // pass0 reads.
      project.with(pass).with(pass0).dispatchWorkgroups(...gridGroups);
      bake.with(pass).with(bakeGroups[phase]).dispatchWorkgroups(...gridGroups);

      phase = phase === 0 ? 1 : 0;
    },

    encodeRender(pass) {
      march.with(pass).with(renderGroup).dispatchWorkgroups(imageGroups, imageGroups);
    },

    clear(pass) {
      for (const group of phases) {
        clearCells.with(pass).with(group.pass0).dispatchWorkgroups(...gridGroups);
      }
      // Baking an empty volume and marching it is what actually blanks the image.
      bake.with(pass).with(bakeGroups[phase]).dispatchWorkgroups(...gridGroups);
      march.with(pass).with(renderGroup).dispatchWorkgroups(imageGroups, imageGroups);
    },

    tune(next) {
      tuning = { ...tuning, ...next };
      write(1 / 60);
    },

    readCells() {
      // `phase` has already flipped, so the last pass wrote the buffer this
      // phase now reads.
      return cells[phase].read();
    },

    destroy() {
      params.destroy();
      for (const buffer of cells) {
        buffer.destroy();
      }
      for (const buffer of pressure) {
        buffer.destroy();
      }
      divergence.destroy();
      curl.destroy();
      volume.destroy();
      image.destroy();
    },
  };
}
