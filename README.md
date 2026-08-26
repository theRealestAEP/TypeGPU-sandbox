# Liquid in a Cup

A 3D position-based fluid and a grid smoke solver that read the real world
through a webcam, treat the depth map as a surface, and pour liquid or smoke onto
it. Cups, bowls and cupped hands fill up on their own. Plumes climb walls and roll
around whatever is in frame. The liquid refracts the actual camera image behind it.

Built on [TypeGPU](https://docs.swmansion.com/TypeGPU/) (WebGPU in TypeScript).

## Run it

```sh
npm install
npm run dev      # http://127.0.0.1:5173
```

Needs a WebGPU browser: Chrome 113+, Edge, or Safari 18+.

Also available: `npm run build`, `npm run lint`, `npm run typecheck`.

## Four scenes

The chips along the bottom pick the source. One source runs at a time, and
switching stops the previous one.

| Scene | Key | What it is | Needs |
|---|---|---|---|
| `Vessels` | `Q` | A hand-built scene: a mug, a wide bowl and a tall glass on a table | nothing |
| `Photo` | `W` | Real monocular depth on a still, an empty pool to start with | one 13 MB model download |
| `Clip` | `E` | The same, frame by frame, on a video: four kittens in a room | the model |
| `Camera` | `R` | Live webcam depth | camera permission + the model |

**Bring in your own** with `Open` (or `O`), or by dropping a file on the scene.
An image replaces the still and selects `Photo`; a video replaces the clip and
selects `Clip`. A clip runs through the same path the camera does, so depth
tracks it frame by frame.

The **Scenarios** list on the left is these scenes plus the handful of settings
that make each one read - where the source sits, how fast it pours, whether it
storms, whether it smokes.

The model is [DepthART](https://huggingface.co/reczkok/depthart-typegpu), cached
in the browser after the first download. `small` / `base` / `large` trade sharpness
against speed.

## The simulation is 3D

x and y are the image plane (x right, y **down**, matching WebGPU's texture
origin). z is distance toward the camera: 0 is as far as the scene goes, larger
is nearer, matching the depth field where 1 is nearest.

A depth map only shows the front-most surface, which gives the whole collision
model in one line: **a particle behind the visible surface is inside something, a
particle in front of it is in open air.** There is no cup detector and no depth
plane. A cup viewed from above shows a near rim ringing a far interior, and that
interior is a real basin in the surface, so it fills because of geometry.

The failure case fails correctly: a mug seen dead-on has its opening facing away,
so there is no basin, and liquid runs off it.

## Gravity is measured, but only so far

A depth map cannot tell a floor seen from above from a wall seen head-on. Both
are flat and both face the camera, so both give the same normal. That ambiguity
is not a detail: the first estimator counted any texel with the faintest upward
lean as ground, depth noise pushed half of every camera-facing wall through the
threshold, and those texels won on area. Measured on the pool photo, gravity came
back 93% *into the scene* against 38% down the image, so every drop slid behind
the visible world and piled against the back of the volume.

What the depth map does give, once gravity is known, is the height field - and
that is the part worth having. Water flows downhill in it and basins fill,
which is the whole simulation. But the direction of downhill has to come from
somewhere else: a prior, a sensor, or semantics. Here it is a prior. A level
camera is the starting assumption, the measurement moves it only as far as the
share of the frame that reads as ground supports, and the lean into the scene is
capped at about 46 degrees. A phone could do better - `DeviceOrientationEvent`
reports gravity directly - and that is the obvious next step for the live camera.

## Gravity is measured, not configured

The one thing a depth map cannot tell you is which way is down, and asking for a
camera angle is a bad question - nobody knows what their webcam's tilt is in
degrees.

So it is measured instead. Liquid rests on surfaces whose normal opposes gravity,
so gravity is the reverse of the average **up-facing** surface normal in view.
Only texels whose normal leans toward the top of the image count: those are the
surfaces you are looking down onto - a tabletop, a floor - while a wall facing
the camera has no vertical lean and is correctly ignored. It is re-estimated with
every depth update, so it tracks a moving camera.

Two details make that estimate hold up. It is refit a second time using only the
surfaces that agree with the first pass, because a plain mean gets dragged around
by whatever else is in frame - a wall, a chair - and a few degrees of tilt leaves
a standing sideways pull that slides a pool across a table it should rest flat
on. And the water's floor is a **level plane square to gravity**, not the bottom
of the frame: gathered water finds its own level the way water does.

That second point matters more than it sounds. On a real photographed table,
measured gravity is only about 40% down-screen and 90% *into* the scene - "down"
mostly means away from the camera. A frame-aligned floor therefore made water
stack against the bottom of the picture and squeeze sideways out of the pile,
which reads as water climbing uphill.

## Water accumulates, and the pour has no ceiling

The **fill** meter is real: it is how much of the water supply is currently in
the scene. `Drain` empties it, `Refill` restarts from a falling block.

The supply is one fixed particle budget, which used to be a hard ceiling - once
every drop was in the scene the spout silently stopped, which reads as the pour
randomly refusing to work. At capacity it now retires the deepest settled water
back to the spout instead. Those drops sit at the bottom under everything else,
so removing them is invisible, and the emit window rate-limits it to exactly the
inflow: the level holds while the water circulates.

## The bundled clip

`Cats in water` plays 19 seconds of four kittens playing on a rug, shot wide from
above. Straight cut - no stabilisation, no reversing, no crossfade.

It was found by measurement. Camera motion is scored by block-matching
consecutive downscaled frames, so a stationary camera reads 0.00 px. The source
recording is hand-held for most of its 39 minutes, but the operator sets it down
periodically; sampling every 7 seconds located the window where it is genuinely
still. The cut measures 0.00 px throughout, against 1.0-6.8 px elsewhere in the
same file.

Worth doing that measurement rather than trusting your eye: footage that looks
locked-off often is not, and stabilising footage that is already still makes it
worse - measured 0.18 px before, 2.76 px after, on an earlier candidate.

## Smoke is a grid, not particles

Particles suit water because water has a surface worth finding. Smoke has none:
what you see is the light that survived a path through it, which is an integral
over a volume. So smoke gets its own Eulerian grid - 64x64x40 on desktop, 40x40x25
on phones - covering the same box the liquid uses, and the standard four moves
each step:

1. **Advect.** Trace back along the velocity and read what was there. Velocity,
   smoke and heat all ride the same trace, so it is one pass.
2. **Force.** Buoyancy along the measured up, in proportion to heat; the storm's
   gust; drag; the source.
3. **Project.** Twenty Jacobi sweeps for the pressure that makes the flow
   divergence-free, then subtract its gradient.
4. **Bake and march.** Write density and shadow into a volume texture, then
   integrate along z into an image the composite lays over the scene.

The scene enters through the same rule the liquid uses: a cell behind the visible
depth surface is inside something. Those cells are solid, so a plume climbs a wall
and rolls around a mug without anything being told a wall or a mug is there.

Three things had to be right before it looked like smoke rather than a grey ribbon:

- **The grid faces are open, not sealed.** Treating the edge of the box as a wall
  makes a jar: the plume rises about ten cells, meets its own return flow and
  stalls. Measured with the buoyancy turned up eleven-fold, the top of the plume
  moved by one cell. Open faces - pressure zero outside, velocity free to leave,
  clean air on the way back in - and it crosses the frame.
- **Vorticity confinement, weighted by density.** Tracing back along a velocity
  field smears out every eddy smaller than a cell, and those eddies are most of
  what makes smoke read as smoke. Confinement measures the swirl that survived
  and pushes back along it. Applied everywhere it slowly winds the whole box up
  until every cell hits the speed cap; applied only where there is smoke, it
  stays put - measured flat at 0.16 units/s across a further eighteen simulated
  seconds, against a runaway to the 1.88 cap without it.
- **A source that breathes.** A fixed sphere emitting at a fixed rate makes a
  column that never bends. Two slow wobbles at rates that do not divide into each
  other, and a source wide enough to roll over on itself, and the column breaks
  into puffs.

Both amounts are clamped to zero at the write. The back-trace undershoots by a
fraction of a percent where the field is steep, and a negative density flips the
sign of the extinction in the ray march: the plume stops blocking light and starts
subtracting it, which paints a hard-edged black hole in the middle of the smoke.

Shadow is measured per cell rather than per ray step. There are far fewer cells
than screen pixels and every pixel looking at a cell wants the same answer, so one
short march toward the light per cell replaces eight per pixel. An ambient floor
keeps thick smoke from going to the shadow colour and reading as a hole.

The spout carries one medium at a time - `Water` or `Smoke`, keys `M` and `N` -
and `Hold` releases whichever is selected. Switching clears what the other left
behind. There is no coupling to model between them anyway: they share the scene
and the measured gravity, and nothing else.

## Is the solver the limit?

No, and it is worth knowing which end to push on. Position-Based Fluids gets
criticised for soft incompressibility, so it is measurable: fill the pool, let it
settle, and compare particle packing at depth against the rest spacing.

Under a full water column the deep spacing measures 0.00550 against a rest
spacing of 0.00568 - **about 3% compression**. That is well inside what reads as
water, so the pressure solve is not what limits realism here.

FLIP or APIC would still be the stronger choice on paper - a grid pressure solve
propagates instantly rather than over three iterations, and both carry less
numerical damping - but they would not fix the thing that actually reads as
artificial, which is bead size, and bead size is particle count. Spend effort
there, or on the compositor, before rewriting the solver.

## Where water still snags

Depth-field roughness turns into a landscape of shallow pockets, and water parks
in them. Measured on the pool photo, about 3% of the liquid ended up stranded -
motionless, well above the water line. Widening the collision-surface filter to
roughly one kernel radius cleared it: stranding fell to 0.05%.

It scales with `scene depth`, though. The same pockets get deeper as the scene
does, and past about 0.5 they are deep enough to hold a particle again - a third
of the water strands. That is why the slider stops at 0.48. Deeper scenes would
need a collision filter that widens with them.

## Where smoke still misses

Smoke is stopped by the visible surface all the way to the back of the box, not
by a slab the way water is. A plume cannot pass behind a mug and come out the
other side. Water needs the slab because you can see it back there; smoke does
not, because the ray march stops at the same surface, so the simpler rule stays.

The march is a straight column along z, which is exactly right for the
orthographic simulation and gives the volume no parallax. Nothing in the picture
depends on it, but a smoke plume will not swing across the frame as a real camera
would make it.

Smoke draws over water rather than through it. The two occupy different halves of
a scene often enough that it has not mattered.

## What this is not

The liquid occupies the 3D space the camera can *see into*. It is not a
reconstruction of the room: the far side of an object is guessed at with one
thickness number, not measured.

Two limits are inherent to reading depth from a single camera:

- **Relative depth drifts.** The model emits relative disparity, renormalised
  against a stabilised range, so something entering the frame shifts the whole
  range. A slow EMA damps it; nothing removes it.
- **Depth flickers.** The surface push-out is capped per step, so a surface
  appearing on top of a particle nudges it instead of launching it.

## Layout

```
src/
  main.ts                 wiring, frame loop, controls
  camera.ts               getUserMedia session and orientation
  hud.ts                  game overlay: chips, action bar, fill meter, keys
  depth/                  DepthSource interface + synthetic, model, model store
  sim/                    schemas, SPH kernels, 3D hash grid, PBF, surface field
  sim/smoke.ts            Eulerian smoke: advect, confine, project, bake, march
  render/liquid.ts        depth pass, thickness pass, composite
  gpu/blur.ts             separable Gaussian, for scene depth and thickness
  gpu/surface-filter.ts   depth-aware filter for the fluid surface
  vendor/depthart/        DepthART runtime, vendored from TypeGPU (MIT). Do not edit.
public/scene.jpg          the bundled still; see scene.txt for attribution
public/scene.webm         the bundled clip; same file for attribution
```

`src/vendor/` is third-party and is excluded from the anti-slop lint on purpose.
Everything else is covered by it.

See `PLAN.md` for the design rationale, the cost/payoff ranking and the critique.
