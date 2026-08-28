# TypeGPU Sandbox

Pour water or smoke into a photo, a video, or your webcam. A depth model turns
the picture into a surface, a particle fluid runs against it, and detected
cups become real containers you can fill.

Built on [TypeGPU](https://docs.swmansion.com/TypeGPU/) (WebGPU in TypeScript).

## What runs under the hood

- **Depth** — DepthART, a monocular depth model vendored from TypeGPU, run in
  WebGPU. Each frame becomes a heightfield the fluid collides with. A second
  "background" layer remembers what passing occluders cover.
- **Fluid** — position-based fluids (PBF): 120Hz substeps, three solver
  iterations, a 3D hash grid for neighbours, poly6/spiky kernels. Gravity
  direction is measured from the scene's own geometry (a scatter-matrix fit of
  the surface normals, gated on how floor-like the evidence is).
- **Cups** — MediaPipe's EfficientDet finds glasses and mugs; the box becomes
  a container. The interior is carved into the heightfield, and analytic
  walls, floor, and mouth complete what the depth model cannot see - a clear
  glass reads as void and still holds water.
- **Light** — analytic point lamps shade the scene's depth-derived normals;
  nothing is path-traced. The sun direction comes from a luminance-vs-normal
  regression over the image itself.
- **Smoke** — an Eulerian grid: semi-Lagrangian advection, vorticity
  confinement, pressure projection, then ray-marched front to back with baked
  sun shadows and sky occlusion.
- **Water rendering** — particles splat into a screen-space surface, filtered
  and temporally smoothed, then shaded with refraction, Beer-Lambert
  absorption, reflection, foam, and caustics. Fast water stretches along its
  velocity so streams read as streams.

## Run it

```sh
npm install
npm run dev      # http://127.0.0.1:5173
```

Needs a WebGPU browser: Chrome 113+, Edge, or Safari 18+.
Also `npm run build`, `npm run lint`, `npm run typecheck`.

Hold space to pour. Scroll to set the spout's depth. Drop in your own image or
video, or hold a glass up to the camera. `T` opens the tuning panel; `D`
resets the scene and every slider. `?particles=N` overrides the particle
budget. On camera scenes, `H` toggles hand control (off by default - it costs
a GPU model per frame): your index fingertip steers the spout and a pinch
pours.

## Picking a scene that works

Liquid gathers where the scene has a closed depression under gravity. A basin
shot from near eye level is edge-on and holds almost nothing, however deep it
looks.

```sh
node tools/score-scene.mjs <image> [--region x0,x1,y0,y1]
```

Runs the real depth pipeline, floods the potential field, and reports the
camera pitch and how much of your region can hold water. Needs the dev server
running, plus `ffmpeg`.

## Layout

```
src/
  main.ts                 wiring, frame loop, controls
  camera.ts               getUserMedia session and orientation
  hud.ts                  overlay: chips, action bar, fill meter, keys
  depth/                  DepthSource interface, model + synthetic depth,
                          lighting, vessel flood, cup carve
  track/gestures.ts       MediaPipe vessel detection (face/hand optional)
  sim/                    schemas, SPH kernels, 3D hash grid, PBF, surface field
  sim/smoke.ts            Eulerian smoke: advect, confine, project, bake, march
  render/liquid.ts        depth pass, thickness pass, composite
  gpu/                    separable blur, depth-aware surface filter
  vendor/depthart/        DepthART runtime, vendored from TypeGPU (MIT). Do not edit.
tools/score-scene.mjs     scene containment scorer
tools/verify.mjs          headless smoke test
public/                   bundled stills and clips (scene.txt has attribution);
                          benchmark.png is the cup-filling test frame
```

`src/vendor/` is third-party and is excluded from the anti-slop lint on purpose.
