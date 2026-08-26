# Liquid in a Cup

Pour water or smoke into a photo, a video, or your webcam. A depth model turns
the picture into a surface, and a 3D position-based fluid runs against it, so
basins fill, plumes climb walls, and the liquid refracts the image behind it.

Built on [TypeGPU](https://docs.swmansion.com/TypeGPU/) (WebGPU in TypeScript).

## Run it

```sh
npm install
npm run dev      # http://127.0.0.1:5173
```

Needs a WebGPU browser: Chrome 113+, Edge, or Safari 18+.
Also `npm run build`, `npm run lint`, `npm run typecheck`.

Hold space to pour. Scroll to set the spout's depth. Drop in your own image or
video. `T` opens the tuning panel.

## Picking a scene that works

Liquid gathers where the scene has a closed depression under gravity. A basin
shot from near eye level is edge-on and holds almost nothing, however deep it
looks. Roughly 35° above the vessel or steeper is what you want.

```sh
node tools/score-scene.mjs <image> [--region x0,x1,y0,y1]
```

Runs the real depth pipeline, then floods the potential field and reports the
camera pitch, how much of your region can hold water, and how deep it gets.
Depth is the number that matters: a bathroom shot from 21° wets five times more
area than a sink shot from 39° and still reads as empty.

Needs the dev server running, plus `ffmpeg`.

## Layout

```
src/
  main.ts                 wiring, frame loop, controls
  camera.ts               getUserMedia session and orientation
  hud.ts                  overlay: chips, action bar, fill meter, keys
  depth/                  DepthSource interface, synthetic + model depth, lighting
  sim/                    schemas, SPH kernels, 3D hash grid, PBF, surface field
  sim/smoke.ts            Eulerian smoke: advect, confine, project, bake, march
  render/liquid.ts        depth pass, thickness pass, composite
  gpu/                    separable blur, depth-aware surface filter
  vendor/depthart/        DepthART runtime, vendored from TypeGPU (MIT). Do not edit.
tools/score-scene.mjs     scene containment scorer
tools/verify.mjs          headless smoke test
public/                   bundled stills and clips; see scene.txt for attribution
```

`src/vendor/` is third-party and is excluded from the anti-slop lint on purpose.
