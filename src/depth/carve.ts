import { d, std, tgpu } from 'typegpu';
import type { TgpuComputePass, TgpuRoot } from 'typegpu';
import { createFieldTexture, type SingleChannelTexture } from '../gpu/blur.ts';
import { FIELD_RES } from '../sim/schemas.ts';

/**
 * Gives detected drinking vessels an interior the depth model cannot see.
 *
 * A monocular depth map only knows front surfaces, so a glass held up to the
 * camera reads as a solid bump and poured water slides off it. But cups are
 * self-similar: their interior depth is close to their rim width, and the
 * detector supplies the rim width. So inside each detected vessel the raw
 * depth is pushed back by that much - carved - before anything downstream
 * reads it. Collision, occlusion, the vessel probe and the renderer then all
 * agree the glass is hollow, because in the one world model they share, it is.
 *
 * The carve happens at the raw depth, with margins for the walls and only on
 * texels near the vessel's own front (so a box that overshoots onto the
 * background does not dig a moat around the glass).
 */

const MAX_CUPS = 3;

/** Fraction of the box width kept as wall on each side. */
export const WALL = 0.14;
/** Fraction of the box height kept as rim at the top and base at the bottom. */
export const RIM = 0.06;
export const BASE = 0.12;

const carveLayout = tgpu.bindGroupLayout({
  /** Boxes in field space: x0, y0, x1, y1. Empty slots have x1 <= x0. */
  boxes: { uniform: d.arrayOf(d.vec4f, MAX_CUPS) },
  /** Per cup: the vessel's front depth, and how far back to carve. */
  depths: { uniform: d.arrayOf(d.vec4f, MAX_CUPS) },
  source: { texture: d.texture2d(d.f32) },
  target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
});

const carveKernel = tgpu.computeFn({
  workgroupSize: [8, 8],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= d.u32(FIELD_RES) || gid.y >= d.u32(FIELD_RES)) {
    return;
  }
  const texel = std.textureLoad(carveLayout.$.source, gid.xy, 0);
  let depth = texel.x;
  const uv = (d.vec2f(d.f32(gid.x), d.f32(gid.y)) + 0.5) / d.f32(FIELD_RES);

  for (const slot of std.range(MAX_CUPS)) {
    const box = carveLayout.$.boxes[slot];
    if (box.z <= box.x) {
      continue;
    }
    const width = box.z - box.x;
    const height = box.w - box.y;
    const inside =
      uv.x > box.x + width * d.f32(WALL) &&
      uv.x < box.z - width * d.f32(WALL) &&
      uv.y > box.y + height * d.f32(RIM) &&
      uv.y < box.w - height * d.f32(BASE);
    if (inside) {
      const front = carveLayout.$.depths[slot].x;
      const carve = carveLayout.$.depths[slot].y;
      // A bowl, never a slab. A cavity with cliff edges cannot hold water
      // under a level camera: a z-cliff seen edge-on has no height, so the
      // weir prices crossing it at nothing and the water pours out over the
      // base lip - measured as 99% drainage on a level webcam frame. Grading
      // the interior into ramps gives its floor and walls image-space slope,
      // and slopes hold water the ordinary way: their normals lean against
      // gravity through plain contact.
      const spanX = (uv.x - (box.x + box.z) * 0.5) / (width * (0.5 - d.f32(WALL)));
      const sideRamp = 1 - std.smoothstep(0.45, 1, std.abs(spanX));
      const floorRamp = 1 -
        std.smoothstep(box.w - height * 0.36, box.w - height * d.f32(BASE), uv.y);
      const bowl = carve * std.min(sideRamp, floorRamp);
      // Two cases, one bowl. A texel near the vessel's front is the glass the
      // model DID see: dig it down to the bowl. A texel far behind it is the
      // glass the model saw THROUGH - a clear wall against a dark room reads
      // as void, and a bowl with holes in its floor drains from the bottom,
      // which is exactly what the user's depth view showed. The detector has
      // confirmed a vessel here for three rounds; inside its interior the
      // vessel exists whether or not the depth model can see it, so the void
      // is raised up to the same bowl.
      if (bowl > 0.001) {
        const floor = std.max(front - bowl, 0.02);
        if (std.abs(depth - front) < 0.16) {
          // Clamped above the far plane: a vessel against a deep surface
          // would otherwise carve past the back of the world.
          depth = std.max(std.min(depth, floor), 0.02);
        } else if (depth < floor) {
          depth = floor;
        }
      }
    }
  }

  std.textureStore(carveLayout.$.target, gid.xy, d.vec4f(depth, texel.y, texel.z, texel.w));
});

const copyLayout = tgpu.bindGroupLayout({
  source: { texture: d.texture2d(d.f32) },
  target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
});

const copyKernel = tgpu.computeFn({
  workgroupSize: [8, 8],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= d.u32(FIELD_RES) || gid.y >= d.u32(FIELD_RES)) {
    return;
  }
  std.textureStore(
    copyLayout.$.target,
    gid.xy,
    std.textureLoad(copyLayout.$.source, gid.xy, 0),
  );
});

export interface CupCarve {
  /** The vessel's box in field space plus its measured front depth (0..1). */
  readonly box: readonly [number, number, number, number];
  readonly front: number;
  /** How deep the interior goes; use carveDepth so every consumer agrees. */
  readonly carve: number;
}

/**
 * Interior depth for a vessel. Width-proportional - cups are self-similar -
 * with two floors: a cavity shallower than the collision shell cannot dam
 * anything, and the shell is a USER SLIDER. Cranked past the old fixed carve
 * it dissolved every glass wall: burial never exceeded the shell, so the
 * weir waved the water straight through the sides.
 */
export function carveDepth(width: number, shell: number): number {
  return Math.max(width * 0.9, 0.18, shell + 0.06);
}

export interface Carver {
  initAsync(): Promise<void>;
  /** Replaces the active set. Carve depth derives from each box's width. */
  set(cups: readonly CupCarve[]): void;
  /** Carves the raw depth in place; encode right after the depth source. */
  encode(pass: TgpuComputePass): void;
  destroy(): void;
}

export function createCarver(root: TgpuRoot, depth: SingleChannelTexture): Carver {
  const boxes = root.createBuffer(d.arrayOf(d.vec4f, MAX_CUPS)).$usage('uniform');
  const depths = root.createBuffer(d.arrayOf(d.vec4f, MAX_CUPS)).$usage('uniform');
  const scratch = createFieldTexture(root, FIELD_RES);
  let active = 0;

  const carveBindGroup = root.createBindGroup(carveLayout, {
    boxes,
    depths,
    source: depth.createView(),
    target: scratch.createView(d.textureStorage2d('rgba16float', 'write-only')),
  });
  const publishBindGroup = root.createBindGroup(copyLayout, {
    source: scratch.createView(),
    target: depth.createView(d.textureStorage2d('rgba16float', 'write-only')),
  });
  const carve = root.createComputePipeline({ compute: carveKernel });
  const publish = root.createComputePipeline({ compute: copyKernel });
  const groups = Math.ceil(FIELD_RES / 8);

  return {
    async initAsync() {
      await Promise.all([carve.initAsync(), publish.initAsync()]);
    },

    set(cups) {
      active = Math.min(cups.length, MAX_CUPS);
      const a: [number, number, number, number][] = [];
      const b: [number, number, number, number][] = [];
      for (let i = 0; i < MAX_CUPS; i++) {
        const cup = cups[i];
        if (cup) {
          a.push([cup.box[0], cup.box[1], cup.box[2], cup.box[3]]);
          b.push([cup.front, cup.carve, 0, 0]);
        } else {
          a.push([0, 0, 0, 0]);
          b.push([0, 0, 0, 0]);
        }
      }
      boxes.write(a);
      depths.write(b);
    },

    encode(pass) {
      if (active === 0) {
        return;
      }
      carve.with(pass).with(carveBindGroup).dispatchWorkgroups(groups, groups);
      publish.with(pass).with(publishBindGroup).dispatchWorkgroups(groups, groups);
    },

    destroy() {
      boxes.destroy();
      depths.destroy();
      scratch.destroy();
    },
  };
}
