import { d, std, tgpu } from 'typegpu';
import type { StorageFlag, TgpuBuffer, TgpuComputePass, TgpuRoot } from 'typegpu';
import { hashLayout } from './bindings.ts';
import { cellCoordOf, cellIndexOf, clampCell } from './sph.ts';
import {
  GRID_CELLS,
  PARTICLE_COUNT,
  PARTICLE_WORKGROUPS,
  ParticleArray,
  WORKGROUP_SIZE,
} from './schemas.ts';

const RESET_WORKGROUPS = Math.ceil(GRID_CELLS / WORKGROUP_SIZE);

/** One cell per thread, so a block covers this many cells. */
const SCAN_THREADS = 256;
const SCAN_BLOCKS = Math.ceil(GRID_CELLS / SCAN_THREADS);
/** Hillis-Steele needs log2(SCAN_THREADS) doubling steps. */
const SCAN_STEPS = [1, 2, 4, 8, 16, 32, 64, 128] as const;

/** Matches the fluid's drain line. */
const DORMANT_LINE = 1.1;

/**
 * Drained particles waiting to re-enter. They are left out of the grid entirely
 * so no live particle ever iterates them.
 */
const isDormant = (index: number) => {
  'use gpu';
  return hashLayout.$.particles[index].pos.y > DORMANT_LINE;
};

/** Cell of a particle, clamped so strays outside the volume still bin. */
const binOf = (index: number) => {
  'use gpu';
  return clampCell(cellCoordOf(hashLayout.$.particles[index].pos));
};

const resetKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP_SIZE],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= GRID_CELLS) {
    return;
  }
  std.atomicStore(hashLayout.$.cellCount[gid.x], 0);
  std.atomicStore(hashLayout.$.cellCursor[gid.x], 0);
});

const countKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP_SIZE],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= PARTICLE_COUNT || isDormant(gid.x)) {
    return;
  }
  std.atomicAdd(hashLayout.$.cellCount[cellIndexOf(binOf(gid.x))], 1);
});

const scanShared = tgpu.workgroupVar(d.arrayOf(d.u32, SCAN_THREADS));

/**
 * Block-local exclusive prefix sum. In 3D the grid is far too large to scan on
 * one thread, so this is the real thing: a Hillis-Steele scan per block, then a
 * scan of the block totals, then one add pass.
 */
const scanBlocksKernel = tgpu.computeFn({
  workgroupSize: [SCAN_THREADS],
  in: {
    gid: d.builtin.globalInvocationId,
    lid: d.builtin.localInvocationIndex,
    wid: d.builtin.workgroupId,
  },
})(({ gid, lid, wid }) => {
  'use gpu';
  let own = d.u32(0);
  if (gid.x < GRID_CELLS) {
    own = std.atomicLoad(hashLayout.$.cellCount[gid.x]);
  }
  scanShared.$[lid] = own;
  std.workgroupBarrier();

  for (const step of tgpu.unroll(SCAN_STEPS)) {
    let addend = d.u32(0);
    if (lid >= step) {
      addend = scanShared.$[lid - step];
    }
    std.workgroupBarrier();
    scanShared.$[lid] = scanShared.$[lid] + addend;
    std.workgroupBarrier();
  }

  if (gid.x < GRID_CELLS) {
    // Inclusive minus own value is the exclusive scan.
    hashLayout.$.cellStart[gid.x] = scanShared.$[lid] - own;
  }
  if (lid === SCAN_THREADS - 1) {
    hashLayout.$.blockSums[wid.x] = scanShared.$[lid];
  }
});

/** Scan of the per-block totals. A few hundred entries, so one thread is fine. */
const scanTotalsKernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
  let running = d.u32(0);
  for (const block of std.range(SCAN_BLOCKS)) {
    const total = hashLayout.$.blockSums[block];
    hashLayout.$.blockSums[block] = running;
    running += total;
  }
  hashLayout.$.cellStart[GRID_CELLS] = running;
});

const addOffsetsKernel = tgpu.computeFn({
  workgroupSize: [SCAN_THREADS],
  in: { gid: d.builtin.globalInvocationId, wid: d.builtin.workgroupId },
})(({ gid, wid }) => {
  'use gpu';
  if (gid.x >= GRID_CELLS) {
    return;
  }
  hashLayout.$.cellStart[gid.x] =
    hashLayout.$.cellStart[gid.x] + hashLayout.$.blockSums[wid.x];
});

const scatterKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP_SIZE],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  if (gid.x >= PARTICLE_COUNT || isDormant(gid.x)) {
    return;
  }
  const cell = cellIndexOf(binOf(gid.x));
  const slot = hashLayout.$.cellStart[cell] + std.atomicAdd(hashLayout.$.cellCursor[cell], 1);
  hashLayout.$.sortedIndex[slot] = gid.x;
});

export interface HashGrid {
  /** Exclusive start offset per cell, plus a terminating total at GRID_CELLS. */
  readonly cellStart: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
  /** Particle indices ordered by cell. */
  readonly sortedIndex: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
  initAsync(): Promise<void>;
  encode(pass: TgpuComputePass): void;
  destroy(): void;
}

export function createHashGrid(
  root: TgpuRoot,
  particles: TgpuBuffer<typeof ParticleArray> & StorageFlag,
): HashGrid {
  const cellCount = root.createBuffer(d.arrayOf(d.atomic(d.u32), GRID_CELLS)).$usage('storage');
  const cellCursor = root.createBuffer(d.arrayOf(d.atomic(d.u32), GRID_CELLS)).$usage('storage');
  const cellStart = root.createBuffer(d.arrayOf(d.u32, GRID_CELLS + 1)).$usage('storage');
  const blockSums = root.createBuffer(d.arrayOf(d.u32, SCAN_BLOCKS)).$usage('storage');
  const sortedIndex = root.createBuffer(d.arrayOf(d.u32, PARTICLE_COUNT)).$usage('storage');

  const bindGroup = root.createBindGroup(hashLayout, {
    particles,
    cellCount,
    cellCursor,
    cellStart,
    blockSums,
    sortedIndex,
  });

  const reset = root.createComputePipeline({ compute: resetKernel });
  const count = root.createComputePipeline({ compute: countKernel });
  const scanBlocks = root.createComputePipeline({ compute: scanBlocksKernel });
  const scanTotals = root.createComputePipeline({ compute: scanTotalsKernel });
  const addOffsets = root.createComputePipeline({ compute: addOffsetsKernel });
  const scatter = root.createComputePipeline({ compute: scatterKernel });

  const stages = [reset, count, scanBlocks, scanTotals, addOffsets, scatter];

  return {
    cellStart,
    sortedIndex,

    async initAsync() {
      await Promise.all(stages.map((stage) => stage.initAsync()));
    },

    encode(pass) {
      reset.with(pass).with(bindGroup).dispatchWorkgroups(RESET_WORKGROUPS);
      count.with(pass).with(bindGroup).dispatchWorkgroups(PARTICLE_WORKGROUPS);
      scanBlocks.with(pass).with(bindGroup).dispatchWorkgroups(SCAN_BLOCKS);
      scanTotals.with(pass).with(bindGroup).dispatchWorkgroups(1);
      addOffsets.with(pass).with(bindGroup).dispatchWorkgroups(SCAN_BLOCKS);
      scatter.with(pass).with(bindGroup).dispatchWorkgroups(PARTICLE_WORKGROUPS);
    },

    destroy() {
      cellCount.destroy();
      cellCursor.destroy();
      cellStart.destroy();
      blockSums.destroy();
      sortedIndex.destroy();
    },
  };
}
