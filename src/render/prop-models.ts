/**
 * Minimal GLB reader for the prop models. The files it loads are known
 * quantities - Quaternius exports with solid-colour materials, packed float
 * attributes and 16-bit indices - so it parses exactly that and throws on
 * anything else rather than growing into a general glTF loader.
 */

export interface PropModelSpec {
  /** Material names to leave out: baked fire cones, when the flame is drawn live. */
  readonly skip?: readonly string[];
  /** Which extent sets the size - tall props fit height, ground props width. */
  readonly fit: 'height' | 'width';
  /** Target extent in field units at reach = 1. */
  readonly size: number;
  /** Fraction up the model's height that lands on the planted point. */
  readonly anchor: number;
}

/** De-indexed triangles: packed xyz / xyz / rgb per vertex, y up, anchor at origin. */
export interface PropModel {
  readonly vertexCount: number;
  readonly position: Float32Array;
  readonly normal: Float32Array;
  readonly tint: Float32Array;
}

interface GltfNode {
  readonly mesh?: number;
  readonly children?: readonly number[];
  readonly translation?: readonly [number, number, number];
  readonly rotation?: readonly [number, number, number, number];
  readonly scale?: readonly [number, number, number];
}

interface GltfDoc {
  readonly scene?: number;
  readonly scenes: readonly { readonly nodes: readonly number[] }[];
  readonly nodes: readonly GltfNode[];
  readonly meshes: readonly {
    readonly primitives: readonly {
      readonly attributes: { readonly POSITION: number; readonly NORMAL: number };
      readonly indices: number;
      readonly material: number;
    }[];
  }[];
  readonly materials: readonly {
    readonly name?: string;
    readonly pbrMetallicRoughness?: {
      readonly baseColorFactor?: readonly [number, number, number, number];
    };
  }[];
  readonly accessors: readonly {
    readonly bufferView: number;
    readonly byteOffset?: number;
    readonly componentType: number;
    readonly count: number;
    readonly type: string;
  }[];
  readonly bufferViews: readonly {
    readonly byteOffset?: number;
    readonly byteLength: number;
    readonly byteStride?: number;
  }[];
}

/** Row-major 4x4. */
type Mat = Float32Array;

function trsMatrix(node: GltfNode): Mat {
  const [tx, ty, tz] = node.translation ?? [0, 0, 0];
  const [qx, qy, qz, qw] = node.rotation ?? [0, 0, 0, 1];
  const [sx, sy, sz] = node.scale ?? [1, 1, 1];
  return Float32Array.of(
    (1 - 2 * (qy * qy + qz * qz)) * sx, 2 * (qx * qy - qz * qw) * sy, 2 * (qx * qz + qy * qw) * sz, tx,
    2 * (qx * qy + qz * qw) * sx, (1 - 2 * (qx * qx + qz * qz)) * sy, 2 * (qy * qz - qx * qw) * sz, ty,
    2 * (qx * qz - qy * qw) * sx, 2 * (qy * qz + qx * qw) * sy, (1 - 2 * (qx * qx + qy * qy)) * sz, tz,
    0, 0, 0, 1,
  );
}

function mulMat(a: Mat, b: Mat): Mat {
  const out = new Float32Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      out[row * 4 + col] =
        a[row * 4] * b[col] +
        a[row * 4 + 1] * b[4 + col] +
        a[row * 4 + 2] * b[8 + col] +
        a[row * 4 + 3] * b[12 + col];
    }
  }
  return out;
}

export async function loadPropModel(url: string, spec: PropModelSpec): Promise<PropModel> {
  const bytes = await (await fetch(url)).arrayBuffer();
  const head = new DataView(bytes);
  if (head.getUint32(0, true) !== 0x46546c67 || head.getUint32(4, true) !== 2) {
    throw new Error(`${url} is not a glTF 2 binary`);
  }
  const jsonLength = head.getUint32(12, true);
  // SAFETY: the header check above says this is glTF 2.0; every field the
  // walk below touches is validated by use, and anything missing throws.
  const doc = JSON.parse(
    new TextDecoder().decode(new Uint8Array(bytes, 20, jsonLength)),
  ) as GltfDoc;
  // The binary chunk follows the JSON chunk; slicing re-aligns it to zero so
  // typed-array views over the accessors line up.
  const binStart = 20 + jsonLength + 8;
  const bin = bytes.slice(binStart, binStart + head.getUint32(20 + jsonLength, true));

  function floats(accessor: number, perVertex: number): Float32Array {
    const acc = doc.accessors[accessor];
    const view = doc.bufferViews[acc.bufferView];
    if (acc.componentType !== 5126 || (view.byteStride ?? 0) !== 0) {
      throw new Error(`${url}: expected packed float attributes`);
    }
    return new Float32Array(bin, (view.byteOffset ?? 0) + (acc.byteOffset ?? 0), acc.count * perVertex);
  }

  function indices(accessor: number): Uint16Array | Uint32Array {
    const acc = doc.accessors[accessor];
    const view = doc.bufferViews[acc.bufferView];
    const start = (view.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    if (acc.componentType === 5123) {
      return new Uint16Array(bin, start, acc.count);
    }
    if (acc.componentType === 5125) {
      return new Uint32Array(bin, start, acc.count);
    }
    throw new Error(`${url}: expected 16- or 32-bit indices`);
  }

  const position: number[] = [];
  const normal: number[] = [];
  const tint: number[] = [];

  function bakeNode(index: number, parent: Mat): void {
    const node = doc.nodes[index];
    const world = mulMat(parent, trsMatrix(node));
    if (node.mesh !== undefined) {
      for (const prim of doc.meshes[node.mesh].primitives) {
        const material = doc.materials[prim.material];
        if (spec.skip?.includes(material.name ?? '')) {
          continue;
        }
        // Base colour is linear; the compositor works in display-ish space,
        // so gamma the bake or every prop comes out near black.
        const base = material.pbrMetallicRoughness?.baseColorFactor ?? [1, 1, 1, 1];
        const shown = base.slice(0, 3).map((c) => c ** (1 / 2.2));
        const pos = floats(prim.attributes.POSITION, 3);
        const norm = floats(prim.attributes.NORMAL, 3);
        for (const vertex of indices(prim.indices)) {
          const [x, y, z] = pos.subarray(vertex * 3, vertex * 3 + 3);
          position.push(
            world[0] * x + world[1] * y + world[2] * z + world[3],
            world[4] * x + world[5] * y + world[6] * z + world[7],
            world[8] * x + world[9] * y + world[10] * z + world[11],
          );
          const [nx, ny, nz] = norm.subarray(vertex * 3, vertex * 3 + 3);
          // Uniform node scales, so the upper 3x3 plus a normalise is exact.
          const wx = world[0] * nx + world[1] * ny + world[2] * nz;
          const wy = world[4] * nx + world[5] * ny + world[6] * nz;
          const wz = world[8] * nx + world[9] * ny + world[10] * nz;
          const len = Math.hypot(wx, wy, wz) || 1;
          normal.push(wx / len, wy / len, wz / len);
          tint.push(shown[0], shown[1], shown[2]);
        }
      }
    }
    for (const child of node.children ?? []) {
      bakeNode(child, world);
    }
  }

  const identity = trsMatrix({});
  for (const rootIndex of doc.scenes[doc.scene ?? 0].nodes) {
    bakeNode(rootIndex, identity);
  }

  // Normalise into field units: anchor point at the origin, y up, sized so a
  // prop at reach = 1 has the extent the spec asks for.
  const bound = { lo: [Infinity, Infinity, Infinity], hi: [-Infinity, -Infinity, -Infinity] };
  for (let i = 0; i < position.length; i += 3) {
    for (let axis = 0; axis < 3; axis++) {
      bound.lo[axis] = Math.min(bound.lo[axis], position[i + axis]);
      bound.hi[axis] = Math.max(bound.hi[axis], position[i + axis]);
    }
  }
  const height = bound.hi[1] - bound.lo[1];
  const width = Math.max(bound.hi[0] - bound.lo[0], bound.hi[2] - bound.lo[2]);
  const scale = spec.size / (spec.fit === 'height' ? height : width);
  const centre = [
    (bound.lo[0] + bound.hi[0]) / 2,
    bound.lo[1] + spec.anchor * height,
    (bound.lo[2] + bound.hi[2]) / 2,
  ];
  const packed = new Float32Array(position.length);
  for (let i = 0; i < position.length; i += 3) {
    packed[i] = (position[i] - centre[0]) * scale;
    packed[i + 1] = (position[i + 1] - centre[1]) * scale;
    packed[i + 2] = (position[i + 2] - centre[2]) * scale;
  }

  return {
    vertexCount: packed.length / 3,
    position: packed,
    normal: Float32Array.from(normal),
    tint: Float32Array.from(tint),
  };
}
