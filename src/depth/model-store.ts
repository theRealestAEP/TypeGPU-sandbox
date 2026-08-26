/**
 * DepthART bundle fetching, lifted from the TypeGPU monocular-light-injection
 * example so the vendored runtime gets the weights it expects.
 */
export const MODEL_SIZES = ['small', 'base', 'large'] as const;
export type ModelSize = (typeof MODEL_SIZES)[number];

export interface ModelVariant {
  readonly bundle: string;
  readonly megabytes: number;
}

interface ModelPair {
  readonly fp16: ModelVariant;
  /** Absent where the bundle is only published in half precision. */
  readonly f32: ModelVariant | undefined;
}

const MODEL_VARIANTS = {
  small: {
    fp16: { bundle: 'depthart-relative-s-448-balanced', megabytes: 13 },
    f32: { bundle: 'depthart-relative-s-448-f32', megabytes: 23 },
  },
  base: {
    fp16: { bundle: 'depthart-relative-b-448-balanced', megabytes: 24 },
    f32: { bundle: 'depthart-relative-b-448-f32', megabytes: 43 },
  },
  large: {
    fp16: { bundle: 'depthart-relative-l-448-balanced', megabytes: 68 },
    f32: undefined,
  },
} satisfies Record<ModelSize, ModelPair>;

export const RECOMMENDED_MODEL: ModelSize = 'small';

const MODEL_HOST =
  'https://huggingface.co/reczkok/depthart-typegpu/resolve/913a7c13ddfbd48549279555d1db98172e8e5e0d';
const MODEL_CACHE = 'depthart-models';

export function modelVariant(size: ModelSize, hasShaderF16: boolean): ModelVariant | undefined {
  const variants = MODEL_VARIANTS[size];
  return hasShaderF16 ? variants.fp16 : variants.f32;
}

export function modelLabel(size: ModelSize, variant: ModelVariant): string {
  return `${size} - ${variant.megabytes} MB`;
}

export async function fetchModel(variant: ModelVariant, signal: AbortSignal): Promise<ArrayBuffer> {
  const url = `${MODEL_HOST}/${variant.bundle}.depthart`;

  let store: Cache | undefined;
  try {
    store = await caches.open(MODEL_CACHE);
    const hit = await store.match(url);
    if (hit) {
      return await hit.arrayBuffer();
    }
  } catch {
    store = undefined;
  }

  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`Model download failed (${response.status}).`);
  }
  if (store) {
    const bytes = await response.clone().arrayBuffer();
    await store.put(url, response).catch(() => undefined);
    return bytes;
  }
  return await response.arrayBuffer();
}
