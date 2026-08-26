import type { d, TgpuComputePass } from 'typegpu';

export interface CameraFrame {
  readonly texture: GPUExternalTexture;
  readonly uvTransform: d.m2x2f;
  readonly swapAxes: boolean;
  /** Front-facing capture is mirrored here, once, and never again downstream. */
  readonly mirror: boolean;
}

/**
 * Fills the shared depth texture: 0 is far, 1 is nearest. Two implementations,
 * so the fluid stays developable with no camera and no model download.
 */
export interface DepthSource {
  /** Shortest gap between refreshes. Inference is expensive; the scene is not. */
  readonly minIntervalMs: number;
  initAsync(): Promise<void>;
  encode(pass: TgpuComputePass, frame: CameraFrame | undefined): void;
  destroy(): void;
}
