import { d } from 'typegpu';

export interface CameraGeometry {
  readonly uvTransform: d.m2x2f;
  /** Whether the transform exchanges the width and height axes. */
  readonly swapAxes: boolean;
}

/** Quarter turns applied to the incoming frame. */
export const ROTATIONS = [0, 90, 180, 270] as const;
export type Rotation = (typeof ROTATIONS)[number];

const UPRIGHT: CameraGeometry = { uvTransform: d.mat2x2f.identity(), swapAxes: false };

/**
 * A quarter-turn about the centre of the frame. The preprocessor applies this as
 * `transform * (uv - 0.5) + 0.5`, so these are plain rotation matrices; 90 and
 * 270 also exchange the width and height axes, which the crop needs to know.
 */
export function rotationGeometry(rotation: Rotation): CameraGeometry {
  if (rotation === 90) {
    return { uvTransform: d.mat2x2f(0, -1, 1, 0), swapAxes: true };
  }
  if (rotation === 180) {
    return { uvTransform: d.mat2x2f(-1, 0, 0, -1), swapAxes: false };
  }
  if (rotation === 270) {
    return { uvTransform: d.mat2x2f(0, 1, -1, 0), swapAxes: true };
  }
  return UPRIGHT;
}

/**
 * How far the frame needs turning to come out upright.
 *
 * Cameras report frames in the sensor's own orientation, and only some platforms
 * rotate them for you. Rather than special-casing vendors, this compares what the
 * track actually delivers against how the device is held:
 *
 *  - A track that is portrait-shaped while the screen is landscape (or the other
 *    way round) is arriving sideways, so it needs a quarter turn.
 *  - Which quarter turn depends on the screen angle, which is also what tells us
 *    whether the device is upside down.
 *
 * On a desktop webcam both checks are no-ops and this returns 0, which is right.
 */
export function detectRotation(track: MediaStreamTrack | undefined): Rotation {
  const angle = screen.orientation?.angle ?? 0;
  const settings = track?.getSettings();
  const width = settings?.width ?? 0;
  const height = settings?.height ?? 0;

  const screenIsPortrait = globalThis.innerHeight >= globalThis.innerWidth;
  const trackIsPortrait = height > width;
  const sideways = width > 0 && height > 0 && trackIsPortrait !== screenIsPortrait;

  if (!sideways) {
    return angle === 180 ? 180 : 0;
  }
  return angle === 270 || angle === -90 ? 270 : 90;
}

export type CameraFacing = 'user' | 'environment';

export interface CameraSession {
  readonly active: boolean;
  facing: CameraFacing;
  start(): Promise<void>;
  stop(): void;
  /** The live element, or undefined when nothing is streaming. */
  source(): HTMLVideoElement | undefined;
  /** The live video track, for reading what the camera is actually sending. */
  track(): MediaStreamTrack | undefined;
}

export function createCameraSession(video: HTMLVideoElement): CameraSession {
  let stream: MediaStream | undefined;
  let facing: CameraFacing = 'user';

  function release(): void {
    for (const track of stream?.getTracks() ?? []) {
      track.stop();
    }
    stream = undefined;
    video.pause();
    video.srcObject = null;
  }

  return {
    get active() {
      return stream !== undefined;
    },

    get facing() {
      return facing;
    },

    set facing(next: CameraFacing) {
      facing = next;
    },

    async start() {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera capture needs a secure context and a browser that allows it.');
      }
      release();
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: facing },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      video.srcObject = stream;
      await video.play().catch(() => undefined);
    },

    stop() {
      release();
    },

    source() {
      return stream && video.readyState >= 2 ? video : undefined;
    },

    track() {
      return stream?.getVideoTracks()[0];
    },
  };
}
