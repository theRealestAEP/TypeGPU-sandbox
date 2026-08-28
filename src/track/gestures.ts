import { FaceLandmarker, FilesetResolver, HandLandmarker, ObjectDetector } from '@mediapipe/tasks-vision';

/**
 * Face and hand tracking over the live video, in the field's own coordinates.
 *
 * MediaPipe reports landmarks in the source frame's normalised space. The depth
 * pipeline sees that frame mirrored, centre-cropped square, and possibly
 * quarter-turned, so every landmark is pushed through the inverse of exactly
 * that chain before anyone downstream sees it. Get this wrong and a tear runs
 * down a cheek that is three centimetres left of the face.
 *
 * The models come from the MediaPipe CDN the first time and live in the browser
 * cache after that, the same arrangement as the depth model.
 */

const WASM_ROOT =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const FACE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const HAND_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const DETECT_MODEL =
  'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite';

/** COCO classes that mean "a thing you could pour into". */
const VESSEL_CLASSES = new Set(['cup', 'wine glass', 'bowl', 'vase']);
/** Runs of the detector are throttled; boxes move slowly compared to hands. */
const DETECT_EVERY_MS = 160;

/** How the source frame maps into the field; mirrors the depth preprocessor. */
export interface FrameFit {
  readonly mirror: boolean;
  readonly swapAxes: boolean;
  /** Row-major 2x2, as the rotation matrices in camera.ts. */
  readonly uv: readonly [number, number, number, number];
}

export interface TrackedPoint {
  readonly x: number;
  readonly y: number;
}

export interface HeldVessel {
  /** Box in field coordinates. */
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  label: string;
}

export interface Tracked {
  /** Cups, glasses and bowls the detector currently sees. */
  readonly vessels: HeldVessel[];
  /** Index fingertip, when a hand is in frame. */
  readonly tip?: TrackedPoint;
  /** Thumb tip, for the pinch gesture. */
  readonly thumb?: TrackedPoint;
  /** Sweat springs: high on the forehead and at the temples. */
  readonly brow: TrackedPoint[];
  /** Tear ducts: just under each eye. */
  readonly eyes: TrackedPoint[];
  /** Iris centres, for anything worn on the eyes. */
  readonly lenses: TrackedPoint[];
  /** 0..1, how furrowed the brow is; scales the sweat. */
  readonly effort: number;
}

/** Forehead centre, left temple, right temple. */
const BROW_MARKS = [10, 103, 332];
/** Under each lower eyelid, where a tear would gather. */
const EYE_MARKS = [153, 380];

export interface Gestures {
  /** Runs both trackers against the video's current frame. */
  read(video: HTMLVideoElement, fit: FrameFit, now: number): Tracked;
  /** Runs only the vessel detector, for still photos. */
  readStill(image: HTMLCanvasElement, fit: FrameFit, now: number): Tracked;
  destroy(): void;
}

export async function createGestures(): Promise<Gestures> {
  const files = await FilesetResolver.forVisionTasks(WASM_ROOT);
  const [face, hands, finder] = await Promise.all([
    FaceLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      // oxlint-disable-next-line anti-slop/no-shape-in-symbol-names -- MediaPipe's API name
      outputFaceBlendshapes: true,
    }),
    HandLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numHands: 1,
    }),
    ObjectDetector.createFromOptions(files, {
      // CPU on purpose: with three tasks sharing one context, the GPU-delegate
      // detector returned the same hallucination for every frame - it never
      // saw the video at all. Lite0-int8 is built for CPU and runs in tens of
      // milliseconds at the 6Hz this needs.
      baseOptions: { modelAssetPath: DETECT_MODEL, delegate: 'CPU' },
      runningMode: 'VIDEO',
      // Low on purpose: a glass held close to the camera scores in the 0.2
      // band, and the persistence rule below is what keeps the noise out -
      // a detection only becomes a vessel after three consecutive rounds.
      scoreThreshold: 0.2,
      maxResults: 4,
    }),
  ]);

  let lastVideoTime = -1;
  let lastDetectAt = -1e9;
  /** Candidates carry how long they have persisted and how long been missing. */
  let candidates: (HeldVessel & { streak: number; misses: number })[] = [];
  let heldVessels: HeldVessel[] = [];
  let held: Tracked = { vessels: [], brow: [], eyes: [], lenses: [], effort: 0 };

  /** Source-frame normalised coords to field coords: the preprocessor, undone. */
  function toField(
    point: { x: number; y: number },
    fit: FrameFit,
    width: number,
    height: number,
  ): TrackedPoint {
    // Undo the quarter turn.
    const [m00, m01, m10, m11] = fit.uv;
    const cx = point.x - 0.5;
    const cy = point.y - 0.5;
    // The rotation matrices are orthonormal, so the inverse is the transpose.
    const rx = m00 * cx + m10 * cy + 0.5;
    const ry = m01 * cx + m11 * cy + 0.5;
    // Undo the centre crop.
    const sw = fit.swapAxes ? height : width;
    const sh = fit.swapAxes ? width : height;
    const scaleX = Math.min(sw, sh) / sw;
    const scaleY = Math.min(sw, sh) / sh;
    let x = (rx - (1 - scaleX) * 0.5) / scaleX;
    const y = (ry - (1 - scaleY) * 0.5) / scaleY;
    // Undo the mirror.
    if (fit.mirror) {
      x = 1 - x;
    }
    return { x, y };
  }

  /**
   * The vessel detector, shared by video frames and still photos. It runs a
   * few times a second; a held cup moves at hand speed, and the persistence
   * smoothing absorbs the gap.
   */
  function findVessels(
    source: HTMLVideoElement | HTMLCanvasElement,
    w: number,
    h: number,
    fit: FrameFit,
    now: number,
  ): void {
    if (now - lastDetectAt < DETECT_EVERY_MS) {
      return;
    }
    lastDetectAt = now;
    const found = finder.detectForVideo(source, now);

      const fresh: HeldVessel[] = [];
      for (const detection of found.detections) {
        const category = detection.categories[0];
        const box = detection.boundingBox;
        if (!category || !box || !VESSEL_CLASSES.has(category.categoryName)) {
          continue;
        }
        const a = toField({ x: box.originX / w, y: box.originY / h }, fit, w, h);
        const b = toField(
          { x: (box.originX + box.width) / w, y: (box.originY + box.height) / h },
          fit,
          w,
          h,
        );
        fresh.push({
          x0: Math.min(a.x, b.x),
          y0: Math.min(a.y, b.y),
          x1: Math.max(a.x, b.x),
          y1: Math.max(a.y, b.y),
          label: category.categoryName,
        });
      }
      // Persistence over confidence. The threshold above is low enough to
      // let genuine close-up glasses through, so single-round noise gets in
      // too; requiring three consecutive rounds keeps phantom cups from
      // carving cavities into the scene. The other direction gets grace as
      // well - a vessel survives two missed rounds - because every
      // appear/disappear flap re-digs the carve, and that quake throws the
      // water it held. Matched boxes ease toward the new detection.
      const matched = new Set<number>();
      for (const candidate of candidates) {
        const index = fresh.findIndex(
          (next, i) =>
            !matched.has(i) &&
            Math.abs(candidate.x0 - next.x0) < 0.15 &&
            Math.abs(candidate.y0 - next.y0) < 0.15,
        );
        if (index >= 0) {
          matched.add(index);
          const next = fresh[index];
          // Asymmetric on purpose: growing edges follow the detection at
          // full speed, shrinking edges crawl. A box that breathes sheds
          // water - every upward jitter of the bottom edge strands the base
          // layer outside the container for a frame, and on live footage
          // that was a steady bottom leak. A held glass's true extent almost
          // never shrinks quickly; jitter does.
          const ease = (current: number, target: number, grow: boolean) =>
            current + (target - current) * ((target > current) === grow ? 0.55 : 0.12);
          candidate.x0 = ease(candidate.x0, next.x0, false);
          candidate.y0 = ease(candidate.y0, next.y0, false);
          candidate.x1 = ease(candidate.x1, next.x1, true);
          candidate.y1 = ease(candidate.y1, next.y1, true);
          candidate.label = next.label;
          candidate.streak += 1;
          candidate.misses = 0;
        } else {
          candidate.misses += 1;
        }
      }
      candidates = candidates.filter((candidate) => candidate.misses <= 2);
      fresh.forEach((next, i) => {
        if (!matched.has(i)) {
          candidates.push({ ...next, streak: 1, misses: 0 });
        }
      });
      heldVessels = candidates
        .filter((candidate) => candidate.streak >= 3)
        .map(({ x0, y0, x1, y1, label }) => ({ x0, y0, x1, y1, label }));
  }

  return {
    read(video, fit, now) {
      if (video.readyState < 2) {
        return held;
      }
      if (video.currentTime === lastVideoTime) {
        return held;
      }
      lastVideoTime = video.currentTime;
      const w = video.videoWidth;
      const h = video.videoHeight;
      if (w === 0 || h === 0) {
        return held;
      }

      const faces = face.detectForVideo(video, now);
      const palms = hands.detectForVideo(video, now);

      findVessels(video, w, h, fit, now);

      const map = (p: { x: number; y: number }) => toField(p, fit, w, h);
      let tip: TrackedPoint | undefined;
      let thumb: TrackedPoint | undefined;
      const palm = palms.landmarks[0];
      if (palm) {
        tip = map(palm[8]);
        thumb = map(palm[4]);
      }

      const brow: TrackedPoint[] = [];
      const eyes: TrackedPoint[] = [];
      const lenses: TrackedPoint[] = [];
      let effort = 0;
      const marks = faces.faceLandmarks[0];
      if (marks) {
        for (const index of BROW_MARKS) {
          brow.push(map(marks[index]));
        }
        for (const index of EYE_MARKS) {
          eyes.push(map(marks[index]));
        }
        // Iris centres arrive as landmarks 468 and 473 when the model refines
        // eyes, which this one does.
        if (marks.length > 473) {
          lenses.push(map(marks[468]), map(marks[473]));
        }
        // oxlint-disable-next-line anti-slop/no-shape-in-symbol-names -- MediaPipe's API name
        const blend = faces.faceBlendshapes[0];
        if (blend) {
          for (const category of blend.categories) {
            if (category.categoryName === 'browDownLeft' || category.categoryName === 'browDownRight') {
              effort = Math.max(effort, category.score);
            }
          }
        }
      }

      held = { vessels: heldVessels, tip, thumb, brow, eyes, lenses, effort };
      return held;
    },

    readStill(image, fit, now) {
      findVessels(image, image.width, image.height, fit, now);
      held = { vessels: heldVessels, brow: [], eyes: [], lenses: [], effort: 0 };
      return held;
    },

    destroy() {
      face.close();
      hands.close();
      finder.close();
    },
  };
}
