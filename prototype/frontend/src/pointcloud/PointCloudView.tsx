import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ConvexGeometry } from "three/examples/jsm/geometries/ConvexGeometry.js";
import type { PointPayload } from "../api/types";

export type PointInfo = {
  index: number;
  eventId: number;
  x: number;
  y: number;
  z: number;
  tUs: number;
  polarity: number;
  label: number;
  screenX: number;
  screenY: number;
};

export type PointCloudHandle = {
  focusOnPoint: (pointIndex: number) => void;
};

export type ToolMode = "box" | "circle" | "lasso" | "pan" | "wand" | "tscrub";
export type SelectionOp = "set" | "add" | "subtract";

// Live (pre-Load) Ranges & Filters draft, in raw sensor units — drawn as a
// translucent yellow region box in 3D so the user can see what will be kept.
export type FilterPreview = { xMin: number; xMax: number; yMin: number; yMax: number; startUs: number; endUs: number };

type Props = {
  payload: PointPayload | null;
  // Multi-points props (will replace `payload` once refactor lands).
  manifest?: import("../api/types").ChunkManifest | null;
  chunksMap?: Map<number, PointPayload>;
  visibleChunkIds?: number[];
  viewMode: "3d" | "xy";
  selectedIds: Set<number>;
  gpuPreference: "high-performance" | "low-power" | "default";
  theme: "light" | "dark";
  labels: { id: number; name: string; color: string }[];
  meta?: { width: number; height: number; t_min_us: number; t_max_us: number };
  toolMode?: ToolMode;
  brushRadius?: number;
  selectThrough?: boolean;
  xyRotation?: number; // 0/90/180/270 degrees, only applied in xy mode
  xyFlipY?: boolean;   // mirror across horizontal axis in xy mode
  xyFlipX?: boolean;   // mirror across vertical axis in xy mode
  currentFrameUs?: number;
  frameWindowUs?: number;
  onPointClick?: (info: PointInfo | null) => void;
  onSelection?: (indices: number[], op: SelectionOp) => void;
  /** Wand click — parent looks up the seed event's connected component on
   *  the backend and pushes the resulting ids into the selection set. */
  onWandSeed?: (seedEventId: number, op: SelectionOp) => void;
  onBrushRadiusChange?: (r: number) => void;
  tooltipRef?: React.RefObject<HTMLDivElement | null>;
  handleRef?: React.RefObject<PointCloudHandle | null>;
  /** Optimistic label edit: { chunkId → changed local indices }. Bumping `seq`
   *  triggers an incremental in-place recolour of just those points — no
   *  geometry rebuild, no chunk re-fetch. */
  labelPatch?: { seq: number; perChunk: Map<number, number[]> } | null;
  /** Display aids for reading the cloud. colorMode "time" colours unlabelled
   *  events by timestamp (motion becomes visible); pointSizeScale multiplies the
   *  rendered point size; window[Start/End]Us is the time range the gradient
   *  normalises over (the loaded window). */
  colorMode?: "polarity" | "time";
  pointSizeScale?: number;
  windowStartUs?: number;
  windowEndUs?: number;
  /** Polarity colour scheme config: base colour for unlabelled points (hex) and
   *  the brightness contrast (0..1) applied per polarity. Wired to setPolarityDisplay. */
  unlabeledColor?: string;
  polarityContrast?: number;
  /** Ranges & Filters: events outside these bounds (pixel x/y, polarity, and the
   *  windowStart/EndUs time range) are culled from the render. Defaults are the
   *  full sensor / "all", i.e. no filtering. */
  filterXMin?: number;
  filterXMax?: number;
  filterYMin?: number;
  filterYMax?: number;
  filterPolarity?: string;
  /** Live preview (pre-Load) of the Ranges & Filters draft region, drawn as a
   *  translucent yellow box in 3D. null = nothing to preview. */
  filterPreview?: FilterPreview | null;
  /** Time-scrub on the T axis (3D): reports the time under the cursor + the
   *  screen point to anchor a preview, or null when the scrub ends. */
  onTimeScrub?: (info: { tUs: number; sx: number; sy: number } | null) => void;
  /** Preview-frame orientation, so the red corner marker in the scrub plane
   *  follows the preview's current displayed top-left. */
  previewRot?: number;          // 0 / 90 / 180 / 270
  previewFlipX?: boolean;
  previewFlipY?: boolean;
};

// Colors are kept as uint8 (normalised) to cut GPU upload to ¼ vs Float32.
// Critical for 8M+ event datasets where the upload otherwise blocks the UI.
type CloudBuffers = { positions: Float32Array; colors: Uint8Array; times: Float32Array; count: number; norm: NormTransform };

/** Per-chunk renderable. Each loaded chunk owns its own GPU geometry so we
 *  can dispose evicted chunks independently without tearing down the whole
 *  point cloud. The `payload` reference is kept alongside for fast picker
 *  + tooltip lookup (no extra copy of the data). */
type ChunkRender = {
  chunkId: number;
  eventStartIndex: number;        // global event_id of payload[0]
  eventCount: number;
  payload: PointPayload;          // raw arrays (shared with chunksMap)
  colors: Uint8Array;              // owned per-chunk
  geometry: THREE.BufferGeometry; // owned per-chunk
  points: THREE.Points;
  // When a Ranges & Filters filter is active, a copy of payload.positions with
  // culled points set to NaN (which WebGL discards). null when no filter active
  // (the geometry then uses payload.positions directly — no extra allocation).
  displayPositions?: Float32Array;
};

// Ranges & Filters criteria, in raw sensor units (pixel x/y, time µs).
type PointFilter = { xMin: number; xMax: number; yMin: number; yMax: number; pol: string; tMin: number; tMax: number };

// Does the filter actually exclude anything? (Avoids the per-point scan + the
// positions copy in the common "no filter" case.) Time is only "active" when the
// requested window is narrower than the dataset's real t range.
function filterIsActive(f: PointFilter, width: number, height: number, dataTMin: number, dataTMax: number): boolean {
  if (f.pol === "on" || f.pol === "off") return true;
  if (f.xMin > 0 || (width > 0 && f.xMax < width - 1)) return true;
  if (f.yMin > 0 || (height > 0 && f.yMax < height - 1)) return true;
  if (f.tMin > dataTMin || f.tMax < dataTMax) return true;
  return false;
}

// Apply (or clear) the filter on one chunk by swapping its geometry position
// attribute between the canonical payload.positions and a NaN-culled copy.
// payload.positions is never mutated, so picking/selection still see real coords.
function applyChunkFilter(render: ChunkRender, f: PointFilter, width: number, height: number, active: boolean): void {
  const geom = render.geometry;
  const src = render.payload.positions;
  if (!active) {
    if (render.displayPositions) {
      render.displayPositions = undefined;
      geom.setAttribute("position", new THREE.BufferAttribute(src, 3));
      (geom.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    }
    return;
  }
  let disp = render.displayPositions;
  if (!disp || disp.length !== src.length) {
    disp = new Float32Array(src.length);
    render.displayPositions = disp;
    geom.setAttribute("position", new THREE.BufferAttribute(disp, 3));
  }
  const cx = (width - 1) / 2, cy = (height - 1) / 2;
  const { xMin, xMax, yMin, yMax, pol, tMin, tMax } = f;
  const tarr = render.payload.t_us, parr = render.payload.polarity;
  for (let i = 0; i < render.payload.count; i++) {
    const o = i * 3;
    const sx = src[o], sy = src[o + 1];
    const px = sx + cx, py = cy - sy;
    const t = tarr[i];
    const on = !!parr[i];
    const pass =
      px >= xMin && px <= xMax && py >= yMin && py <= yMax &&
      t >= tMin && t <= tMax &&
      (pol === "all" || (pol === "on" ? on : !on));
    if (pass) { disp[o] = sx; disp[o + 1] = sy; disp[o + 2] = src[o + 2]; }
    else { disp[o] = NaN; disp[o + 1] = NaN; disp[o + 2] = NaN; }
  }
  (geom.attributes.position as THREE.BufferAttribute).needsUpdate = true;
}

type Bounds3 = { minX: number; maxX: number; minY: number; maxY: number; minZ: number; maxZ: number };

type NormTransform = { cx: number; cy: number; cz: number; scale: number; bounds: Bounds3; flipY?: boolean; flipX?: boolean };

// Map a normalised time fraction [0,1] → a 5-stop blue→cyan→green→yellow→red
// ramp (0–255). In "time" colour mode unlabelled events are coloured by their
// timestamp, so the temporal axis — i.e. motion over time — becomes directly
// visible in the 3D view instead of the structure-less polarity wash.
function timeGradient8(frac: number): [number, number, number] {
  const f = frac < 0 ? 0 : frac > 1 ? 1 : frac;
  const stops: [number, number, number][] = [
    [40, 70, 210],   // early  — blue
    [40, 190, 220],  // cyan
    [70, 200, 90],   // green
    [240, 210, 50],  // yellow
    [230, 60, 40],   // late   — red
  ];
  const seg = f * (stops.length - 1);
  const i = Math.min(stops.length - 2, seg | 0);
  const tt = seg - i;
  const a = stops[i], b = stops[i + 1];
  return [
    (a[0] + (b[0] - a[0]) * tt) | 0,
    (a[1] + (b[1] - a[1]) * tt) | 0,
    (a[2] + (b[2] - a[2]) * tt) | 0,
  ];
}

/** Fill an existing Uint8 colour buffer for one chunk. Rules: selected → white,
 *  label colour wins over everything else; otherwise either a time-gradient
 *  (colorMode="time", normalised over [tMin,tMax]) or the polarity ON/OFF
 *  colour. Pre-converts the label colour map to 0–255 ints once. */
function buildChunkColors(
  payload: PointPayload,
  eventStartIndex: number,
  selectedIds: Set<number>,
  labels: { id: number; name: string; color: string }[],
  out?: Uint8Array,
  colorMode: "polarity" | "time" = "polarity",
  tMin = 0,
  tMax = 1,
): Uint8Array {
  const N = payload.count;
  const tSpan = tMax > tMin ? tMax - tMin : 1;
  const colors = out && out.length === N * 3 ? out : new Uint8Array(N * 3);
  const labelLut = new Map<number, [number, number, number]>();
  for (const l of labels) {
    const c = hexToRGB(l.color);
    labelLut.set(l.id, [(c[0] * 255) | 0, (c[1] * 255) | 0, (c[2] * 255) | 0]);
  }
  const checkSel = selectedIds.size > 0;
  for (let i = 0; i < N; i++) {
    const o = i * 3;
    if (checkSel && selectedIds.has(eventStartIndex + i)) {
      colors[o] = 255; colors[o + 1] = 255; colors[o + 2] = 255;
      continue;
    }
    const labelId = payload.label[i];
    const lc = labelLut.get(labelId);
    if (colorMode === "time") {
      // Time mode unchanged: labelled → label colour, unlabelled → time gradient.
      if (lc !== undefined) {
        colors[o] = lc[0]; colors[o + 1] = lc[1]; colors[o + 2] = lc[2];
      } else {
        const g = timeGradient8((payload.t_us[i] - tMin) / tSpan);
        colors[o] = g[0]; colors[o + 1] = g[1]; colors[o + 2] = g[2];
      }
    } else {
      // Polarity mode: base = label colour if labelled else the unlabelled
      // colour, then modulate brightness by polarity for ALL points.
      const base = lc !== undefined ? lc : UNLABELED8;
      const on = !!payload.polarity[i];
      colors[o] = polarityModulate(base[0], on);
      colors[o + 1] = polarityModulate(base[1], on);
      colors[o + 2] = polarityModulate(base[2], on);
    }
  }
  return colors;
}

/** Build an id → uint8 RGB lookup from the label schema (once per recolour). */
function buildLabelLut(labels: { id: number; color: string }[]): Map<number, [number, number, number]> {
  const lut = new Map<number, [number, number, number]>();
  for (const l of labels) {
    const c = hexToRGB(l.color);
    lut.set(l.id, [(c[0] * 255) | 0, (c[1] * 255) | 0, (c[2] * 255) | 0]);
  }
  return lut;
}

/** Recolour only the given local indices of one chunk, in place. Same rules as
 *  buildChunkColors (selected → white, label wins over polarity) but touches a
 *  handful of points instead of rescanning the whole chunk — this is what makes
 *  Assign/selection feel instant on multi-million-point clouds. */
function recolorChunkIndices(
  payload: PointPayload,
  eventStartIndex: number,
  localIdxs: Iterable<number>,
  selectedIds: Set<number>,
  labelLut: Map<number, [number, number, number]>,
  colors: Uint8Array,
  colorMode: "polarity" | "time" = "polarity",
  tMin = 0,
  tMax = 1,
): void {
  const checkSel = selectedIds.size > 0;
  const tSpan = tMax > tMin ? tMax - tMin : 1;
  for (const i of localIdxs) {
    const o = i * 3;
    if (checkSel && selectedIds.has(eventStartIndex + i)) {
      colors[o] = 255; colors[o + 1] = 255; colors[o + 2] = 255; continue;
    }
    const lc = labelLut.get(payload.label[i]);
    if (colorMode === "time") {
      // Time mode unchanged.
      if (lc !== undefined) { colors[o] = lc[0]; colors[o + 1] = lc[1]; colors[o + 2] = lc[2]; }
      else { const g = timeGradient8((payload.t_us[i] - tMin) / tSpan); colors[o] = g[0]; colors[o + 1] = g[1]; colors[o + 2] = g[2]; }
    } else {
      // Polarity mode: label colour (or unlabelled colour) modulated by polarity.
      const base = lc !== undefined ? lc : UNLABELED8;
      const on = !!payload.polarity[i];
      colors[o] = polarityModulate(base[0], on);
      colors[o + 1] = polarityModulate(base[1], on);
      colors[o + 2] = polarityModulate(base[2], on);
    }
  }
}

/** Merge sorted indices into contiguous [start,count] runs. */
function coalesceIndices(sorted: number[]): { start: number; count: number }[] {
  if (sorted.length === 0) return [];
  const ranges: { start: number; count: number }[] = [];
  let start = sorted[0], prev = start;
  for (let i = 1; i < sorted.length; i++) {
    const v = sorted[i];
    if (v <= prev + 1) { prev = v; continue; }
    ranges.push({ start, count: prev - start + 1 });
    start = prev = v;
  }
  ranges.push({ start, count: prev - start + 1 });
  return ranges;
}

/** Upload only the changed colour ranges (gl.bufferSubData via Three's
 *  addUpdateRange). For a large or very fragmented change set, fall back to a
 *  single full-attribute upload (thresholds per the perf README §6.1). */
function uploadChangedColors(attribute: THREE.BufferAttribute, localIdxs: number[], pointCount: number): void {
  if (localIdxs.length === 0) return;
  const attr = attribute as THREE.BufferAttribute & {
    addUpdateRange?: (start: number, count: number) => void;
    clearUpdateRanges?: () => void;
  };
  const frac = localIdxs.length / Math.max(1, pointCount);
  const sorted = localIdxs.slice().sort((a, b) => a - b);
  const ranges = coalesceIndices(sorted);
  if (frac > 0.2 || ranges.length > 256 || typeof attr.addUpdateRange !== "function") {
    attribute.needsUpdate = true; // empty updateRanges + needsUpdate ⇒ full upload
    return;
  }
  attr.clearUpdateRanges?.();
  for (const r of ranges) attr.addUpdateRange(r.start * 3, r.count * 3);
  attribute.needsUpdate = true;
}

/**
 * Build a single global norm from the dataset manifest. All chunks share this
 * transform so per-chunk geometries align in world space without anyone having
 * to allocate a combined payload to recompute bounds.
 *
 * Coordinate convention matches `coordinate_positions` in backend/sampling.py:
 *   - x_world = x_pixel - centerX
 *   - y_world = centerY - y_pixel  (Y flipped so y_pixel=0 is +Y world)
 *   - z_world = (t_us - t_min) * (sensor_width / duration_us) - half_z_span
 *
 * The manifest gives us sensor width/height + total t range; that's everything
 * we need to mirror those formulas in `bounds`.
 */
function computeManifestNorm(
  manifest: { width: number; height: number; t_min_us: number; t_max_us: number },
  flipY = false,
  flipX = false,
): NormTransform {
  const w = Math.max(1, manifest.width);
  const h = Math.max(1, manifest.height);
  const cxPix = (w - 1) / 2;
  const cyPix = (h - 1) / 2;
  // Backend uses time_scale = width / duration_us so total z_world span = width.
  // The bounds of payload.positions[:,2] thus span [-width/2, +width/2] (centred).
  const zHalf = w / 2;
  const bounds: Bounds3 = {
    minX: -cxPix, maxX: cxPix,
    minY: -cyPix, maxY: cyPix,
    minZ: -zHalf, maxZ: zHalf,
  };
  const cx = (bounds.minX + bounds.maxX) / 2;
  const cy = (bounds.minY + bounds.maxY) / 2;
  const cz = (bounds.minZ + bounds.maxZ) / 2;
  const scale = 2 / Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, bounds.maxZ - bounds.minZ, 1);
  return { cx, cy, cz, scale, bounds, flipY, flipX };
}

function hexToRGB(hex: string): [number, number, number] {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16 & 0xff) / 255, (n >> 8 & 0xff) / 255, (n & 0xff) / 255];
}

// Match the backend's ON / OFF polarity colors (config.py: (255,100,40) and
// (40,140,255)) so the live XY view looks like the rendered video.
const POLARITY_ON_COLOR: [number, number, number] = [255 / 255, 100 / 255, 40 / 255];
const POLARITY_OFF_COLOR: [number, number, number] = [40 / 255, 140 / 255, 255 / 255];

// Pre-converted 0–255 polarity colours for the incremental recolour helpers.
const ON8: [number, number, number] = [(POLARITY_ON_COLOR[0] * 255) | 0, (POLARITY_ON_COLOR[1] * 255) | 0, (POLARITY_ON_COLOR[2] * 255) | 0];
const OFF8: [number, number, number] = [(POLARITY_OFF_COLOR[0] * 255) | 0, (POLARITY_OFF_COLOR[1] * 255) | 0, (POLARITY_OFF_COLOR[2] * 255) | 0];

// Configurable polarity-display state (the "polarity" colour scheme). The base
// colour of each point is its label colour if labelled, else UNLABELED8; that
// base is then brightened (ON) or darkened (OFF) by POL_CONTRAST. Mutated via
// setPolarityDisplay from the host component when the user changes settings.
let UNLABELED8: [number, number, number] = [136, 136, 136];
let POL_CONTRAST = 0.5;
export function setPolarityDisplay(unlabeledHex: string, contrast: number) {
  const c = hexToRGB(unlabeledHex);
  UNLABELED8 = [(c[0] * 255) | 0, (c[1] * 255) | 0, (c[2] * 255) | 0];
  POL_CONTRAST = contrast;
}

/** Apply ON/OFF brightness modulation to a base 0–255 channel value:
 *  ON (polarity truthy) brightens toward white by POL_CONTRAST; OFF darkens
 *  toward black by POL_CONTRAST. Returns a clamped 0–255 int. */
function polarityModulate(c: number, on: boolean): number {
  const v = on ? c + (255 - c) * POL_CONTRAST : c * (1 - POL_CONTRAST);
  return v < 0 ? 0 : v > 255 ? 255 : v | 0;
}

function computeBounds(payload: PointPayload): Bounds3 {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < payload.count; i++) {
    const x = payload.positions[i * 3], y = payload.positions[i * 3 + 1], z = payload.positions[i * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  return Number.isFinite(minX) ? { minX, maxX, minY, maxY, minZ, maxZ } : { minX: -1, maxX: 1, minY: -1, maxY: 1, minZ: -1, maxZ: 1 };
}

/**
 * Async, chunked version of buildBuffers — yields the UI thread every CHUNK
 * events so a 16M-event load doesn't freeze the browser. The caller passes a
 * cancel flag so navigations during a long build abort cleanly.
 */
async function buildBuffersChunked(
  payload: PointPayload,
  viewMode: Props["viewMode"],
  selectedIds: Set<number>,
  labels: { id: number; color: string }[],
  flipY: boolean,
  isCancelled: () => boolean,
  onProgress?: (frac: number) => void,
): Promise<CloudBuffers | null> {
  const bounds = computeBounds(payload);
  const cx = (bounds.minX + bounds.maxX) / 2, cy = (bounds.minY + bounds.maxY) / 2, cz = (bounds.minZ + bounds.maxZ) / 2;
  const scale = 2 / Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, bounds.maxZ - bounds.minZ, 1);
  // Re-use the payload's raw positions and Float32 t_us directly — saves
  // ~250 MB JS heap on 16 M events. Shader does (position - centre) × scale
  // per vertex; t_us was already downcast to Float32 by the streaming parser.
  const positions = payload.positions;
  const times = payload.t_us;
  // Uint8 normalised colours – ¼ the size of Float32 and identical on-screen.
  const colors = new Uint8Array(payload.count * 3);

  const labelColorMap = new Map<number, [number, number, number]>();
  for (const l of labels) labelColorMap.set(l.id, hexToRGB(l.color));

  // Hoisted hot-loop invariants. Colour values pre-converted to 0–255 ints.
  const yMul = (viewMode === "xy" && flipY) ? -1 : 1;
  const checkSel = selectedIds.size > 0;
  const onR8 = (POLARITY_ON_COLOR[0] * 255) | 0, onG8 = (POLARITY_ON_COLOR[1] * 255) | 0, onB8 = (POLARITY_ON_COLOR[2] * 255) | 0;
  const offR8 = (POLARITY_OFF_COLOR[0] * 255) | 0, offG8 = (POLARITY_OFF_COLOR[1] * 255) | 0, offB8 = (POLARITY_OFF_COLOR[2] * 255) | 0;
  // Cache label colours as uint8 triples too.
  const labelColorMap8 = new Map<number, [number, number, number]>();
  for (const [id, c] of labelColorMap) labelColorMap8.set(id, [(c[0] * 255) | 0, (c[1] * 255) | 0, (c[2] * 255) | 0]);

  const CHUNK = 200_000;
  const total = payload.count;

  for (let s = 0; s < total; s += CHUNK) {
    if (isCancelled()) return null;
    const e = Math.min(s + CHUNK, total);
    for (let i = s; i < e; i++) {
      const o = i * 3;
      // positions and times are reused from payload — only colours need
      // computing per event in this loop.

      let sel = false;
      if (checkSel) sel = selectedIds.has(Number(payload.event_id[i]));
      if (sel) {
        colors[o] = 255; colors[o + 1] = 255; colors[o + 2] = 255;
      } else {
        const labelId = payload.label[i];
        const labelColor8 = labelColorMap8.get(labelId);
        if (labelColor8 !== undefined) {
          colors[o] = labelColor8[0];
          colors[o + 1] = labelColor8[1];
          colors[o + 2] = labelColor8[2];
        } else if (payload.polarity[i]) {
          colors[o] = onR8; colors[o + 1] = onG8; colors[o + 2] = onB8;
        } else {
          colors[o] = offR8; colors[o + 1] = offG8; colors[o + 2] = offB8;
        }
      }
    }
    onProgress?.(e / total);
    // Yield to the event loop so the browser can paint / handle input.
    await new Promise<void>((r) => setTimeout(r, 0));
  }

  return {
    positions, colors, times, count: total,
    norm: { cx, cy, cz, scale, bounds, flipY: yMul === -1 },
  };
}

const sphereVertexShader = `
attribute vec3 color;
attribute float aTime;
varying vec3 vColor;
varying float vIsXy;
uniform float uPointSize;
uniform float uSizeScale;    // user point-size multiplier (legibility)
uniform float uFrameStart;
uniform float uFrameEnd;
uniform float uUseFrameFilter;
uniform float uZoom;
uniform float uIsXy;        // 1.0 in XY mode → render as image-pixel-sized squares
uniform float uXyPxScale;   // framebuffer pixels per image pixel at zoom=1
uniform vec3  uPosCenter;   // payload-space centre (cx, cy, cz)
uniform float uPosScale;    // payload→world scale factor
uniform float uYMul;        // ±1 for vertical flip in XY mode
uniform float uXMul;        // ±1 for horizontal flip in XY mode
uniform float uIs3d;        // 1.0 keeps z; 0.0 collapses to z=0 (XY mode)
uniform float uAxisScaleX;  // user-driven per-axis stretch (default 1)
uniform float uAxisScaleY;
uniform float uAxisScaleZ;

void main() {
  vColor = color;
  vIsXy = uIsXy;
  if (uUseFrameFilter > 0.5 && (aTime < uFrameStart || aTime > uFrameEnd)) {
    gl_PointSize = 0.0;
    gl_Position = vec4(9.0, 9.0, 9.0, 1.0);
    return;
  }
  // Normalise raw payload coords here so we don't need a duplicate 192 MB
  // Float32 buffer on the JS heap.
  vec3 p = (position - uPosCenter) * uPosScale;
  p.x *= uXMul * uAxisScaleX;
  p.y *= uYMul * uAxisScaleY;
  p.z *= uIs3d * uAxisScaleZ;
  vec4 mvPos = modelViewMatrix * vec4(p, 1.0);
  if (uIsXy > 0.5) {
    gl_PointSize = max(2.0, uXyPxScale * uZoom) * uSizeScale;
  } else {
    gl_PointSize = uPointSize * (7.0 / -mvPos.z) * uZoom * uSizeScale;
  }
  gl_Position = projectionMatrix * mvPos;
}
`;

const sphereFragmentShader = `
varying vec3 vColor;
varying float vIsXy;

void main() {
  if (vIsXy > 0.5) {
    // Flat-shaded square in XY mode — each fragment is 1 image-pixel.
    gl_FragColor = vec4(vColor, 1.0);
    return;
  }
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float r = length(uv);
  float aaWidth = fwidth(r) * 1.5;
  float edgeAlpha = 1.0 - smoothstep(1.0 - aaWidth, 1.0, r);
  if (edgeAlpha < 0.01) discard;
  float z = sqrt(max(0.0, 1.0 - r * r));
  float rim = 1.0 - z;
  float brightness = 0.75 + 0.25 * z + 0.15 * pow(rim, 3.0);
  gl_FragColor = vec4(vColor * brightness, edgeAlpha);
}
`;

function createGlowTexture(): THREE.CanvasTexture {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const cx = size / 2, cy = size / 2;
  const g = ctx.createRadialGradient(cx, cy, size * 0.3, cx, cy, size * 0.5);
  g.addColorStop(0, "rgba(255,255,255,0)");
  g.addColorStop(0.3, "rgba(255,255,255,0.15)");
  g.addColorStop(0.6, "rgba(255,255,255,0.7)");
  g.addColorStop(0.8, "rgba(255,255,255,0.9)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

/**
 * Bin candidate indices into screen-space cells; in each cell keep only the
 * points that are within `depthFrac` of the closest point's depth. This
 * approximates Blender's "X-Ray off / Visible-only" behaviour: hidden points
 * behind closer ones are dropped.
 *
 * `projected` is the [sx, sy, wc] array from projectAllPoints; wc is camera-
 * space depth (smaller = closer for our perspective + ortho setups).
 */
function filterVisibleIndices(
  candidates: number[],
  projected: Float32Array,
  cellPx: number,
  depthFrac: number
): number[] {
  if (candidates.length < 2) return candidates;
  // First pass: find candidate-set depth range (for relative tolerance).
  let minWcAll = Infinity, maxWcAll = -Infinity;
  for (const i of candidates) {
    const wc = projected[i * 3 + 2];
    if (!isFinite(wc) || wc <= 0) continue;
    if (wc < minWcAll) minWcAll = wc;
    if (wc > maxWcAll) maxWcAll = wc;
  }
  if (!isFinite(minWcAll) || maxWcAll <= minWcAll) return candidates; // no depth diff → nothing to filter
  const eps = (maxWcAll - minWcAll) * depthFrac;

  // Second pass: bin by screen cell, track per-cell min wc.
  const cells = new Map<number, { min: number; ids: number[] }>();
  const inv = 1 / cellPx;
  for (const i of candidates) {
    const sx = projected[i * 3], sy = projected[i * 3 + 1], wc = projected[i * 3 + 2];
    if (sx < -999) continue;
    const key = (Math.floor(sx * inv) << 16) | (Math.floor(sy * inv) & 0xffff);
    let cell = cells.get(key);
    if (!cell) {
      cell = { min: wc, ids: [] };
      cells.set(key, cell);
    } else if (wc < cell.min) {
      cell.min = wc;
    }
    cell.ids.push(i);
  }
  const out: number[] = [];
  for (const cell of cells.values()) {
    const cutoff = cell.min + eps;
    for (const i of cell.ids) {
      if (projected[i * 3 + 2] <= cutoff) out.push(i);
    }
  }
  return out;
}

/**
 * Multi-Points helpers — operate on per-chunk ChunkRender records, never
 * allocating a combined payload. Each function applies the GLOBAL norm so
 * positions land in the same world space the GPU shader uses.
 *
 * Allocation note: `selectAcrossChunks` does not retain projections — the
 * inline math in each chunk loop avoids the big Float32Array(N*3) we'd
 * otherwise need for `projectAllPoints`. For chunk counts × event counts
 * encountered in this app (≤16M total) the redundant compute beats the
 * memory cost.
 */
function pickClosestAcrossChunks(
  chunks: Iterable<ChunkRender>,
  cam: THREE.Camera,
  norm: NormTransform,
  is3d: boolean,
  clickX: number, clickY: number,
  vw: number, vh: number,
  asX = 1, asY = 1, asZ = 1, // per-axis stretch (must match the shader's uAxisScale*)
): { render: ChunkRender; indexInChunk: number; eventId: number; worldPos: THREE.Vector3 } | null {
  cam.updateMatrixWorld();
  const pv = new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
  const m = pv.elements;
  const { cx, cy, cz, scale, flipY: nFlipY, flipX: nFlipX } = norm;
  const yMul = nFlipY ? -1 : 1;
  const xMul = nFlipX ? -1 : 1;
  const uPointSize = is3d ? 2.0 : 1.5;
  let bestDepth = Infinity;
  let best: { render: ChunkRender; idx: number; wx: number; wy: number; wz: number } | null = null;
  for (const render of chunks) {
    const N = render.payload.count;
    const pos = render.payload.positions;
    for (let i = 0; i < N; i++) {
      const px = (pos[i * 3] - cx) * scale * xMul * asX;
      const py = (pos[i * 3 + 1] - cy) * scale * yMul * asY;
      const pz = is3d ? (pos[i * 3 + 2] - cz) * scale * asZ : 0;
      const wc = m[3] * px + m[7] * py + m[11] * pz + m[15];
      if (wc <= 0.001) continue;
      const xc = (m[0] * px + m[4] * py + m[8] * pz + m[12]) / wc;
      const yc = (m[1] * px + m[5] * py + m[9] * pz + m[13]) / wc;
      const sx = (xc + 1) * 0.5 * vw;
      const sy = (1 - yc) * 0.5 * vh;
      const dd = (sx - clickX) ** 2 + (sy - clickY) ** 2;
      const renderedRadius = (uPointSize * 7.0 / wc) * 0.5;
      const pickRadius = Math.max(renderedRadius, 8);
      if (dd > pickRadius * pickRadius) continue;
      if (wc < bestDepth) {
        bestDepth = wc;
        best = { render, idx: i, wx: px, wy: py, wz: pz };
      }
    }
  }
  if (!best) return null;
  return {
    render: best.render,
    indexInChunk: best.idx,
    eventId: best.render.eventStartIndex + best.idx,
    worldPos: new THREE.Vector3(best.wx, best.wy, best.wz),
  };
}

// First index whose value is >= target (a is ascending). Standard lower_bound.
function lowerBound(a: ArrayLike<number>, value: number): number {
  let lo = 0, hi = a.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (a[mid] < value) lo = mid + 1; else hi = mid; }
  return lo;
}
// First index whose value is > target.
function upperBound(a: ArrayLike<number>, value: number): number {
  let lo = 0, hi = a.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (a[mid] <= value) lo = mid + 1; else hi = mid; }
  return lo;
}

// Selectable index range [lo,hi) for one chunk. Full range in 3D; in XY only
// the events inside the current playback frame window — because the shader hides
// the rest (uUseFrameFilter), so scanning them would be both wasted work and a
// correctness bug (selecting invisible same-pixel events from other times).
// payload.t_us is ascending (backend emits time-sorted chunks), so this is a
// binary search; whole non-overlapping chunks collapse to [0,0).
function frameIndexRange(t: ArrayLike<number>, n: number, xyFrame: { start: number; end: number } | null): [number, number] {
  if (!xyFrame || n === 0) return [0, n];
  if (t[n - 1] < xyFrame.start || t[0] > xyFrame.end) return [0, 0];
  return [lowerBound(t, xyFrame.start), upperBound(t, xyFrame.end)];
}

function selectAcrossChunks(
  chunks: Iterable<ChunkRender>,
  cam: THREE.Camera,
  norm: NormTransform,
  is3d: boolean,
  vw: number, vh: number,
  predicate: (sx: number, sy: number, wc: number) => boolean,
  xyFrame: { start: number; end: number } | null,
  asX = 1, asY = 1, asZ = 1, // per-axis stretch (must match the shader's uAxisScale*)
): number[] {
  cam.updateMatrixWorld();
  const pv = new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
  const m = pv.elements;
  const { cx, cy, cz, scale, flipY: nFlipY, flipX: nFlipX } = norm;
  const yMul = nFlipY ? -1 : 1;
  const xMul = nFlipX ? -1 : 1;
  const ids: number[] = [];
  for (const render of chunks) {
    const N = render.payload.count;
    const [lo, hi] = frameIndexRange(render.payload.t_us, N, xyFrame);
    if (lo === hi) continue;
    const pos = render.payload.positions;
    const base = render.eventStartIndex;
    for (let i = lo; i < hi; i++) {
      const px = (pos[i * 3] - cx) * scale * xMul * asX;
      const py = (pos[i * 3 + 1] - cy) * scale * yMul * asY;
      const pz = is3d ? (pos[i * 3 + 2] - cz) * scale * asZ : 0;
      const wc = m[3] * px + m[7] * py + m[11] * pz + m[15];
      if (wc <= 0.001) continue;
      const xc = (m[0] * px + m[4] * py + m[8] * pz + m[12]) / wc;
      const yc = (m[1] * px + m[5] * py + m[9] * pz + m[13]) / wc;
      const sx = (xc + 1) * 0.5 * vw;
      const sy = (1 - yc) * 0.5 * vh;
      if (predicate(sx, sy, wc)) ids.push(base + i);
    }
  }
  return ids;
}

/** Unproject a screen-space brush AABB into payload-space x/y bounds.
 *  Only valid when the camera is OrthographicCamera (XY view) — in 3D
 *  perspective the inverse of a screen rect is a frustum slice that varies
 *  with depth, so this short-cut doesn't apply. Returns null when the
 *  caller should fall back to per-event full projection. */
function computeBrushPayloadAABB(
  cam: THREE.Camera,
  norm: NormTransform,
  brushCx: number, brushCy: number, brushR: number,
  vw: number, vh: number,
): { xMin: number; xMax: number; yMin: number; yMax: number } | null {
  if (!(cam instanceof THREE.OrthographicCamera)) return null;
  const { cx, cy, scale, flipY: nFlipY, flipX: nFlipX } = norm;
  const yMul = nFlipY ? -1 : 1;
  const xMul = nFlipX ? -1 : 1;
  const denomX = scale * xMul;
  const denomY = scale * yMul;
  cam.updateMatrixWorld();
  const tmp = new THREE.Vector3();
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  // Four screen corners → unproject → invert payload-space mapping.
  // ndc.z=0 lands on the camera's mid-clip plane; for ortho that's fine since
  // the projection is depth-independent.
  const corners: [number, number][] = [
    [brushCx - brushR, brushCy - brushR],
    [brushCx + brushR, brushCy - brushR],
    [brushCx - brushR, brushCy + brushR],
    [brushCx + brushR, brushCy + brushR],
  ];
  for (let i = 0; i < 4; i++) {
    const [sx, sy] = corners[i];
    tmp.set((sx / vw) * 2 - 1, -((sy / vh) * 2 - 1), 0).unproject(cam);
    const px = tmp.x / denomX + cx;
    const py = tmp.y / denomY + cy;
    if (px < minX) minX = px; if (px > maxX) maxX = px;
    if (py < minY) minY = py; if (py > maxY) maxY = py;
  }
  if (!isFinite(minX)) return null;
  return { xMin: minX, xMax: maxX, yMin: minY, yMax: maxY };
}

/**
 * Brush-stroke colouring for circle/lasso during drag. In one pass it
 * projects each event, hit-tests, accumulates new event_ids into
 * `strokeIds`, AND mutates the chunk's colour buffer to white in-place so
 * the GPU shows what's currently selected without any overlay drawing.
 *
 * XY-mode fast path: when an `aabb` is supplied (computed once per move via
 * `computeBrushPayloadAABB`), each event's raw payload x/y is bbox-tested
 * with two comparisons before doing the full projection. With a typical
 * brush radius covering <5% of the canvas, this rejects ~95% of events in
 * a couple of branches and is the dominant speed-up for 16M-event datasets.
 */
function brushStrokeChunks(
  renders: Iterable<ChunkRender>,
  cam: THREE.Camera,
  norm: NormTransform,
  is3d: boolean,
  vw: number, vh: number,
  hitTest: (sx: number, sy: number, wc: number) => boolean,
  strokeIds: Set<number>,
  touchedChunks: Set<number>,
  aabb: { xMin: number; xMax: number; yMin: number; yMax: number } | null,
  xyFrame: { start: number; end: number } | null,
  asX = 1, asY = 1, asZ = 1, // per-axis stretch (must match the shader's uAxisScale*)
): void {
  cam.updateMatrixWorld();
  const pv = new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
  const m = pv.elements;
  const { cx, cy, cz, scale, flipY: nFlipY, flipX: nFlipX } = norm;
  const yMul = nFlipY ? -1 : 1;
  const xMul = nFlipX ? -1 : 1;
  const useAabb = aabb !== null;
  const aabbXMin = aabb ? aabb.xMin : 0;
  const aabbXMax = aabb ? aabb.xMax : 0;
  const aabbYMin = aabb ? aabb.yMin : 0;
  const aabbYMax = aabb ? aabb.yMax : 0;
  for (const render of renders) {
    const N = render.payload.count;
    const [lo, hi] = frameIndexRange(render.payload.t_us, N, xyFrame);
    if (lo === hi) continue;
    const pos = render.payload.positions;
    const colors = render.colors;
    const base = render.eventStartIndex;
    let touched = false;
    for (let i = lo; i < hi; i++) {
      // Fast reject: payload-space AABB. Two compares per axis cuts 95%+
      // of events without doing the full projection math.
      if (useAabb) {
        const rx = pos[i * 3];
        if (rx < aabbXMin || rx > aabbXMax) continue;
        const ry = pos[i * 3 + 1];
        if (ry < aabbYMin || ry > aabbYMax) continue;
      }
      const eid = base + i;
      if (strokeIds.has(eid)) continue;
      const ppx = (pos[i * 3] - cx) * scale * xMul * asX;
      const ppy = (pos[i * 3 + 1] - cy) * scale * yMul * asY;
      const ppz = is3d ? (pos[i * 3 + 2] - cz) * scale * asZ : 0;
      const wc = m[3] * ppx + m[7] * ppy + m[11] * ppz + m[15];
      if (wc <= 0.001) continue;
      const xc = (m[0] * ppx + m[4] * ppy + m[8] * ppz + m[12]) / wc;
      const yc = (m[1] * ppx + m[5] * ppy + m[9] * ppz + m[13]) / wc;
      const sx = (xc + 1) * 0.5 * vw;
      const sy = (1 - yc) * 0.5 * vh;
      if (!hitTest(sx, sy, wc)) continue;
      strokeIds.add(eid);
      const o = i * 3;
      colors[o] = 255; colors[o + 1] = 255; colors[o + 2] = 255;
      touched = true;
    }
    if (touched) {
      (render.geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
      touchedChunks.add(render.chunkId);
    }
  }
}

/** Look up a single event_id across loaded chunks. Used by focusOnPoint and
 *  the convex-hull selection mesh. Returns the resolved world position +
 *  source chunk metadata, or null if the event isn't currently loaded. */
function findEventAcrossChunks(
  chunks: Iterable<ChunkRender>,
  norm: NormTransform,
  is3d: boolean,
  eventId: number,
  asX = 1, asY = 1, asZ = 1, // per-axis stretch (must match the shader's uAxisScale*)
): { render: ChunkRender; indexInChunk: number; worldPos: THREE.Vector3; payloadX: number; payloadY: number; payloadZ: number; tUs: number; polarity: number; label: number } | null {
  const { cx, cy, cz, scale, flipY: nFlipY, flipX: nFlipX } = norm;
  const yMul = nFlipY ? -1 : 1;
  const xMul = nFlipX ? -1 : 1;
  for (const render of chunks) {
    const start = render.eventStartIndex;
    const end = start + render.payload.count;
    if (eventId < start || eventId >= end) continue;
    const i = eventId - start;
    const px = render.payload.positions[i * 3];
    const py = render.payload.positions[i * 3 + 1];
    const pz = render.payload.positions[i * 3 + 2];
    const worldPos = new THREE.Vector3(
      (px - cx) * scale * xMul * asX,
      (py - cy) * scale * yMul * asY,
      is3d ? (pz - cz) * scale * asZ : 0,
    );
    return {
      render, indexInChunk: i, worldPos,
      payloadX: px, payloadY: py, payloadZ: pz,
      tUs: render.payload.t_us[i],
      polarity: render.payload.polarity[i],
      label: render.payload.label[i],
    };
  }
  return null;
}

// Point-in-polygon test (ray casting)
function pointInPolygon(px: number, py: number, polygon: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

// Screen-space bbox of a polygon — a 4-compare reject in front of the much more
// expensive point-in-polygon test.
function polygonBounds(polygon: number[][]): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of polygon) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  return { minX, minY, maxX, maxY };
}

// Ramer–Douglas–Peucker simplification. A freehand lasso accumulates 100–300
// vertices; point-in-polygon is O(candidates × vertices), so collapsing
// near-collinear runs to ~12–40 vertices cuts the per-point cost ~10× with a
// visually identical boundary. Iterative (explicit stack) to avoid deep
// recursion on long strokes.
function simplifyRDP(points: number[][], epsilon: number): number[][] {
  const n = points.length;
  if (n < 3) return points.slice();
  const keep = new Uint8Array(n);
  keep[0] = 1; keep[n - 1] = 1;
  const stack: [number, number][] = [[0, n - 1]];
  while (stack.length) {
    const [a, b] = stack.pop()!;
    const [ax, ay] = points[a];
    const [bx, by] = points[b];
    const dx = bx - ax, dy = by - ay;
    const denom = Math.hypot(dx, dy) || 1;
    let maxDist = 0, idx = -1;
    for (let i = a + 1; i < b; i++) {
      const [px, py] = points[i];
      const d = Math.abs(dy * px - dx * py + bx * ay - by * ax) / denom;
      if (d > maxDist) { maxDist = d; idx = i; }
    }
    if (idx !== -1 && maxDist > epsilon) {
      keep[idx] = 1;
      stack.push([a, idx], [idx, b]);
    }
  }
  const out: number[][] = [];
  for (let i = 0; i < n; i++) if (keep[i]) out.push(points[i]);
  return out;
}

// Compute nice tick values for a range [min, max] with target count
function niceTicks(min: number, max: number, targetCount: number): number[] {
  if (max <= min) return [min];
  const range = max - min;
  const rawStep = range / targetCount;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normStep = rawStep / mag;
  let niceStep: number;
  if (normStep <= 1.5) niceStep = mag;
  else if (normStep <= 3.5) niceStep = 2 * mag;
  else if (normStep <= 7.5) niceStep = 5 * mag;
  else niceStep = 10 * mag;
  const ticks: number[] = [];
  const start = Math.ceil(min / niceStep) * niceStep;
  for (let v = start; v <= max; v += niceStep) ticks.push(Math.round(v * 1000) / 1000);
  return ticks;
}

function formatTickValue(v: number, unit: string): string {
  if (unit === "us") {
    if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}s`;
    if (v >= 1_000) return `${(v / 1_000).toFixed(1)}ms`;
    return `${Math.round(v)}us`;
  }
  return `${Math.round(v)}`;
}

type AxisTick = { worldPos: [number, number, number]; label: string; axis: "x" | "y" | "t" };

function computePayloadTRange(payload: PointPayload | null): { minT: number; maxT: number } | null {
  if (!payload || payload.count === 0) return null;
  let minT = payload.t_us[0];
  let maxT = minT;
  for (let i = 1; i < payload.count; i++) {
    const t = payload.t_us[i];
    if (t < minT) minT = t;
    else if (t > maxT) maxT = t;
  }
  return { minT, maxT };
}

function computeAxisTicks(
  bounds: Bounds3,
  norm: NormTransform,
  meta: { width: number; height: number } | null,
  tRange: { minT: number; maxT: number } | null,
  // 1 = default density. Higher = more ticks (used when the user zooms in
  // so the visible axis gets sub-divided more finely). Capped to keep the
  // label clutter manageable.
  densityScale = 1,
): AxisTick[] {
  const { cx, cy, cz, scale } = norm;
  const wMinZ = (bounds.minZ - cz) * scale;
  const wMaxZ = (bounds.maxZ - cz) * scale;
  const ticks: AxisTick[] = [];
  const xN = Math.round(Math.max(2, Math.min(40, 6 * densityScale)));
  const yN = Math.round(Math.max(2, Math.min(40, 5 * densityScale)));
  const tN = Math.round(Math.max(2, Math.min(40, 5 * densityScale)));
  if (meta) {
    // Axis origin: (x_pixel = 0, y_pixel = 0). Convert to world space.
    const ox = (-(meta.width - 1) / 2 - cx) * scale;
    const oy = ((meta.height - 1) / 2 - cy) * scale;
    for (const px of niceTicks(0, meta.width - 1, xN)) {
      const wx = (px - (meta.width - 1) / 2 - cx) * scale;
      ticks.push({ worldPos: [wx, oy, wMinZ], label: `${Math.round(px)}`, axis: "x" });
    }
    for (const py of niceTicks(0, meta.height - 1, yN)) {
      const wy = ((meta.height - 1) / 2 - py - cy) * scale;
      ticks.push({ worldPos: [ox, wy, wMinZ], label: `${Math.round(py)}`, axis: "y" });
    }
    if (tRange && tRange.maxT > tRange.minT) {
      const span = tRange.maxT - tRange.minT;
      // Relative microseconds — the origin tick reads "0us".
      for (const dt of niceTicks(0, span, tN)) {
        const wz = wMinZ + (dt / span) * (wMaxZ - wMinZ);
        ticks.push({ worldPos: [ox, oy, wz], label: formatTickValue(dt, "us"), axis: "t" });
      }
    }
  }
  return ticks;
}

type LabelStyle = {
  fg: string;       // text colour
  bg: string;       // pill background colour (rgba ok)
  border: string;   // pill border colour (rgba ok)
  fontSize?: number;
  hWorld?: number;
  emphasis?: boolean; // bolder for axis-end labels
};

function makeTextSprite(text: string, style: LabelStyle): THREE.Sprite {
  const fontSize = style.fontSize ?? 10;
  const dpr = Math.min(window.devicePixelRatio || 1, 2) * 3;  // supersample → labels stay crisp when zoomed in
  const padX = 6, padY = 3;
  const fontWeight = style.emphasis ? "600" : "500";
  const fontSpec = `${fontWeight} ${fontSize}px Inter, "Segoe UI", monospace`;
  const measure = document.createElement("canvas").getContext("2d")!;
  measure.font = fontSpec;
  const tw = Math.ceil(measure.measureText(text).width);
  const cssW = tw + padX * 2;
  const cssH = fontSize + padY * 2;
  const c = document.createElement("canvas");
  c.width = Math.ceil(cssW * dpr);
  c.height = Math.ceil(cssH * dpr);
  const ctx = c.getContext("2d")!;
  ctx.scale(dpr, dpr);
  // Rounded-rect pill background
  const radius = Math.min(cssH / 2, 6);
  ctx.beginPath();
  ctx.moveTo(radius, 0);
  ctx.lineTo(cssW - radius, 0);
  ctx.quadraticCurveTo(cssW, 0, cssW, radius);
  ctx.lineTo(cssW, cssH - radius);
  ctx.quadraticCurveTo(cssW, cssH, cssW - radius, cssH);
  ctx.lineTo(radius, cssH);
  ctx.quadraticCurveTo(0, cssH, 0, cssH - radius);
  ctx.lineTo(0, radius);
  ctx.quadraticCurveTo(0, 0, radius, 0);
  ctx.closePath();
  ctx.fillStyle = style.bg;
  ctx.fill();
  ctx.strokeStyle = style.border;
  ctx.lineWidth = 0.75;
  ctx.stroke();
  // Text
  ctx.font = fontSpec;
  ctx.fillStyle = style.fg;
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  ctx.fillText(text, cssW / 2, cssH / 2 + 0.5);
  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  const mat = new THREE.SpriteMaterial({
    map: tex,
    transparent: true,
    depthTest: true,
    depthWrite: false,
    sizeAttenuation: true,
  });
  const sprite = new THREE.Sprite(mat);
  const hWorld = style.hWorld ?? 0.05;
  sprite.scale.set(hWorld * (cssW / cssH), hWorld, 1);
  sprite.center.set(0.5, 0.5);
  return sprite;
}

/** Slider-thumb texture: a white vertical capsule (stadium) with an indigo
 *  border + 3 grip lines, drawn at high resolution so it stays crisp. Mapped
 *  onto a flat plane (not a sprite) so the handle sits in the slider plane. */
function makeThumbTexture(): THREE.CanvasTexture {
  // High native resolution → stays sharp when the plane is magnified.
  const W = 168, H = 252, lw = 14;
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const ctx = c.getContext("2d")!;
  // Capsule body: rounded rect with radius = half-width = a true stadium.
  const x0 = lw / 2, y0 = lw / 2, ww = W - lw, hh = H - lw, rr = ww / 2;
  ctx.beginPath();
  ctx.moveTo(x0 + rr, y0);
  ctx.arcTo(x0 + ww, y0, x0 + ww, y0 + rr, rr);
  ctx.lineTo(x0 + ww, y0 + hh - rr);
  ctx.arcTo(x0 + ww, y0 + hh, x0 + ww - rr, y0 + hh, rr);
  ctx.lineTo(x0 + rr, y0 + hh);
  ctx.arcTo(x0, y0 + hh, x0, y0 + hh - rr, rr);
  ctx.lineTo(x0, y0 + rr);
  ctx.arcTo(x0, y0, x0 + rr, y0, rr);
  ctx.closePath();
  ctx.fillStyle = "#ffffff"; ctx.fill();
  ctx.lineWidth = lw; ctx.strokeStyle = "#6c5ce7"; ctx.stroke();
  // Three horizontal grip lines, centred.
  ctx.strokeStyle = "#6c5ce7"; ctx.lineWidth = 12; ctx.lineCap = "round";
  const cx = W / 2, gx = W * 0.24;
  for (const dy of [-34, 0, 34]) {
    ctx.beginPath(); ctx.moveTo(cx - gx, H / 2 + dy); ctx.lineTo(cx + gx, H / 2 + dy); ctx.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter; tex.anisotropy = 8; tex.needsUpdate = true;
  return tex;
}

/** Axis-end chip: a big coloured axis letter with a small unit beneath it
 *  (e.g. "X" / "pixels"). Matches the Plotly-style coordinate labels. */
function makeAxisChip(letter: string, unit: string, fg: string, bg: string, border: string, unitFg: string): THREE.Sprite {
  const dpr = Math.min(window.devicePixelRatio || 1, 2) * 3;  // supersample → labels stay crisp when zoomed in
  const padX = 8, padY = 5, gap = 1, lf = 14, uf = 8.5;
  const m = document.createElement("canvas").getContext("2d")!;
  m.font = `700 ${lf}px Inter, "Segoe UI", monospace`;
  const lw = m.measureText(letter).width;
  m.font = `500 ${uf}px Inter, "Segoe UI", monospace`;
  const uw = m.measureText(unit).width;
  const cssW = Math.ceil(Math.max(lw, uw) + padX * 2);
  const cssH = Math.ceil(lf + gap + uf + padY * 2 + 2);
  const c = document.createElement("canvas");
  c.width = Math.ceil(cssW * dpr); c.height = Math.ceil(cssH * dpr);
  const ctx = c.getContext("2d")!; ctx.scale(dpr, dpr);
  const rad = 7;
  ctx.beginPath();
  ctx.moveTo(rad, 0); ctx.lineTo(cssW - rad, 0); ctx.quadraticCurveTo(cssW, 0, cssW, rad);
  ctx.lineTo(cssW, cssH - rad); ctx.quadraticCurveTo(cssW, cssH, cssW - rad, cssH);
  ctx.lineTo(rad, cssH); ctx.quadraticCurveTo(0, cssH, 0, cssH - rad);
  ctx.lineTo(0, rad); ctx.quadraticCurveTo(0, 0, rad, 0); ctx.closePath();
  ctx.fillStyle = bg; ctx.fill(); ctx.strokeStyle = border; ctx.lineWidth = 1; ctx.stroke();
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.font = `700 ${lf}px Inter, "Segoe UI", monospace`; ctx.fillStyle = fg;
  ctx.fillText(letter, cssW / 2, padY);
  ctx.font = `500 ${uf}px Inter, "Segoe UI", monospace`; ctx.fillStyle = unitFg;
  ctx.fillText(unit, cssW / 2, padY + lf + gap);
  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter; tex.anisotropy = 8; tex.needsUpdate = true;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: true, depthWrite: false, sizeAttenuation: true });
  const sprite = new THREE.Sprite(mat);
  const hWorld = 0.08;
  sprite.scale.set(hWorld * (cssW / cssH), hWorld, 1);
  sprite.center.set(0.5, 0.5);
  return sprite;
}

function buildBoxWireGeometry(corners: [number, number, number][]): THREE.BufferGeometry {
  const edges: [number, number][] = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const verts: number[] = [];
  for (const [a, b] of edges) {
    verts.push(...corners[a], ...corners[b]);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  return g;
}

function makeArrowhead(
  tip: [number, number, number],
  dir: [number, number, number],
  size: number,
  colorHex: number
): THREE.Mesh {
  // Cone is +Y aligned by default. Place its base centre at `tip - dir*size`.
  const d = new THREE.Vector3(...dir).normalize();
  const cone = new THREE.ConeGeometry(size * 0.45, size, 18, 1, false);
  const mat = new THREE.MeshBasicMaterial({
    color: colorHex,
    transparent: true,
    opacity: 0.95,
    depthTest: true,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(cone, mat);
  // Cone apex sits at +size/2 by default; we want apex at `tip`, base at `tip - d*size`.
  // Translate centre to (tip - d*size/2), then rotate +Y → d.
  mesh.position.set(tip[0] - d.x * size * 0.5, tip[1] - d.y * size * 0.5, tip[2] - d.z * size * 0.5);
  const yUp = new THREE.Vector3(0, 1, 0);
  if (Math.abs(d.dot(yUp) - 1) < 1e-6) {
    // already +Y
  } else if (Math.abs(d.dot(yUp) + 1) < 1e-6) {
    mesh.quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
  } else {
    mesh.quaternion.setFromUnitVectors(yUp, d);
  }
  return mesh;
}

function disposeAxesGroupChildren(group: THREE.Group): void {
  while (group.children.length) {
    const c = group.children[0] as THREE.Object3D;
    group.remove(c);
    const anyC = c as unknown as { geometry?: THREE.BufferGeometry; material?: THREE.Material | THREE.Material[] };
    anyC.geometry?.dispose();
    const m = anyC.material;
    if (m) {
      const mats = Array.isArray(m) ? m : [m];
      for (const mm of mats) {
        const sm = mm as THREE.SpriteMaterial;
        sm.map?.dispose();
        mm.dispose();
      }
    }
  }
}

function populateAxesGroup(
  group: THREE.Group,
  buffers: CloudBuffers,
  meta: { width: number; height: number } | null,
  payload: PointPayload | null,
  theme: "light" | "dark",
  // Optional override — passed by callers in multi-Points mode where
  // `payload` is null but the time range is available from the manifest.
  // When omitted the range is derived from `payload` (legacy behaviour).
  tRangeOverride?: { minT: number; maxT: number } | null,
  // 1 = baseline density. Bumped by the camera-zoom watcher so a tightly
  // zoomed-in view still gets enough graduations to be readable.
  tickDensityScale = 1,
  // Per-axis stretch (default 1). All POSITIONS are scaled (so labels and
  // arrowheads land at the same place the stretched points do), but SIZES
  // (arrow length, sphere radius, tick nub length) come from the unscaled
  // span — keeps decorations from inflating.
  axisScale: { x: number; y: number; z: number } = { x: 1, y: 1, z: 1 },
  // Optional output: the T (time) axis world endpoints + the bounding-box far
  // corner, filled for the time scrubber. bottom = earliest (t_min), top =
  // latest (t_max); xEnd/yEnd let the scrub plane span the full XY extent.
  outTAxis?: { bottom: THREE.Vector3; top: THREE.Vector3; xEnd: number; yEnd: number; valid: boolean },
): void {
  disposeAxesGroupChildren(group);
  const tRange = tRangeOverride ?? computePayloadTRange(payload);
  const ticks = computeAxisTicks(buffers.norm.bounds, buffers.norm, meta, tRange, tickDensityScale);
  // Apply per-axis scale to tick world positions in-place so all callers
  // below see scaled coordinates without per-tick branching.
  for (const tk of ticks) {
    tk.worldPos = [tk.worldPos[0] * axisScale.x, tk.worldPos[1] * axisScale.y, tk.worldPos[2] * axisScale.z];
  }

  const isLight = theme === "light";
  const axisColorHex = isLight ? 0x2c3744 : 0xc9d2dc;
  const boxColorHex = isLight ? 0x9aa4b1 : 0x4a5566;
  const tickStyle: LabelStyle = isLight
    ? { fg: "#1a2230", bg: "rgba(255,255,255,0.92)", border: "rgba(40,55,75,0.18)", fontSize: 9, hWorld: 0.038 }
    : { fg: "#e6eef6", bg: "rgba(18,24,34,0.85)", border: "rgba(180,195,215,0.22)", fontSize: 9, hWorld: 0.038 };
  const endStyle: LabelStyle = isLight
    ? { fg: "#0a1320", bg: "rgba(255,255,255,0.96)", border: "rgba(40,55,75,0.35)", fontSize: 10, hWorld: 0.052, emphasis: true }
    : { fg: "#f4f8fc", bg: "rgba(20,28,40,0.92)", border: "rgba(200,215,235,0.4)", fontSize: 10, hWorld: 0.052, emphasis: true };

  const { cx, cy, cz, scale } = buffers.norm;
  // Unscaled (base) world anchors used to derive decoration sizes.
  const baseZMin = (buffers.norm.bounds.minZ - cz) * scale;
  const baseZMax = (buffers.norm.bounds.maxZ - cz) * scale;
  const baseOx = meta ? (-(meta.width - 1) / 2 - cx) * scale : (buffers.norm.bounds.minX - cx) * scale;
  const baseOy = meta ? ((meta.height - 1) / 2 - cy) * scale : (buffers.norm.bounds.minY - cy) * scale;
  const baseXEnd = meta ? ((meta.width - 1) / 2 - cx) * scale : (buffers.norm.bounds.maxX - cx) * scale;
  const baseYEnd = meta ? (-(meta.height - 1) / 2 - cy) * scale : (buffers.norm.bounds.maxY - cy) * scale;
  const baseSpan = Math.max(Math.abs(baseXEnd - baseOx), Math.abs(baseYEnd - baseOy), baseZMax - baseZMin, 1e-6);
  // Scaled positions used to place every visual element.
  const ox = baseOx * axisScale.x;
  const oy = baseOy * axisScale.y;
  const xEnd = baseXEnd * axisScale.x;
  const yEnd = baseYEnd * axisScale.y;
  const wMinZ = baseZMin * axisScale.z;
  const wMaxZ = baseZMax * axisScale.z;
  const off = baseSpan * 0.03;
  const yTickDir = Math.sign(yEnd - oy) || 1;

  if (outTAxis) {
    // T axis runs at the (minX,minY) corner; wMinZ = earliest, wMaxZ = latest.
    outTAxis.bottom.set(ox, oy, wMinZ);
    outTAxis.top.set(ox, oy, wMaxZ);
    outTAxis.xEnd = xEnd;
    outTAxis.yEnd = yEnd;
    outTAxis.valid = true;
  }

  // 1. Bounding box wireframe.
  const boxCorners: [number, number, number][] = [
    [ox, oy, wMinZ], [xEnd, oy, wMinZ], [xEnd, yEnd, wMinZ], [ox, yEnd, wMinZ],
    [ox, oy, wMaxZ], [xEnd, oy, wMaxZ], [xEnd, yEnd, wMaxZ], [ox, yEnd, wMaxZ],
  ];
  group.add(new THREE.LineSegments(
    buildBoxWireGeometry(boxCorners),
    new THREE.LineBasicMaterial({ color: boxColorHex, transparent: true, opacity: isLight ? 0.22 : 0.18, depthTest: true, depthWrite: false })
  ));

  // Per-axis colours (Plotly-style): X = blue, Y = green, T = purple.
  const AXC = { x: "#4a90e2", y: "#23b26d", t: "#8b5cf6" } as const;
  const AXH = { x: 0x4a90e2, y: 0x23b26d, t: 0x8b5cf6 } as const;
  const tickLen = baseSpan * 0.02;
  const segGeom = (segs: number[]) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(segs, 3));
    return g;
  };
  const lineMat = (hex: number) => new THREE.LineBasicMaterial({ color: hex, transparent: true, opacity: 0.95, depthTest: true, depthWrite: false });

  // 2. Three coloured axis lines, each with its own tick nubs.
  const xSeg: number[] = [ox, oy, wMinZ, xEnd, oy, wMinZ];
  const ySeg: number[] = [ox, oy, wMinZ, ox, yEnd, wMinZ];
  const tSeg: number[] = [ox, oy, wMinZ, ox, oy, wMaxZ];
  for (const tk of ticks) {
    const [wx, wy, wz] = tk.worldPos;
    if (tk.axis === "x") xSeg.push(wx, oy, wMinZ, wx, oy - yTickDir * tickLen, wMinZ);
    else if (tk.axis === "y") ySeg.push(ox, wy, wMinZ, ox - tickLen, wy, wMinZ);
    else tSeg.push(ox, oy, wz, ox - tickLen, oy, wz);
  }
  group.add(new THREE.LineSegments(segGeom(xSeg), lineMat(AXH.x)));
  group.add(new THREE.LineSegments(segGeom(ySeg), lineMat(AXH.y)));
  group.add(new THREE.LineSegments(segGeom(tSeg), lineMat(AXH.t)));

  // 3. Origin "0" label at the corner.
  const zeroSprite = makeTextSprite("0", { fg: tickStyle.fg, bg: tickStyle.bg, border: tickStyle.border, fontSize: 9, hWorld: 0.03 });
  zeroSprite.position.set(ox - off * 0.85, oy - yTickDir * off * 0.85, wMinZ);
  group.add(zeroSprite);

  // 4. Coloured arrowheads.
  const arrowSize = baseSpan * 0.018;
  group.add(makeArrowhead([xEnd, oy, wMinZ], [Math.sign(xEnd - ox) || 1, 0, 0], arrowSize, AXH.x));
  group.add(makeArrowhead([ox, yEnd, wMinZ], [0, yTickDir, 0], arrowSize, AXH.y));
  group.add(makeArrowhead([ox, oy, wMaxZ], [0, 0, 1], arrowSize, AXH.t));

  // 5. Tick number sprites — text coloured to match each axis.
  const tickStyleFor = (axis: "x" | "y" | "t"): LabelStyle => ({ fg: AXC[axis], bg: tickStyle.bg, border: tickStyle.border, fontSize: 9, hWorld: 0.032 });
  for (const tk of ticks) {
    const sp = makeTextSprite(tk.label, tickStyleFor(tk.axis));
    const [wx, wy, wz] = tk.worldPos;
    if (tk.axis === "x") sp.position.set(wx, oy - yTickDir * off, wMinZ);
    else if (tk.axis === "y") sp.position.set(ox - off, wy, wMinZ);
    else sp.position.set(ox - off, oy, wz);
    group.add(sp);
  }

  // 6. Axis-end unit chips (letter + unit, coloured per axis).
  const unitFg = isLight ? "rgba(40,55,75,0.62)" : "rgba(205,218,236,0.6)";
  const endOff = arrowSize + off * 1.6;
  const xChip = makeAxisChip("X", "pixels", AXC.x, endStyle.bg, endStyle.border, unitFg);
  xChip.position.set(xEnd + endOff * (Math.sign(xEnd - ox) || 1), oy, wMinZ);
  group.add(xChip);
  const yChip = makeAxisChip("Y", "pixels", AXC.y, endStyle.bg, endStyle.border, unitFg);
  yChip.position.set(ox, yEnd + endOff * yTickDir, wMinZ);
  group.add(yChip);
  const tChip = makeAxisChip("T", "time", AXC.t, endStyle.bg, endStyle.border, unitFg);
  tChip.position.set(ox, oy, wMaxZ + endOff);
  group.add(tChip);
}

function buildAxesLineGeometry(
  bounds: Bounds3,
  norm: NormTransform,
  // ticks must already have axisScale applied to their worldPos (caller
  // populateAxesGroup does this in-place).
  ticks: AxisTick[],
  meta: { width: number; height: number } | null,
  axisScale: { x: number; y: number; z: number } = { x: 1, y: 1, z: 1 },
  unscaledSpan?: number,
): THREE.BufferGeometry {
  const { cx, cy, cz, scale } = norm;
  const baseZMin = (bounds.minZ - cz) * scale, baseZMax = (bounds.maxZ - cz) * scale;
  const baseOx = meta ? (-(meta.width - 1) / 2 - cx) * scale : (bounds.minX - cx) * scale;
  const baseOy = meta ? ((meta.height - 1) / 2 - cy) * scale : (bounds.minY - cy) * scale;
  const baseXEnd = meta ? ((meta.width - 1) / 2 - cx) * scale : (bounds.maxX - cx) * scale;
  const baseYEnd = meta ? (-(meta.height - 1) / 2 - cy) * scale : (bounds.maxY - cy) * scale;
  const ox = baseOx * axisScale.x;
  const oy = baseOy * axisScale.y;
  const xEnd = baseXEnd * axisScale.x;
  const yEnd = baseYEnd * axisScale.y;
  const wMinZ = baseZMin * axisScale.z;
  const wMaxZ = baseZMax * axisScale.z;
  // Tick nub length comes from the unscaled span so nubs stay uniform.
  const refSpan = unscaledSpan ?? Math.max(
    Math.abs(baseXEnd - baseOx), Math.abs(baseYEnd - baseOy), baseZMax - baseZMin, 1e-6,
  );
  const tickLen = refSpan * 0.02;

  const verts: number[] = [];
  verts.push(ox, oy, wMinZ, xEnd, oy, wMinZ);
  verts.push(ox, oy, wMinZ, ox, yEnd, wMinZ);
  verts.push(ox, oy, wMinZ, ox, oy, wMaxZ);
  const yTickDir = Math.sign(yEnd - oy) || 1;
  for (const tk of ticks) {
    const [wx, wy, wz] = tk.worldPos;
    if (tk.axis === "x") verts.push(wx, oy, wMinZ, wx, oy - yTickDir * tickLen, wMinZ);
    else if (tk.axis === "y") verts.push(ox, wy, wMinZ, ox - tickLen, wy, wMinZ);
    else verts.push(ox, oy, wz, ox - tickLen, oy, wz);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  return geom;
}

export function PointCloudView({
  payload, manifest, chunksMap, visibleChunkIds, viewMode, selectedIds, gpuPreference, theme, labels, meta,
  toolMode, brushRadius, selectThrough, xyRotation = 0, xyFlipY = false, xyFlipX = false, currentFrameUs, frameWindowUs,
  onPointClick, onSelection, onWandSeed, onBrushRadiusChange, tooltipRef, handleRef, labelPatch,
  colorMode = "polarity", pointSizeScale = 1, windowStartUs, windowEndUs, onTimeScrub,
  unlabeledColor = "#888888", polarityContrast = 0.5,
  filterXMin = 0, filterXMax = Infinity, filterYMin = 0, filterYMax = Infinity, filterPolarity = "all",
  filterPreview = null,
  previewRot = 0, previewFlipX = false, previewFlipY = false
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const mountRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 1, height: 1 });
  const cameraRef = useRef<THREE.Camera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const buffersRef = useRef<CloudBuffers | null>(null);
  const geometryRef = useRef<THREE.BufferGeometry | null>(null);
  const materialRef = useRef<THREE.ShaderMaterial | null>(null);
  // Multi-Points renderer state — one ChunkRender per loaded chunk so we
  // never allocate a single combined payload (which is the OOM source on
  // the user's borderline machine).
  const chunkRendersRef = useRef<Map<number, ChunkRender>>(new Map());
  const axesRef = useRef<THREE.AxesHelper | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const axesGroupRef = useRef<THREE.Group | null>(null);
  const tRangeRef = useRef<{ minT: number; maxT: number } | null>(null);
  // T-axis world endpoints (bottom = earliest, top = latest), filled by the
  // axis builder and projected by the time scrubber.
  const tAxisRef = useRef<{ bottom: THREE.Vector3; top: THREE.Vector3; xEnd: number; yEnd: number; valid: boolean }>({ bottom: new THREE.Vector3(), top: new THREE.Vector3(), xEnd: 1, yEnd: 1, valid: false });
  const onTimeScrubRef = useRef(onTimeScrub);
  useEffect(() => { onTimeScrubRef.current = onTimeScrub; }, [onTimeScrub]);
  // Bumped each time the main scene-setup effect (re-)builds the renderer, so
  // scene-dependent effects (chunk reconciliation, scrub visuals) re-attach to
  // the fresh scene. Declared here so those effects can list it as a dep.
  const [sceneVersion, setSceneVersion] = useState(0);
  // Time-scrub 3D visuals: a blue plane at z=t spanning the XY extent + a small
  // handle on the T axis. Created with the scene, toggled by the tscrub tool.
  const scrubGroupRef = useRef<THREE.Group | null>(null);
  const scrubPlaneRef = useRef<THREE.Mesh | null>(null);
  // Yellow Ranges & Filters preview box (the draft region, before Load).
  const filterBoxGroupRef = useRef<THREE.Group | null>(null);
  // The slider is a flat 2D widget living in a vertical plane through the T axis,
  // angled at 45° in XY (normal along the XY diagonal) so it faces the default
  // camera. sliderGroup carries that orientation; fill/handle/label are flat
  // planes in its local XY.
  const scrubSliderGroupRef = useRef<THREE.Group | null>(null);
  const scrubHandleRef = useRef<THREE.Mesh | null>(null);   // thumb capsule (flat)
  const scrubFillRef = useRef<THREE.Mesh | null>(null);     // purple filled track 0→t (flat)
  const scrubTriRef = useRef<THREE.Mesh | null>(null);  // red ▸ marks the video top-left
  const scrubLabelRef = useRef<THREE.Mesh | null>(null);  // time tooltip bubble (flat)
  const scrubLabelCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const scrubLabelTexRef = useRef<THREE.CanvasTexture | null>(null);
  // Preview orientation (mirrors MainViewport's buttons) so the red corner
  // marker follows the preview's displayed top-left. + last scrubbed time so we
  // can redraw the marker when the buttons change without an active drag.
  const previewRotRef = useRef(previewRot);
  const previewFlipXRef = useRef(previewFlipX);
  const previewFlipYRef = useRef(previewFlipY);
  const lastScrubTUsRef = useRef<number | null>(null);

  // Position the blue scrub plane + axis handle at a given time and show them.
  const updateScrubAt = useCallback((tUs: number) => {
    const ta = tAxisRef.current, tr = tRangeRef.current;
    const group = scrubGroupRef.current, plane = scrubPlaneRef.current, handle = scrubHandleRef.current;
    if (!ta.valid || !tr || !group || !plane || !handle) return;
    const span = Math.max(1, tr.maxT - tr.minT);
    const s = Math.min(1, Math.max(0, (tUs - tr.minT) / span));
    const z = ta.bottom.z + s * (ta.top.z - ta.bottom.z);
    const ox = ta.bottom.x, oy = ta.bottom.y, xEnd = ta.xEnd, yEnd = ta.yEnd;
    plane.position.set((ox + xEnd) / 2, (oy + yEnd) / 2, z);
    plane.scale.set(Math.abs(xEnd - ox) || 1, Math.abs(yEnd - oy) || 1, 1);
    // The slider plane lives in sliderGroup; its orientation is updated every
    // frame in the render loop (camera-facing about the T axis). Here we just
    // keep its anchor at the T-axis corner. local Y → world Z, local X → right.
    const z0 = ta.bottom.z;
    const sg = scrubSliderGroupRef.current;
    if (sg) sg.position.set(ox, oy, 0);
    // Flat filled track (local x = width, local y = world z).
    const fill = scrubFillRef.current;
    if (fill) {
      fill.position.set(0, (z0 + z) / 2, 0);
      fill.scale.set(0.03, Math.max(1e-4, z - z0), 1);
    }
    // Flat capsule thumb.
    const handleH = 0.11, handleW = handleH * (168 / 252);
    handle.position.set(0, z, 0.001);
    handle.scale.set(handleW, handleH, 1);
    // Flat time tooltip bubble, to the right of the handle in-plane.
    const label = scrubLabelRef.current, lcanvas = scrubLabelCanvasRef.current, ltex = scrubLabelTexRef.current;
    if (label && lcanvas && ltex) {
      const tipH = 0.08, tipW = tipH * (lcanvas.width / lcanvas.height);
      label.position.set(handleW / 2 + 0.015 + tipW / 2, z, 0.001);
      label.scale.set(tipW, tipH, 1);
      const lctx = lcanvas.getContext("2d");
      if (lctx) {
        const W = lcanvas.width, H = lcanvas.height, my = H / 2;
        lctx.clearRect(0, 0, W, H);
        const text = `${(tUs / 1e6).toFixed(1)} s`;
        lctx.font = `600 46px Inter, "Segoe UI", monospace`;
        const tw = lctx.measureText(text).width;
        const tail = 20, padX = 26, bh = 80, r = 16;
        const bx = tail + 40, by = my - bh / 2, bw = tw + padX * 2;
        lctx.beginPath();
        lctx.moveTo(bx + r, by);
        lctx.lineTo(bx + bw - r, by); lctx.quadraticCurveTo(bx + bw, by, bx + bw, by + r);
        lctx.lineTo(bx + bw, by + bh - r); lctx.quadraticCurveTo(bx + bw, by + bh, bx + bw - r, by + bh);
        lctx.lineTo(bx + r, by + bh); lctx.quadraticCurveTo(bx, by + bh, bx, by + bh - r);
        lctx.lineTo(bx, my + 13); lctx.lineTo(bx - tail, my); lctx.lineTo(bx, my - 13);
        lctx.lineTo(bx, by + r); lctx.quadraticCurveTo(bx, by, bx + r, by);
        lctx.closePath();
        lctx.fillStyle = "#ffffff"; lctx.fill();
        lctx.strokeStyle = "rgba(108,92,231,0.3)"; lctx.lineWidth = 2; lctx.stroke();
        lctx.fillStyle = "#5b4fd6"; lctx.textBaseline = "middle"; lctx.textAlign = "left";
        lctx.fillText(text, bx + padX, my + 2);
        ltex.needsUpdate = true;
      }
    }
    const tri = scrubTriRef.current;
    if (tri) {
      // Mark the corner that the PREVIEW currently shows as its top-left. Invert
      // the preview's flip+rotation to find which raw image corner maps to the
      // displayed (0,0): solve forward tf for output (0,0) per rotation, then
      // undo the flips. px'∈{0,W}, py'∈{0,H}.
      const rot = previewRotRef.current, fx = previewFlipXRef.current, fy = previewFlipYRef.current;
      let pxF: 0 | 1, pyF: 0 | 1; // flipped-pixel corner: 0 = min, 1 = max
      if (rot === 90) { pxF = 0; pyF = 1; }
      else if (rot === 180) { pxF = 1; pyF = 1; }
      else if (rot === 270) { pxF = 1; pyF = 0; }
      else { pxF = 0; pyF = 0; }
      const pxR = fx ? (pxF ? 0 : 1) : pxF; // undo horizontal flip
      const pyR = fy ? (pyF ? 0 : 1) : pyF; // undo vertical flip
      // raw pixel x: 0 → ox (min X), W → xEnd; raw pixel y: 0 → oy (+Y top), H → yEnd.
      const cornerX = pxR ? xEnd : ox;
      const cornerY = pyR ? yEnd : oy;
      const signX = cornerX === ox ? 1 : -1; // legs point toward the interior
      const signY = cornerY === oy ? 1 : -1;
      const d = 0.06 * Math.min(Math.abs(xEnd - ox), Math.abs(yEnd - oy)) || 0.025;
      tri.position.set(cornerX, cornerY, z + 0.002);
      tri.scale.set(signX * d, signY * d, 1);
    }
    lastScrubTUsRef.current = tUs;
    group.visible = true;
  }, []);

  // Keep preview-orientation refs current; redraw the corner marker when the
  // preview's mirror/rotation changes (no active drag needed).
  useEffect(() => {
    previewRotRef.current = previewRot;
    previewFlipXRef.current = previewFlipX;
    previewFlipYRef.current = previewFlipY;
    if (scrubGroupRef.current?.visible && lastScrubTUsRef.current != null) {
      updateScrubAt(lastScrubTUsRef.current);
    }
  }, [previewRot, previewFlipX, previewFlipY, updateScrubAt]);

  // Build the scrub visuals (blue XY plane + handle) with the 3D scene.
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || viewMode !== "3d") return;
    const group = new THREE.Group();
    group.visible = false;
    const planeGeo = new THREE.PlaneGeometry(1, 1);
    const planeMat = new THREE.MeshBasicMaterial({ color: 0x3c8cff, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false });
    const plane = new THREE.Mesh(planeGeo, planeMat);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(planeGeo), new THREE.LineBasicMaterial({ color: 0x3c8cff, transparent: true, opacity: 0.85, depthWrite: false }));
    plane.add(edges);
    // The slider as a flat 2D widget inside an angled vertical plane.
    const sliderGroup = new THREE.Group();
    // Slider fill: a flat purple rectangle from the origin up to the current
    // time (the "filled track"). Unit plane, scaled per-frame.
    const fillGeo = new THREE.PlaneGeometry(1, 1);
    const fillMat = new THREE.MeshBasicMaterial({ color: 0x8b5cf6, transparent: true, opacity: 0.95, side: THREE.DoubleSide, depthTest: false, depthWrite: false });
    const fill = new THREE.Mesh(fillGeo, fillMat); fill.renderOrder = 10;
    // Slider thumb: a flat capsule (texture on a plane) in the slider plane.
    const thumbTex = makeThumbTexture();
    const handleGeo = new THREE.PlaneGeometry(1, 1);
    const handleMat = new THREE.MeshBasicMaterial({ map: thumbTex, transparent: true, side: THREE.DoubleSide, depthTest: false, depthWrite: false });
    const handle = new THREE.Mesh(handleGeo, handleMat); handle.renderOrder = 12;
    // Time tooltip bubble — flat plane in the slider plane; persistent
    // canvas/texture redrawn each frame (no per-frame allocation).
    const labelCanvas = document.createElement("canvas");
    labelCanvas.width = 384; labelCanvas.height = 112;
    const labelTex = new THREE.CanvasTexture(labelCanvas);
    labelTex.minFilter = THREE.LinearFilter; labelTex.magFilter = THREE.LinearFilter; labelTex.anisotropy = 8;
    const labelGeo = new THREE.PlaneGeometry(1, 1);
    const labelMat = new THREE.MeshBasicMaterial({ map: labelTex, transparent: true, side: THREE.DoubleSide, depthTest: false, depthWrite: false });
    const label = new THREE.Mesh(labelGeo, labelMat); label.renderOrder = 13;
    sliderGroup.add(fill); sliderGroup.add(handle); sliderGroup.add(label);
    // Red right-triangle filling the plane's top-left corner = the video's
    // (0,0) pixel, so the plane's orientation is unambiguous. Unit triangle
    // (right angle at origin, extends +x and -y); positioned/scaled per frame.
    const triGeo = new THREE.BufferGeometry();
    triGeo.setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, -1, 0], 3));
    const triMat = new THREE.MeshBasicMaterial({ color: 0xff3b30, transparent: true, opacity: 0.95, side: THREE.DoubleSide, depthWrite: false });
    const tri = new THREE.Mesh(triGeo, triMat);
    group.add(plane); group.add(sliderGroup); group.add(tri);
    scene.add(group);
    scrubGroupRef.current = group; scrubPlaneRef.current = plane; scrubSliderGroupRef.current = sliderGroup;
    scrubHandleRef.current = handle; scrubFillRef.current = fill; scrubTriRef.current = tri;
    scrubLabelRef.current = label; scrubLabelCanvasRef.current = labelCanvas; scrubLabelTexRef.current = labelTex;
    return () => {
      scene.remove(group);
      planeGeo.dispose(); planeMat.dispose();
      (edges.geometry as THREE.BufferGeometry).dispose(); (edges.material as THREE.Material).dispose();
      fillGeo.dispose(); fillMat.dispose();
      handleGeo.dispose(); handleMat.dispose(); thumbTex.dispose();
      triGeo.dispose(); triMat.dispose();
      labelGeo.dispose(); labelTex.dispose(); labelMat.dispose();
      scrubGroupRef.current = null; scrubPlaneRef.current = null; scrubSliderGroupRef.current = null;
      scrubHandleRef.current = null; scrubFillRef.current = null; scrubTriRef.current = null;
      scrubLabelRef.current = null; scrubLabelCanvasRef.current = null; scrubLabelTexRef.current = null;
    };
  }, [sceneVersion, viewMode]);

  // Yellow Ranges & Filters preview box: created with the scene, positioned by
  // the effect below. Translucent fill + bright edges; a 3D box in 3D view, a
  // flat X/Y rectangle in XY view.
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const group = new THREE.Group();
    group.visible = false;
    const boxGeo = new THREE.BoxGeometry(1, 1, 1);
    const boxMat = new THREE.MeshBasicMaterial({ color: 0xffcc33, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false });
    const box = new THREE.Mesh(boxGeo, boxMat);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeo), new THREE.LineBasicMaterial({ color: 0xffcc33, transparent: true, opacity: 0.9, depthWrite: false }));
    box.add(edges);
    group.add(box);
    scene.add(group);
    filterBoxGroupRef.current = group;
    return () => {
      scene.remove(group);
      boxGeo.dispose(); boxMat.dispose();
      (edges.geometry as THREE.BufferGeometry).dispose(); (edges.material as THREE.Material).dispose();
      filterBoxGroupRef.current = null;
    };
  }, [sceneVersion, viewMode]);

  // Position/size the yellow preview box from the live filter draft, in the same
  // world space as the cloud. 3D: a box over X/Y/time (time→z via the T axis).
  // XY: a flat rectangle over the X/Y range (time collapsed) using the norm
  // transform. Hidden when there's no draft or the mapping isn't ready.
  useEffect(() => {
    const group = filterBoxGroupRef.current;
    if (!group) return;
    if (!filterPreview || !meta) { group.visible = false; return; }
    const W = meta.width || 1, H = meta.height || 1;
    if (viewMode === "3d") {
      const ta = tAxisRef.current, tr = tRangeRef.current;
      if (!ta.valid || !tr) { group.visible = false; return; }
      const ox = ta.bottom.x, oy = ta.bottom.y, xEnd = ta.xEnd, yEnd = ta.yEnd;
      const span = Math.max(1, tr.maxT - tr.minT);
      const wx = (px: number) => ox + (px / W) * (xEnd - ox);
      const wy = (py: number) => oy + (py / H) * (yEnd - oy);
      const wz = (tt: number) => ta.bottom.z + Math.min(1, Math.max(0, (tt - tr.minT) / span)) * (ta.top.z - ta.bottom.z);
      const x0 = wx(filterPreview.xMin), x1 = wx(filterPreview.xMax);
      const y0 = wy(filterPreview.yMin), y1 = wy(filterPreview.yMax);
      const z0 = wz(filterPreview.startUs), z1 = wz(filterPreview.endUs);
      group.position.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
      group.scale.set(Math.abs(x1 - x0) || 1e-3, Math.abs(y1 - y0) || 1e-3, Math.abs(z1 - z0) || 1e-3);
    } else {
      // XY: flat rectangle over the X/Y range, via the norm transform (same map
      // the points/picking use), z collapsed. Camera handles xyRotation; flip is
      // baked in via the norm's mul.
      const norm = normRef.current;
      if (!norm) { group.visible = false; return; }
      const xMul = norm.flipX ? -1 : 1, yMul = norm.flipY ? -1 : 1;
      const wx = (px: number) => (px - (W - 1) / 2 - norm.cx) * norm.scale * xMul;
      const wy = (py: number) => ((H - 1) / 2 - py - norm.cy) * norm.scale * yMul;
      const x0 = wx(filterPreview.xMin), x1 = wx(filterPreview.xMax);
      const y0 = wy(filterPreview.yMin), y1 = wy(filterPreview.yMax);
      // Nudge toward the camera (XY camera sits at +z, or -z when rotated 90°)
      // so depth-test doesn't hide the rect behind the z=0 points.
      const bz = (cameraRef.current && cameraRef.current.position.z < 0) ? -0.02 : 0.02;
      group.position.set((x0 + x1) / 2, (y0 + y1) / 2, bz);
      group.scale.set(Math.abs(x1 - x0) || 1e-3, Math.abs(y1 - y0) || 1e-3, 1e-3);
    }
    group.visible = true;
  }, [filterPreview, viewMode, sceneVersion, meta, manifest, xyFlipX, xyFlipY, xyRotation]);

  // Show the scrub plane/handle (at the current frame) while the tscrub tool is
  // active; hide it + dismiss the preview otherwise.
  useEffect(() => {
    if (toolMode === "tscrub" && viewMode === "3d") {
      updateScrubAt(currentFrameUsRef.current ?? (tRangeRef.current?.minT ?? 0));
    } else {
      if (scrubGroupRef.current) scrubGroupRef.current.visible = false;
      onTimeScrubRef.current?.(null);
    }
  }, [toolMode, viewMode, updateScrubAt]);
  const selectionMeshRef = useRef<THREE.Mesh | null>(null);
  const highlightWorldPos = useRef<THREE.Vector3 | null>(null);
  const normRef = useRef<NormTransform | null>(null);
  const metaRef = useRef(meta);  const onPointClickRef = useRef(onPointClick);
  const onSelectionRef = useRef(onSelection);
  const onWandSeedRef = useRef(onWandSeed);
  const onBrushRadiusChangeRef = useRef(onBrushRadiusChange);
  const tooltipRefStable = useRef(tooltipRef);
  const payloadRef = useRef(payload);
  const toolModeRef = useRef(toolMode);
  const brushRadiusRef = useRef(brushRadius);
  const selectThroughRef = useRef(selectThrough);
  // Multi-Points refs (read by the long-lived main effect's mouse handlers).
  const manifestRef = useRef(manifest);
  const chunksMapRef = useRef(chunksMap);
  const multiActiveRef = useRef(false);
  // Latest playback state — read by the main setup effect when (re-)creating
  // the ShaderMaterial so its frame-filter uniforms are seeded correctly.
  // Not in main effect's deps because that would tear down the renderer on
  // every playback tick. The dedicated frame-filter useEffect keeps these in
  // sync after the initial seed.
  const currentFrameUsRef = useRef(currentFrameUs);
  const frameWindowUsRef = useRef(frameWindowUs);
  // True when the chunk-paged renderer is the source of truth (manifest +
  // chunksMap supplied by the parent). Hoisted up here so refs syncs and
  // dependent effects below can read it without forward-ref hazards.
  const multiPointsActive = !!(manifest && chunksMap);
  useEffect(() => { onPointClickRef.current = onPointClick; }, [onPointClick]);
  useEffect(() => { onSelectionRef.current = onSelection; }, [onSelection]);
  useEffect(() => { onWandSeedRef.current = onWandSeed; }, [onWandSeed]);
  useEffect(() => { onBrushRadiusChangeRef.current = onBrushRadiusChange; }, [onBrushRadiusChange]);
  useEffect(() => { tooltipRefStable.current = tooltipRef; }, [tooltipRef]);
  useEffect(() => { payloadRef.current = payload; }, [payload]);
  useEffect(() => { toolModeRef.current = toolMode; }, [toolMode]);
  useEffect(() => { brushRadiusRef.current = brushRadius; }, [brushRadius]);
  useEffect(() => { selectThroughRef.current = selectThrough; }, [selectThrough]);
  useEffect(() => { metaRef.current = meta; }, [meta]);
  useEffect(() => { manifestRef.current = manifest; }, [manifest]);
  useEffect(() => { chunksMapRef.current = chunksMap; }, [chunksMap]);
  useEffect(() => { multiActiveRef.current = multiPointsActive; }, [multiPointsActive]);
  useEffect(() => { currentFrameUsRef.current = currentFrameUs; }, [currentFrameUs]);
  useEffect(() => { frameWindowUsRef.current = frameWindowUs; }, [frameWindowUs]);
  // Read by the in-place brush colouring helper (which lives inside the
  // long-lived main effect) when it needs to revert chunk colours on a
  // cancelled stroke.
  const labelsRef = useRef(labels);
  const selectedIdsRef = useRef(selectedIds);
  useEffect(() => { labelsRef.current = labels; }, [labels]);
  useEffect(() => { selectedIdsRef.current = selectedIds; }, [selectedIds]);
  // Display-aid refs so the colour helpers can read current values without
  // adding them to the geometry effect's deps.
  const colorModeRef = useRef(colorMode);
  const pointSizeScaleRef = useRef(pointSizeScale);
  const tMinRef = useRef(windowStartUs ?? 0);
  const tMaxRef = useRef(windowEndUs ?? 1);
  // Latest Ranges & Filters criteria, read by the chunk-build effect so newly
  // loaded chunks are culled to match (kept current by the filter effect below).
  const filterRef = useRef<PointFilter>({
    xMin: filterXMin, xMax: filterXMax, yMin: filterYMin, yMax: filterYMax,
    pol: filterPolarity, tMin: windowStartUs ?? -Infinity, tMax: windowEndUs ?? Infinity,
  });

  // Live point-size uniform — no geometry touch.
  useEffect(() => {
    pointSizeScaleRef.current = pointSizeScale;
    const mat = materialRef.current;
    if (mat) mat.uniforms.uSizeScale.value = pointSizeScale;
  }, [pointSizeScale]);

  // Colour mode or the time-window range changed → fully recolour the loaded
  // chunks (the time gradient is normalised over the window; polarity/label are
  // not, but a single rescan here is simplest and these changes are user-driven
  // and infrequent).
  useEffect(() => {
    colorModeRef.current = colorMode;
    tMinRef.current = windowStartUs ?? 0;
    tMaxRef.current = windowEndUs ?? 1;
    const renders = chunkRendersRef.current;
    if (renders.size === 0) return;
    for (const r of renders.values()) {
      buildChunkColors(r.payload, r.eventStartIndex, selectedIdsRef.current, labelsRef.current, r.colors, colorModeRef.current, tMinRef.current, tMaxRef.current);
      (r.geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    }
  }, [colorMode, windowStartUs, windowEndUs]);

  // Polarity-display config (unlabelled base colour + contrast) changed → push
  // the new values into the module-level state and fully recolour + re-upload
  // every loaded chunk so the polarity scheme updates live. Mirrors the
  // colorMode recolour effect above.
  useEffect(() => {
    setPolarityDisplay(unlabeledColor, polarityContrast);
    const renders = chunkRendersRef.current;
    if (renders.size === 0) return;
    for (const r of renders.values()) {
      buildChunkColors(r.payload, r.eventStartIndex, selectedIdsRef.current, labelsRef.current, r.colors, colorModeRef.current, tMinRef.current, tMaxRef.current);
      (r.geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    }
  }, [unlabeledColor, polarityContrast]);

  // Ranges & Filters changed (XY bounds / polarity / time window) → cull the
  // matching points from every loaded chunk by NaN-ing their positions. Points
  // are not re-fetched; payload.positions stays intact so picking is unaffected.
  useEffect(() => {
    const f: PointFilter = {
      xMin: filterXMin, xMax: filterXMax, yMin: filterYMin, yMax: filterYMax,
      pol: filterPolarity,
      tMin: windowStartUs ?? -Infinity, tMax: windowEndUs ?? Infinity,
    };
    filterRef.current = f;
    const w = meta?.width ?? 0, h = meta?.height ?? 0;
    const dataTMin = manifest?.t_min_us ?? -Infinity;
    const dataTMax = manifest?.t_max_us ?? Infinity;
    const active = filterIsActive(f, w, h, dataTMin, dataTMax);
    for (const r of chunkRendersRef.current.values()) applyChunkFilter(r, f, w, h, active);
  }, [filterXMin, filterXMax, filterYMin, filterYMax, filterPolarity, windowStartUs, windowEndUs, meta, manifest]);

  // Buffer build is async + chunked so multi-million-event loads do not
  // freeze the browser. Cancel an in-flight build when deps change.
  // SKIP when the multi-Points pipeline is active (manifest + chunksMap):
  // building a combined CloudBuffers there allocates ~135 MB of typed arrays
  // and triggers OOM on borderline machines — the whole point of the refactor.
  const [buffers, setBuffers] = useState<CloudBuffers | null>(null);
  const [buildProgress, setBuildProgress] = useState(0);
  // Bottom-right zoom slider state. zoomValue=1 is the camera's "default"
  // (the position OrbitControls.reset() returns to). Slider is logarithmic
  // so 0.1×–10× takes the same UI distance either side of 1.
  const [zoomValue, setZoomValue] = useState(1.0);
  // Per-axis stretch factors. Default 1; range 0.2–5. Implemented via
  // scene.scale (non-uniform), so points + axis lines + labels all stay
  // aligned. Independent of camera zoom (which is uniform).
  const [axisScaleX, setAxisScaleX] = useState(1.0);
  const [axisScaleY, setAxisScaleY] = useState(1.0);
  const [axisScaleZ, setAxisScaleZ] = useState(1.0);
  // Which per-axis popup is currently open (null = none).
  const [openAxisPanel, setOpenAxisPanel] = useState<null | "x" | "y" | "z">(null);
  // Close any open axis-scale popup if we leave 3D — it isn't shown in XY.
  useEffect(() => {
    if (viewMode === "xy") setOpenAxisPanel(null);
  }, [viewMode]);
  useEffect(() => {
    let cancelled = false;
    if (multiPointsActive) {
      setBuffers(null);
      setBuildProgress(1);
      return;
    }
    if (!payload) {
      setBuffers(null);
      setBuildProgress(0);
      return;
    }
    setBuildProgress(0);
    (async () => {
      const result = await buildBuffersChunked(
        payload, viewMode, selectedIds, labels, xyFlipY,
        () => cancelled,
        (frac) => { if (!cancelled) setBuildProgress(frac); },
      );
      if (!cancelled && result) {
        setBuffers(result);
        setBuildProgress(1);
      }
    })();
    return () => { cancelled = true; };
  }, [payload, viewMode, selectedIds, labels, xyFlipY, multiPointsActive]);

  useEffect(() => {
    buffersRef.current = buffers;
    if (buffers) normRef.current = buffers.norm;
    const geo = geometryRef.current;
    // Hide the legacy single-payload geometry when multi-Points mode is active
    // OR when buffers is null. Axes stay visible whenever data exists.
    if (axesRef.current) axesRef.current.visible = !!buffers || multiPointsActive;
    if (!geo || !buffers) {
      if (geo) geo.setDrawRange(0, 0);
      return;
    }
    geo.setAttribute("position", new THREE.BufferAttribute(buffers.positions, 3));
    // colours are uint8 normalised (range 0–255 → 0–1 in shader): 4× smaller
    // upload than Float32, identical visuals.
    geo.setAttribute("color", new THREE.BufferAttribute(buffers.colors, 3, true));
    geo.setAttribute("aTime", new THREE.BufferAttribute(buffers.times, 1));
    // Skip computeBoundingSphere — it iterates every position. Use the data's
    // known normalised bounds to set a static sphere covering all points.
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 2);
    geo.boundingBox = new THREE.Box3(new THREE.Vector3(-1.5, -1.5, -1.5), new THREE.Vector3(1.5, 1.5, 1.5));
    geo.setDrawRange(0, buffers.count);
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    geo.attributes.aTime.needsUpdate = true;
    // Push the dataset normalisation into shader uniforms — the shader does
    // (position - centre) × scale per vertex so we don't need a duplicate
    // Float32Array of normalised positions on the JS heap.
    const matRef = materialRef.current;
    if (matRef) {
      const { cx, cy, cz, scale, flipY: nFlipY, flipX: nFlipX } = buffers.norm;
      (matRef.uniforms.uPosCenter.value as THREE.Vector3).set(cx, cy, cz);
      matRef.uniforms.uPosScale.value = scale;
      matRef.uniforms.uYMul.value = nFlipY ? -1 : 1;
      matRef.uniforms.uXMul.value = nFlipX ? -1 : 1;
    }
  }, [buffers, multiPointsActive]);

  // Multi-Points reconciliation — add new chunks to the scene, dispose evicted
  // ones, refresh colours when selection/labels change. Each chunk owns its own
  // BufferGeometry + Points so we never allocate a combined payload (the
  // borderline-machine OOM source). The shared material does the
  // (position - centre) × scale normalisation in the shader, so all chunks
  // align in world space using the manifest-derived global norm.
  useEffect(() => {
    const scene = sceneRef.current;
    const material = materialRef.current;
    if (!scene || !material) return;
    const renders = chunkRendersRef.current;

    // No dataset (e.g. Close Project) → dispose every chunk geometry and wipe
    // the 2D axis overlay so the viewport actually empties. Without this the old
    // cloud + axes linger behind the "no point cloud" message.
    if (!manifest || !chunksMap || !visibleChunkIds) {
      for (const [chunkId, render] of [...renders]) {
        scene.remove(render.points);
        render.geometry.dispose();
        renders.delete(chunkId);
      }
      const ov = overlayRef.current;
      const octx = ov ? ov.getContext("2d") : null;
      if (ov && octx) octx.clearRect(0, 0, ov.width, ov.height);
      return;
    }

    const norm = computeManifestNorm(manifest, viewMode === "xy" && xyFlipY, viewMode === "xy" && xyFlipX);
    normRef.current = norm;
    (material.uniforms.uPosCenter.value as THREE.Vector3).set(norm.cx, norm.cy, norm.cz);
    material.uniforms.uPosScale.value = norm.scale;
    material.uniforms.uYMul.value = norm.flipY ? -1 : 1;
    material.uniforms.uXMul.value = norm.flipX ? -1 : 1;

    const visible = new Set(visibleChunkIds);

    // Evict chunks not in the visible set (or removed from chunksMap).
    for (const [chunkId, render] of [...renders]) {
      if (!visible.has(chunkId) || !chunksMap.has(chunkId)) {
        scene.remove(render.points);
        render.geometry.dispose();
        renders.delete(chunkId);
      }
    }

    // Index manifest by chunk_id for O(1) lookup.
    const manifestById = new Map<number, import("../api/types").ChunkManifestEntry>();
    for (const c of manifest.chunks) manifestById.set(c.chunk_id, c);

    for (const chunkId of visibleChunkIds) {
      const payload = chunksMap.get(chunkId);
      const entry = manifestById.get(chunkId);
      if (!payload || !entry) continue;
      const existing = renders.get(chunkId);
      if (existing && existing.payload === payload) {
        // Same data reference — geometry is already current. Colour changes
        // (selection / label) are applied incrementally by the dedicated
        // effects below, so we deliberately do NOT rescan the chunk here.
        // (This is what removes the O(all visible points) recolour that used to
        // run on every selection change / Assign.)
        continue;
      }
      if (existing) {
        scene.remove(existing.points);
        existing.geometry.dispose();
      }
      // New chunk: colour once using the latest selection/labels/colour-mode
      // via refs (so this effect needn't depend on them and re-run for all).
      const colors = buildChunkColors(payload, entry.event_start_index, selectedIdsRef.current, labelsRef.current, undefined, colorModeRef.current, tMinRef.current, tMaxRef.current);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(payload.positions, 3));
      const colorAttr = new THREE.BufferAttribute(colors, 3, true);
      colorAttr.setUsage(THREE.DynamicDrawUsage); // colours mutate often (label/selection)
      geometry.setAttribute("color", colorAttr);
      geometry.setAttribute("aTime", new THREE.BufferAttribute(payload.t_us, 1));
      // Static bounding sphere — all chunks share the manifest-derived norm so
      // their normalised positions live in the unit cube. Skip computeBounding*
      // (which would scan every position).
      geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 2);
      geometry.boundingBox = new THREE.Box3(new THREE.Vector3(-1.5, -1.5, -1.5), new THREE.Vector3(1.5, 1.5, 1.5));
      const points = new THREE.Points(geometry, material);
      points.frustumCulled = false; // bounding sphere is conservative; avoid culls
      scene.add(points);
      const renderObj: ChunkRender = {
        chunkId,
        eventStartIndex: entry.event_start_index,
        eventCount: payload.count,
        payload,
        colors,
        geometry,
        points,
      };
      renders.set(chunkId, renderObj);
      // Cull to the current Ranges & Filters criteria so freshly loaded chunks
      // match the already-filtered ones.
      const fW = meta?.width ?? 0, fH = meta?.height ?? 0;
      applyChunkFilter(renderObj, filterRef.current, fW, fH,
        filterIsActive(filterRef.current, fW, fH, manifest.t_min_us, manifest.t_max_us));
    }

    tRangeRef.current = { minT: manifest.t_min_us, maxT: manifest.t_max_us };
    // sceneVersion bumps every time the main setup effect (re-)builds the
    // renderer; depending on it ensures recon re-attaches chunks to the new
    // scene, regardless of whether viewMode / xyRotation / gpuPreference /
    // theme triggered the rebuild.
    // NOTE: selectedIds & labels are intentionally NOT deps — colour updates are
    // incremental (selection-delta / label-patch / palette effects below). New
    // chunks read the current selection/labels via refs at creation time.
  }, [manifest, chunksMap, visibleChunkIds, viewMode, xyFlipY, xyFlipX, sceneVersion]);

  // ── Incremental colour updates (no geometry rebuild, no chunk re-fetch) ────
  // Recolour an arbitrary set of GLOBAL event ids across loaded chunks. Used by
  // the selection-delta effect. Binary-searches sorted chunk ranges so a large
  // changed set stays ~O(n log chunks).
  const recolorGlobalIds = useCallback((globalIds: Iterable<number>) => {
    const renders = chunkRendersRef.current;
    if (renders.size === 0) return;
    const labelLut = buildLabelLut(labelsRef.current);
    const sids = selectedIdsRef.current;
    const ranges: { r: ChunkRender; start: number; end: number }[] = [];
    for (const r of renders.values()) ranges.push({ r, start: r.eventStartIndex, end: r.eventStartIndex + r.eventCount });
    ranges.sort((a, b) => a.start - b.start);
    const perRender = new Map<ChunkRender, number[]>();
    for (const id of globalIds) {
      let lo = 0, hi = ranges.length - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const R = ranges[mid];
        if (id < R.start) hi = mid - 1;
        else if (id >= R.end) lo = mid + 1;
        else { let arr = perRender.get(R.r); if (!arr) { arr = []; perRender.set(R.r, arr); } arr.push(id - R.start); break; }
      }
    }
    for (const [r, idxs] of perRender) {
      recolorChunkIndices(r.payload, r.eventStartIndex, idxs, sids, labelLut, r.colors, colorModeRef.current, tMinRef.current, tMaxRef.current);
      uploadChangedColors(r.geometry.attributes.color as THREE.BufferAttribute, idxs, r.payload.count);
    }
  }, []);

  // Selection-delta: only recolour points whose selected-state actually changed
  // (symmetric difference of prev vs current). Replaces the old full rescan, so
  // clearing a 200k selection no longer touches every visible point.
  const prevSelectedRef = useRef<Set<number>>(new Set());
  useEffect(() => {
    const prev = prevSelectedRef.current;
    const cur = selectedIds;
    const changed: number[] = [];
    for (const id of cur) if (!prev.has(id)) changed.push(id);
    for (const id of prev) if (!cur.has(id)) changed.push(id);
    prevSelectedRef.current = new Set(cur);
    if (changed.length) recolorGlobalIds(changed);
  }, [selectedIds, recolorGlobalIds]);

  // Label-patch: optimistic Assign/Clear already mutated payload.label in place;
  // here we repaint just those points (grouped per chunk by useAppState).
  useEffect(() => {
    if (!labelPatch) return;
    const renders = chunkRendersRef.current;
    const labelLut = buildLabelLut(labelsRef.current);
    const sids = selectedIdsRef.current;
    for (const [cid, idxs] of labelPatch.perChunk) {
      const r = renders.get(cid);
      if (!r) continue;
      recolorChunkIndices(r.payload, r.eventStartIndex, idxs, sids, labelLut, r.colors, colorModeRef.current, tMinRef.current, tMaxRef.current);
      uploadChangedColors(r.geometry.attributes.color as THREE.BufferAttribute, idxs, r.payload.count);
    }
  }, [labelPatch]);

  // Palette change (user edits a label's colour / adds-removes a class). This is
  // infrequent, so a full rescan of loaded chunks is acceptable here.
  useEffect(() => {
    const renders = chunkRendersRef.current;
    if (renders.size === 0) return;
    for (const r of renders.values()) {
      buildChunkColors(r.payload, r.eventStartIndex, selectedIdsRef.current, labels, r.colors, colorModeRef.current, tMinRef.current, tMaxRef.current);
      (r.geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    }
  }, [labels]);

  // Recompute the XY-mode pixel-size uniform when relevant state changes.
  // Keep this OUT of the [buffers] effect above so window resizing doesn't
  // re-upload all GPU vertex buffers.
  // Deps include `viewMode` + `sceneVersion`: viewMode flips and scene
  // rebuilds both produce a fresh OrthographicCamera; without re-running, the
  // uniform stays at the init-time fallback (1.0) and points render at 2 px
  // until the first window resize.
  useEffect(() => {
    const mat = materialRef.current;
    const cam = cameraRef.current;
    if (!mat || !mat.uniforms.uXyPxScale || !(cam instanceof THREE.OrthographicCamera)) return;
    const norm = buffers?.norm ?? normRef.current;
    if (!norm) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const span = cam.top - cam.bottom;
    mat.uniforms.uXyPxScale.value = (norm.scale * size.height * dpr) / span;
  }, [buffers, manifest, size, viewMode, sceneVersion]);

  // Rebuild 3D axes when bounds / theme / meta / payload change without a
  // viewMode flip. (Mode flips are handled inline in the main effect because
  // the main effect creates a fresh axesGroup AFTER this effect's setup runs.)
  useEffect(() => {
    const group = axesGroupRef.current;
    if (!group) return;
    // In multi-Points mode there's no combined `buffers`; synthesise an
    // axes-only stand-in from the manifest norm + manifest sensor metadata so
    // populateAxesGroup keeps working unchanged.
    let axesBuffers = buffers;
    if (!axesBuffers && multiPointsActive && manifest && normRef.current) {
      axesBuffers = {
        positions: new Float32Array(0),
        colors: new Uint8Array(0),
        times: new Float32Array(0),
        count: 0,
        norm: normRef.current,
      };
    }
    if (viewMode !== "3d" || !axesBuffers) {
      disposeAxesGroupChildren(group);
      tRangeRef.current = null;
      return;
    }
    const axesMeta = meta ?? (manifest ? { width: manifest.width, height: manifest.height } : null);
    // Compute tRange explicitly so the "T · <duration>" axis-end label is
    // populated even when there's no combined `payload` (multi-Points mode).
    const tRange = payload
      ? computePayloadTRange(payload)
      : manifest
        ? { minT: manifest.t_min_us, maxT: manifest.t_max_us }
        : null;
    // Tick density scales with the camera-zoom level so a 4× zoom-in shows
    // ~4× more graduations across the visible axis. Quantised to powers of
    // 2 to avoid rebuilding axes on every micro-zoom-change.
    const tickDensity = Math.pow(2, Math.max(0, Math.round(Math.log2(Math.max(1, zoomValue)))));
    populateAxesGroup(group, axesBuffers, axesMeta, payload, theme, tRange, tickDensity,
      { x: axisScaleX, y: axisScaleY, z: axisScaleZ }, tAxisRef.current);
    tRangeRef.current = tRange;
    // sceneVersion in deps: when the main setup effect rebuilds the renderer
    // (viewMode / xyRotation / theme / gpuPreference change), it bumps
    // sceneVersion AFTER axesGroupRef is reset; this rerun ensures the new
    // axesGroup gets repopulated. Without it, XY → 3D leaves the 3D scene
    // axes-less.
  }, [buffers, theme, viewMode, meta, payload, manifest, multiPointsActive, sceneVersion, zoomValue, axisScaleX, axisScaleY, axisScaleZ]);

  // Drive the camera from the zoom slider. For OrthographicCamera we use
  // camera.zoom directly (cheap, no projection rebuild). For Perspective we
  // change distance to target — keeps the view centred and matches what
  // OrbitControls' wheel does.
  useEffect(() => {
    const cam = cameraRef.current;
    const ctrl = controlsRef.current;
    if (!cam || !ctrl) return;
    if (cam instanceof THREE.OrthographicCamera) {
      cam.zoom = zoomValue;
      cam.updateProjectionMatrix();
    } else if (cam instanceof THREE.PerspectiveCamera) {
      const target = ctrl.target;
      const offset = cam.position.clone().sub(target);
      const ud = cam.userData as { baseDistance?: number };
      if (ud.baseDistance === undefined) ud.baseDistance = offset.length();
      const newDist = ud.baseDistance / Math.max(0.0001, zoomValue);
      const dir = offset.lengthSq() > 1e-9 ? offset.normalize() : new THREE.Vector3(0, 0, 1);
      cam.position.copy(target).add(dir.multiplyScalar(newDist));
    }
    ctrl.update();
  }, [zoomValue, sceneVersion]);

  // Per-axis stretch — write shader uniforms so the points stretch but
  // text labels / arrowheads / origin marker keep their uniform world-space
  // sizes. The axes-rebuild effect re-runs on the same deps and places the
  // labels at the matching scaled world positions.
  useEffect(() => {
    const mat = materialRef.current;
    if (!mat) return;
    // Per-axis stretch is a 3D-only aid. In XY view it must be 1 — otherwise an
    // X/Y stretch set in 3D distorts (and pushes off-screen) the XY view.
    const is3d = viewMode === "3d";
    mat.uniforms.uAxisScaleX.value = is3d ? axisScaleX : 1;
    mat.uniforms.uAxisScaleY.value = is3d ? axisScaleY : 1;
    mat.uniforms.uAxisScaleZ.value = is3d ? axisScaleZ : 1;
  }, [axisScaleX, axisScaleY, axisScaleZ, sceneVersion, viewMode]);

  // Sync the zoom slider value FROM the camera every render frame so wheel
  // zoom (OrbitControls) reflects in the slider position. Uses a ref for
  // the comparison value so the rAF loop isn't torn down + restarted every
  // time setZoomValue fires (would otherwise thrash on each wheel tick).
  const zoomValueRef = useRef(zoomValue);
  useEffect(() => { zoomValueRef.current = zoomValue; }, [zoomValue]);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const cam = cameraRef.current;
      if (!cam) return;
      let v: number | null = null;
      if (cam instanceof THREE.OrthographicCamera) {
        v = cam.zoom;
      } else if (cam instanceof THREE.PerspectiveCamera) {
        const ctrl = controlsRef.current;
        const ud = cam.userData as { baseDistance?: number };
        if (ctrl && ud.baseDistance !== undefined) {
          const d = cam.position.distanceTo(ctrl.target);
          if (d > 1e-6) v = ud.baseDistance / d;
        }
      }
      const prev = zoomValueRef.current;
      if (v !== null && Math.abs(v - prev) / Math.max(1e-6, prev) > 0.01) {
        setZoomValue(v);
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [sceneVersion]);

  // Update frame filter uniforms reactively (no buffer rebuild).
  // sceneVersion is in deps so this re-applies after the main setup effect
  // creates a FRESH ShaderMaterial (viewMode / theme / xyRotation /
  // gpuPreference change). Without this dep the brand-new material keeps its
  // init defaults (uFrameStart=0, uFrameEnd=10000), which rejects every event
  // outside the first 10 ms of the file → blank XY view when paused.
  useEffect(() => {
    const mat = materialRef.current;
    if (!mat) return;
    mat.uniforms.uFrameStart.value = currentFrameUs ?? 0;
    mat.uniforms.uFrameEnd.value = (currentFrameUs ?? 0) + (frameWindowUs ?? 10000);
    mat.uniforms.uUseFrameFilter.value = viewMode === "xy" ? 1.0 : 0.0;
  }, [currentFrameUs, frameWindowUs, viewMode, sceneVersion]);

  // Reactively update OrbitControls mouse buttons when toolMode changes
  useEffect(() => {
    const controls = controlsRef.current;
    // Clear overlay and hide selection surface when switching to pan
    const overlay = overlayRef.current;
    if (overlay) {
      const ctx = overlay.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, overlay.width, overlay.height);
    }
    const selMesh = selectionMeshRef.current;
    if (toolMode === "pan" && selMesh) selMesh.visible = false;
    if (!controls) return;
    if (toolMode && toolMode !== "pan") {
      controls.mouseButtons = { LEFT: undefined as any, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: viewMode === "3d" ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN };
    } else {
      controls.mouseButtons = { LEFT: viewMode === "3d" ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    }
  }, [toolMode, viewMode]);

  // Update blue selection surface — convex hull wrapping selected points
  useEffect(() => {
    const mesh = selectionMeshRef.current;
    if (!mesh) return;
    // Skip the hull entirely for empty OR large selections. Building it walks
    // every selected id (findEventAcrossChunks is O(selected × chunks)) plus a
    // Vector3 per point and a ConvexGeometry — a multi-hundred-ms second long
    // task right after a big "select all". The brush/lasso overlay + the white
    // point highlight already convey the selection, so the hull is just an
    // optional decoration for small selections.
    const MAX_HULL_SELECTION = 5_000;
    if (selectedIds.size === 0 || selectedIds.size > MAX_HULL_SELECTION) { mesh.visible = false; return; }
    const isMulti = multiActiveRef.current;
    const is3d = viewMode === "3d";
    const allPts: THREE.Vector3[] = [];

    if (isMulti) {
      const norm = normRef.current;
      if (!norm) { mesh.visible = false; return; }
      // Lookup each selected event_id across loaded chunks. O(|selected| ·
      // chunks) — fine because chunks are bounded (~64) and selectedIds is
      // typically a few thousand. For very large selections we could index
      // chunks by event_start_index and binary-search, but this is good enough
      // until profiling shows otherwise.
      const su = materialRef.current?.uniforms;
      const sfx = su ? su.uAxisScaleX.value : 1, sfy = su ? su.uAxisScaleY.value : 1, sfz = su ? su.uAxisScaleZ.value : 1;
      for (const eid of selectedIds) {
        const ev = findEventAcrossChunks(chunkRendersRef.current.values(), norm, is3d, eid, sfx, sfy, sfz);
        if (ev) allPts.push(ev.worldPos);
      }
    } else {
      if (!buffersRef.current || !payloadRef.current) { mesh.visible = false; return; }
      const buf = buffersRef.current;
      const pl = payloadRef.current;
      const { cx, cy, cz, scale, flipY: nFlipY, flipX: nFlipX } = buf.norm;
      const yMul = nFlipY ? -1 : 1;
      const xMul = nFlipX ? -1 : 1;
      for (let i = 0; i < buf.count; i++) {
        if (!selectedIds.has(Number(pl.event_id[i]))) continue;
        const wx = (buf.positions[i * 3] - cx) * scale * xMul;
        const wy = (buf.positions[i * 3 + 1] - cy) * scale * yMul;
        const wz = is3d ? (buf.positions[i * 3 + 2] - cz) * scale : 0;
        allPts.push(new THREE.Vector3(wx, wy, wz));
      }
    }
    if (allPts.length < 4) { mesh.visible = false; return; }

    // Sample for convex hull performance (max 1500 points)
    let pts: THREE.Vector3[];
    if (allPts.length > 1500) {
      pts = [];
      const step = allPts.length / 1500;
      for (let s = 0; s < 1500; s++) {
        pts.push(allPts[Math.floor(s * step)]);
      }
    } else {
      pts = allPts;
    }

    // Compute centroid
    const centroid = new THREE.Vector3();
    for (const p of pts) centroid.add(p);
    centroid.divideScalar(pts.length);

    // Expand points outward from centroid by 15%
    const expandFactor = 1.15;
    const expandedPts = pts.map(p => {
      const dir = new THREE.Vector3().subVectors(p, centroid);
      return centroid.clone().add(dir.multiplyScalar(expandFactor));
    });

    try {
      const hullGeo = new ConvexGeometry(expandedPts);
      mesh.geometry.dispose();
      mesh.geometry = hullGeo;
      mesh.visible = true;

      // Remove old edge lines
      const oldEdges = mesh.getObjectByName("selectionEdges");
      if (oldEdges) { (oldEdges as THREE.LineSegments).geometry.dispose(); mesh.remove(oldEdges); }

      // Add blue edge lines
      const edgesGeo = new THREE.EdgesGeometry(hullGeo, 15);
      const edgeMat = new THREE.LineBasicMaterial({ color: 0x3c8cff, transparent: true, opacity: 0.6 });
      const edgeLines = new THREE.LineSegments(edgesGeo, edgeMat);
      edgeLines.name = "selectionEdges";
      mesh.add(edgeLines);
    } catch {
      // ConvexGeometry can fail with degenerate input — fall back to invisible
      mesh.visible = false;
    }
  }, [selectedIds]);

  useEffect(() => {
    if (handleRef) {
      handleRef.current = {
        focusOnPoint: (pointIndex: number) => {
          const cam = cameraRef.current;
          const ctrl = controlsRef.current;
          if (!cam || !ctrl) return;
          const fIs3d = viewMode === "3d";
          let target: THREE.Vector3 | null = null;
          const isMulti = multiActiveRef.current;
          if (isMulti) {
            // pointIndex is a global event_id in multi-Points mode.
            const norm = normRef.current;
            if (!norm) return;
            const fu = materialRef.current?.uniforms;
            const ev = findEventAcrossChunks(chunkRendersRef.current.values(), norm, fIs3d, pointIndex,
              fu ? fu.uAxisScaleX.value : 1, fu ? fu.uAxisScaleY.value : 1, fu ? fu.uAxisScaleZ.value : 1);
            if (!ev) return;
            target = ev.worldPos;
          } else {
            const buf = buffersRef.current;
            if (!buf || pointIndex < 0 || pointIndex >= buf.count) return;
            const { cx, cy, cz, scale, flipY: nFlipY, flipX: nFlipX } = buf.norm;
            const fyMul = nFlipY ? -1 : 1;
            const fxMul = nFlipX ? -1 : 1;
            target = new THREE.Vector3(
              (buf.positions[pointIndex * 3] - cx) * scale * fxMul,
              (buf.positions[pointIndex * 3 + 1] - cy) * scale * fyMul,
              fIs3d ? (buf.positions[pointIndex * 3 + 2] - cz) * scale : 0,
            );
          }
          ctrl.target.copy(target);
          if (cam instanceof THREE.PerspectiveCamera) {
            const dir = new THREE.Vector3().subVectors(cam.position, target).normalize();
            cam.position.copy(target.clone().add(dir.multiplyScalar(0.5)));
          } else {
            cam.position.set(target.x, target.y, cam.position.z);
          }
          ctrl.update();
        },
      };
    }
  }, [handleRef]);

  useEffect(() => {
    if (!hostRef.current) return;
    const observer = new ResizeObserver(([entry]) => {
      const r = entry.contentRect;
      setSize({ width: Math.max(1, r.width), height: Math.max(1, r.height) });
    });
    observer.observe(hostRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const mount = mountRef.current;
    const host = hostRef.current;
    if (!mount || !host) return;
    mount.replaceChildren();

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: gpuPreference });
    renderer.setClearColor(theme === "light" ? 0xe8e8e8 : 0x10151d, 1);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(size.width, size.height, false);
    renderer.domElement.className = "pointCanvas";
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    sceneRef.current = scene;
    const aspect = size.width / Math.max(size.height, 1);
    // XY-mode ortho is fitted tightly to the data box. After buildBuffers the
    // data world extents are halfW = sensorW / max(W,H), halfH = sensorH / max(W,H)
    // (because everything is normalised to the largest of W / H / time-axis).
    // For a 90°-rotated XY view the screen vertical/horizontal axes are swapped
    // because camera.up rotates onto world -X.
    const metaW = meta?.width ?? size.width;
    const metaH = meta?.height ?? size.height;
    const maxDim = Math.max(metaW, metaH);
    const halfW_world = metaW / maxDim;
    const halfH_world = metaH / maxDim;
    const screenHalfV = xyRotation === 90 ? halfW_world : halfH_world;
    const screenHalfH = xyRotation === 90 ? halfH_world : halfW_world;
    const dataDisplayAspect = screenHalfH / Math.max(screenHalfV, 1e-6);
    const pad = 1.03;
    const xyHalfH = aspect >= dataDisplayAspect
      ? screenHalfV * pad                          // canvas wider than data → fit vertically
      : (screenHalfH * pad) / aspect;              // canvas narrower → fit horizontally
    const camera = viewMode === "3d"
      ? new THREE.PerspectiveCamera(50, aspect, 0.01, 100)
      : new THREE.OrthographicCamera(-xyHalfH * aspect, xyHalfH * aspect, xyHalfH, -xyHalfH, 0.01, 100);
    if (viewMode === "3d") { camera.position.set(1.65, -2.15, 1.55); camera.up.set(0, 0, 1); }
    else if (xyRotation === 90) {
      // Transposed view: X axis vertical (left edge, going down),
      // Y axis horizontal (top edge, going right). Origin still top-left.
      // Achieved by looking from -Z with up = world -X.
      camera.position.set(0, 0, -4);
      camera.up.set(-1, 0, 0);
    } else {
      // Default: X axis horizontal (top edge), Y axis vertical (left edge).
      camera.position.set(0, 0, 4);
      camera.up.set(0, 1, 0);
    }
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    controls.enableRotate = viewMode === "3d";
    // Initial mouse buttons — updated reactively by the toolMode useEffect
    controls.mouseButtons = { LEFT: viewMode === "3d" ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    controlsRef.current = controls;

    const updateZoomSpeed = () => {
      if (viewMode === "3d" && camera instanceof THREE.PerspectiveCamera) {
        const dist = camera.position.distanceTo(controls.target);
        controls.zoomSpeed = Math.max(0.05, dist * 0.3);
      } else if (camera instanceof THREE.OrthographicCamera) {
        const effectiveRange = (camera.right - camera.left) / camera.zoom;
        controls.zoomSpeed = Math.max(0.05, effectiveRange * 0.25);
      }
    };
    controls.addEventListener("change", updateZoomSpeed);
    updateZoomSpeed();
    controls.update();

    // ── Overlay canvas (used by axis + selection drawing below) ──
    const overlay = overlayRef.current;
    const octx = overlay ? overlay.getContext("2d") : null;

    // Declared up here (before drawAxesOverlay, which reads it) so an early
    // controls "change" event during setup can't hit a temporal-dead-zone.
    let isDragging = false;
    // True while a time-scrub grab is in progress (started near the T axis).
    let scrubbing = false;

    // ── Axis overlay drawing ──
    const axisColor = theme === "light" ? "rgba(0,0,0,0.6)" : "rgba(255,255,255,0.5)";
    const axisLabelColor = theme === "light" ? "rgba(0,0,0,0.8)" : "rgba(255,255,255,0.7)";
    // Per-axis colours, matching the 3D Plotly-style axes (X = blue, Y = green).
    const AX_X = "#4a90e2", AX_Y = "#23b26d";
    const boxCol = theme === "light" ? "rgba(40,55,75,0.18)" : "rgba(180,195,215,0.18)";

    function drawAxesOverlay() {
      if (!octx || !overlay) return;
      // Don't clear if currently dragging for selection
      if (toolModeRef.current !== "pan" && isDragging) return;

      const norm = normRef.current;
      const m = metaRef.current;
      if (!norm || !m) return;

      // Use CSS dimensions (overlay.width/height are raw device pixels and
      // the 2D context is scaled by dpr so all drawing is in CSS units).
      const w = size.width, h = size.height;

      // Clear previous frame's axes so they don't accumulate while dragging.
      // Pass raw pixel dimensions divided by dpr is just `size`; cover whole canvas.
      octx.clearRect(0, 0, w, h);

      camera.updateMatrixWorld();
      const pv = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);

      function project(wx: number, wy: number, wz: number): [number, number] {
        const e = pv.elements;
        const wc = e[3] * wx + e[7] * wy + e[11] * wz + e[15];
        if (wc <= 0.001) return [-9999, -9999];
        const xc = (e[0] * wx + e[4] * wy + e[8] * wz + e[12]) / wc;
        const yc = (e[1] * wx + e[5] * wy + e[9] * wz + e[13]) / wc;
        return [(xc + 1) * 0.5 * w, (1 - yc) * 0.5 * h];
      }

      octx.save();
      octx.font = "10px monospace";
      octx.fillStyle = axisLabelColor;
      octx.strokeStyle = axisColor;
      octx.lineWidth = 1;
      octx.textAlign = "center";
      octx.textBaseline = "top";

      const { cx, cy, cz, scale, bounds } = norm;

      if (viewMode === "xy") {
        // Frame around the data canvas: trace the four world-space corners
        // (x_pixel∈[0, width-1], y_pixel∈[0, height-1]) so the frame, ticks
        // and labels follow the data box even when the user pans/zooms.
        // When `flipY` is set, world Y is negated so y_pixel=0 sits at the
        // bottom of the screen rather than the top.
        const yMul = norm.flipY ? -1 : 1;
        const ox = (-(m.width - 1) / 2 - cx) * scale;
        const oy = ((m.height - 1) / 2 - cy) * scale * yMul;       // y_pixel = 0
        const xMaxW = ((m.width - 1) / 2 - cx) * scale;
        const yMaxW = (-(m.height - 1) / 2 - cy) * scale * yMul;   // y_pixel = height-1

        const [tlX, tlY] = project(ox, oy, 0);              // top-left corner in screen
        const [trX, trY] = project(xMaxW, oy, 0);
        const [blX, blY] = project(ox, yMaxW, 0);
        const [brX, brY] = project(xMaxW, yMaxW, 0);

        // 3D-style frame: the two FAR edges drawn faint (the box), then the X
        // and Y axes coloured along their own edges (X = tl→tr, Y = tl→bl).
        octx.lineWidth = 1;
        octx.strokeStyle = boxCol;
        octx.beginPath();
        octx.moveTo(trX, trY); octx.lineTo(brX, brY); octx.lineTo(blX, blY);
        octx.stroke();
        octx.lineWidth = 2;
        octx.strokeStyle = AX_X;                       // X axis edge
        octx.beginPath(); octx.moveTo(tlX, tlY); octx.lineTo(trX, trY); octx.stroke();
        octx.strokeStyle = AX_Y;                       // Y axis edge
        octx.beginPath(); octx.moveTo(tlX, tlY); octx.lineTo(blX, blY); octx.stroke();

        const tickLen = 5;
        const labelOffset = 4;
        const fillBgPill = (cx_: number, cy_: number, text: string, anchor: "above" | "below" | "left" | "right", fg: string = axisLabelColor) => {
          octx.font = "10px Inter, 'Segoe UI', monospace";
          const metrics = octx.measureText(text);
          const tw = Math.ceil(metrics.width);
          const th = 14;
          const padX = 5, padY = 1;
          let x: number, y: number;
          if (anchor === "above") { x = cx_ - tw / 2 - padX; y = cy_ - th - padY * 2; }
          else if (anchor === "below") { x = cx_ - tw / 2 - padX; y = cy_ + padY; }
          else if (anchor === "left") { x = cx_ - tw - padX * 2; y = cy_ - th / 2 - padY; }
          else { x = cx_; y = cy_ - th / 2 - padY; }
          const r = 4, w_ = tw + padX * 2, h_ = th + padY * 2;
          octx.beginPath();
          octx.moveTo(x + r, y);
          octx.lineTo(x + w_ - r, y);
          octx.quadraticCurveTo(x + w_, y, x + w_, y + r);
          octx.lineTo(x + w_, y + h_ - r);
          octx.quadraticCurveTo(x + w_, y + h_, x + w_ - r, y + h_);
          octx.lineTo(x + r, y + h_);
          octx.quadraticCurveTo(x, y + h_, x, y + h_ - r);
          octx.lineTo(x, y + r);
          octx.quadraticCurveTo(x, y, x + r, y);
          octx.closePath();
          octx.fillStyle = theme === "light" ? "rgba(255,255,255,0.92)" : "rgba(18,24,34,0.85)";
          octx.fill();
          octx.lineWidth = 0.75;
          octx.strokeStyle = theme === "light" ? "rgba(40,55,75,0.18)" : "rgba(180,195,215,0.22)";
          octx.stroke();
          octx.fillStyle = fg;
          octx.textBaseline = "middle";
          octx.textAlign = "center";
          octx.fillText(text, x + w_ / 2, y + h_ / 2 + 0.5);
        };

        // Helper to draw a rotated (vertical) pill centred at (cxPos, cyPos).
        const fillBgPillVertical = (cxPos: number, cyPos: number, text: string, fg: string = axisLabelColor) => {
          octx.font = "10px Inter, 'Segoe UI', monospace";
          const metrics = octx.measureText(text);
          const tw = Math.ceil(metrics.width);
          const th = 14, padX = 5, padY = 1, r = 4;
          const w_ = tw + padX * 2, h_ = th + padY * 2;
          octx.save();
          octx.translate(cxPos, cyPos);
          octx.rotate(-Math.PI / 2);
          const x = -w_ / 2, y = -h_ / 2;
          octx.beginPath();
          octx.moveTo(x + r, y);
          octx.lineTo(x + w_ - r, y);
          octx.quadraticCurveTo(x + w_, y, x + w_, y + r);
          octx.lineTo(x + w_, y + h_ - r);
          octx.quadraticCurveTo(x + w_, y + h_, x + w_ - r, y + h_);
          octx.lineTo(x + r, y + h_);
          octx.quadraticCurveTo(x, y + h_, x, y + h_ - r);
          octx.lineTo(x, y + r);
          octx.quadraticCurveTo(x, y, x + r, y);
          octx.closePath();
          octx.fillStyle = theme === "light" ? "rgba(255,255,255,0.92)" : "rgba(18,24,34,0.85)";
          octx.fill();
          octx.lineWidth = 0.75;
          octx.strokeStyle = theme === "light" ? "rgba(40,55,75,0.18)" : "rgba(180,195,215,0.22)";
          octx.stroke();
          octx.fillStyle = fg;
          octx.textBaseline = "middle";
          octx.textAlign = "center";
          octx.fillText(text, 0, 0.5);
          octx.restore();
        };

        // In mode 0 the X axis lies horizontally on the top edge; in mode 90
        // it lies vertically on the left edge. Same world projection in both
        // cases, but the tick stub direction (and label anchor) rotates 90°.
        octx.lineWidth = 1;
        octx.strokeStyle = axisColor;
        const xModeVertical = xyRotation === 90;
        const xTickDir: [number, number] = xModeVertical ? [-1, 0] : [0, -1];
        const yTickDir: [number, number] = xModeVertical ? [0, -1] : [-1, 0];
        const xAnchor = xModeVertical ? "left" : "above";
        const yAnchor = xModeVertical ? "above" : "left";

        // X ticks (blue, matching the X axis)
        octx.strokeStyle = AX_X;
        for (const px of niceTicks(0, m.width - 1, 6)) {
          const rawX = px - (m.width - 1) / 2;
          const wx = (rawX - cx) * scale;
          const [sx, sy] = project(wx, oy, 0);
          octx.beginPath();
          octx.moveTo(sx, sy);
          octx.lineTo(sx + xTickDir[0] * tickLen, sy + xTickDir[1] * tickLen);
          octx.stroke();
          fillBgPill(sx + xTickDir[0] * (tickLen + labelOffset), sy + xTickDir[1] * (tickLen + labelOffset), `${Math.round(px)}`, xAnchor as any, AX_X);
        }
        // X end chip — "X · pixels" in the axis colour.
        const [xEndSX, xEndSY] = project(xMaxW, oy, 0);
        if (xModeVertical) {
          fillBgPillVertical(xEndSX + xTickDir[0] * 30, xEndSY + xTickDir[1] * 30, "X · pixels", AX_X);
        } else {
          fillBgPill(xEndSX + xTickDir[0] * 24, xEndSY + xTickDir[1] * 24, "X · pixels", xAnchor as any, AX_X);
        }

        // Y ticks (green, matching the Y axis)
        octx.strokeStyle = AX_Y;
        for (const py of niceTicks(0, m.height - 1, 5)) {
          const rawY = (m.height - 1) / 2 - py;
          const wy = (rawY - cy) * scale * yMul;
          const [sx, sy] = project(ox, wy, 0);
          octx.beginPath();
          octx.moveTo(sx, sy);
          octx.lineTo(sx + yTickDir[0] * tickLen, sy + yTickDir[1] * tickLen);
          octx.stroke();
          fillBgPill(sx + yTickDir[0] * (tickLen + labelOffset), sy + yTickDir[1] * (tickLen + labelOffset), `${Math.round(py)}`, yAnchor as any, AX_Y);
        }
        // Y end chip — "Y · pixels" in the axis colour.
        const [yEndSX, yEndSY] = project(ox, yMaxW, 0);
        if (!xModeVertical) {
          fillBgPillVertical(yEndSX + yTickDir[0] * 30, yEndSY + yTickDir[1] * 30, "Y · pixels", AX_Y);
        } else {
          fillBgPill(yEndSX + yTickDir[0] * 24, yEndSY + yTickDir[1] * 24, "Y · pixels", yAnchor as any, AX_Y);
        }
      } else {
        // 3D: lines, ticks AND text labels are all rendered as Three.js objects
        // in the scene (LineSegments + Sprite) so they participate in depth
        // testing. Nothing more to draw on the 2D overlay here apart from the
        // optional 100-px scale bar below.
        // Draw scale bar for pixel distance
        // 100 pixels in world space = 100 * scale
        const dist100 = 100 * scale;
        const barStart3d = new THREE.Vector3(0, 0, 0);
        const barEnd3d = new THREE.Vector3(dist100, 0, 0);
        const [bsx] = project(barStart3d.x, barStart3d.y, barStart3d.z);
        const [bex] = project(barEnd3d.x, barEnd3d.y, barEnd3d.z);
        const barPx = Math.abs(bex - bsx);
        if (barPx > 30 && barPx < w / 2) {
          const by = h - 16;
          octx.beginPath(); octx.moveTo(w - 20 - barPx, by); octx.lineTo(w - 20, by); octx.stroke();
          octx.beginPath(); octx.moveTo(w - 20 - barPx, by - 3); octx.lineTo(w - 20 - barPx, by + 3); octx.stroke();
          octx.beginPath(); octx.moveTo(w - 20, by - 3); octx.lineTo(w - 20, by + 3); octx.stroke();
          octx.textAlign = "center"; octx.textBaseline = "bottom";
          octx.fillText("100 px", w - 20 - barPx / 2, by - 4);
        }
      }
      octx.restore();
    }

    // Draw axes on camera change
    controls.addEventListener("change", drawAxesOverlay);
    drawAxesOverlay();

    const geometry = new THREE.BufferGeometry();
    geometryRef.current = geometry;
    if (buffers) {
      geometry.setAttribute("position", new THREE.BufferAttribute(buffers.positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(buffers.colors, 3));
      geometry.setAttribute("aTime", new THREE.BufferAttribute(buffers.times, 1));
      geometry.computeBoundingSphere();
    }

    // Framebuffer pixels per image-pixel at zoom=1 (used only in XY mode).
    // The ortho camera now fits the data tightly so visible vertical range is
    // 2 × xyHalfH world units. Multiply by renderer pixel ratio so the point
    // fully covers an image-pixel on HiDPI displays.
    const orthoVerticalSpan = viewMode === "xy" ? 2 * xyHalfH : 2.5;
    const rendererDpr = renderer.getPixelRatio();
    const xyPxScale = buffers
      ? (buffers.norm.scale * size.height * rendererDpr) / orthoVerticalSpan
      : 1.0;

    // Seed frame-filter uniforms from current playback state so a freshly-
    // built material doesn't reject every event before the dedicated
    // frame-filter useEffect runs (post-paint). Without this seed, switching
    // 3D → XY shows blank until the next currentFrameUs tick.
    const seedFrameStart = currentFrameUsRef.current ?? 0;
    const seedFrameEnd = seedFrameStart + (frameWindowUsRef.current ?? 10000);
    const material = new THREE.ShaderMaterial({
      vertexShader: sphereVertexShader,
      fragmentShader: sphereFragmentShader,
      uniforms: {
        uPointSize: { value: viewMode === "3d" ? 2.0 : 1.5 },
        uSizeScale: { value: pointSizeScaleRef.current },
        uFrameStart: { value: seedFrameStart },
        uFrameEnd: { value: seedFrameEnd },
        uUseFrameFilter: { value: viewMode === "xy" ? 1.0 : 0.0 },
        uZoom: { value: 1.0 },
        uIsXy: { value: viewMode === "xy" ? 1.0 : 0.0 },
        uXyPxScale: { value: xyPxScale },
        uPosCenter: { value: new THREE.Vector3(0, 0, 0) },
        uPosScale: { value: 1.0 },
        uYMul: { value: 1.0 },
        uXMul: { value: 1.0 },
        uAxisScaleX: { value: 1.0 },
        uAxisScaleY: { value: 1.0 },
        uAxisScaleZ: { value: 1.0 },
        uIs3d: { value: viewMode === "3d" ? 1.0 : 0.0 },
      },
      transparent: true,
      depthWrite: true,
    });
    materialRef.current = material;
    const points = new THREE.Points(geometry, material);
    scene.add(points);

    // Corner-anchored axis lines + tick marks. Lives in 3D scene so it
    // gets occluded by the point cloud's depth writes.
    const axesGroup = new THREE.Group();
    axesGroupRef.current = axesGroup;
    if (viewMode === "3d") {
      scene.add(axesGroup);
      // Populate immediately — the dedicated axes-rebuild useEffect runs BEFORE
      // this one on viewMode toggles, so it cannot see the freshly-created group.
      if (buffers) {
        populateAxesGroup(axesGroup, buffers, meta ?? null, payload, theme);
        tRangeRef.current = computePayloadTRange(payload);
      }
    }

    const glowTexture = createGlowTexture();
    const glowMat = new THREE.SpriteMaterial({ map: glowTexture, color: 0xffffff, transparent: true, opacity: 0.9, depthTest: true, depthWrite: false });
    const glowSprite = new THREE.Sprite(glowMat);
    glowSprite.scale.set(0.025, 0.025, 1);
    glowSprite.renderOrder = 2;
    glowSprite.visible = false;
    scene.add(glowSprite);

    // Semi-transparent blue selection surface (convex hull of selected points)
    const selectionMesh = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshBasicMaterial({ color: 0x3c8cff, transparent: true, opacity: 0.10, side: THREE.DoubleSide, depthWrite: false, depthTest: true })
    );
    selectionMesh.renderOrder = 0;
    selectionMesh.visible = false;
    scene.add(selectionMesh);
    selectionMeshRef.current = selectionMesh;

    let frame = 0;
    const isXyView = viewMode === "xy";
    // Scratch objects for the per-frame slider-plane orientation.
    const _slN = new THREE.Vector3(), _slR = new THREE.Vector3(), _slUp = new THREE.Vector3(0, 0, 1), _slM = new THREE.Matrix4();
    const renderLoop = () => {
      frame = requestAnimationFrame(renderLoop);
      controls.update();
      // Keep the time-slider plane facing the camera (rotating about the
      // vertical T axis) so it never turns edge-on / vanishes when orbiting.
      const sg = scrubSliderGroupRef.current;
      if (sg && scrubGroupRef.current?.visible) {
        const ta = tAxisRef.current;
        if (ta.valid) {
          _slN.set(camera.position.x - ta.bottom.x, camera.position.y - ta.bottom.y, 0);
          if (_slN.lengthSq() < 1e-9) _slN.set(1, 0, 0);
          _slN.normalize();
          _slR.copy(_slUp).cross(_slN).normalize();
          _slM.makeBasis(_slR, _slUp, _slN);
          sg.quaternion.setFromRotationMatrix(_slM);
        }
      }
      material.uniforms.uZoom.value = camera.zoom;
      // Push the latest frame-filter values every frame. Pulling these from
      // refs (instead of relying on the dedicated useEffect to fire) makes
      // the XY view robust to React's post-paint effect scheduling: a
      // freshly-mounted material no longer renders even one black frame
      // while waiting for the useEffect to push the right uniforms.
      const fStart = currentFrameUsRef.current ?? 0;
      const fWin = frameWindowUsRef.current ?? 10000;
      material.uniforms.uFrameStart.value = fStart;
      material.uniforms.uFrameEnd.value = fStart + fWin;
      material.uniforms.uUseFrameFilter.value = isXyView ? 1.0 : 0.0;
      renderer.render(scene, camera);
      // Keep the 2D axis overlay in sync every frame so view toggles, payload
      // arrivals, and any unexpected canvas wipes don't leave it blank.
      // The function early-returns during active selection drags.
      drawAxesOverlay();

      const wp = highlightWorldPos.current;
      if (wp) {
        // Re-assert glow visibility — the effect may have just rebuilt the sprite
        // (e.g. after window resize) with visible=false.
        glowSprite.visible = true;
        glowSprite.position.copy(wp);
        const pulse = 0.025 + Math.sin(Date.now() * 0.005) * 0.002;
        glowSprite.scale.set(pulse, pulse, 1);
        if (tooltipRefStable.current?.current) {
          const clipPos = wp.clone().project(camera);
          const rect = host.getBoundingClientRect();
          const sx = (clipPos.x + 1) * 0.5 * rect.width;
          const sy = (1 - clipPos.y) * 0.5 * rect.height;
          tooltipRefStable.current.current.style.left = (sx + 18) + "px";
          tooltipRefStable.current.current.style.top = (sy - 20) + "px";
        }
      } else {
        glowSprite.visible = false;
      }
    };
    renderLoop();

    // ── Projection helper ──
    function projectAllPoints(cam: THREE.Camera, buf: CloudBuffers, vw: number, vh: number): Float32Array {
      cam.updateMatrixWorld();
      const pv = new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
      const m = pv.elements;
      // buf.positions are RAW payload coords; apply norm to get world coords
      // (mirrors the GPU vertex shader transform).
      const { cx, cy, cz, scale, flipY: nFlipY, flipX: nFlipX } = buf.norm;
      const yMul = nFlipY ? -1 : 1;
      const xMul = nFlipX ? -1 : 1;
      const is3d = viewMode === "3d";
      const result = new Float32Array(buf.count * 3);
      for (let i = 0; i < buf.count; i++) {
        const px = (buf.positions[i * 3] - cx) * scale * xMul;
        const py = (buf.positions[i * 3 + 1] - cy) * scale * yMul;
        const pz = is3d ? (buf.positions[i * 3 + 2] - cz) * scale : 0;
        const wc = m[3] * px + m[7] * py + m[11] * pz + m[15];
        const o = i * 3;
        if (wc <= 0.001) { result[o] = -9999; result[o + 1] = -9999; result[o + 2] = wc; continue; }
        const xc = (m[0] * px + m[4] * py + m[8] * pz + m[12]) / wc;
        const yc = (m[1] * px + m[5] * py + m[9] * pz + m[13]) / wc;
        result[o] = (xc + 1) * 0.5 * vw;
        result[o + 1] = (1 - yc) * 0.5 * vh;
        result[o + 2] = wc;
      }
      return result;
    }

    // ── Selection overlay drawing ──
    // overlay + octx already initialised above
    const selColor = theme === "light" ? "#0066cc" : "#3c8cff";
    const selGlow = theme === "light" ? "rgba(0,102,204,0.18)" : "rgba(60,140,255,0.18)";
    const selFill = theme === "light" ? "rgba(0,102,204,0.06)" : "rgba(60,140,255,0.06)";
    const selStroke50 = theme === "light" ? "rgba(0,102,204,0.5)" : "rgba(60,140,255,0.5)";
    const selStroke60 = theme === "light" ? "rgba(0,102,204,0.6)" : "rgba(60,140,255,0.6)";

    function clearOverlay() {
      if (octx && overlay) octx.clearRect(0, 0, overlay.width, overlay.height);
    }

    function drawBox(x1: number, y1: number, x2: number, y2: number) {
      if (!octx) return;
      clearOverlay();
      const rx = Math.min(x1, x2), ry = Math.min(y1, y2);
      const rw = Math.abs(x2 - x1), rh = Math.abs(y2 - y1);
      // Glow layer (wider, more transparent)
      octx.shadowColor = selColor;
      octx.shadowBlur = 16;
      octx.fillStyle = selGlow;
      octx.fillRect(rx, ry, rw, rh);
      octx.shadowBlur = 0;
      // Fill
      octx.fillStyle = selFill;
      octx.fillRect(rx, ry, rw, rh);
      // Dashed border
      octx.setLineDash([6, 4]);
      octx.strokeStyle = selColor;
      octx.lineWidth = 1.5;
      octx.strokeRect(rx, ry, rw, rh);
      octx.setLineDash([]);
    }

    function drawCircle(cx: number, cy: number, r: number) {
      if (!octx) return;
      octx.beginPath();
      octx.arc(cx, cy, r, 0, Math.PI * 2);
      octx.strokeStyle = selStroke60;
      octx.lineWidth = 1.5;
      octx.stroke();
      // Glow
      octx.shadowColor = selColor;
      octx.shadowBlur = 10;
      octx.strokeStyle = selGlow;
      octx.stroke();
      octx.shadowBlur = 0;
    }

    function drawLasso(points: number[][]) {
      if (!octx || points.length < 2) return;
      clearOverlay();
      octx.beginPath();
      octx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i++) octx.lineTo(points[i][0], points[i][1]);
      octx.closePath();
      // Glow
      octx.shadowColor = selColor;
      octx.shadowBlur = 16;
      octx.fillStyle = selGlow;
      octx.fill();
      octx.shadowBlur = 0;
      // Fill
      octx.fillStyle = selFill;
      octx.fill();
      // Border
      octx.strokeStyle = selColor;
      octx.lineWidth = 1.5;
      octx.stroke();
    }

    // ── Interaction state ── (isDragging is declared earlier, above drawAxesOverlay)
    let dragStartX = 0, dragStartY = 0;
    let dragCurrentX = 0, dragCurrentY = 0;
    let lassoPoints: number[][] = [];
    // In legacy mode this holds buffer-row indices; in multi-Points mode it
    // holds global event_ids. Both are passed to onSelection as a number[];
    // the parent maps appropriately.
    let circleStrokeIndices = new Set<number>();
    // Multi-Points only: chunks whose colour buffers have been mutated to
    // white in-place by the brush. On cancel we rebuild their colours from
    // the current selectedIds + labels so the white spots disappear.
    let touchedStrokeChunks = new Set<number>();
    const revertStrokeColours = () => {
      const renders = chunkRendersRef.current;
      const sids = selectedIdsRef.current;
      const lbls = labelsRef.current;
      for (const cid of touchedStrokeChunks) {
        const r = renders.get(cid);
        if (!r) continue;
        buildChunkColors(r.payload, r.eventStartIndex, sids, lbls, r.colors, colorModeRef.current, tMinRef.current, tMaxRef.current);
        (r.geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
      }
      touchedStrokeChunks = new Set();
    };

    // Persistently show circle cursor when tool is circle
    function drawCircleCursor(e: MouseEvent) {
      if (toolModeRef.current !== "circle") return;
      const rect = host?.getBoundingClientRect();
      if (!rect) return;
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      if (cx < 0 || cy < 0 || cx > rect.width || cy > rect.height) return;
      clearOverlay();
      drawCircle(cx, cy, brushRadiusRef.current ?? 15);
    }

    // Determine selection operation from keyboard modifiers
    function getOp(e: MouseEvent): SelectionOp {
      if (e.shiftKey) return "add";
      if (e.ctrlKey || e.metaKey) return "subtract";
      return "set";
    }

    // Time-scrub: map the cursor onto the screen projection of the T axis and
    // report the time under it (for the XY preview). bottom = earliest (t_min),
    // top = latest (t_max).
    // Perspective-correct cursor→time: find the point on the world-space T axis
    // closest to the cursor's view ray. (The old code linearly interpolated the
    // cursor along the *projected* screen segment, which only holds for an ortho
    // camera; under perspective the error grows as you dolly/zoom in — that was
    // the cursor↔slider offset.)
    const SCRUB_NEAR_PX = 40; // a scrub only STARTS this close to the T axis
    const _ray = new THREE.Raycaster();
    const _ndc = new THREE.Vector2();
    const _u = new THREE.Vector3(), _w0 = new THREE.Vector3(), _axisPt = new THREE.Vector3();
    const doTimeScrub = (clientX: number, clientY: number, requireNear = false): boolean => {
      const cam = cameraRef.current;
      const ta = tAxisRef.current;
      const tr = tRangeRef.current;
      if (!cam || !ta.valid || !tr) return false;
      const rect = host.getBoundingClientRect();
      const vw = rect.width || 1, vh = rect.height || 1;
      const cx = clientX - rect.left, cy = clientY - rect.top;
      _ndc.set((cx / vw) * 2 - 1, -((cy / vh) * 2 - 1));
      _ray.setFromCamera(_ndc, cam);
      const O = _ray.ray.origin, v = _ray.ray.direction;
      // Closest point parameter s on the segment bottom + s·(top-bottom) to the ray.
      _u.subVectors(ta.top, ta.bottom);
      _w0.subVectors(ta.bottom, O);
      const a = _u.dot(_u), b = _u.dot(v), c = v.dot(v), d = _u.dot(_w0), e = v.dot(_w0);
      const denom = a * c - b * b;
      let s = denom > 1e-9 ? (b * e - c * d) / denom : 0;
      s = s < 0 ? 0 : s > 1 ? 1 : s;
      // Only grab the scrubber when the click lands near the T axis on screen.
      if (requireNear) {
        _axisPt.copy(_u).multiplyScalar(s).add(ta.bottom).project(cam);
        const sxp = (_axisPt.x + 1) * 0.5 * vw, syp = (1 - _axisPt.y) * 0.5 * vh;
        const pdx = cx - sxp, pdy = cy - syp;
        if (pdx * pdx + pdy * pdy > SCRUB_NEAR_PX * SCRUB_NEAR_PX) return false;
      }
      const tUs = tr.minT + s * (tr.maxT - tr.minT);
      updateScrubAt(tUs);  // move the blue plane + axis handle
      onTimeScrubRef.current?.({ tUs, sx: clientX, sy: clientY });
      return true;
    };

    const onDown = (e: MouseEvent) => {
      if (e.button !== 0) return;
      // Only react to clicks that begin INSIDE the WebGL canvas; otherwise a
      // click on a sibling UI control (e.g. the play button) would later run
      // the picking branch in onUp and wipe the overlay.
      const rect = host.getBoundingClientRect();
      if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) return;
      // Time scrubber (3D only): grab only when pressing near the T axis, then
      // follow the drag. Clicks away from the axis fall through (orbit/pick).
      if (toolModeRef.current === "tscrub" && cameraRef.current instanceof THREE.PerspectiveCamera) {
        scrubbing = doTimeScrub(e.clientX, e.clientY, true);
        return;
      }
      // Record drag start so pan-mode short-clicks can be detected as picks.
      dragStartX = e.clientX; dragStartY = e.clientY;
      dragCurrentX = e.clientX; dragCurrentY = e.clientY;
      isDragging = false;
      lassoPoints = [];
      circleStrokeIndices = new Set();
    };

    // rAF coalescing — large-dataset selection (16M+ events) projects every
    // event on every mousemove. Without throttling, fast drags queue up
    // dozens of expensive moves per frame and the UI freezes. We keep only
    // the latest mouse position between frames and process it once per rAF.
    let pendingMove: MouseEvent | null = null;
    let movePassiveScheduled = false;

    const handleMove = (e: MouseEvent) => {
      const tm = toolModeRef.current;
      if (tm === "circle" && !isDragging && e.buttons === 0) {
        drawCircleCursor(e);
      }
      if (tm === "tscrub") { if (e.buttons === 1 && scrubbing) doTimeScrub(e.clientX, e.clientY, false); return; }

      if (e.buttons !== 1) return; // Only when left button held
      if (tm === "pan") return;

      const rect = host.getBoundingClientRect();
      if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) return;

      const dx = e.clientX - dragStartX, dy = e.clientY - dragStartY;
      if (dx * dx + dy * dy < 9) return;
      isDragging = true;

      const cam = cameraRef.current;
      const buf = buffersRef.current;
      const isMulti = multiActiveRef.current;
      const norm = normRef.current;
      if (!cam) return;
      if (!isMulti && !buf) return;
      if (isMulti && !norm) return;

      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      dragCurrentX = e.clientX; dragCurrentY = e.clientY;

      if (tm === "box") {
        drawBox(dragStartX - rect.left, dragStartY - rect.top, cx, cy);
      } else if (tm === "circle") {
        const radius = brushRadiusRef.current ?? 15;
        const r2 = radius * radius;
        if (isMulti) {
          // Multi-Points: do projection + hit-test + GPU colour update in one
          // pass per chunk. Painting the actual point cloud white (instead of
          // an overlay canvas of fillRects) is what gives instant visual
          // feedback at 16M+ events without bogging down the drag.
          const is3d = viewMode === "3d";
          // XY-mode AABB pre-filter — rejects most events in 2 comparisons
          // each before the per-event projection. Null in 3D (perspective
          // unprojection of a screen rect isn't depth-invariant).
          const aabb = is3d ? null : computeBrushPayloadAABB(cam, norm!, cx, cy, radius, rect.width, rect.height);
          // XY: restrict the stroke to the visible playback window so each drag
          // frame scans only those events (a few thousand) instead of the whole
          // loaded time range (millions). 3D has no frame filter → full scan.
          const xyFrame = is3d ? null : { start: currentFrameUsRef.current ?? 0, end: (currentFrameUsRef.current ?? 0) + (frameWindowUsRef.current ?? 10000) };
          const bu = materialRef.current?.uniforms;
          brushStrokeChunks(
            chunkRendersRef.current.values(), cam, norm!, is3d, rect.width, rect.height,
            (sx, sy) => (sx - cx) ** 2 + (sy - cy) ** 2 <= r2,
            circleStrokeIndices, touchedStrokeChunks, aabb, xyFrame,
            bu ? bu.uAxisScaleX.value : 1, bu ? bu.uAxisScaleY.value : 1, bu ? bu.uAxisScaleZ.value : 1,
          );
          clearOverlay();
          drawCircle(cx, cy, radius);
        } else if (buf) {
          const projected = projectAllPoints(cam, buf, rect.width, rect.height);
          for (let i = 0; i < buf.count; i++) {
            const o = i * 3;
            const sx = projected[o], sy = projected[o + 1];
            if (sx < -999) continue;
            if ((sx - cx) ** 2 + (sy - cy) ** 2 <= r2) {
              circleStrokeIndices.add(i);
            }
          }
          clearOverlay();
          drawCircle(cx, cy, radius);
          if (octx) {
            octx.fillStyle = selStroke50;
            for (const idx of circleStrokeIndices) {
              octx.fillRect(projected[idx * 3] - 2, projected[idx * 3 + 1] - 2, 4, 4);
            }
          }
        }
      } else if (tm === "lasso") {
        // Only append when the cursor has moved ≥4px — keeps the vertex count
        // (and the pointer-up point-in-polygon cost) bounded on fast drags.
        const last = lassoPoints[lassoPoints.length - 1];
        if (!last || (cx - last[0]) ** 2 + (cy - last[1]) ** 2 >= 16) lassoPoints.push([cx, cy]);
        drawLasso(lassoPoints);
      }
    };

    const onMove = (e: MouseEvent) => {
      pendingMove = e;
      if (movePassiveScheduled) return;
      movePassiveScheduled = true;
      requestAnimationFrame(() => {
        movePassiveScheduled = false;
        const ev = pendingMove;
        pendingMove = null;
        if (ev) handleMove(ev);
      });
    };

    const onUp = (e: MouseEvent) => {
      if (e.button !== 0) return;
      // Time scrub: keep the preview + plane after release (cleared when the
      // tool is switched off), so the user can study the frame they landed on.
      if (toolModeRef.current === "tscrub") { isDragging = false; scrubbing = false; return; }
      // Classify click vs drag by total mouse displacement.
      const totalDx = e.clientX - dragStartX;
      const totalDy = e.clientY - dragStartY;
      const totalMovedSq = totalDx * totalDx + totalDy * totalDy;
      const isPan = toolModeRef.current === "pan";
      const isWand = toolModeRef.current === "wand";
      const clickPicks = isPan || isWand;
      // In pan mode, OrbitControls handles real drags (rotate/pan/zoom);
      // we only intercept short-click releases as picks. Wand is click-only
      // — a drag does nothing.
      if (isPan && totalMovedSq > 64) return;
      if (isWand && totalMovedSq > 64) { isDragging = false; return; }
      // In selection modes (box / circle / lasso), a short click should NOT
      // pop the picking tooltip. Just discard any half-drawn selection visual.
      if (!clickPicks && totalMovedSq <= 64) {
        clearOverlay();
        drawAxesOverlay();
        // Revert any chunk colours we painted white during a partial brush —
        // the user clicked rather than dragged, so no selection happened.
        revertStrokeColours();
        circleStrokeIndices = new Set();
        lassoPoints = [];
        isDragging = false;
        return;
      }
      if (totalMovedSq <= 64) {
        // Short click — point picking. Validate the click landed inside the
        // canvas BEFORE touching any visual state, otherwise clicks on outside
        // UI (e.g. play button) will wipe the overlay axes.
        const cam = cameraRef.current;
        const buf = buffersRef.current;
        const pl = payloadRef.current;
        const isMulti = multiActiveRef.current;
        const norm = normRef.current;
        if (!host || !cam) { isDragging = false; return; }
        if (!isMulti && (!buf || !pl)) { isDragging = false; return; }
        if (isMulti && !norm) { isDragging = false; return; }
        const rect = host.getBoundingClientRect();
        if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) { isDragging = false; return; }
        clearOverlay();
        // Re-emit the axis overlay so the click doesn't visually wipe it.
        drawAxesOverlay();
        revertStrokeColours();
        circleStrokeIndices = new Set();
        lassoPoints = [];

        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;
        const vw = rect.width, vh = rect.height;

        if (isMulti) {
          const pu = materialRef.current?.uniforms;
          const hit = pickClosestAcrossChunks(
            chunkRendersRef.current.values(), cam, norm!, viewMode === "3d",
            clickX, clickY, vw, vh,
            pu ? pu.uAxisScaleX.value : 1, pu ? pu.uAxisScaleY.value : 1, pu ? pu.uAxisScaleZ.value : 1,
          );
          if (isWand) {
            // Wand mode — hand the seed event_id to the parent to run a
            // server-side connected-component selection. Shift/Ctrl modifiers
            // do add/subtract relative to the existing selection.
            if (hit && onWandSeedRef.current) {
              onWandSeedRef.current(hit.eventId, getOp(e));
              highlightWorldPos.current = hit.worldPos;
              glowSprite.position.copy(hit.worldPos);
              glowSprite.visible = true;
            }
            isDragging = false;
            return;
          }
          if (hit) {
            highlightWorldPos.current = hit.worldPos;
            glowSprite.position.copy(hit.worldPos);
            glowSprite.visible = true;
            const r = hit.render;
            if (onPointClickRef.current) {
              onPointClickRef.current({
                index: hit.eventId, eventId: hit.eventId,
                x: r.payload.positions[hit.indexInChunk * 3],
                y: r.payload.positions[hit.indexInChunk * 3 + 1],
                z: r.payload.positions[hit.indexInChunk * 3 + 2],
                tUs: r.payload.t_us[hit.indexInChunk],
                polarity: r.payload.polarity[hit.indexInChunk],
                label: r.payload.label[hit.indexInChunk],
                screenX: clickX, screenY: clickY,
              });
            }
          } else {
            highlightWorldPos.current = null;
            glowSprite.visible = false;
            if (onPointClickRef.current) onPointClickRef.current(null);
          }
          isDragging = false;
          return;
        }

        // Legacy single-payload pick path.
        const projected = projectAllPoints(cam, buf!, vw, vh);
        const uPointSize = viewMode === "3d" ? 2.0 : 1.5;
        let bestDepth = Infinity, minIdx = -1;
        for (let i = 0; i < buf!.count; i++) {
          const o = i * 3;
          const sx = projected[o], sy = projected[o + 1], wc = projected[o + 2];
          if (sx < -999) continue;
          const dd = (sx - clickX) ** 2 + (sy - clickY) ** 2;
          const renderedRadius = (uPointSize * 7.0 / wc) * 0.5;
          const pickRadius = Math.max(renderedRadius, 8);
          if (dd > pickRadius * pickRadius) continue;
          if (wc < bestDepth) { bestDepth = wc; minIdx = i; }
        }

        if (minIdx >= 0) {
          const { cx: _cx, cy: _cy, cz: _cz, scale: _s, flipY: _f, flipX: _fx } = buf!.norm;
          const _yM = _f ? -1 : 1;
          const _xM = _fx ? -1 : 1;
          const _is3d = viewMode === "3d";
          const worldPos = new THREE.Vector3(
            (buf!.positions[minIdx * 3] - _cx) * _s * _xM,
            (buf!.positions[minIdx * 3 + 1] - _cy) * _s * _yM,
            _is3d ? (buf!.positions[minIdx * 3 + 2] - _cz) * _s : 0,
          );
          highlightWorldPos.current = worldPos;
          glowSprite.position.copy(worldPos);
          glowSprite.visible = true;
          if (onPointClickRef.current) {
            onPointClickRef.current({
              index: minIdx, eventId: Number(pl!.event_id[minIdx]),
              x: pl!.positions[minIdx * 3], y: pl!.positions[minIdx * 3 + 1], z: pl!.positions[minIdx * 3 + 2],
              tUs: pl!.t_us[minIdx], polarity: pl!.polarity[minIdx], label: pl!.label[minIdx],
              screenX: clickX, screenY: clickY,
            });
          }
        } else {
          highlightWorldPos.current = null;
          glowSprite.visible = false;
          if (onPointClickRef.current) onPointClickRef.current(null);
        }
        isDragging = false;
        return;
      }

      // End of drag — finalize selection
      const tm = toolModeRef.current;
      const cam = cameraRef.current;
      const buf = buffersRef.current;
      const pl = payloadRef.current;
      const isMulti = multiActiveRef.current;
      const norm = normRef.current;
      if (!cam || !onSelectionRef.current) { isDragging = false; clearOverlay(); return; }
      if (!isMulti && (!buf || !pl)) { isDragging = false; clearOverlay(); return; }
      if (isMulti && !norm) { isDragging = false; clearOverlay(); return; }

      const rect = host.getBoundingClientRect();
      const vw = rect.width, vh = rect.height;
      const op = getOp(e);

      const through = selectThroughRef.current ?? false;
      // Cell ≈ 1.5× the rendered point size in 3D; depth tolerance keeps points
      // co-planar with the closest hit. In XY mode all points share z=0 so the
      // filter is a no-op there.
      const visCellPx = 5;
      const visDepthFrac = 0.04;
      const is3d = viewMode === "3d";
      // XY: confine selection to the visible playback window (see frameIndexRange).
      const xyFrame = is3d ? null : { start: currentFrameUsRef.current ?? 0, end: (currentFrameUsRef.current ?? 0) + (frameWindowUsRef.current ?? 10000) };

      if (tm === "box") {
        const x1 = Math.min(dragStartX, dragCurrentX) - rect.left;
        const y1 = Math.min(dragStartY, dragCurrentY) - rect.top;
        const x2 = Math.max(dragStartX, dragCurrentX) - rect.left;
        const y2 = Math.max(dragStartY, dragCurrentY) - rect.top;
        if (isMulti) {
          // Multi-Points: emit event_ids directly.
          const su = materialRef.current?.uniforms;
          const eventIds = selectAcrossChunks(
            chunkRendersRef.current.values(), cam, norm!, is3d, vw, vh,
            (sx, sy) => sx >= x1 && sx <= x2 && sy >= y1 && sy <= y2,
            xyFrame,
            su ? su.uAxisScaleX.value : 1, su ? su.uAxisScaleY.value : 1, su ? su.uAxisScaleZ.value : 1,
          );
          // TODO: visible-only filter for multi-Points (cell binning across
          // chunks). For now `through` is implicitly true in multi mode.
          onSelectionRef.current(eventIds, op);
        } else {
          const projected = projectAllPoints(cam, buf!, vw, vh);
          let indices: number[] = [];
          for (let i = 0; i < buf!.count; i++) {
            const o = i * 3;
            const sx = projected[o], sy = projected[o + 1];
            if (sx < -999) continue;
            if (sx >= x1 && sx <= x2 && sy >= y1 && sy <= y2) indices.push(i);
          }
          if (!through) indices = filterVisibleIndices(indices, projected, visCellPx, visDepthFrac);
          // Convert to event_ids so MainViewport.handleSelection has a
          // uniform input shape across legacy + multi modes.
          const eventIds = indices.map(i => Number(pl!.event_id[i]));
          onSelectionRef.current(eventIds, op);
        }
      } else if (tm === "circle") {
        if (isMulti) {
          // circleStrokeIndices already contains event_ids in multi mode.
          onSelectionRef.current(Array.from(circleStrokeIndices), op);
          // Don't revert chunk colours here — the selection-delta effect will
          // recolour the changed points from the new selection state (and
          // un-white brushed events on a "subtract" op).
          touchedStrokeChunks = new Set();
        } else {
          let indices = Array.from(circleStrokeIndices);
          if (!through && indices.length > 1) {
            const projected = projectAllPoints(cam, buf!, vw, vh);
            indices = filterVisibleIndices(indices, projected, visCellPx, visDepthFrac);
          }
          const eventIds = indices.map(i => Number(pl!.event_id[i]));
          onSelectionRef.current(eventIds, op);
        }
        circleStrokeIndices = new Set();
        } else if (tm === "lasso") {
        if (lassoPoints.length >= 3) {
          // Simplify the freehand path and precompute its bbox so each point
          // does a 4-compare reject before the O(vertices) point-in-polygon.
          const poly = simplifyRDP(lassoPoints, 2);
          const pb = polygonBounds(poly);
          const inLasso = (sx: number, sy: number) =>
            sx >= pb.minX && sx <= pb.maxX && sy >= pb.minY && sy <= pb.maxY && pointInPolygon(sx, sy, poly);
          if (isMulti) {
            const lu = materialRef.current?.uniforms;
            const eventIds = selectAcrossChunks(
              chunkRendersRef.current.values(), cam, norm!, is3d, vw, vh,
              (sx, sy) => inLasso(sx, sy),
              xyFrame,
              lu ? lu.uAxisScaleX.value : 1, lu ? lu.uAxisScaleY.value : 1, lu ? lu.uAxisScaleZ.value : 1,
            );
            onSelectionRef.current(eventIds, op);
          } else {
            const projected = projectAllPoints(cam, buf!, vw, vh);
            let indices: number[] = [];
            for (let i = 0; i < buf!.count; i++) {
              const o = i * 3;
              const sx = projected[o], sy = projected[o + 1];
              if (sx < -999) continue;
              if (inLasso(sx, sy)) indices.push(i);
            }
            if (!through) indices = filterVisibleIndices(indices, projected, visCellPx, visDepthFrac);
            const eventIds = indices.map(i => Number(pl!.event_id[i]));
            onSelectionRef.current(eventIds, op);
          }
        }
        lassoPoints = [];
      }

      clearOverlay();
      isDragging = false;
    };

    // Scroll wheel adjusts brush radius in circle mode
    const onWheel = (e: WheelEvent) => {
      if (toolModeRef.current !== "circle") return;
      const rect = host.getBoundingClientRect();
      if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) return;
      e.preventDefault();
      const delta = e.deltaY > 0 ? -2 : 2;
      const next = Math.max(5, Math.min(120, (brushRadiusRef.current ?? 15) + delta));
      if (onBrushRadiusChangeRef.current) onBrushRadiusChangeRef.current(next);
      // Redraw cursor at current mouse position
      clearOverlay();
      drawCircle(e.clientX - rect.left, e.clientY - rect.top, next);
    };

    // Sizing overlay canvas — high-DPI aware so text isn't blurry.
    // Must happen BEFORE the first drawAxesOverlay() so the canvas isn't still
    // its default 300×150 when we draw (otherwise axes/frame won't appear
    // until the user triggers a redraw via interaction).
    if (overlay) {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      overlay.width = Math.round(size.width * dpr);
      overlay.height = Math.round(size.height * dpr);
      overlay.style.width = size.width + "px";
      overlay.style.height = size.height + "px";
      const octx2 = overlay.getContext("2d");
      if (octx2) {
        octx2.setTransform(dpr, 0, 0, dpr, 0, 0);
        octx2.font = "11px Inter, 'Segoe UI', monospace";
      }
    }
    // Re-draw axes now that the overlay has correct dimensions.
    drawAxesOverlay();

    window.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    host.addEventListener("wheel", onWheel, { passive: false });

    // Bump scene version so the chunk-reconciliation effect knows to re-attach
    // its Points to the freshly-built scene. (The main setup effect rebuilds
    // the scene on viewMode/gpuPreference/theme/xyRotation changes; recon must
    // run AFTER that rebuild to add chunks back.)
    setSceneVersion((v) => v + 1);

    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      host.removeEventListener("wheel", onWheel);
      highlightWorldPos.current = null;
      cameraRef.current = null;
      controlsRef.current = null;
      materialRef.current = null;
      geometryRef.current = null;
      axesRef.current = null;
      selectionMeshRef.current = null;
      cancelAnimationFrame(frame);
      controls.dispose();
      geometry.dispose();
      material.dispose();
      selectionMesh.geometry.dispose();
      (selectionMesh.material as THREE.Material).dispose();
      // Dispose axes line geometry/materials
      for (const c of axesGroup.children) {
        const obj = c as THREE.LineSegments;
        obj.geometry?.dispose();
        const mm = obj.material as THREE.Material | THREE.Material[];
        if (Array.isArray(mm)) mm.forEach((m) => m.dispose());
        else mm?.dispose();
      }
      axesGroupRef.current = null;
      // Dispose any per-chunk geometries owned by the multi-Points pipeline.
      // The reconciliation effect rebuilds them on next mount because its deps
      // (manifest / chunksMap) emit fresh objects on Map regeneration there;
      // here we just need to free GPU memory tied to the old WebGL context.
      for (const render of chunkRendersRef.current.values()) {
        render.geometry.dispose();
      }
      chunkRendersRef.current.clear();
      sceneRef.current = null;
      glowMat.dispose();
      glowTexture.dispose();
      rendererRef.current = null;
      renderer.dispose();
      mount.replaceChildren();
    };
  }, [viewMode, gpuPreference, theme, xyRotation]);

  // Lightweight resize handler — adjusts the existing renderer / camera /
  // overlay canvas without tearing down the WebGL context. Avoids the white /
  // black flash and the GL-context-loss race when the user drags the window.
  useEffect(() => {
    const renderer = rendererRef.current;
    const camera = cameraRef.current;
    const overlay = overlayRef.current;
    if (!renderer || !camera || size.width < 2 || size.height < 2) return;
    renderer.setSize(size.width, size.height, false);
    const aspect = size.width / size.height;
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
    } else if (camera instanceof THREE.OrthographicCamera) {
      // Tight-fit to data box (matches creation logic in main effect).
      const metaW = meta?.width ?? size.width;
      const metaH = meta?.height ?? size.height;
      const maxDim = Math.max(metaW, metaH);
      const halfW_world = metaW / maxDim;
      const halfH_world = metaH / maxDim;
      // Note: this resize handler doesn't currently know about xyRotation, but
      // the main effect rebuilds whenever rotation changes, so the orientation
      // is fresh each time we get here. Still, default to mode 0 swap rules.
      const screenHalfV = halfH_world;
      const screenHalfH = halfW_world;
      const dataDisplayAspect = screenHalfH / Math.max(screenHalfV, 1e-6);
      const pad = 1.03;
      const halfH = aspect >= dataDisplayAspect
        ? screenHalfV * pad
        : (screenHalfH * pad) / aspect;
      camera.left = -halfH * aspect;
      camera.right = halfH * aspect;
      camera.top = halfH;
      camera.bottom = -halfH;
      camera.updateProjectionMatrix();
    }
    if (overlay) {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      overlay.width = Math.round(size.width * dpr);
      overlay.height = Math.round(size.height * dpr);
      overlay.style.width = size.width + "px";
      overlay.style.height = size.height + "px";
      const ctx = overlay.getContext("2d");
      if (ctx) {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.font = "11px Inter, 'Segoe UI', monospace";
      }
    }
    // Recompute the XY-mode pixel-size uniform since it depends on size.height
    // and the ortho frustum height (which we just updated above).
    const mat = materialRef.current;
    const buf = buffersRef.current;
    if (mat && mat.uniforms.uXyPxScale && buf && camera instanceof THREE.OrthographicCamera) {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const span = camera.top - camera.bottom;
      mat.uniforms.uXyPxScale.value = (buf.norm.scale * size.height * dpr) / span;
    }
  }, [size]);

  const showBuildOverlay = !!payload && buildProgress < 1;
  // Zoom slider is logarithmic — equal slider distance for 0.5× and 2×.
  const zoomLog = Math.log10(Math.max(1e-6, zoomValue));
  // Per-axis stretch maxima scale purely with the data, rounded, floored at 1×
  // (so the slider stays valid). T (time): 10× per 60 s of recording. X/Y: 5× per
  // 1280×720 of sensor area — area-based so it's orientation-agnostic (works
  // whether the wide side is X or Y for landscape vs. portrait recordings).
  const tDurationSec = meta ? Math.max(0, (meta.t_max_us - meta.t_min_us) / 1e6) : 0;
  const maxTZoom = Math.max(1, Math.round((tDurationSec / 60) * 10));
  const maxXYZoom = meta ? Math.max(1, Math.round((meta.width * meta.height) / (1280 * 720) * 5)) : 5;
  const axisMaxZoom = (a: "x" | "y" | "z") => (a === "z" ? maxTZoom : maxXYZoom);
  const axisMaxLog = (a: "x" | "y" | "z") => Math.log10(axisMaxZoom(a));
  const resetZoomAndAxes = () => {
    setZoomValue(1.0);
    setAxisScaleX(1.0);
    setAxisScaleY(1.0);
    setAxisScaleZ(1.0);
    const ctrl = controlsRef.current;
    if (ctrl) ctrl.reset();
  };
  const axisLabel = (a: "x" | "y" | "z") => (a === "x" ? "X" : a === "y" ? "Y" : "T");
  const axisValue = (a: "x" | "y" | "z") => (a === "x" ? axisScaleX : a === "y" ? axisScaleY : axisScaleZ);
  const setAxisValue = (a: "x" | "y" | "z", v: number) => {
    if (a === "x") setAxisScaleX(v);
    else if (a === "y") setAxisScaleY(v);
    else setAxisScaleZ(v);
  };

  return (
    <div className="viewport" ref={hostRef}>
      <div className="webglMount" ref={mountRef} />
      <canvas ref={overlayRef} className="selectionOverlay" />
      {showBuildOverlay && (
        <div className="buildProgress">
          Building point buffers… {Math.round(buildProgress * 100)}%
          <div className="buildProgressBar"><div style={{ width: `${Math.round(buildProgress * 100)}%` }} /></div>
        </div>
      )}
      <div className={viewMode === "xy" ? "zoomControls zoomControlsXy" : "zoomControls"}
        onMouseDown={(e) => e.stopPropagation()} onMouseUp={(e) => e.stopPropagation()}>
        <div className="zoomRow">
          <button className="zoomReset" onClick={resetZoomAndAxes} title="Reset view">⟲</button>
          <input
            className="zoomSlider"
            type="range"
            min={-1}
            max={1.3}
            step={0.02}
            value={zoomLog}
            onChange={(e) => setZoomValue(Math.pow(10, Number(e.target.value)))}
            title={`Zoom ${(zoomValue * 100).toFixed(0)}%`}
          />
          <span className="zoomLabel">{(zoomValue * 100).toFixed(0)}%</span>
        </div>
        {/* Per-axis stretch only makes sense in 3D — XY collapses Z and the
            X/Y axes are 1:1 with the sensor pixels, so we hide the row. */}
        {viewMode === "3d" && (
          <>
            <div className="axisRow">
              {(["x", "y", "z"] as const).map((a) => (
                <button
                  key={a}
                  className={openAxisPanel === a ? "axisBtn axisBtnActive" : "axisBtn"}
                  onClick={() => setOpenAxisPanel(openAxisPanel === a ? null : a)}
                  title={`Scale ${axisLabel(a)} axis (${axisValue(a).toFixed(2)}×)`}
                >
                  {axisLabel(a)} · {axisValue(a).toFixed(2)}×
                </button>
              ))}
            </div>
            {openAxisPanel && (
              <div className="axisPanel">
                <div className="axisPanelHeader">{axisLabel(openAxisPanel)} axis scale (max {axisMaxZoom(openAxisPanel)}×)</div>
                <input
                  className="zoomSlider"
                  type="range"
                  min={-0.7}
                  max={axisMaxLog(openAxisPanel)}
                  step={0.02}
                  value={Math.log10(Math.max(1e-6, axisValue(openAxisPanel)))}
                  onChange={(e) => setAxisValue(openAxisPanel, Math.pow(10, Number(e.target.value)))}
                />
                <div className="axisPanelFooter">
                  <span>{axisValue(openAxisPanel).toFixed(2)}×</span>
                  <button onClick={() => setAxisValue(openAxisPanel, 1.0)}>Reset</button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
