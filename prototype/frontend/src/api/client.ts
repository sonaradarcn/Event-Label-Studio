import type { DatasetMeta, LabelClass, PointPayload, Recording } from "./types";

const API = "/api";

type ManifestArray = { name: keyof PointPayload; dtype: string; shape: number[]; bytes: number };
type Manifest = { version: number; count: number; arrays: ManifestArray[] };

export async function listRecordings(): Promise<Recording[]> {
  return fetch(`${API}/recordings`).then((r) => r.json());
}

export async function uploadFile(file: File): Promise<{ path: string }> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API}/datasets/upload`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function openDataset(path: string, name?: string): Promise<{ dataset_id: string; meta: DatasetMeta }> {
  const response = await fetch(`${API}/datasets/open`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(name ? { path, name } : { path })
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

// Existing cached project names (= cache folder ids), used to suggest a unique
// default name and warn about collisions when importing a new file.
export async function listDatasetIds(): Promise<string[]> {
  const response = await fetch(`${API}/datasets`);
  if (!response.ok) return [];
  const metas = (await response.json()) as { dataset_id?: string }[];
  return metas.map((m) => String(m.dataset_id ?? "")).filter(Boolean);
}

export type OpenTask = {
  task_id: string;
  status: "running" | "ready" | "failed";
  stage?: string;
  progress?: number;
  message?: string;
  dataset_id?: string;
  meta?: DatasetMeta;
};

export async function openDatasetAsync(path: string, name?: string): Promise<OpenTask> {
  const response = await fetch(`${API}/datasets/open_async`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(name ? { path, name } : { path })
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function getTask(taskId: string): Promise<OpenTask> {
  const response = await fetch(`${API}/tasks/${taskId}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function getLabelSchema(datasetId: string): Promise<{ labels: LabelClass[] }> {
  return fetch(`${API}/datasets/${datasetId}/labels/schema`).then((r) => r.json());
}

export async function fetchChunkManifest(datasetId: string, chunkUs?: number): Promise<import("./types").ChunkManifest> {
  const url = new URL(`${API}/datasets/${datasetId}/manifest`, window.location.origin);
  if (chunkUs !== undefined) url.searchParams.set("chunk_us", String(chunkUs));
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchChunk(datasetId: string, chunkId: number, chunkUs?: number): Promise<PointPayload> {
  const url = new URL(`${API}/datasets/${datasetId}/chunks/${chunkId}.bin`, window.location.origin);
  if (chunkUs !== undefined) url.searchParams.set("chunk_us", String(chunkUs));
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  return parsePointPayloadStreaming(response);
}

export async function fetchPoints(datasetId: string, params: { start_us?: number; end_us?: number; sample?: number; polarity: string }): Promise<PointPayload> {
  const url = new URL(`${API}/datasets/${datasetId}/points.bin`, window.location.origin);
  if (params.start_us !== undefined) url.searchParams.set("start_us", String(params.start_us));
  if (params.end_us !== undefined) url.searchParams.set("end_us", String(params.end_us));
  if (params.sample !== undefined && params.sample > 0) url.searchParams.set("sample", String(params.sample));
  url.searchParams.set("polarity", params.polarity);
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  return parsePointPayloadStreaming(response);
}

async function recordAlloc(label: string, bytes: number) {
  try {
    const m = await import("../devLog");
    m.trackAlloc(label, bytes);
  } catch { /* devLog optional */ }
}

async function parsePointPayloadStreaming(response: Response): Promise<PointPayload> {
  if (!response.body) throw new Error("response has no body stream");
  const reader = response.body.getReader();

  let manifest: (Manifest & { arrays: (Manifest["arrays"][number] & { offset: number })[] }) | null = null;
  let positions: Float32Array | undefined;
  let event_id: BigUint64Array | undefined;
  let tUsRaw: BigUint64Array | undefined;
  let polarity: Uint8Array | undefined;
  let label: Int16Array | undefined;
  const arraySpec: { name: string; offset: number; bytes: number }[] = [];

  // Header is buffered as a list of small Uint8Array chunks until complete,
  // then concatenated once.
  let headerChunks: Uint8Array[] = [];
  let headerSize = 0;
  let payloadCursor = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value || value.length === 0) continue;

    let chunkData: Uint8Array | null = null;

    if (!manifest) {
      headerChunks.push(value);
      headerSize += value.length;
      if (headerSize >= 4) {
        const concat = new Uint8Array(headerSize);
        let off = 0;
        for (const c of headerChunks) { concat.set(c, off); off += c.length; }
        const headerLen = new DataView(concat.buffer, concat.byteOffset, 4).getUint32(0, true);
        const totalHeaderBytes = 4 + headerLen;
        if (concat.length >= totalHeaderBytes) {
          const headerText = new TextDecoder().decode(concat.subarray(4, totalHeaderBytes));
          manifest = JSON.parse(headerText);
          recordAlloc(`manifest-parsed (count=${manifest!.count.toLocaleString()})`, 0);
          for (const item of manifest!.arrays) {
            const length = item.shape.reduce((a: number, b: number) => a * b, 1);
            switch (item.name) {
              case "positions":
                positions = new Float32Array(length);
                recordAlloc("positions Float32", positions.byteLength);
                break;
              case "event_id":
                event_id = new BigUint64Array(length);
                recordAlloc("event_id BigUint64", event_id.byteLength);
                break;
              case "t_us":
                tUsRaw = new BigUint64Array(length);
                recordAlloc("t_us_raw BigUint64", tUsRaw.byteLength);
                break;
              case "polarity":
                polarity = new Uint8Array(length);
                recordAlloc("polarity Uint8", polarity.byteLength);
                break;
              case "label":
                label = new Int16Array(length);
                recordAlloc("label Int16", label.byteLength);
                break;
              // colours are intentionally skipped — recomputed on the fly.
            }
            arraySpec.push({ name: item.name, offset: item.offset, bytes: item.bytes });
          }
          chunkData = concat.subarray(totalHeaderBytes);
          headerChunks = [];
          headerSize = 0;
          if (chunkData.length === 0) continue;
        } else {
          continue;
        }
      } else {
        continue;
      }
    } else {
      chunkData = value;
    }

    // Dispatch chunk bytes into the right typed arrays' backing buffers.
    const chunkStart = payloadCursor;
    const chunkEnd = chunkStart + chunkData.length;
    for (const spec of arraySpec) {
      if (spec.name === "colors") continue;
      const arrStart = spec.offset;
      const arrEnd = arrStart + spec.bytes;
      const ovLo = Math.max(chunkStart, arrStart);
      const ovHi = Math.min(chunkEnd, arrEnd);
      if (ovHi <= ovLo) continue;
      const dstOff = ovLo - arrStart;
      const srcOff = ovLo - chunkStart;
      const n = ovHi - ovLo;
      const dst = spec.name === "positions" ? positions
        : spec.name === "event_id" ? event_id
        : spec.name === "t_us" ? tUsRaw
        : spec.name === "polarity" ? polarity
        : spec.name === "label" ? label
        : null;
      if (!dst) continue;
      const dstBytes = new Uint8Array(dst.buffer, dst.byteOffset + dstOff, n);
      dstBytes.set(chunkData.subarray(srcOff, srcOff + n));
    }
    payloadCursor = chunkEnd;
  }

  if (!manifest || !positions || !event_id || !tUsRaw || !polarity || !label) {
    throw new Error("incomplete streaming payload");
  }

  // Convert t_us BigUint64 → Float32. Read each u64 via DataView's getUint32
  // halves and combine in Float64 math — never Number(bigint[i]) because
  // BigInt indexing allocates a fresh BigInt per access (16 M garbage objects
  // overwhelm GC). Pre-alloc Float32 first, then drop the BigUint64 buffer.
  const tUsCount = tUsRaw.length;
  const tUsDv = new DataView(tUsRaw.buffer, tUsRaw.byteOffset, tUsRaw.byteLength);
  const tUs = new Float32Array(tUsCount);
  recordAlloc("t_us Float32 (final)", tUs.byteLength);
  for (let i = 0; i < tUsCount; i++) {
    const lo = tUsDv.getUint32(i * 8, true);
    const hi = tUsDv.getUint32(i * 8 + 4, true);
    tUs[i] = hi * 4294967296 + lo;
  }
  // Detach reference so the GC can reclaim the BigUint64 buffer.
  tUsRaw = undefined;
  recordAlloc("t_us_raw dropped", -1 * tUsCount * 8);

  return {
    count: manifest.count,
    positions,
    event_id,
    t_us: tUs,
    polarity,
    label,
  };
}

export async function updateLabels(datasetId: string, eventIds: number[], label: number, operation: "assign" | "clear") {
  const response = await fetch(`${API}/datasets/${datasetId}/labels/update`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ event_ids: eventIds, label, operation })
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export type LabelUpdateResult = {
  updated: number;
  label: number;
  label_version: number;
  // Per-label-id count change ("-1" == unlabelled). Lets the client patch its
  // stats without an O(N) rescan of the whole label array.
  stats_delta: Record<string, number>;
};

/** Single binary label update — sends the ids as a raw little-endian
 *  Uint32Array (one request, no 50k JSON batching). */
export async function updateLabelsBinary(
  datasetId: string,
  eventIds: Uint32Array,
  label: number,
  operation: "assign" | "clear",
): Promise<LabelUpdateResult> {
  const url = new URL(`${API}/datasets/${datasetId}/labels/update.bin`, window.location.origin);
  url.searchParams.set("label", String(label));
  url.searchParams.set("operation", operation);
  // Copy out the exact byte range backing the typed array.
  const body = eventIds.slice().buffer as ArrayBuffer;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body,
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export type WandParams = {
  seed_event_id: number;
  /** Foolproof mode — backend auto-picks all parameters from the seed's
   *  existing label. Overrides the manual ones below. */
  auto?: boolean;
  r_xy?: number;
  r_t_us?: number;
  polarity_match?: boolean;
  label_match?: boolean;
  max_size?: number;
  /** Min events per voxel for it to count as occupied (density gate). 1 =
   *  legacy; >1 stops sparse noise bridging everything into one blob. */
  min_pts?: number;
  voxel_xy?: number;
  voxel_t_us?: number;
};

export async function selectComponent(datasetId: string, params: WandParams): Promise<{ event_ids: number[]; size: number; truncated: boolean }> {
  const response = await fetch(`${API}/datasets/${datasetId}/select_component`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function command(datasetId: string, name: "undo" | "redo") {
  const response = await fetch(`${API}/datasets/${datasetId}/labels/${name}`, { method: "POST" });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export type ExportParams = {
  /** false → full export, true → time-stratified sub-sample of the export set */
  sample: boolean;
  sample_size: number;
  /** export only events that carry a real label (skip label === -1) */
  only_labelled: boolean;
  include_npz: boolean;
  include_ply: boolean;
  include_csv: boolean;
};

export type ExportResult = {
  project?: string; labels?: string; csv?: string; ply?: string;
  zip?: string; zip_url?: string; zip_bytes?: number;
};

export async function exportLabels(datasetId: string, params: ExportParams): Promise<ExportResult> {
  const response = await fetch(`${API}/datasets/${datasetId}/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

// ─── Video render ─────────────────────────────────────

export type RenderVideoParams = {
  start_us: number;
  end_us: number;
  fps: number;
  frame_window_us: number;
  color_mode: "label_polarity" | "polarity_only" | "label_only";
  background: "black" | "white";
  format: "mp4" | "webm";
};

export async function renderVideoStart(datasetId: string, params: RenderVideoParams & { unlabeled_color?: string; polarity_contrast?: number }): Promise<{ task_id: string }> {
  const response = await fetch(`${API}/datasets/${datasetId}/render_video`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function renderVideoPreview(datasetId: string, params: Partial<RenderVideoParams> & { t_us: number; unlabeled_color?: string; polarity_contrast?: number }): Promise<Blob> {
  const response = await fetch(`${API}/datasets/${datasetId}/render_video/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.blob();
}

export function renderVideoUrl(datasetId: string, fmt: "mp4" | "webm"): string {
  return `${API}/datasets/${datasetId}/render_video.${fmt}`;
}

export async function updateLabelsByFilter(
  datasetId: string,
  params: {
    label: number;
    operation: string;
    start_us?: number;
    end_us?: number;
    polarity?: string;
    x_min?: number;
    x_max?: number;
    y_min?: number;
    y_max?: number;
  }
): Promise<{ updated: number; label: number; label_version?: number; stats_delta?: Record<string, number> }> {
  const response = await fetch(`${API}/datasets/${datasetId}/labels/update_by_filter`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

// ─── Propagation tracking ─────────────────────────────────────────────

export type TrackSlice = {
  t_start: number; t_end: number; t_center: number;
  cx: number; cy: number; count: number;
  confidence: number; status: "ok" | "low" | "ambiguous" | "lost";
  event_ids: number[];
};
export type TrackCandidate = {
  cx: number; cy: number; count: number; score: number; event_ids: number[];
};
export type TrackDiag = {
  roi: number; blobs: number; merged_blobs: number; competitors: number;
  best_count: number; kept: number; kept_frac_roi: number; status: string;
};
export type TrackPropagateResult = {
  slices: TrackSlice[];
  frontier_event_ids: number[];
  pause_time_us: number | null;
  candidates: TrackCandidate[];
  diag?: TrackDiag[];
  stop_reason: string;
  step_us: number;
  direction: "forward" | "backward";
};

export async function propagateTrack(
  datasetId: string,
  params: { seed_event_ids: number[]; step_us: number; direction: "forward" | "backward"; max_slices: number; stop_on_low: boolean },
): Promise<TrackPropagateResult> {
  const response = await fetch(`${API}/datasets/${datasetId}/tracks/propagate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function getStats(datasetId: string): Promise<{ total: number; unlabelled: number; per_class: Record<string, number> }> {
  const response = await fetch(`${API}/datasets/${datasetId}/stats`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

// ─── Cache management ─────────────────────────────────────────────────

export type CacheEntry = {
  dataset_id: string;
  source_path: string;
  width: number;
  height: number;
  event_count: number;
  t_min_us: number;
  t_max_us: number;
  /** total bytes of the dataset's cache folder */
  size_bytes: number;
  /** mtime epoch seconds */
  modified: number;
};

export async function listCache(): Promise<CacheEntry[]> {
  const response = await fetch(`${API}/cache`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function deleteDataset(id: string): Promise<void> {
  const response = await fetch(`${API}/datasets/${id}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await response.text());
}

export async function renameDataset(id: string, name: string): Promise<{ ok: boolean; dataset_id?: string; error?: string }> {
  const response = await fetch(`${API}/datasets/${id}/rename`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const data = (await response.json().catch(() => ({}))) as { ok?: boolean; dataset_id?: string; error?: string };
  if (!response.ok || !data.ok) return { ok: false, error: data.error || "Rename failed" };
  return { ok: true, dataset_id: data.dataset_id };
}

export async function revealDataset(id: string): Promise<void> {
  const response = await fetch(`${API}/datasets/${id}/reveal`, { method: "POST" });
  if (!response.ok) throw new Error(await response.text());
}

/** Run a FULL default export (no sub-sampling, all events, all formats) for a
 *  dataset, then trigger a browser download of the resulting zip. Mirrors the
 *  download pattern used elsewhere in the app (create an <a>, click it). */
export async function exportDatasetDefault(id: string): Promise<void> {
  const params: ExportParams = {
    sample: false,
    sample_size: 0,
    only_labelled: false,
    include_npz: true,
    include_ply: true,
    include_csv: true,
  };
  const result = await exportLabels(id, params);
  const href = result.zip_url ?? `${API}/datasets/${id}/export.zip`;
  const a = document.createElement("a");
  a.href = href;
  a.download = `${id}_labels.zip`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// Legacy non-streaming parser — kept only so existing tests / scripts can
// still parse a buffered response. Matches the streaming parser's output
// shape (Float32 t_us). Not used in the live fetch path.
export function parsePointPayload(buffer: ArrayBuffer): PointPayload {
  const headerLength = new DataView(buffer, 0, 4).getUint32(0, true);
  const headerText = new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength));
  const manifest = JSON.parse(headerText) as Manifest & { arrays: (Manifest["arrays"][number] & { offset?: number })[] };
  const payloadStart = 4 + headerLength;
  const output: Partial<PointPayload> = { count: manifest.count };

  let cursor = 0;
  for (const item of manifest.arrays) {
    const length = item.shape.reduce((a, b) => a * b, 1);
    if (typeof item.offset === "number") cursor = item.offset;
    const absolute = payloadStart + cursor;
    cursor += item.bytes;

    const make = <T>(Ctor: new (b: ArrayBuffer, off: number, len: number) => T, alignBytes: number): T => {
      if (absolute % alignBytes === 0) return new Ctor(buffer, absolute, length);
      const sliced = buffer.slice(absolute, absolute + item.bytes);
      return new Ctor(sliced, 0, length);
    };

    if (item.name === "positions") output.positions = make(Float32Array, 4);
    else if (item.name === "colors") output.colors = make(Uint8Array, 1);
    else if (item.name === "event_id") output.event_id = make(BigUint64Array, 8);
    else if (item.name === "t_us") {
      const u64 = make(BigUint64Array, 8);
      const f32 = new Float32Array(u64.length);
      for (let i = 0; i < u64.length; i++) f32[i] = Number(u64[i]);
      output.t_us = f32;
    }
    else if (item.name === "polarity") output.polarity = make(Uint8Array, 1);
    else if (item.name === "label") output.label = make(Int16Array, 2);
  }
  return output as PointPayload;
}
