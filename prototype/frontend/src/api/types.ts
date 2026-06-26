export type Recording = { name: string; path: string };

export type DatasetMeta = {
  dataset_id: string;
  source_path: string;
  width: number;
  height: number;
  event_count: number;
  t_min_us: number;
  t_max_us: number;
  default_window_us: number;
};

export type LabelClass = { id: number; name: string; color: string };

export type ChunkManifestEntry = {
  chunk_id: number;
  start_us: number;
  end_us: number;
  event_start_index: number;
  event_count: number;
};

export type ChunkManifest = {
  dataset_id: string;
  total_events: number;
  duration_us: number;
  t_min_us: number;
  t_max_us: number;
  chunk_duration_us: number;
  n_chunks: number;
  width: number;
  height: number;
  chunks: ChunkManifestEntry[];
};

export type PointPayload = {
  positions: Float32Array;
  colors?: Uint8Array;          // Backend value not used; frontend recomputes.
  event_id: BigUint64Array;
  // Downcast to Float32 by the streaming parser — saves 64 MB on big payloads
  // and avoids per-event BigInt↔Number conversions.
  t_us: Float32Array;
  polarity: Uint8Array;
  label: Int16Array;
  count: number;
};

