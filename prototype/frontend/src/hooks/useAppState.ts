import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { command, exportLabels, fetchChunk, fetchChunkManifest, fetchPoints, getLabelSchema, getStats, getTask, listCache, listDatasetIds, listRecordings, openDatasetAsync, propagateTrack, renderVideoStart, renderVideoUrl, selectComponent as selectComponentApi, updateLabelsBinary, updateLabelsByFilter, uploadFile, type ExportParams, type RenderVideoParams } from "../api/client";
import type { ChunkManifest, ChunkManifestEntry } from "../api/types";
import type { DatasetMeta, LabelClass, PointPayload, Recording } from "../api/types";
import type { PointInfo, ToolMode, SelectionOp, FilterPreview } from "../pointcloud/PointCloudView";
import type { TrackCandidate } from "../api/client";
import { captureView, captureSvgView, copyDataUrlToClipboard, printDataUrl, printSvgMarkup } from "../pointcloud/capture";

export type ProgressState = { visible: boolean; value: number; title: string; detail: string; elapsed: number; indeterminate?: boolean };
// Naming dialog shown when importing a recording file (the name becomes the
// project's cache id, so one recording can be imported as several projects).
export type ImportPromptState = { visible: boolean; fileName: string; suggested: string; existingNames: string[] };

// Derive the file stem and, if a project of that name already exists, suffix it
// (-2, -3, …) so the suggested default never silently collides with a project.
function deriveStem(fileName: string): string {
  return (fileName.split(/[\\/]/).pop() ?? fileName).replace(/\.[^.]+$/, "");
}
function uniqueProjectName(base: string, existing: string[]): string {
  const taken = new Set(existing);
  if (!taken.has(base)) return base;
  let i = 2;
  while (taken.has(`${base}-${i}`)) i++;
  return `${base}-${i}`;
}
// Propagation-tracking session surfaced to the UI. Heavy data (the accepted
// event set + the next-propagation seed) lives in refs; this is just what the
// TrackPanel renders.
export type TrackUiState = {
  status: "ready" | "propagating" | "paused";
  labelId: number;
  direction: "forward" | "backward";
  acceptedCount: number;
  slices: number;
  spanMs: number;
  lastConfidence: number | null;
  lastReason: string | null;
  // On an ambiguous pause (nearby/crossing target), the competing candidates so
  // the user can pick which one to keep tracking; previewCand = index currently
  // previewed (highlighted) but not yet confirmed.
  candidates?: TrackCandidate[];
  previewCand?: number | null;
  // Time range the track currently covers (start→end of propagation), for the
  // panel's draggable progress bar.
  coverStartUs?: number;
  coverEndUs?: number;
};
export type StatsData = { total: number; unlabelled: number; per_class: Record<string, number> } | null;
export type MessageBoxAction = { label: string; onClick: () => void; primary?: boolean };
export type MessageBoxState = { visible: boolean; title: string; body: string; kind: "info" | "success" | "error"; actions?: MessageBoxAction[] };

export function useAppState() {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [meta, setMeta] = useState<DatasetMeta | null>(null);
  const [labels, setLabels] = useState<LabelClass[]>([]);
  const [activeLabel, setActiveLabel] = useState(1);
  const [payload, setPayload] = useState<PointPayload | null>(null);
  const [viewMode, setViewMode] = useState<"3d" | "xy">("3d");
  // Display aids for reading the cloud (purely visual — not data filters).
  // colorMode "time" tints unlabelled events by timestamp so motion is visible;
  // pointSizeScale multiplies rendered point size for sparse/dense legibility.
  const [colorMode, setColorMode] = useState<"polarity" | "time">("polarity");
  // Persisted so a chosen point size survives a page reload / reopen.
  const [pointSizeScale, setPointSizeScale] = useState<number>(() => {
    const v = parseFloat(localStorage.getItem("display.pointSizeScale") ?? "");
    return Number.isFinite(v) && v > 0 ? v : 1;
  });
  // Display settings (visual only). unlabeledColor tints events with no label;
  // polarityContrast (0..1) controls how strongly ON/OFF polarity brightens /
  // darkens the base colour in the "polarity" colour scheme.
  // Persisted across sessions via localStorage so the user's display tuning
  // survives a page reload / reopen.
  const [unlabeledColor, setUnlabeledColor] = useState<string>(
    () => localStorage.getItem("display.unlabeledColor") || "#888888",
  );
  const [polarityContrast, setPolarityContrast] = useState<number>(() => {
    const v = parseFloat(localStorage.getItem("display.polarityContrast") ?? "");
    return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0.5;
  });
  useEffect(() => { localStorage.setItem("display.unlabeledColor", unlabeledColor); }, [unlabeledColor]);
  useEffect(() => { localStorage.setItem("display.polarityContrast", String(polarityContrast)); }, [polarityContrast]);
  useEffect(() => { localStorage.setItem("display.pointSizeScale", String(pointSizeScale)); }, [pointSizeScale]);
  // Settings dialog visibility.
  const [settingsVisible, setSettingsVisible] = useState(false);
  const [settingsSection, setSettingsSection] = useState<"display" | "general" | "cache" | "shortcuts" | "about">("general");
  const openSettings = useCallback((section: "display" | "general" | "cache" | "shortcuts" | "about" = "general") => {
    setSettingsSection(section);
    setSettingsVisible(true);
  }, []);
  const closeSettings = useCallback(() => setSettingsVisible(false), []);
  // Propagation tracking. UI state in `track`; the (potentially large) accepted
  // event set + next seed live in refs to avoid huge React updates.
  const [track, setTrack] = useState<TrackUiState | null>(null);
  const trackAcceptedRef = useRef<Set<number>>(new Set());
  const trackSeedRef = useRef<number[]>([]);
  const TRACK_STEP_US = 33_333; // fixed slice (independent of the view window)
  const [polarity, setPolarity] = useState("all");
  // Resident-event cap (soft). Default 0 = unbounded: every chunk in the
  // visible window stays resident. Users on low-RAM machines can dial a cap
  // back in via the right-panel Sample dropdown (the chunk pager then
  // decimates the visible window evenly across time).
  const [sample, setSample] = useState(0);
  const [startUs, setStartUs] = useState(0);
  const [endUs, setEndUs] = useState(1000000);
  const [xMin, setXMin] = useState(0);
  const [xMax, setXMax] = useState(1279);
  const [yMin, setYMin] = useState(0);
  const [yMax, setYMax] = useState(719);
  // Live (pre-Load) filter draft, surfaced as a yellow preview box in the 3D view.
  const [filterPreview, setFilterPreview] = useState<FilterPreview | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [status, setStatus] = useState("Ready");
  const [progress, setProgress] = useState<ProgressState>({ visible: false, value: 0, title: "", detail: "", elapsed: 0 });
  const [renderVideoVisible, setRenderVideoVisible] = useState(false);
  const openRenderVideo = useCallback(() => setRenderVideoVisible(true), []);
  const closeRenderVideo = useCallback(() => setRenderVideoVisible(false), []);
  const [messageBox, setMessageBox] = useState<MessageBoxState>({ visible: false, title: "", body: "", kind: "info" });
  const showMessage = useCallback((title: string, body: string, kind: MessageBoxState["kind"] = "info", actions?: MessageBoxAction[]) => {
    setMessageBox({ visible: true, title, body, kind, actions });
  }, []);
  const hideMessage = useCallback(() => setMessageBox((m) => ({ ...m, visible: false })), []);
  // Name-project dialog state (the picked file waits in a ref until confirmed).
  const [importPrompt, setImportPrompt] = useState<ImportPromptState>({ visible: false, fileName: "", suggested: "", existingNames: [] });
  const pendingFileRef = useRef<File | null>(null);
  const [stats, setStats] = useState<StatsData>(null);
  const [newLabelName, setNewLabelName] = useState("");
  const [pointInfo, setPointInfo] = useState<PointInfo | null>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [toolMode, setToolMode] = useState<ToolMode>("pan");
  const [brushRadius, setBrushRadius] = useState(20);
  const [selectThrough, setSelectThrough] = useState(true);
  // Wand (spatiotemporal connected component) parameters.
  // r_xy in pixels — neighbour distance on the sensor plane.
  // r_t_us in microseconds — neighbour gap along the time axis.
  // polarityMatch: when true, only walk events with the seed's polarity.
  // Wand = voxel connected-components. Three knobs directly size the voxel "box"
  // and its occupancy threshold (no more advanced/radius mode):
  //  - voxel width (px) and voxel duration (ms) = the largest gap still treated
  //    as "connected" in space / time;
  //  - min_pts = min events per voxel for it to count as occupied (drops sparse
  //    noise that would otherwise bridge everything into one blob).
  const [wandVoxelXy, setWandVoxelXy] = useState(15);
  const [wandVoxelTms, setWandVoxelTms] = useState(100);
  const [wandMinPts, setWandMinPts] = useState(2);
  // Whether a wand seed exists yet (enables the manual "Apply" button). Param
  // changes no longer auto-apply — the user clicks Apply to re-run on the
  // current selection, or just clicks a new point (which uses the current params).
  const [wandHasSeed, setWandHasSeed] = useState(false);
  // XY view orientation: 0 = X horizontal / Y vertical, 90 = X vertical / Y horizontal.
  // Origin (x=0, y=0) is always at top-left in either mode.
  const [xyRotation, setXyRotation] = useState(0);
  const rotateXyView = useCallback(() => setXyRotation((r) => (r === 0 ? 90 : 0)), []);
  // Vertical flip (mirror across horizontal axis) for the XY view.
  // Independent of rotation. Off by default → image-coord convention.
  const [xyFlipY, setXyFlipY] = useState(false);
  const toggleXyFlipY = useCallback(() => setXyFlipY((v) => !v), []);
  // Horizontal flip (mirror across vertical axis) for the XY view, parallel
  // to xyFlipY. Independent of rotation and Y flip.
  const [xyFlipX, setXyFlipX] = useState(false);
  const toggleXyFlipX = useCallback(() => setXyFlipX((v) => !v), []);

  // Playback state (XY frame view)
  const [currentFrameUs, setCurrentFrameUs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [frameWindowUs, setFrameWindowUs] = useState(33333); // 33ms = matches default video render frame window

  // "Recent Projects" is auto-scanned from the cache (every openable project),
  // most-recently-modified first — not a manually tracked list. So a project
  // imported anywhere (UI, CLI, MCP) shows up here. `project` is the cache id;
  // re-opening uses it to load the existing cache by name (no re-decode).
  const RECENTS_MAX = 10;
  const [recents, setRecents] = useState<{ path: string; name: string; project?: string }[]>([]);
  const refreshRecents = useCallback(async () => {
    try {
      const entries = await listCache();
      entries.sort((a, b) => b.modified - a.modified);
      setRecents(entries.slice(0, RECENTS_MAX).map((e) => ({
        path: e.source_path, name: e.dataset_id, project: e.dataset_id,
      })));
    } catch { /* backend down: keep the current list */ }
  }, []);

  useEffect(() => {
    listRecordings().then(setRecordings).catch((error) => setStatus(String(error)));
    refreshRecents();
  }, [refreshRecents]);

  const refreshStats = useCallback(async () => {
    if (!datasetId) return;
    try { setStats(await getStats(datasetId)); } catch { /* ignore */ }
  }, [datasetId]);

  const handleOpen = useCallback(async (path: string, name?: string) => {
    const startedAt = performance.now();
    setStatus("Opening dataset...");
    setProgress({ visible: true, value: 0.02, title: "Opening dataset", detail: "Starting backend cache task", elapsed: 0 });
    // If the open request itself fails (backend down, path rejected, …) the
    // progress dialog would otherwise sit at 2% forever — fail loudly instead.
    let task: { task_id: string };
    try {
      task = await openDatasetAsync(path, name);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      setProgress({ visible: false, value: 0, title: "", detail: "", elapsed: 0 });
      setStatus(`Open failed: ${detail}`);
      showMessage("Open failed", detail, "error");
      return;
    }
    const timer = window.setInterval(async () => {
      const elapsed = (performance.now() - startedAt) / 1000;
      try {
        const current = await getTask(task.task_id);
        setProgress({
          visible: current.status !== "ready",
          value: current.progress ?? 0,
          title: current.status === "failed" ? "Open failed" : "Opening dataset",
          detail: current.message ?? current.stage ?? "Working",
          elapsed
        });
        setStatus(`${current.message ?? "Opening"} (${Math.round((current.progress ?? 0) * 100)}%)`);
        if (current.status === "ready" && current.dataset_id && current.meta) {
          window.clearInterval(timer);
          setDatasetId(current.dataset_id);
          setMeta(current.meta);
          setStartUs(current.meta.t_min_us);
          setEndUs(current.meta.t_max_us);
          setCurrentFrameUs(current.meta.t_min_us);
          setXMin(0);
          setXMax(current.meta.width - 1);
          setYMin(0);
          setYMax(current.meta.height - 1);
          setLabels((await getLabelSchema(current.dataset_id)).labels);
          setProgress({ visible: false, value: 1, title: "Dataset ready", detail: "Cache ready", elapsed });
          setStatus(`Dataset ready in ${elapsed.toFixed(1)}s`);
          // Re-scan the cache so the just-opened project appears in Recent Projects.
          refreshRecents();
          refreshStats();
        }
        if (current.status === "failed") {
          window.clearInterval(timer);
          setStatus(current.message ?? "Open failed");
        }
      } catch (error) {
        window.clearInterval(timer);
        setStatus(String(error));
        setProgress({ visible: false, value: 0, title: "", detail: "", elapsed });
      }
    }, 350);
  }, [refreshStats, refreshRecents, showMessage]);

  const autoOpenRef = useRef(false);
  useEffect(() => {
    if (autoOpenRef.current) return;
    const params = new URLSearchParams(window.location.search);
    const autoOpenPath = params.get("open");
    if (!autoOpenPath) return;
    autoOpenRef.current = true;
    handleOpen(autoOpenPath);
  }, [handleOpen]);

  const startRenderVideo = useCallback(async (params: RenderVideoParams) => {
    if (!datasetId) return;
    setRenderVideoVisible(false);
    const startedAt = performance.now();
    setProgress({ visible: true, value: 0.02, title: "Rendering video", detail: "Queued", elapsed: 0 });
    try {
      const { task_id } = await renderVideoStart(datasetId, { ...params, unlabeled_color: unlabeledColor, polarity_contrast: polarityContrast });
      const poll = window.setInterval(async () => {
        try {
          const current = await getTask(task_id) as { status: string; progress?: number; message?: string; result?: { path?: string; bytes?: number; frames?: number; fps?: number; url?: string } };
          const elapsed = (performance.now() - startedAt) / 1000;
          setProgress({
            visible: current.status === "running",
            value: current.progress ?? 0,
            title: current.status === "failed" ? "Render failed" : "Rendering video",
            detail: current.message ?? "Working",
            elapsed,
          });
          if (current.status === "ready") {
            window.clearInterval(poll);
            setProgress({ visible: false, value: 1, title: "Rendered", detail: "", elapsed });
            const r = current.result ?? {};
            const fmt = params.format;
            const url = r.url ?? renderVideoUrl(datasetId, fmt);
            // Auto-trigger browser download.
            const a = document.createElement("a");
            a.href = url;
            a.download = `${datasetId}.${fmt}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            const sizeMb = r.bytes ? (r.bytes / 1_048_576).toFixed(1) + " MB" : "";
            const body = [
              `${r.frames ?? "?"} frames @ ${r.fps ?? params.fps} fps`,
              sizeMb ? `Size: ${sizeMb}` : "",
              r.path ? `File:\n  ${r.path}` : "",
            ].filter(Boolean).join("\n\n");
            setStatus(`Video rendered in ${elapsed.toFixed(1)}s`);
            showMessage(`Video ready (${elapsed.toFixed(1)}s)`, body, "success", [
              { label: "Download again", onClick: () => {
                const a2 = document.createElement("a");
                a2.href = url; a2.download = `${datasetId}.${fmt}`;
                document.body.appendChild(a2); a2.click(); document.body.removeChild(a2);
              } },
            ]);
          } else if (current.status === "failed") {
            window.clearInterval(poll);
            setProgress({ visible: false, value: 0, title: "", detail: "", elapsed });
            setStatus(`Render failed: ${current.message}`);
            showMessage("Render failed", current.message ?? "Unknown error", "error");
          }
        } catch (e) {
          window.clearInterval(poll);
          setProgress({ visible: false, value: 0, title: "", detail: "", elapsed: 0 });
          showMessage("Render failed", String(e), "error");
        }
      }, 400);
    } catch (e) {
      setProgress({ visible: false, value: 0, title: "", detail: "", elapsed: 0 });
      showMessage("Render failed", String(e), "error");
    }
  }, [datasetId, showMessage, unlabeledColor, polarityContrast]);

  const closeProject = useCallback(() => {
    setDatasetId("");
    setMeta(null);
    setPayload(null);
    setLabels([]);
    setSelectedIds(new Set());
    setStats(null);
    setStatus("Project closed");
    setProgress({ visible: false, value: 0, title: "", detail: "", elapsed: 0 });
  }, []);

  // ── Chunk-paged loading (multi-Points architecture) ───────────────────
  // Each chunk lives independently in `chunksMap`. PointCloudView renders
  // one Three.js Points per chunk, so we never allocate a single combined
  // payload (no 2× transition peaks) and per-chunk GPU uploads stay small.
  const [manifest, setManifest] = useState<ChunkManifest | null>(null);
  const [chunksMap, setChunksMap] = useState<Map<number, PointPayload>>(new Map());
  // Read-side ref so async fetch handlers can check current state without
  // including chunksMap in their effect deps (which would loop).
  const chunksReadRef = useRef(chunksMap);
  useEffect(() => { chunksReadRef.current = chunksMap; }, [chunksMap]);
  // Bumped on invalidateChunks (e.g. after a label mutation). The fetch
  // effect lists this in its deps so a manual cache flush triggers a
  // re-fetch — we DON'T put `chunksMap` itself in the fetch effect's deps
  // because every successful chunk arrival mutates it and would loop.
  const [chunksGeneration, setChunksGeneration] = useState(0);
  // True from the moment Assign is clicked until the freshly-relabelled
  // chunks have all been re-fetched. Used by LeftPanel to disable the
  // button so the user can't re-fire while the re-render is still resolving.
  const [assignBusy, setAssignBusy] = useState(false);
  // Optimistic label-patch channel. Declared up here (not next to assign) so the
  // histogram useMemo can list its seq as a dep — an in-place label edit keeps
  // the same chunksMap identity, so without this the histogram wouldn't refresh.
  const [labelPatch, setLabelPatch] = useState<{ seq: number; perChunk: Map<number, number[]> } | null>(null);
  const labelSeqRef = useRef(0);
  const labelVersionRef = useRef(0);

  // Fetch manifest whenever dataset changes.
  useEffect(() => {
    if (!datasetId) {
      setManifest(null);
      setChunksMap(new Map());
      return;
    }
    let cancelled = false;
    fetchChunkManifest(datasetId).then((mf) => {
      if (cancelled) return;
      setManifest(mf);
      setChunksMap(new Map());
    }).catch((e) => setStatus(`manifest failed: ${e}`));
    return () => { cancelled = true; };
  }, [datasetId]);

  // Soft cap on resident events. The Sample dropdown is the user-facing knob;
  // 0 → "no cap" but we still default to a safe number for huge datasets.
  // If the visible window exceeds the cap we evenly sub-sample chunks across
  // time and BUDGET-CHECK greedily so chunk-size variance doesn't push us
  // past the cap.
  const visibleChunkIds = useMemo<number[]>(() => {
    if (!manifest) return [];
    const win = Math.max(1, endUs - startUs);
    const buf = win * 0.15;                      // 15 % buffer either side
    const lo = startUs - buf;
    const hi = endUs + buf;
    const intersecting = manifest.chunks.filter((c) => c.end_us > lo && c.start_us < hi);
    if (intersecting.length === 0) return [];
    const totalEvents = intersecting.reduce((n, c) => n + c.event_count, 0);
    // sample === 0 ⇒ "No cap" — keep every chunk in the visible window,
    // never stride-decimate. The previous 8M fallback caused gaps in the
    // time axis when datasets exceeded 8M events: the playback head would
    // alternate between "in a kept chunk" (visible) and "in a dropped
    // chunk" (blank), producing the XY flicker.
    if (sample <= 0) return intersecting.map((c) => c.chunk_id);
    const cap = sample;
    if (totalEvents <= cap) return intersecting.map((c) => c.chunk_id);
    // Stride decimation as starting point; then greedily prune until under cap.
    const stride = Math.max(2, Math.ceil(totalEvents / cap));
    const candidates = intersecting.filter((_, i) => i % stride === 0);
    let running = 0;
    const result: number[] = [];
    for (const c of candidates) {
      if (running + c.event_count > cap) continue;   // skip rather than break (preserve coverage at later times)
      result.push(c.chunk_id);
      running += c.event_count;
    }
    return result;
  }, [manifest, startUs, endUs, sample]);

  // Sync chunksMap with visibleChunkIds: evict invisible chunks, fetch the
  // missing ones in parallel. Each chunk lives independently — no combined
  // payload allocation, so memory peaks are per-chunk (small).
  useEffect(() => {
    if (!manifest || !datasetId) return;
    let cancelled = false;
    const visibleSet = new Set(visibleChunkIds);

    // Evict chunks that fell out of the visible window.
    setChunksMap((prev) => {
      let dirty = false;
      const next = new Map(prev);
      for (const id of Array.from(next.keys())) {
        if (!visibleSet.has(id)) { next.delete(id); dirty = true; }
      }
      return dirty ? next : prev;
    });

    const queue = visibleChunkIds.filter((id) => !chunksReadRef.current.has(id));
    if (queue.length === 0) return;

    const debounceTimer = window.setTimeout(async () => {
      if (cancelled) return;
      const startedAt = performance.now();
      setStatus(`Fetching ${queue.length} chunk(s)...`);
      setProgress({ visible: true, value: 0.05, title: "Loading chunks", detail: `0 / ${queue.length}`, elapsed: 0 });
      const total = queue.length;
      let done = 0;
      const PARALLEL = 4;

      const runOne = async () => {
        while (!cancelled) {
          const id = queue.shift();
          if (id === undefined) return;
          try {
            const chunk = await fetchChunk(datasetId, id, manifest.chunk_duration_us);
            if (cancelled || !visibleSet.has(id)) return;
            // Insert chunk via functional update so concurrent inserts don't race.
            setChunksMap((prev) => {
              if (!visibleSet.has(id)) return prev;
              const next = new Map(prev);
              next.set(id, chunk);
              return next;
            });
          } catch (e) {
            setStatus(`chunk ${id} failed: ${e}`);
          } finally {
            done++;
            setProgress({
              visible: true,
              value: 0.05 + 0.9 * (done / total),
              title: "Loading chunks",
              detail: `${done} / ${total}`,
              elapsed: (performance.now() - startedAt) / 1000,
            });
          }
        }
      };
      const workers = Array.from({ length: PARALLEL }, () => runOne());
      await Promise.all(workers);
      if (cancelled) return;
      const elapsed = (performance.now() - startedAt) / 1000;
      setProgress({ visible: false, value: 1, title: "", detail: "", elapsed });
      setStatus(`Loaded ${visibleChunkIds.length} chunks in ${elapsed.toFixed(1)}s`);
    }, 250);
    return () => { cancelled = true; window.clearTimeout(debounceTimer); };
  }, [manifest, datasetId, visibleChunkIds, chunksGeneration]);

  // Histogram across currently-resident chunks. Iterates per-chunk label
  // arrays directly so no combined payload is required. Falls back to the
  // legacy `payload` if it ever gets set (kept for compat).
  const histogram = useMemo(() => {
    const counts = new Map<number, number>();
    if (chunksMap.size > 0) {
      for (const p of chunksMap.values()) {
        for (const label of p.label) counts.set(label, (counts.get(label) ?? 0) + 1);
      }
      return counts;
    }
    if (payload) {
      for (const label of payload.label) counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    return counts;
    // labelPatch dep: in-place label edits keep chunksMap identity, so the
    // patch seq is what tells the histogram to recount.
  }, [payload, chunksMap, labelPatch]);

  // Total visible event count across loaded chunks (or legacy payload). Used
  // in StatusBar / RightPanel where the old code referenced `payload.count`.
  const visibleEventCount = useMemo(() => {
    if (chunksMap.size > 0) {
      let n = 0;
      for (const p of chunksMap.values()) n += p.count;
      return n;
    }
    return payload?.count ?? 0;
  }, [payload, chunksMap]);

  // (assignBusy is now released directly in the assign() finally block — the
  // optimistic path no longer re-fetches chunks, so there's nothing to wait on.)

  // Invalidate cached chunks (e.g. after a label mutation) so the visible
  // window re-fetches fresh bytes with up-to-date labels. Bumping the
  // generation counter is what re-triggers the fetch effect — clearing the
  // map alone wouldn't, since chunksMap is not in that effect's deps.
  const invalidateChunks = useCallback(() => {
    setChunksMap(new Map());
    setChunksGeneration((g) => g + 1);
  }, []);

  // Targeted invalidation: drop only the chunks that contain at least one
  // of the given event IDs. Used after Assign / Assign-single so a small
  // edit doesn't force a re-fetch of the whole visible window.
  // Chunks are stored time-sequentially with monotonic event_start_index,
  // so we can binary-search the manifest to map an event_id → chunk_id.
  const invalidateChunksByEventIds = useCallback((eventIds: Iterable<number>) => {
    if (!manifest) { setChunksMap(new Map()); setChunksGeneration((g) => g + 1); return; }
    const chunks = manifest.chunks;
    const affected = new Set<number>();
    for (const eid of eventIds) {
      let lo = 0, hi = chunks.length - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const c = chunks[mid];
        if (eid < c.event_start_index) hi = mid - 1;
        else if (eid >= c.event_start_index + c.event_count) lo = mid + 1;
        else { affected.add(c.chunk_id); break; }
      }
    }
    if (affected.size === 0) return;
    setChunksMap((prev) => {
      const next = new Map(prev);
      for (const cid of affected) next.delete(cid);
      return next;
    });
    setChunksGeneration((g) => g + 1);
  }, [manifest]);

  // Invalidate every chunk overlapping a time range. Used after region-based
  // labelling (filter assign), where many more chunks may be stale than the
  // ones that contained the visually-selected events.
  const invalidateChunksByTimeRange = useCallback((startUs: number, endUs: number) => {
    if (!manifest) { setChunksMap(new Map()); setChunksGeneration((g) => g + 1); return; }
    const affected = new Set<number>();
    for (const c of manifest.chunks) {
      if (c.end_us > startUs && c.start_us < endUs) affected.add(c.chunk_id);
    }
    if (affected.size === 0) return;
    setChunksMap((prev) => {
      const next = new Map(prev);
      for (const cid of affected) next.delete(cid);
      return next;
    });
    setChunksGeneration((g) => g + 1);
  }, [manifest]);

  // (Multi-Points refactor) The transitional combined-payload effect is gone.
  // PointCloudView reads `chunksMap` + `manifest` directly via the chunk
  // reconciliation effect, so we never allocate a single ~135 MB combined
  // typed-array set on the JS heap — that allocation was the OOM trigger on
  // borderline machines when loading rain.es at 5M+ events.
  //
  // `payload` stays null in this code path; it remains in the surface API only
  // so legacy consumers that don't yet read chunks can no-op gracefully.

  // Legacy single-shot reload — preserved so label-mutating callbacks below
  // can call `await loadPoints()` after a label change. With chunks we just
  // invalidate the cache and let the visibility effect re-fetch.
  const loadPoints = useCallback(async () => {
    if (!datasetId) return;
    invalidateChunks();
  }, [datasetId, invalidateChunks]);

  const addToSelection = useCallback((ids: number[]) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      for (const id of ids) next.add(id);
      return next;
    });
  }, []);

  const removeFromSelection = useCallback((ids: number[]) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      for (const id of ids) next.delete(id);
      return next;
    });
  }, []);

  // Range selection across all currently-resident chunks. Each chunk's
  // payload.event_id[i] is sequential (= chunk.event_start_index + i) by
  // construction of the backend chunk endpoint, so we use the chunk
  // start-index from the manifest to avoid BigInt allocations in the loop.
  // Single-flight guard — at large radii BFS can take seconds, and Flask
  // is single-threaded, so click-spam stacks up. Drop subsequent clicks
  // while one is in flight rather than queueing them.
  const wandInFlightRef = useRef(false);
  // Last successful wand seed. Set on each successful selectComponent response
  // so the debounced live-refine effect can re-run the wand on parameter
  // changes. null until the first wand click (guards the effect on mount).
  const lastWandSeedRef = useRef<number | null>(null);
  // Wand action — backend BFS yields the seed's connected component, then
  // applies it to selectedIds with the user's modifier op (set / add / sub).
  const selectComponent = useCallback(async (seedEventId: number, op: SelectionOp) => {
    if (!datasetId) return;
    if (wandInFlightRef.current) {
      setStatus("Wand: previous request still running — ignored");
      return;
    }
    wandInFlightRef.current = true;
    setProgress({ visible: true, value: 0, title: "Wand: finding component", detail: "Tracing connected events…", elapsed: 0, indeterminate: true });
    setStatus("Wand: finding component…");
    const params: import("../api/client").WandParams = {
      seed_event_id: seedEventId,
      auto: true,
      voxel_xy: wandVoxelXy,
      voxel_t_us: wandVoxelTms * 1000,
      min_pts: wandMinPts,
    };
    // eslint-disable-next-line no-console
    console.log("[wand] request", params);
    try {
      const result = await selectComponentApi(datasetId, params);
      // Remember the seed so the manual "Apply" button can re-run on it.
      lastWandSeedRef.current = seedEventId;
      setWandHasSeed(true);
      const ids = result.event_ids;
      // eslint-disable-next-line no-console
      console.log("[wand] response", { size: result.size, truncated: result.truncated, reason: (result as { truncated_reason?: string }).truncated_reason, elapsed: (result as { elapsed_s?: number }).elapsed_s, firstFew: ids.slice(0, 5) });
      if (op === "add") {
        setSelectedIds((prev) => {
          const next = new Set(prev);
          for (const id of ids) next.add(id);
          return next;
        });
      } else if (op === "subtract") {
        setSelectedIds((prev) => {
          const next = new Set(prev);
          for (const id of ids) next.delete(id);
          return next;
        });
      } else {
        setSelectedIds(new Set(ids));
      }
      const reason = (result as { truncated_reason?: string }).truncated_reason;
      const suffix = result.truncated
        ? reason === "time_budget"
          ? " (time budget hit — try smaller radii)"
          : reason === "max_size"
            ? " (size cap — component is huge)"
            : " (truncated)"
        : "";
      setStatus(`Wand: ${result.size.toLocaleString()} events${suffix}`);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error("[wand] failed", e);
      setStatus(`Wand failed: ${e}`);
      showMessage("Wand failed", String(e), "error");
    } finally {
      wandInFlightRef.current = false;
      setProgress({ visible: false, value: 0, title: "", detail: "", elapsed: 0 });
    }
  }, [datasetId, wandVoxelXy, wandVoxelTms, wandMinPts, showMessage]);

  // Manual apply: re-run the wand on the LAST seed with the CURRENT parameters,
  // replacing the current selection. Parameter changes no longer auto-apply —
  // the user clicks "Apply" to refine the current selection, or simply clicks a
  // new point (a fresh wand click always uses the current parameters).
  const applyWandParams = useCallback(() => {
    const seed = lastWandSeedRef.current;
    if (seed == null) {
      setStatus("Wand: click a point first, then Apply");
      return;
    }
    selectComponent(seed, "set");
  }, [selectComponent]);

  const selectByRanges = useCallback(() => {
    if (!meta) return;
    const selected = new Set<number>();
    if (chunksMap.size > 0 && manifest) {
      const manifestById = new Map<number, ChunkManifestEntry>();
      for (const c of manifest.chunks) manifestById.set(c.chunk_id, c);
      for (const [chunkId, p] of chunksMap) {
        const entry = manifestById.get(chunkId);
        if (!entry) continue;
        const base = entry.event_start_index;
        for (let i = 0; i < p.count; i++) {
          const t = p.t_us[i];
          const px = Number(p.positions[i * 3]) + (meta.width - 1) / 2;
          const py = (meta.height - 1) / 2 - Number(p.positions[i * 3 + 1]);
          if (t >= startUs && t <= endUs && px >= xMin && px <= xMax && py >= yMin && py <= yMax) selected.add(base + i);
        }
      }
    } else if (payload) {
      for (let i = 0; i < payload.count; i++) {
        const t = payload.t_us[i];
        const id = Number(payload.event_id[i]);
        const px = Number(payload.positions[i * 3]) + (meta.width - 1) / 2;
        const py = (meta.height - 1) / 2 - Number(payload.positions[i * 3 + 1]);
        if (t >= startUs && t <= endUs && px >= xMin && px <= xMax && py >= yMin && py <= yMax) selected.add(id);
      }
    } else {
      return;
    }
    setSelectedIds(selected);
    setStatus(`Selected ${selected.size.toLocaleString()} events`);
  }, [payload, manifest, chunksMap, meta, startUs, endUs, xMin, xMax, yMin, yMax]);

  // ── Optimistic local label patching ──────────────────────────────────────
  // Assign / Clear no longer invalidate + re-fetch chunks. They mutate the
  // already-loaded chunk payloads' `label` arrays IN PLACE and notify
  // PointCloudView via `labelPatch` to repaint only those points on the GPU.
  // chunksMap identity is deliberately NOT changed, so the geometry
  // reconciliation effect does not run — no chunk re-download, no BufferGeometry
  // rebuild. Persistence happens in the background; on failure we roll back.
  // (labelPatch / labelSeqRef / labelVersionRef are declared earlier so the
  // histogram can depend on the patch seq.)
  const emitLabelPatch = useCallback((perChunk: Map<number, number[]>) => {
    if (perChunk.size === 0) return;
    labelSeqRef.current += 1;
    setLabelPatch({ seq: labelSeqRef.current, perChunk });
  }, []);

  // Map global event ids → { chunkId: [local indices] } across loaded chunks.
  // payload.event_id[i] == chunk.event_start_index + i (contiguous), so the
  // local index is just id - base.
  const collectByIds = useCallback((ids: Iterable<number>): Map<number, number[]> => {
    const per = new Map<number, number[]>();
    if (!manifest) return per;
    const entryById = new Map<number, ChunkManifestEntry>();
    for (const c of manifest.chunks) entryById.set(c.chunk_id, c);
    const loaded: { cid: number; start: number; end: number }[] = [];
    for (const cid of chunksMap.keys()) {
      const e = entryById.get(cid);
      if (e) loaded.push({ cid, start: e.event_start_index, end: e.event_start_index + e.event_count });
    }
    loaded.sort((a, b) => a.start - b.start);
    for (const id of ids) {
      let lo = 0, hi = loaded.length - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const L = loaded[mid];
        if (id < L.start) hi = mid - 1;
        else if (id >= L.end) lo = mid + 1;
        else { let arr = per.get(L.cid); if (!arr) { arr = []; per.set(L.cid, arr); } arr.push(id - L.start); break; }
      }
    }
    return per;
  }, [manifest, chunksMap]);

  // Same, but for a spatiotemporal filter — mirrors the backend's predicate so
  // the loaded view stays consistent with a filter/envelope-based label write.
  const collectByFilter = useCallback((f: { start_us: number; end_us: number; polarity: string; x_min: number; x_max: number; y_min: number; y_max: number }): Map<number, number[]> => {
    const per = new Map<number, number[]>();
    if (!meta) return per;
    const w = meta.width, h = meta.height;
    for (const [cid, p] of chunksMap) {
      let arr: number[] | undefined;
      for (let i = 0; i < p.count; i++) {
        const t = p.t_us[i];
        if (t < f.start_us || t > f.end_us) continue;
        const px = p.positions[i * 3] + (w - 1) / 2;
        const py = (h - 1) / 2 - p.positions[i * 3 + 1];
        if (px < f.x_min || px > f.x_max || py < f.y_min || py > f.y_max) continue;
        if (f.polarity === "on" && !p.polarity[i]) continue;
        if (f.polarity === "off" && p.polarity[i]) continue;
        if (!arr) { arr = []; per.set(cid, arr); }
        arr.push(i);
      }
    }
    return per;
  }, [meta, chunksMap]);

  // Mutate label arrays in place; returns a rollback closure.
  const mutatePerChunk = useCallback((per: Map<number, number[]>, newValue: number) => {
    const undo: { cid: number; idx: number[]; old: Int16Array }[] = [];
    for (const [cid, idxs] of per) {
      const p = chunksMap.get(cid);
      if (!p) continue;
      const old = new Int16Array(idxs.length);
      for (let k = 0; k < idxs.length; k++) { old[k] = p.label[idxs[k]]; p.label[idxs[k]] = newValue; }
      undo.push({ cid, idx: idxs, old });
    }
    return () => {
      for (const u of undo) {
        const p = chunksMap.get(u.cid);
        if (!p) continue;
        for (let k = 0; k < u.idx.length; k++) p.label[u.idx[k]] = u.old[k];
      }
    };
  }, [chunksMap]);

  // Fold a backend stats_delta (keyed by label id, "-1" = unlabelled) into the
  // existing stats without an O(N) /stats rescan.
  const applyStatsDelta = useCallback((delta: Record<string, number>) => {
    setStats((prev) => {
      if (!prev) return prev;
      const next = { total: prev.total, unlabelled: prev.unlabelled, per_class: { ...prev.per_class } };
      for (const [idStr, d] of Object.entries(delta)) {
        const id = Number(idStr);
        if (id === -1) { next.unlabelled += d; continue; }
        const cls = labels.find((l) => l.id === id);
        if (cls) next.per_class[cls.name] = (next.per_class[cls.name] ?? 0) + d;
      }
      return next;
    });
  }, [labels]);

  const assign = useCallback(async (operation: "assign" | "clear") => {
    if (!datasetId || selectedIds.size === 0 || assignBusy) return;
    const newValue = operation === "clear" ? -1 : activeLabel;
    const allIds = Array.from(selectedIds);
    // Switch to pan mode so a lingering selection tool can't start a fresh
    // selection on the points we're about to repaint.
    setToolMode("pan");
    setAssignBusy(true);

    // Subsampled (cap below dataset total): the backend labels the whole
    // spatiotemporal envelope of the selection (incl. unloaded points), so we
    // patch every loaded point in that same envelope. Otherwise: exact ids.
    const totalEvents = manifest?.total_events ?? 0;
    const subsampled = totalEvents > 0 && sample > 0 && sample < totalEvents;

    let per: Map<number, number[]>;
    let persist: () => Promise<{ updated?: number; label_version?: number; stats_delta?: Record<string, number> }>;

    if (subsampled && manifest && meta) {
      const w = meta.width, h = meta.height;
      const entryById = new Map<number, ChunkManifestEntry>();
      for (const c of manifest.chunks) entryById.set(c.chunk_id, c);
      let xMinP = Infinity, xMaxP = -Infinity, yMinP = Infinity, yMaxP = -Infinity, tMinP = Infinity, tMaxP = -Infinity;
      for (const [cid, p] of chunksMap) {
        const entry = entryById.get(cid);
        if (!entry) continue;
        const base = entry.event_start_index;
        for (let i = 0; i < p.count; i++) {
          if (!selectedIds.has(base + i)) continue;
          const px = p.positions[i * 3] + (w - 1) / 2;
          const py = (h - 1) / 2 - p.positions[i * 3 + 1];
          const t = p.t_us[i];
          if (px < xMinP) xMinP = px; if (px > xMaxP) xMaxP = px;
          if (py < yMinP) yMinP = py; if (py > yMaxP) yMaxP = py;
          if (t < tMinP) tMinP = t; if (t > tMaxP) tMaxP = t;
        }
      }
      if (!isFinite(xMinP)) { setAssignBusy(false); return; }
      const env = {
        start_us: tMinP, end_us: tMaxP, polarity: "all",
        x_min: Math.max(0, Math.floor(xMinP)), x_max: Math.min(w - 1, Math.ceil(xMaxP)),
        y_min: Math.max(0, Math.floor(yMinP)), y_max: Math.min(h - 1, Math.ceil(yMaxP)),
      };
      per = collectByFilter(env);
      persist = () => updateLabelsByFilter(datasetId, { label: activeLabel, operation, ...env });
    } else {
      per = collectByIds(allIds);
      persist = () => updateLabelsBinary(datasetId, Uint32Array.from(allIds), activeLabel, operation);
    }

    const rollback = mutatePerChunk(per, newValue);
    emitLabelPatch(per);
    setSelectedIds(new Set());
    setStatus(`${operation === "assign" ? "Assigned" : "Cleared"} ${allIds.length.toLocaleString()} events`);

    try {
      const result = await persist();
      if (result.stats_delta) applyStatsDelta(result.stats_delta);
      else refreshStats();
      if (typeof result.label_version === "number") labelVersionRef.current = result.label_version;
    } catch (err) {
      rollback();
      emitLabelPatch(per);
      setSelectedIds(new Set(allIds));
      setStatus(`Label update failed: ${String(err)}`);
      showMessage("Label update failed", String(err), "error");
    } finally {
      setAssignBusy(false);
    }
  }, [datasetId, selectedIds, activeLabel, manifest, meta, sample, chunksMap, assignBusy, collectByIds, collectByFilter, mutatePerChunk, emitLabelPatch, applyStatsDelta, refreshStats, showMessage]);

  const assignSingle = useCallback(async (eventId: number, labelId: number, operation: "assign" | "clear") => {
    if (!datasetId) return;
    const newValue = operation === "clear" ? -1 : labelId;
    const per = collectByIds([eventId]);
    const rollback = mutatePerChunk(per, newValue);
    emitLabelPatch(per);
    try {
      const result = await updateLabelsBinary(datasetId, Uint32Array.from([eventId]), labelId, operation);
      if (result.stats_delta) applyStatsDelta(result.stats_delta);
      else refreshStats();
    } catch (err) {
      rollback();
      emitLabelPatch(per);
      setStatus(`Label update failed: ${String(err)}`);
    }
  }, [datasetId, collectByIds, mutatePerChunk, emitLabelPatch, applyStatsDelta, refreshStats]);

  const assignByFilter = useCallback(async () => {
    if (!datasetId) return;
    const f = { start_us: startUs, end_us: endUs, polarity, x_min: xMin, x_max: xMax, y_min: yMin, y_max: yMax };
    const per = collectByFilter(f);
    const rollback = mutatePerChunk(per, activeLabel);
    emitLabelPatch(per);
    try {
      const result = await updateLabelsByFilter(datasetId, { label: activeLabel, operation: "assign", ...f });
      setStatus(`Assigned ${result.updated.toLocaleString()} labels by filter`);
      if (result.stats_delta) applyStatsDelta(result.stats_delta);
      else refreshStats();
    } catch (err) {
      rollback();
      emitLabelPatch(per);
      setStatus(`Paint failed: ${String(err)}`);
      showMessage("Paint failed", String(err), "error");
    }
  }, [datasetId, activeLabel, startUs, endUs, polarity, xMin, xMax, yMin, yMax, collectByFilter, mutatePerChunk, emitLabelPatch, applyStatsDelta, refreshStats, showMessage]);

  const clearByFilter = useCallback(async () => {
    if (!datasetId) return;
    const f = { start_us: startUs, end_us: endUs, polarity, x_min: xMin, x_max: xMax, y_min: yMin, y_max: yMax };
    const per = collectByFilter(f);
    const rollback = mutatePerChunk(per, -1);
    emitLabelPatch(per);
    try {
      const result = await updateLabelsByFilter(datasetId, { label: -1, operation: "clear", ...f });
      setStatus(`Cleared ${result.updated.toLocaleString()} labels by filter`);
      if (result.stats_delta) applyStatsDelta(result.stats_delta);
      else refreshStats();
    } catch (err) {
      rollback();
      emitLabelPatch(per);
      setStatus(`Clear failed: ${String(err)}`);
      showMessage("Clear failed", String(err), "error");
    }
  }, [datasetId, startUs, endUs, polarity, xMin, xMax, yMin, yMax, collectByFilter, mutatePerChunk, emitLabelPatch, applyStatsDelta, refreshStats, showMessage]);

  // Reset every event to unlabelled (-1) — a clean slate for re-testing. Clears
  // the loaded chunks locally (instant histogram/recolour) and the whole
  // spatiotemporal range on the backend (covers unloaded events too).
  const clearAllLabels = useCallback(async () => {
    if (!datasetId || !meta) return;
    const per = new Map<number, number[]>();
    for (const [cid, p] of chunksMap) {
      let arr: number[] | undefined;
      for (let i = 0; i < p.count; i++) if (p.label[i] !== -1) { if (!arr) { arr = []; per.set(cid, arr); } arr.push(i); }
    }
    const rollback = mutatePerChunk(per, -1);
    emitLabelPatch(per);
    setSelectedIds(new Set());
    const f = { start_us: meta.t_min_us, end_us: meta.t_max_us, polarity: "all", x_min: 0, x_max: meta.width - 1, y_min: 0, y_max: meta.height - 1 };
    try {
      const result = await updateLabelsByFilter(datasetId, { label: -1, operation: "clear", ...f });
      setStatus(`Cleared all labels (${result.updated.toLocaleString()} events)`);
      refreshStats();
    } catch (err) {
      rollback();
      emitLabelPatch(per);
      setStatus(`Clear failed: ${String(err)}`);
      showMessage("Clear failed", String(err), "error");
    }
  }, [datasetId, meta, chunksMap, mutatePerChunk, emitLabelPatch, refreshStats, showMessage]);

  // Exact-id optimistic label write (used by the track freeze; mirrors the
  // exact branch of assign without the envelope path).
  const labelEventIds = useCallback(async (ids: number[], labelId: number, operation: "assign" | "clear") => {
    if (!datasetId || ids.length === 0) return;
    const newValue = operation === "clear" ? -1 : labelId;
    const per = collectByIds(ids);
    const rollback = mutatePerChunk(per, newValue);
    emitLabelPatch(per);
    try {
      const result = await updateLabelsBinary(datasetId, Uint32Array.from(ids), labelId, operation);
      if (result.stats_delta) applyStatsDelta(result.stats_delta);
      else refreshStats();
    } catch (err) {
      rollback();
      emitLabelPatch(per);
      setStatus(`Label update failed: ${String(err)}`);
      showMessage("Label update failed", String(err), "error");
    }
  }, [datasetId, collectByIds, mutatePerChunk, emitLabelPatch, applyStatsDelta, refreshStats, showMessage]);

  // ── Propagation tracking ──────────────────────────────────────────────────
  // Create → propagate (auto-pauses at first low-confidence slice) → optionally
  // re-seed from a correction → freeze. The propagated events are a
  // non-destructive PREVIEW shown via the selection highlight; nothing is
  // written to labels until Freeze.
  const trackSetDirection = useCallback((direction: "forward" | "backward") => {
    setTrack((t) => (t ? { ...t, direction } : t));
  }, []);

  const trackCreate = useCallback(() => {
    if (!datasetId) return;
    if (selectedIds.size === 0) {
      showMessage("Create track", "Select the target first, then create a track.", "error");
      return;
    }
    const seed = Array.from(selectedIds);
    trackAcceptedRef.current = new Set(seed);
    trackSeedRef.current = seed;
    setTrack({ status: "ready", labelId: activeLabel, direction: "forward", acceptedCount: seed.length, slices: 0, spanMs: 0, lastConfidence: null, lastReason: null, coverStartUs: currentFrameUs, coverEndUs: currentFrameUs });
    setStatus(`Track created from ${seed.length.toLocaleString()} events`);
  }, [datasetId, selectedIds, activeLabel, showMessage, currentFrameUs]);

  const trackPropagate = useCallback(async () => {
    if (!datasetId || !track || trackSeedRef.current.length === 0) return;
    setTrack((t) => (t ? { ...t, status: "propagating" } : t));
    try {
      const res = await propagateTrack(datasetId, {
        seed_event_ids: trackSeedRef.current,
        step_us: TRACK_STEP_US,
        direction: track.direction,
        max_slices: 60,
        stop_on_low: true,
      });
      const okSlices = res.slices.filter((s) => s.status === "ok");
      for (const s of okSlices) for (const id of s.event_ids) trackAcceptedRef.current.add(id);
      // Advance the seed to the frontier (last confident slice) so a repeat
      // Propagate continues onward.
      if (res.frontier_event_ids.length > 0) trackSeedRef.current = res.frontier_event_ids;
      // Extend the loaded window so the freshly-propagated preview is visible,
      // and jump the playback head to where it paused.
      if (okSlices.length > 0) {
        const lastEnd = Math.max(...okSlices.map((s) => s.t_end));
        const firstStart = Math.min(...okSlices.map((s) => s.t_start));
        if (track.direction === "forward") setEndUs((e) => Math.max(e, lastEnd));
        else setStartUs((s) => Math.min(s, firstStart));
      }
      const pause = res.pause_time_us;
      if (pause != null) setCurrentFrameUs(pause);
      else if (okSlices.length > 0) setCurrentFrameUs(okSlices[okSlices.length - 1].t_center);
      // Preview = all accepted events (selection highlight).
      setSelectedIds(new Set(trackAcceptedRef.current));
      const lastConf = res.slices.length > 0 ? res.slices[res.slices.length - 1].confidence : null;
      const coveredMs = okSlices.length * (TRACK_STEP_US / 1000);
      // Extend the covered time range for the panel's progress bar.
      const sliceStarts = res.slices.map((s) => s.t_start);
      const sliceEnds = res.slices.map((s) => s.t_end);
      setTrack((t) => t ? {
        ...t,
        status: "paused",
        acceptedCount: trackAcceptedRef.current.size,
        slices: t.slices + okSlices.length,
        spanMs: t.spanMs + coveredMs,
        lastConfidence: lastConf,
        lastReason: res.stop_reason,
        candidates: res.candidates && res.candidates.length > 1 ? res.candidates : undefined,
        previewCand: undefined,
        coverStartUs: sliceStarts.length ? Math.min(t.coverStartUs ?? Infinity, ...sliceStarts) : t.coverStartUs,
        coverEndUs: sliceEnds.length ? Math.max(t.coverEndUs ?? -Infinity, ...sliceEnds) : t.coverEndUs,
      } : t);
      // Diagnostics for the first slice: how much of the ROI did we keep, and
      // was the object split into many blobs? (helps diagnose under-selection)
      const d0 = res.diag && res.diag[0];
      const diagStr = d0
        ? ` | slice1: kept ${d0.kept}/${d0.roi} ROI (${Math.round(d0.kept_frac_roi * 100)}%), blobs ${d0.merged_blobs}+${d0.competitors}`
        : "";
      setStatus(`Track: +${okSlices.length} slices (${coveredMs.toFixed(0)} ms), paused: ${res.stop_reason}${diagStr}`);
    } catch (err) {
      setTrack((t) => (t ? { ...t, status: "paused" } : t));
      setStatus(`Propagate failed: ${String(err)}`);
      showMessage("Propagate failed", String(err), "error");
    }
  }, [datasetId, track, showMessage]);

  // Correction: take the user's current (re-)selection at the pause frame as the
  // new keyframe seed, fold it into the accepted preview, then propagate on.
  const trackUseSelectionAsSeed = useCallback(() => {
    if (!track) return;
    if (selectedIds.size === 0) {
      showMessage("Set keyframe", "Re-select the target at this frame first.", "error");
      return;
    }
    const sel = Array.from(selectedIds);
    trackSeedRef.current = sel;
    for (const id of sel) trackAcceptedRef.current.add(id);
    setTrack((t) => (t ? { ...t, acceptedCount: trackAcceptedRef.current.size, candidates: undefined, previewCand: undefined } : t));
    setStatus(`Keyframe set from ${sel.length.toLocaleString()} events — Propagate to continue`);
  }, [track, selectedIds, showMessage]);

  // Ambiguous-pause resolution, step 1 — PREVIEW: clicking a candidate just
  // highlights its events (so the user can SEE which object it is) and remembers
  // the choice. Nothing is committed yet.
  const trackPreviewCandidate = useCallback((index: number) => {
    setTrack((t) => {
      if (!t || !t.candidates || !t.candidates[index]) return t;
      setSelectedIds(new Set(t.candidates[index].event_ids));
      return { ...t, previewCand: index };
    });
  }, []);

  // Step 2 — CONFIRM: commit the previewed candidate as the new keyframe seed
  // (folded into the accepted set), so the next Propagate continues from it.
  const trackConfirmCandidate = useCallback(() => {
    setTrack((t) => {
      if (!t || !t.candidates || t.previewCand == null || !t.candidates[t.previewCand]) return t;
      const ids = t.candidates[t.previewCand].event_ids;
      trackSeedRef.current = ids;
      for (const id of ids) trackAcceptedRef.current.add(id);
      setSelectedIds(new Set(trackAcceptedRef.current));
      setStatus(`Candidate ${t.previewCand + 1} confirmed (${ids.length.toLocaleString()} events) — Propagate to continue`);
      return { ...t, acceptedCount: trackAcceptedRef.current.size, candidates: undefined, previewCand: undefined, lastReason: null };
    });
  }, []);

  const trackCancel = useCallback(() => {
    trackAcceptedRef.current = new Set();
    trackSeedRef.current = [];
    setSelectedIds(new Set());
    setTrack(null);
    setStatus("Track cancelled");
  }, []);

  // Export modal visibility. Export is now parameterised (sample on/off, only
  // labelled, which files), so the toolbar/menu open this dialog instead of
  // firing a one-shot export.
  const [exportVisible, setExportVisible] = useState(false);
  const openExport = useCallback(() => {
    if (!datasetId) {
      showMessage("Export skipped", "No dataset is open. Open a recording first.", "error");
      return;
    }
    setExportVisible(true);
  }, [datasetId, showMessage]);
  const closeExport = useCallback(() => setExportVisible(false), []);

  // Screenshot / Print of the current view (3D or XY, whichever is active).
  // Screenshot goes to the clipboard (image/png) rather than a file.
  const screenshotView = useCallback(async () => {
    if (!datasetId) { showMessage("Screenshot", "Open a recording first.", "error"); return; }
    const url = captureView({ transparent: false });
    if (!url) { showMessage("Screenshot failed", "Could not capture the view.", "error"); return; }
    try {
      await copyDataUrlToClipboard(url);
      setStatus("Screenshot copied to clipboard");
    } catch (e) {
      showMessage("Screenshot failed", `Could not copy to clipboard: ${e instanceof Error ? e.message : String(e)}`, "error");
    }
  }, [datasetId, showMessage]);

  const [printVisible, setPrintVisible] = useState(false);
  const openPrint = useCallback(() => {
    if (!datasetId) { showMessage("Print", "Open a recording first.", "error"); return; }
    setPrintVisible(true);
  }, [datasetId, showMessage]);
  const cancelPrint = useCallback(() => setPrintVisible(false), []);
  const confirmPrint = useCallback((orientation: "landscape" | "portrait") => {
    setPrintVisible(false);
    // Vector PDF: points as a high-res image layer, axes/labels as true SVG.
    const svg = captureSvgView();
    if (svg) { printSvgMarkup(svg, orientation); return; }
    // Fallback to a transparent raster if the SVG export is unavailable.
    const url = captureView({ transparent: true });
    if (!url) { showMessage("Print failed", "Could not capture the view.", "error"); return; }
    printDataUrl(url, orientation);
  }, [showMessage]);

  const startExport = useCallback(async (params: ExportParams) => {
    if (!datasetId) return;
    setExportVisible(false);
    // Export can take several seconds for large datasets — surface progress.
    const startedAt = performance.now();
    setProgress({ visible: true, value: 0.05, title: "Exporting labels", detail: "Writing selected files", elapsed: 0 });
    const ticker = window.setInterval(() => {
      setProgress((p) => p.visible ? { ...p, elapsed: (performance.now() - startedAt) / 1000, value: Math.min(0.9, p.value + 0.04) } : p);
    }, 200);
    try {
      const r = await exportLabels(datasetId, params);
      window.clearInterval(ticker);
      const elapsed = (performance.now() - startedAt) / 1000;
      setProgress({ visible: false, value: 1, title: "Exported", detail: "", elapsed });
      const fmtBytes = (n?: number) => {
        if (!n || n <= 0) return "";
        if (n >= 1_048_576) return ` (${(n / 1_048_576).toFixed(1)} MB)`;
        if (n >= 1024) return ` (${(n / 1024).toFixed(0)} KB)`;
        return ` (${n} B)`;
      };
      const triggerDownload = () => {
        if (!r.zip_url) return;
        const a = document.createElement("a");
        a.href = r.zip_url;
        a.download = `${datasetId}_labels.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      };
      // Auto-trigger the browser download (user explicitly invoked Export).
      triggerDownload();
      const lines = [
        r.zip    ? `Zip bundle${fmtBytes(r.zip_bytes)}\n  ${r.zip}` : "",
        r.labels ? `Labels (.npz)\n  ${r.labels}` : "",
        r.ply    ? `Point cloud (.ply)\n  ${r.ply}`    : "",
        r.csv    ? `Table (.csv)\n  ${r.csv}`    : "",
        r.project? `Project\n  ${r.project}`      : "",
      ].filter(Boolean).join("\n\n");
      setStatus(`Exported in ${elapsed.toFixed(1)}s — ${r.zip ?? r.project ?? "output/"}`);
      // eslint-disable-next-line no-console
      console.info("[export]", r);
      const actions: MessageBoxAction[] = r.zip_url
        ? [{ label: "Download zip again", onClick: triggerDownload }]
        : [];
      showMessage(
        `Export complete (${elapsed.toFixed(1)}s)`,
        lines || "Files written.",
        "success",
        actions
      );
    } catch (err) {
      window.clearInterval(ticker);
      setProgress({ visible: false, value: 0, title: "", detail: "", elapsed: 0 });
      setStatus(`export failed: ${String(err)}`);
      showMessage("Export failed", String(err), "error");
    }
  }, [datasetId, showMessage]);

  const runCommand = useCallback(async (name: "undo" | "redo") => {
    if (!datasetId) return;
    try {
      const result = await command(datasetId, name);
      setStatus(`${name}: ${JSON.stringify(result)}`);
    } catch (err) {
      setStatus(`${name} failed: ${String(err)}`);
      return;
    }
    await loadPoints(); refreshStats();
  }, [datasetId, loadPoints, refreshStats]);

  const addLabel = useCallback(() => {
    if (!newLabelName.trim()) return;
    const newId = labels.length > 0 ? Math.max(...labels.map(l => l.id)) + 1 : 0;
    const colors = ["#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c", "#e67e22", "#3498db"];
    const color = colors[newId % colors.length];
    const updated = [...labels, { id: newId, name: newLabelName.trim(), color }];
    setLabels(updated);
    // First label in an empty project becomes the active one so it's usable at once.
    if (labels.length === 0) setActiveLabel(newId);
    setNewLabelName("");
    if (datasetId) {
      fetch(`/api/datasets/${datasetId}/labels/schema`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ labels: updated }),
      });
    }
  }, [newLabelName, labels, datasetId]);

  const removeLabel = useCallback(async (id: number) => {
    // Events still carrying this label would otherwise linger as an orphan
    // "#<id>" bar in the histogram. Relabel them to unlabelled (-1): in the
    // loaded chunks (instant histogram/recolour) and persisted to the backend.
    const per = new Map<number, number[]>();
    const ids: number[] = [];
    const entryById = new Map<number, ChunkManifestEntry>();
    if (manifest) for (const c of manifest.chunks) entryById.set(c.chunk_id, c);
    for (const [cid, p] of chunksMap) {
      const base = entryById.get(cid)?.event_start_index ?? 0;
      let arr: number[] | undefined;
      for (let i = 0; i < p.count; i++) {
        if (p.label[i] === id) {
          if (!arr) { arr = []; per.set(cid, arr); }
          arr.push(i);
          ids.push(base + i);
        }
      }
    }
    if (per.size > 0) {
      mutatePerChunk(per, -1);
      emitLabelPatch(per);
    }

    const updated = labels.filter(l => l.id !== id);
    setLabels(updated);
    if (activeLabel === id) setActiveLabel(updated[0]?.id ?? 0);
    if (datasetId) {
      fetch(`/api/datasets/${datasetId}/labels/schema`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ labels: updated }),
      });
      if (ids.length > 0) {
        try { await updateLabelsBinary(datasetId, Uint32Array.from(ids), -1, "clear"); } catch { /* keep local */ }
        refreshStats();
      }
    }
  }, [labels, activeLabel, datasetId, manifest, chunksMap, mutatePerChunk, emitLabelPatch, refreshStats]);

  const updateLabel = useCallback((id: number, changes: { name?: string; color?: string }) => {
    const updated = labels.map(l => l.id === id ? { ...l, ...changes } : l);
    setLabels(updated);
    if (datasetId) {
      fetch(`/api/datasets/${datasetId}/labels/schema`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ labels: updated }),
      });
    }
  }, [labels, datasetId]);

  // Playback animation loop
  useEffect(() => {
    if (!playing) return;
    const range = endUs - startUs;
    const baseUsPerSecond = range / 30;
    let raf: number;
    let lastTime = performance.now();
    const tick = () => {
      const now = performance.now();
      const dt = (now - lastTime) / 1000;
      lastTime = now;
      setCurrentFrameUs(prev => {
        const next = prev + baseUsPerSecond * playbackSpeed * dt;
        if (next + frameWindowUs > endUs) {
          setPlaying(false);
          return Math.max(startUs, endUs - frameWindowUs);
        }
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, playbackSpeed, frameWindowUs, startUs, endUs]);

  const togglePlayback = useCallback(() => setPlaying(p => !p), []);

  const stepForward = useCallback(() => {
    setCurrentFrameUs(prev => Math.min(prev + frameWindowUs, endUs - frameWindowUs));
  }, [frameWindowUs, endUs]);

  const stepBackward = useCallback(() => {
    setCurrentFrameUs(prev => Math.max(prev - frameWindowUs, startUs));
  }, [frameWindowUs, startUs]);

  const goToStart = useCallback(() => {
    setCurrentFrameUs(startUs);
    setPlaying(false);
  }, [startUs]);

  const goToEnd = useCallback(() => {
    setCurrentFrameUs(Math.max(startUs, endUs - frameWindowUs));
    setPlaying(false);
  }, [startUs, endUs, frameWindowUs]);

  // Select every loaded point (the whole point cloud). selectedIds holds global
  // event ids, so we union each loaded chunk's event_id (or the single payload).
  const selectAll = useCallback(() => {
    if (!datasetId) return;
    const ids = new Set<number>();
    for (const p of chunksMap.values()) {
      const ev = p.event_id;
      for (let i = 0; i < p.count; i++) ids.add(Number(ev[i]));
    }
    if (ids.size === 0 && payload) {
      for (let i = 0; i < payload.count; i++) ids.add(Number(payload.event_id[i]));
    }
    setSelectedIds(ids);
    setStatus(`Selected ${ids.size.toLocaleString()} events`);
  }, [datasetId, chunksMap, payload]);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    if (e.ctrlKey && e.key === "z") { e.preventDefault(); runCommand("undo"); }
    else if (e.ctrlKey && e.key === "y") { e.preventDefault(); runCommand("redo"); }
    else if (e.ctrlKey && e.key === "s") { e.preventDefault(); loadPoints(); }
    // Space toggles XY playback while the default pan tool is active.
    else if (e.key === " " && viewMode === "xy" && toolMode === "pan") { e.preventDefault(); togglePlayback(); }
    else if (e.key === "1" && labels.length > 0) setActiveLabel(labels[0]?.id ?? 0);
    else if (e.key === "2" && labels.length > 1) setActiveLabel(labels[1]?.id ?? 0);
    else if (e.key === "3" && labels.length > 2) setActiveLabel(labels[2]?.id ?? 0);
    else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); assign("clear"); }
    else if (e.key === "Enter" && selectedIds.size > 0) assign("assign");
    else if (e.key === "f") selectByRanges();
    else if (e.key === "e") clearByFilter();
    else if (e.key === "b") setToolMode("box");
    else if (e.key === "c" && viewMode !== "3d") setToolMode("circle"); // circle is XY-only
    else if (e.key === "l") setToolMode("lasso");
    else if (e.key === "w" && viewMode === "3d") setToolMode("wand");   // wand is 3D-only
    else if (e.key === "a" && e.ctrlKey) { e.preventDefault(); selectAll(); }       // Ctrl+A = select all
    else if (e.key === "Escape") {
      // Esc = deselect — but don't steal it from an open dialog (dialogs close on
      // Esc; their state is still true on this same event, so we bail out).
      if (settingsVisible || printVisible || renderVideoVisible || exportVisible ||
          importPrompt.visible || messageBox.visible || document.querySelector(".confirmOverlay")) return;
      setSelectedIds(new Set());
    }
  }, [runCommand, loadPoints, labels, assign, selectedIds, selectByRanges, clearByFilter, viewMode, toolMode, togglePlayback, selectAll, settingsVisible, printVisible, renderVideoVisible, exportVisible, importPrompt, messageBox]);

  // Importing a file no longer opens it straight away: it asks the user to name
  // the project first (the name becomes the cache id). The file waits in a ref
  // until confirmImport / cancelImport.
  const handleFileUpload = useCallback(async (file: File) => {
    pendingFileRef.current = file;
    let existing: string[] = [];
    try { existing = await listDatasetIds(); } catch { /* offline: no suggestions */ }
    const suggested = uniqueProjectName(deriveStem(file.name), existing);
    setImportPrompt({ visible: true, fileName: file.name, suggested, existingNames: existing });
  }, []);

  const confirmImport = useCallback(async (name: string) => {
    const file = pendingFileRef.current;
    pendingFileRef.current = null;
    setImportPrompt((p) => ({ ...p, visible: false }));
    if (!file) return;
    setStatus(`Uploading ${file.name}...`);
    try {
      const result = await uploadFile(file);
      setStatus("File uploaded, opening project...");
      await handleOpen(result.path, name);
      setRecordings(await listRecordings());
    } catch (error) {
      setStatus(`Upload failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, [handleOpen]);

  const cancelImport = useCallback(() => {
    pendingFileRef.current = null;
    setImportPrompt((p) => ({ ...p, visible: false }));
  }, []);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // Switching view can leave a now-hidden tool active (wand only exists in 3D,
  // circle only in XY) — fall back to pan so the active tool is always visible.
  useEffect(() => {
    if (viewMode === "xy" && (toolMode === "wand" || toolMode === "tscrub")) setToolMode("pan");
    else if (viewMode === "3d" && toolMode === "circle") setToolMode("pan");
  }, [viewMode, toolMode]);

  const timeRange = meta ? meta.t_max_us - meta.t_min_us : 1000000;
  const timeSliderMin = meta?.t_min_us ?? 0;
  const timeSliderMax = meta?.t_max_us ?? 1000000;

  return {
    recordings, datasetId, meta, labels, activeLabel, payload, viewMode,
    manifest, chunksMap, visibleChunkIds,
    polarity, sample, startUs, endUs, xMin, xMax, yMin, yMax,
    colorMode, pointSizeScale,
    unlabeledColor, setUnlabeledColor, polarityContrast, setPolarityContrast,
    settingsVisible, settingsSection, openSettings, closeSettings,
    selectedIds, status, progress, stats, newLabelName, pointInfo, tooltipRef,
    toolMode, brushRadius, selectThrough,
    xyRotation, rotateXyView, xyFlipY, toggleXyFlipY, xyFlipX, toggleXyFlipX,
    recents, refreshRecents, closeProject,
    importPrompt, confirmImport, cancelImport,
    messageBox, hideMessage, showMessage,
    renderVideoVisible, openRenderVideo, closeRenderVideo, startRenderVideo,
    exportVisible, openExport, closeExport, startExport,
    screenshotView, printVisible, openPrint, cancelPrint, confirmPrint,
    currentFrameUs, playing, playbackSpeed, frameWindowUs,
    histogram, visibleEventCount, timeRange, timeSliderMin, timeSliderMax, assignBusy, labelPatch,
    setActiveLabel, setViewMode, setPolarity, setSample, setColorMode, setPointSizeScale,
    setStartUs, setEndUs, setXMin, setXMax, setYMin, setYMax,
    filterPreview, setFilterPreview,
    setSelectedIds, setStatus, setProgress, setNewLabelName, setPointInfo,
    setToolMode, setBrushRadius, setSelectThrough,
    setCurrentFrameUs, setPlaying, setPlaybackSpeed, setFrameWindowUs,
    togglePlayback, stepForward, stepBackward, goToStart, goToEnd,
    handleOpen, handleFileUpload, loadPoints, selectByRanges, selectAll, assign, assignSingle,
    assignByFilter, clearByFilter, clearAllLabels, runCommand, addLabel, removeLabel, updateLabel,
    track, trackCreate, trackPropagate, trackUseSelectionAsSeed, trackPreviewCandidate, trackConfirmCandidate, trackCancel, trackSetDirection,
    addToSelection, removeFromSelection,
    selectComponent,
    wandVoxelXy, setWandVoxelXy,
    wandVoxelTms, setWandVoxelTms,
    wandMinPts, setWandMinPts,
    wandHasSeed, applyWandParams,
  };
}
