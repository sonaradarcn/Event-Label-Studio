import { useEffect, useRef, useState } from "react";
import { Crosshair, RotateCw, FlipHorizontal2, FlipVertical2 } from "lucide-react";
import type { ChunkManifest, LabelClass, DatasetMeta, PointPayload } from "../api/types";
import { useI18n } from "../i18n/I18nContext";
import { useTheme } from "../theme/ThemeContext";
import { PointCloudView } from "../pointcloud/PointCloudView";
import type { PointInfo, PointCloudHandle, ToolMode, SelectionOp, FilterPreview } from "../pointcloud/PointCloudView";
import type { ProgressState, TrackUiState } from "../hooks/useAppState";
import { PlaybackBar } from "./PlaybackBar";
import { TrackPanel } from "./TrackPanel";

type Props = {
  payload: PointPayload | null;
  manifest: ChunkManifest | null;
  chunksMap: Map<number, PointPayload>;
  visibleChunkIds: number[];
  viewMode: "3d" | "xy";
  selectedIds: Set<number>;
  labelPatch: { seq: number; perChunk: Map<number, number[]> } | null;
  colorMode: "polarity" | "time";
  pointSizeScale: number;
  unlabeledColor: string;
  polarityContrast: number;
  // Ranges & Filters (committed via the Load button) — cull the rendered cloud.
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  polarity: string;
  filterPreview: FilterPreview | null;
  gpuPreference: "high-performance" | "low-power" | "default";
  labels: LabelClass[];
  meta?: DatasetMeta;
  datasetId: string;
  progress: ProgressState;
  assignSingle: (eventId: number, labelId: number, operation: "assign" | "clear") => void;
  toolMode: ToolMode;
  xyRotation: number;
  xyFlipY: boolean;
  xyFlipX: boolean;
  brushRadius: number;
  selectThrough: boolean;
  addToSelection: (ids: number[]) => void;
  removeFromSelection: (ids: number[]) => void;
  setSelectedIds: (ids: Set<number>) => void;
  setBrushRadius: (r: number) => void;
  selectComponent: (seedEventId: number, op: SelectionOp) => void;
  // Playback
  currentFrameUs: number;
  startUs: number;
  endUs: number;
  playing: boolean;
  playbackSpeed: number;
  frameWindowUs: number;
  togglePlayback: () => void;
  stepForward: () => void;
  stepBackward: () => void;
  goToStart: () => void;
  goToEnd: () => void;
  setCurrentFrameUs: (us: number) => void;
  setPlaybackSpeed: (speed: number) => void;
  setFrameWindowUs: (us: number) => void;
  // Propagation tracking
  track: TrackUiState | null;
  trackCreate: () => void;
  trackPropagate: () => void;
  trackUseSelectionAsSeed: () => void;
  trackPreviewCandidate: (index: number) => void;
  trackConfirmCandidate: () => void;
  trackCancel: () => void;
  trackSetDirection: (d: "forward" | "backward") => void;
};

export function MainViewport({
  payload, manifest, chunksMap, visibleChunkIds, viewMode, selectedIds, labelPatch, colorMode, pointSizeScale, unlabeledColor, polarityContrast, xMin, xMax, yMin, yMax, polarity, filterPreview, gpuPreference, labels, meta, datasetId, progress, assignSingle,
  toolMode, brushRadius, selectThrough, xyRotation, xyFlipY, xyFlipX, addToSelection, removeFromSelection, setSelectedIds, setBrushRadius,
  selectComponent,
  currentFrameUs, startUs, endUs, playing, playbackSpeed, frameWindowUs,
  togglePlayback, stepForward, stepBackward, goToStart, goToEnd,
  setCurrentFrameUs, setPlaybackSpeed, setFrameWindowUs,
  track, trackCreate, trackPropagate, trackUseSelectionAsSeed, trackPreviewCandidate, trackConfirmCandidate, trackCancel, trackSetDirection,
}: Props) {
  const [pointInfo, setPointInfo] = useState<PointInfo | null>(null);
  const [editLabel, setEditLabel] = useState<number>(0);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const tooltipClicked = useRef(false);
  const cloudHandleRef = useRef<PointCloudHandle | null>(null);
  const { t } = useI18n();
  const { theme } = useTheme();

  // ── T-axis time scrubber preview ──────────────────────────────────────────
  // The preview XY frame is rendered CLIENT-SIDE straight from the loaded chunk
  // payloads (which, with the default no-cap load, hold the whole recording).
  // That makes it instant and fully follow the slider — no per-frame backend
  // round-trip (which lagged behind a fast drag).
  const [scrubTUs, setScrubTUs] = useState<number | null>(null);
  const scrubCanvasRef = useRef<HTMLCanvasElement>(null);
  // Preview-only orientation (independent of the main XY view).
  const [previewRot, setPreviewRot] = useState(0); // 0 / 90 / 180 / 270
  const [previewFlipX, setPreviewFlipX] = useState(false);
  const [previewFlipY, setPreviewFlipY] = useState(false);

  function handleTimeScrub(info: { tUs: number; sx: number; sy: number } | null) {
    setScrubTUs(info ? info.tUs : null);
  }

  // (Re)draw the preview frame whenever the scrubbed time (or its orientation)
  // changes — synchronous, so it tracks the slider 1:1.
  useEffect(() => {
    if (scrubTUs == null) return;
    const canvas = scrubCanvasRef.current;
    if (!canvas || !meta) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = meta.width, H = meta.height;
    // 90/270 swap the output aspect.
    const swap = previewRot === 90 || previewRot === 270;
    const outW = swap ? H : W, outH = swap ? W : H;
    const cw = 200, ch = Math.max(1, Math.round(200 * outH / outW));
    canvas.width = cw; canvas.height = ch;
    const s = cw / outW;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cw, ch);
    const win = frameWindowUs || 33333;
    const t0 = scrubTUs, t1 = scrubTUs + win;
    // Same polarity scheme as the main views: base = label colour (or the
    // configurable unlabelled colour), modulated ON brighter / OFF darker by
    // the contrast. Precompute the two strings per base so the hot loop is fast.
    const k = polarityContrast;
    const hexRgb = (hex: string): [number, number, number] => {
      const m = /^#?([0-9a-fA-F]{6})$/.exec(hex.trim());
      if (!m) return [136, 136, 136];
      const n = parseInt(m[1], 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    };
    const modStr = (base: [number, number, number], on: boolean): string => {
      const f = (c: number) => on ? Math.round(c + (255 - c) * k) : Math.round(c * (1 - k));
      return `rgb(${f(base[0])},${f(base[1])},${f(base[2])})`;
    };
    const unlBase = hexRgb(unlabeledColor);
    const unlOn = modStr(unlBase, true), unlOff = modStr(unlBase, false);
    const labelPair = new Map<number, { on: string; off: string }>();
    for (const l of labels) { const b = hexRgb(l.color); labelPair.set(l.id, { on: modStr(b, true), off: modStr(b, false) }); }
    // lower_bound on an ascending Float32 t array.
    const lb = (a: Float32Array, v: number) => {
      let lo = 0, hi = a.length;
      while (lo < hi) { const m = (lo + hi) >>> 1; if (a[m] < v) lo = m + 1; else hi = m; }
      return lo;
    };
    // pixel (px,py) → output canvas coords, applying flip then rotation.
    const tf = (px: number, py: number): [number, number] => {
      if (previewFlipX) px = W - px;
      if (previewFlipY) py = H - py;
      let X: number, Y: number;
      if (previewRot === 90) { X = H - py; Y = px; }
      else if (previewRot === 180) { X = W - px; Y = H - py; }
      else if (previewRot === 270) { X = py; Y = W - px; }
      else { X = px; Y = py; }
      return [X * s, Y * s];
    };
    for (const p of chunksMap.values()) {
      const t = p.t_us, n = p.count;
      if (n === 0 || t[n - 1] < t0 || t[0] > t1) continue;
      const lo = lb(t, t0), hi = lb(t, t1);
      const pos = p.positions, pol = p.polarity, lab = p.label;
      for (let i = lo; i < hi; i++) {
        const px = pos[i * 3] + (W - 1) / 2;
        const py = (H - 1) / 2 - pos[i * 3 + 1];
        // Sync with Ranges & Filters: cull events outside the committed
        // spatial / polarity / time filter so the preview matches the main view.
        if (px < xMin || px > xMax || py < yMin || py > yMax) continue;
        const tt = t[i];
        if (tt < startUs || tt > endUs) continue;
        const on = !!pol[i];
        if (polarity === "on" && !on) continue;
        if (polarity === "off" && on) continue;
        const [cx, cy] = tf(px, py);
        const lp = labelPair.get(lab[i]);
        ctx.fillStyle = lp ? (on ? lp.on : lp.off) : (on ? unlOn : unlOff);
        ctx.fillRect(cx, cy, 1.5, 1.5);
      }
    }
    // Red corner marker at the displayed top-left — matches the red triangle in
    // the 3D scrub plane so its meaning (= the frame's top-left) is obvious.
    const td = Math.max(10, Math.round(cw * 0.11));
    ctx.fillStyle = "#ff3b30";
    ctx.beginPath();
    ctx.moveTo(0, 0); ctx.lineTo(td, 0); ctx.lineTo(0, td); ctx.closePath();
    ctx.fill();
  }, [scrubTUs, chunksMap, meta, labels, frameWindowUs, previewRot, previewFlipX, previewFlipY, unlabeledColor, polarityContrast, xMin, xMax, yMin, yMax, polarity, startUs, endUs]);

  function handlePointClick(info: PointInfo | null) {
    if (tooltipClicked.current) { tooltipClicked.current = false; return; }
    setPointInfo(info);
    if (info) {
      setEditLabel(info.label >= 0 ? info.label : labels[0]?.id ?? 0);
    }
  }

  function focusOnPoint() {
    if (!pointInfo || !cloudHandleRef.current) return;
    cloudHandleRef.current.focusOnPoint(pointInfo.index);
  }

  // PointCloudView now always emits event_ids (multi-Points + legacy paths
  // both pre-resolve indices → event_ids inside the picker), so we no longer
  // dereference the `payload.event_id` BigInt array here. This also lets
  // selection work in multi-Points mode, where there is no combined payload.
  function handleSelection(eventIds: number[], op: SelectionOp) {
    if (eventIds.length === 0) return;
    if (op === "add") addToSelection(eventIds);
    else if (op === "subtract") removeFromSelection(eventIds);
    else setSelectedIds(new Set(eventIds));
  }

  return (
    <section className="mainview">
      <PointCloudView
        payload={payload}
        manifest={manifest}
        chunksMap={chunksMap}
        visibleChunkIds={visibleChunkIds}
        viewMode={viewMode}
        selectedIds={selectedIds}
        labelPatch={labelPatch}
        colorMode={colorMode}
        pointSizeScale={pointSizeScale}
        unlabeledColor={unlabeledColor}
        polarityContrast={polarityContrast}
        filterXMin={xMin}
        filterXMax={xMax}
        filterYMin={yMin}
        filterYMax={yMax}
        filterPolarity={polarity}
        filterPreview={filterPreview}
        windowStartUs={startUs}
        windowEndUs={endUs}
        gpuPreference={gpuPreference} theme={theme} labels={labels}
        onPointClick={handlePointClick} tooltipRef={tooltipRef}
        handleRef={cloudHandleRef}
        toolMode={toolMode} brushRadius={brushRadius} selectThrough={selectThrough}
        xyRotation={xyRotation} xyFlipY={xyFlipY} xyFlipX={xyFlipX}
        onSelection={handleSelection}
        onWandSeed={selectComponent}
        onBrushRadiusChange={setBrushRadius}
        onTimeScrub={handleTimeScrub}
        previewRot={previewRot}
        previewFlipX={previewFlipX}
        previewFlipY={previewFlipY}
        currentFrameUs={currentFrameUs}
        frameWindowUs={frameWindowUs}
        meta={meta}
      />

      {scrubTUs != null && meta && (
        <div className="scrubPreview" onMouseDown={(e) => e.stopPropagation()}>
          <canvas ref={scrubCanvasRef} />
          <div className="scrubPreviewFoot">
            <span className="scrubPreviewLabel">t = {(scrubTUs / 1e6).toFixed(3)} s</span>
            <span className="scrubPreviewBtns">
              <button onClick={() => setPreviewRot((r) => (r + 90) % 360)} title={t("toolbar.rotate")}><RotateCw size={12} /></button>
              <button className={previewFlipX ? "active" : ""} onClick={() => setPreviewFlipX((v) => !v)} title={t("toolbar.flipX")}><FlipHorizontal2 size={12} /></button>
              <button className={previewFlipY ? "active" : ""} onClick={() => setPreviewFlipY((v) => !v)} title={t("toolbar.flipY")}><FlipVertical2 size={12} /></button>
            </span>
          </div>
        </div>
      )}

      {pointInfo && (
        <div ref={tooltipRef} className="pointTooltip" style={{ left: pointInfo.screenX + 14, top: pointInfo.screenY - 10 }}
          onMouseDown={(e) => { e.stopPropagation(); tooltipClicked.current = true; }}
          onMouseUp={(e) => e.stopPropagation()}>
          <div className="pointTooltipHeader">{t("tooltip.event")} #{pointInfo.eventId}</div>
          <div className="pointTooltipRow"><span>X</span><span>{pointInfo.x.toFixed(1)}</span></div>
          <div className="pointTooltipRow"><span>Y</span><span>{pointInfo.y.toFixed(1)}</span></div>
          <div className="pointTooltipRow"><span>T</span><span>{pointInfo.tUs.toLocaleString()} us</span></div>
          <div className="pointTooltipRow">
            <span>{t("tooltip.polarity")}</span>
            <span>{pointInfo.polarity ? t("tooltip.polarityOn") : t("tooltip.polarityOff")}</span>
          </div>
          <div className="pointTooltipRow">
            <span>{t("tooltip.label")}</span>
            <span>{labels.find(l => l.id === pointInfo.label)?.name ?? (pointInfo.label === -1 ? t("tooltip.unlabelled") : String(pointInfo.label))}</span>
          </div>

          <div className="pointTooltipLabel">
            <select value={editLabel} onChange={(e) => setEditLabel(Number(e.target.value))}>
              {labels.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
            </select>
            <div className="pointTooltipActions">
              <button onClick={() => { assignSingle(pointInfo.eventId, editLabel, "assign"); }}>{t("tooltip.apply")}</button>
              <button className="danger" onClick={() => { assignSingle(pointInfo.eventId, -1, "clear"); }}>{t("tooltip.clearLabel")}</button>
            </div>
          </div>

          <div className="pointTooltipActions" style={{ marginTop: 4 }}>
            <button onClick={focusOnPoint} title="Focus on this point"><Crosshair size={12} /> {t("tooltip.focus")}</button>
            <button onClick={() => setPointInfo(null)}>{t("tooltip.close")}</button>
          </div>
        </div>
      )}

      {!payload && !manifest && !progress.visible && (
        <div className="emptyState">
          <strong>{t("empty.title")}</strong>
          <span>{t("empty.hint")}</span>
        </div>
      )}
      {progress.visible && (
        <div className="progressOverlay">
          <div className="progressBox">
            <div className="progressHeader">
              <strong>{progress.title}</strong>
              {!progress.indeterminate && <span>{Math.round(progress.value * 100)}%</span>}
            </div>
            {progress.indeterminate
              ? <div className="progressTrack indeterminate"><div /></div>
              : <div className="progressTrack"><div style={{ width: `${Math.round(progress.value * 100)}%` }} /></div>}
            <p>{progress.detail}</p>
            {!progress.indeterminate && <small>{progress.elapsed.toFixed(1)}s {t("progress.elapsed")}</small>}
          </div>
        </div>
      )}

      {viewMode === "xy" && (
        <PlaybackBar
          currentFrameUs={currentFrameUs} startUs={startUs} endUs={endUs}
          playing={playing} playbackSpeed={playbackSpeed} frameWindowUs={frameWindowUs}
          onTogglePlayback={togglePlayback} onStepForward={stepForward}
          onStepBackward={stepBackward} onGoToStart={goToStart} onGoToEnd={goToEnd}
          onSeek={setCurrentFrameUs} onSetSpeed={setPlaybackSpeed}
          onSetFrameWindow={setFrameWindowUs}
        />
      )}

      {datasetId && viewMode === "xy" && (
        <TrackPanel
          track={track}
          selectedCount={selectedIds.size}
          trackCreate={trackCreate}
          trackPropagate={trackPropagate}
          trackUseSelectionAsSeed={trackUseSelectionAsSeed}
          trackPreviewCandidate={trackPreviewCandidate}
          trackConfirmCandidate={trackConfirmCandidate}
          trackCancel={trackCancel}
          trackSetDirection={trackSetDirection}
          currentFrameUs={currentFrameUs}
          onSeek={setCurrentFrameUs}
        />
      )}
    </section>
  );
}
