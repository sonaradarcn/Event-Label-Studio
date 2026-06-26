import { ChevronDown, ChevronRight } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { DatasetMeta, LabelClass, PointPayload } from "../api/types";
import { useI18n } from "../i18n/I18nContext";

type Props = {
  payload: PointPayload | null;
  visibleEventCount: number;
  selectedIds: Set<number>;
  histogram: Map<number, number>;
  labels: LabelClass[];
  meta: DatasetMeta | null;
  startUs: number;
  endUs: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  polarity: string;
  sample: number;
  timeSliderMin: number;
  timeSliderMax: number;
  timeRange: number;
  setStartUs: (v: number) => void;
  setEndUs: (v: number) => void;
  setXMin: (v: number) => void;
  setXMax: (v: number) => void;
  setYMin: (v: number) => void;
  setYMax: (v: number) => void;
  setPolarity: (v: string) => void;
  setSample: (v: number) => void;
  setFilterPreview: (p: { xMin: number; xMax: number; yMin: number; yMax: number; startUs: number; endUs: number } | null) => void;
};

function Section({ title, defaultOpen, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen ?? true);
  return (
    <div style={{ marginBottom: 4 }}>
      <button className="sectionToggle" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {title}
      </button>
      {open && <div className="sectionContent">{children}</div>}
    </div>
  );
}

// Dual-thumb slider. Native <input type=range> only has one thumb, so the
// track/fill/thumbs are plain divs driven by pointer events. A thumb whose
// checkbox is off is locked: it can't be grabbed, and clicks near it fall
// through to the other (enabled) thumb.
function DualSlider({ min, max, step, lo, hi, loEnabled, hiEnabled, disabled, onChange }: {
  min: number; max: number; step: number;
  lo: number; hi: number;
  loEnabled: boolean; hiEnabled: boolean;
  disabled?: boolean;
  onChange: (lo: number, hi: number) => void;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<"lo" | "hi" | null>(null);
  const span = Math.max(1, max - min);
  const pct = (v: number) => ((v - min) / span) * 100;

  const valueAt = (clientX: number) => {
    const rect = trackRef.current!.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    const snapped = min + Math.round((ratio * span) / step) * step;
    return Math.min(max, Math.max(min, snapped));
  };

  const moveTo = (v: number) => {
    if (dragRef.current === "lo") onChange(Math.min(v, hi), hi);
    else if (dragRef.current === "hi") onChange(lo, Math.max(v, lo));
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!trackRef.current || (!loEnabled && !hiEnabled)) return;
    e.preventDefault(); // stop native text-selection / drag (the "no-drop" cursor that froze the drag)
    const v = valueAt(e.clientX);
    let target: "lo" | "hi";
    if (loEnabled && hiEnabled) {
      const dLo = Math.abs(v - lo);
      const dHi = Math.abs(v - hi);
      // Tie (thumbs stacked): pick by click side so the pair can be pulled apart.
      target = dLo < dHi ? "lo" : dHi < dLo ? "hi" : v < lo ? "lo" : "hi";
    } else {
      target = loEnabled ? "lo" : "hi";
    }
    dragRef.current = target;
    e.currentTarget.setPointerCapture(e.pointerId);
    moveTo(v);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (dragRef.current) moveTo(valueAt(e.clientX));
  };

  const endDrag = () => { dragRef.current = null; };

  return (
    <div
      className={disabled ? "dualSlider disabled" : "dualSlider"}
      ref={trackRef}
      draggable={false}
      onDragStart={(e) => e.preventDefault()}
      onPointerDown={disabled ? undefined : handlePointerDown}
      onPointerMove={disabled ? undefined : handlePointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      <div className="dualSliderTrack" />
      <div className="dualSliderFill" style={{ left: `${pct(lo)}%`, width: `${pct(hi) - pct(lo)}%` }} />
      <div className={loEnabled ? "dualSliderThumb" : "dualSliderThumb locked"} style={{ left: `${pct(lo)}%` }} />
      <div className={hiEnabled ? "dualSliderThumb" : "dualSliderThumb locked"} style={{ left: `${pct(hi)}%` }} />
    </div>
  );
}

// One labelled range row: title + start/end checkboxes, dual slider, and a
// pair of number inputs for precise entry. Checkboxes lock the corresponding
// end so a fixed bound can't be nudged accidentally while dragging the other.
function RangeRow({ title, unit, min, max, step, lo, hi, disabled, onChange }: {
  title: string; unit?: string;
  min: number; max: number; step: number;
  lo: number; hi: number;
  disabled?: boolean;
  onChange: (lo: number, hi: number) => void;
}) {
  const clamp = (v: number) => Math.min(max, Math.max(min, v));
  const onLoInput = (raw: string) => {
    const v = Number(raw);
    if (!Number.isFinite(v)) return;
    onChange(Math.min(clamp(v), hi), hi);
  };
  const onHiInput = (raw: string) => {
    const v = Number(raw);
    if (!Number.isFinite(v)) return;
    onChange(lo, Math.max(clamp(v), lo));
  };

  return (
    <div className="rangeRow">
      <div className="rangeRowHead">
        <span className="rangeRowTitle">{title}{unit ? ` (${unit})` : ""}</span>
      </div>
      <DualSlider min={min} max={max} step={step} lo={lo} hi={hi}
        loEnabled={!disabled} hiEnabled={!disabled} disabled={disabled} onChange={onChange} />
      <div className="rangeRowInputs">
        <input type="number" min={min} max={max} step={step} value={lo}
          disabled={disabled} onChange={(e) => onLoInput(e.target.value)} />
        <span className="rangeDash">–</span>
        <input type="number" min={min} max={max} step={step} value={hi}
          disabled={disabled} onChange={(e) => onHiInput(e.target.value)} />
      </div>
    </div>
  );
}

function HistogramChart({ histogram, labels }: { histogram: Map<number, number>; labels: LabelClass[] }) {
  const { t } = useI18n();
  const byId = new Map(labels.map((l) => [l.id, l]));
  // Fold orphan label ids (present on events but no longer in the schema, e.g.
  // after a label was deleted) into the unlabelled (-1) bucket, so they don't
  // show up as bogus "#id" bars.
  const merged = new Map<number, number>();
  for (const [id, count] of histogram) {
    const key = id === -1 || byId.has(id) ? id : -1;
    merged.set(key, (merged.get(key) ?? 0) + count);
  }
  const entries = [...merged.entries()].sort((a, b) => a[0] - b[0]);
  if (entries.length === 0) {
    return <p style={{ color: "var(--text-muted)" }}>{t("panel.inspector.noData")}</p>;
  }
  const maxCount = Math.max(...entries.map(([, c]) => c));
  return (
    <div className="histChart">
      {entries.map(([id, count]) => {
        const cls = byId.get(id);
        const name = id === -1 ? t("panel.inspector.unlabelled") : cls?.name ?? `#${id}`;
        const color = id === -1 ? "var(--text-muted)" : cls?.color ?? "#888888";
        return (
          <div className="histRow" key={id} title={`${name}: ${count.toLocaleString()}`}>
            <span className="histName">{name}</span>
            <div className="histBarTrack">
              <div className="histBar" style={{ width: `${(count / maxCount) * 100}%`, background: color }} />
            </div>
            <span className="histCount">{count.toLocaleString()}</span>
          </div>
        );
      })}
    </div>
  );
}

export function RightPanel({
  payload: _payload, visibleEventCount, selectedIds, histogram, labels, meta,
  startUs, endUs, xMin, xMax, yMin, yMax, polarity, sample,
  timeSliderMin, timeSliderMax, timeRange,
  setStartUs, setEndUs, setXMin, setXMax, setYMin, setYMax, setPolarity, setSample, setFilterPreview,
}: Props) {
  const { t } = useI18n();
  const step = Math.max(1, Math.floor(timeRange / 1000));
  const maxX = (meta?.width ?? 1280) - 1;
  const maxY = (meta?.height ?? 720) - 1;
  // Foolproofing: nothing to filter until a dataset is open.
  const noData = !meta;

  // Draft (staged) filter values. Edits land here; nothing reloads until the
  // Load button commits them to the app state in one shot. The effect below
  // re-syncs the draft whenever the committed values change from outside
  // (dataset open resets them) — after our own commit it's a no-op.
  const [draft, setDraft] = useState({ startUs, endUs, xMin, xMax, yMin, yMax, polarity, sample });
  useEffect(() => {
    setDraft({ startUs, endUs, xMin, xMax, yMin, yMax, polarity, sample });
  }, [startUs, endUs, xMin, xMax, yMin, yMax, polarity, sample]);

  const dirty =
    draft.startUs !== startUs || draft.endUs !== endUs ||
    draft.xMin !== xMin || draft.xMax !== xMax ||
    draft.yMin !== yMin || draft.yMax !== yMax ||
    draft.polarity !== polarity || draft.sample !== sample;

  // While the spatial/time draft is uncommitted (before Load), show it as a
  // yellow preview box in the 3D view so the user sees the region they're about
  // to keep. Cleared once committed (Load/Reset make draft === committed).
  const spatialDirty =
    draft.startUs !== startUs || draft.endUs !== endUs ||
    draft.xMin !== xMin || draft.xMax !== xMax ||
    draft.yMin !== yMin || draft.yMax !== yMax;
  useEffect(() => {
    setFilterPreview(spatialDirty
      ? { xMin: draft.xMin, xMax: draft.xMax, yMin: draft.yMin, yMax: draft.yMax, startUs: draft.startUs, endUs: draft.endUs }
      : null);
  }, [spatialDirty, draft.xMin, draft.xMax, draft.yMin, draft.yMax, draft.startUs, draft.endUs, setFilterPreview]);
  useEffect(() => () => setFilterPreview(null), [setFilterPreview]); // clear on unmount

  const apply = () => {
    setStartUs(draft.startUs);
    setEndUs(draft.endUs);
    setXMin(draft.xMin);
    setXMax(draft.xMax);
    setYMin(draft.yMin);
    setYMax(draft.yMax);
    setPolarity(draft.polarity);
    setSample(draft.sample);
  };

  // "Full range" = no filtering. Reset stages these and commits them in one shot
  // so the whole cloud reappears immediately (no second Load click needed).
  const full = { startUs: timeSliderMin, endUs: timeSliderMax, xMin: 0, xMax: maxX, yMin: 0, yMax: maxY, polarity: "all", sample: 0 };
  const isFiltered =
    startUs !== full.startUs || endUs !== full.endUs ||
    xMin !== full.xMin || xMax !== full.xMax ||
    yMin !== full.yMin || yMax !== full.yMax ||
    polarity !== full.polarity || sample !== full.sample;
  const reset = () => {
    setDraft(full);
    setStartUs(full.startUs); setEndUs(full.endUs);
    setXMin(full.xMin); setXMax(full.xMax);
    setYMin(full.yMin); setYMax(full.yMax);
    setPolarity(full.polarity); setSample(full.sample);
  };

  return (
    <aside className="panel">
      <Section title={t("panel.inspector")}>
        <p>{t("panel.inspector.visible")}: {visibleEventCount.toLocaleString()}</p>
        <p>{t("panel.inspector.selected")}: {selectedIds.size.toLocaleString()}</p>
        <h3>{t("panel.inspector.labelHistogram")}</h3>
        <HistogramChart histogram={histogram} labels={labels} />
      </Section>

      <Section title={t("panel.inspector.rangesFilters")}>
        <h3>{t("panel.inspector.timeRange")}</h3>
        <RangeRow title={t("panel.inspector.time")} unit="us"
          min={timeSliderMin} max={timeSliderMax} step={step}
          lo={draft.startUs} hi={draft.endUs} disabled={noData}
          onChange={(lo, hi) => setDraft((d) => ({ ...d, startUs: lo, endUs: hi }))} />

        <h3>{t("panel.inspector.spatialRange")}</h3>
        <RangeRow title="X" min={0} max={maxX} step={1}
          lo={draft.xMin} hi={draft.xMax} disabled={noData}
          onChange={(lo, hi) => setDraft((d) => ({ ...d, xMin: lo, xMax: hi }))} />
        <RangeRow title="Y" min={0} max={maxY} step={1}
          lo={draft.yMin} hi={draft.yMax} disabled={noData}
          onChange={(lo, hi) => setDraft((d) => ({ ...d, yMin: lo, yMax: hi }))} />

        <h3>{t("panel.inspector.filters")}</h3>
        <label>{t("panel.inspector.polarity")}
          <select value={draft.polarity} disabled={noData} onChange={(e) => setDraft((d) => ({ ...d, polarity: e.target.value }))}>
            <option value="all">{t("panel.inspector.polarity.all")}</option>
            <option value="on">{t("panel.inspector.polarity.on")}</option>
            <option value="off">{t("panel.inspector.polarity.off")}</option>
          </select>
        </label>
        <div className="filterButtons">
          <button className="resetBtn" onClick={reset} disabled={noData || (!isFiltered && !dirty)}>
            {t("panel.inspector.reset")}
          </button>
          <button className={dirty ? "applyBtn dirty" : "applyBtn"} onClick={apply} disabled={noData || !dirty}>
            {t("panel.inspector.load")}
          </button>
        </div>
        {dirty && <p className="pendingHint">{t("panel.inspector.pending")}</p>}
      </Section>
    </aside>
  );
}
