import { useEffect, useState } from "react";
import { renderVideoPreview, type RenderVideoParams } from "../api/client";
import type { DatasetMeta } from "../api/types";

type Props = {
  visible: boolean;
  datasetId: string;
  meta: DatasetMeta | null;
  defaultStartUs: number;
  defaultEndUs: number;
  unlabeledColor: string;
  polarityContrast: number;
  onCancel: () => void;
  onSubmit: (params: RenderVideoParams) => void;
};

export function RenderVideoModal({ visible, datasetId, meta, defaultStartUs, defaultEndUs, unlabeledColor, polarityContrast, onCancel, onSubmit }: Props) {
  const [startUs, setStartUs] = useState(defaultStartUs);
  const [endUs, setEndUs] = useState(defaultEndUs);
  const [fps, setFps] = useState(30);
  const [frameWindowMs, setFrameWindowMs] = useState(33);
  const [colorMode, setColorMode] = useState<RenderVideoParams["color_mode"]>("label_polarity");
  const [background, setBackground] = useState<"black" | "white">("black");
  const [format, setFormat] = useState<"mp4" | "webm">("mp4");
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewErr, setPreviewErr] = useState<string | null>(null);

  // Reset defaults when dialog (re)opens.
  useEffect(() => {
    if (!visible) return;
    setStartUs(defaultStartUs);
    setEndUs(defaultEndUs);
    setPreviewSrc(null);
    setPreviewErr(null);
  }, [visible, defaultStartUs, defaultEndUs]);

  // Esc closes.
  useEffect(() => {
    if (!visible) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onCancel(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible, onCancel]);

  // Free preview blob URL on close / replace.
  useEffect(() => () => {
    if (previewSrc) URL.revokeObjectURL(previewSrc);
  }, [previewSrc]);

  if (!visible) return null;

  const durationS = Math.max(0, (endUs - startUs) / 1_000_000);
  const nFrames = Math.max(1, Math.round((endUs - startUs) * fps / 1_000_000));
  const frameWindowUs = frameWindowMs * 1000;
  const sensorW = meta?.width ?? 1280;
  const sensorH = meta?.height ?? 720;

  const handlePreview = async () => {
    if (!datasetId) return;
    setPreviewBusy(true);
    setPreviewErr(null);
    try {
      const blob = await renderVideoPreview(datasetId, {
        t_us: startUs,
        frame_window_us: frameWindowUs,
        color_mode: colorMode,
        background,
        unlabeled_color: unlabeledColor,
        polarity_contrast: polarityContrast,
      });
      const url = URL.createObjectURL(blob);
      setPreviewSrc((old) => { if (old) URL.revokeObjectURL(old); return url; });
    } catch (e) {
      setPreviewErr(String(e));
    } finally {
      setPreviewBusy(false);
    }
  };

  const handleSubmit = () => {
    onSubmit({
      start_us: startUs,
      end_us: endUs,
      fps,
      frame_window_us: frameWindowUs,
      color_mode: colorMode,
      background,
      format,
    });
  };

  return (
    <div className="msgBoxOverlay" onMouseDown={onCancel}>
      <div className="msgBox renderVideoModal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="msgBoxHeader">
          <span className="msgBoxIcon" style={{ background: "var(--accent)" }}>▶</span>
          <span className="msgBoxTitle">Render Video</span>
        </div>

        <div className="renderVideoBody">
          <div className="rvField">
            <label>Time range (us)</label>
            <div className="rvTimeRow">
              <input type="number" value={startUs} onChange={(e) => setStartUs(Number(e.target.value))} />
              <span style={{ color: "var(--text-muted)" }}>–</span>
              <input type="number" value={endUs} onChange={(e) => setEndUs(Number(e.target.value))} />
              <span style={{ color: "var(--text-muted)", fontSize: 11 }}>{durationS.toFixed(2)}s</span>
            </div>
          </div>

          <div className="rvFieldRow">
            <div className="rvField">
              <label>Frame rate</label>
              <input type="number" min={1} max={120} value={fps} onChange={(e) => setFps(Math.max(1, Math.min(120, Number(e.target.value) || 30)))} />
              <span className="rvHint">fps</span>
            </div>
            <div className="rvField">
              <label>Frame window</label>
              <input type="number" min={1} max={1000} value={frameWindowMs} onChange={(e) => setFrameWindowMs(Math.max(1, Math.min(1000, Number(e.target.value) || 33)))} />
              <span className="rvHint">ms / frame</span>
            </div>
            <div className="rvField">
              <label>Resolution</label>
              <input type="text" value={`${sensorW} × ${sensorH}`} disabled />
            </div>
          </div>

          <div className="rvField">
            <label>Color mode</label>
            <div className="rvRadioRow">
              <label><input type="radio" checked={colorMode === "label_polarity"} onChange={() => setColorMode("label_polarity")} /> Label (fallback polarity)</label>
              <label><input type="radio" checked={colorMode === "polarity_only"} onChange={() => setColorMode("polarity_only")} /> Polarity only</label>
              <label><input type="radio" checked={colorMode === "label_only"} onChange={() => setColorMode("label_only")} /> Label only</label>
            </div>
          </div>

          <div className="rvFieldRow">
            <div className="rvField">
              <label>Background</label>
              <div className="rvRadioRow">
                <label><input type="radio" checked={background === "black"} onChange={() => setBackground("black")} /> Black</label>
                <label><input type="radio" checked={background === "white"} onChange={() => setBackground("white")} /> White</label>
              </div>
            </div>
            <div className="rvField">
              <label>Format</label>
              <div className="rvRadioRow">
                <label><input type="radio" checked={format === "mp4"} onChange={() => setFormat("mp4")} /> MP4</label>
                <label><input type="radio" checked={format === "webm"} onChange={() => setFormat("webm")} /> WebM</label>
              </div>
            </div>
          </div>

          <div className="rvSummary">
            <span><b>{nFrames}</b> frames</span>
            <span><b>{durationS.toFixed(2)}s</b> duration</span>
            <span>{format.toUpperCase()} · {sensorW}×{sensorH} · {fps} fps</span>
          </div>

          <div className="rvPreviewBox">
            {previewSrc ? <img src={previewSrc} alt="preview" /> : (
              <div className="rvPreviewEmpty">{previewBusy ? "Rendering preview…" : (previewErr ?? "Click Preview to render the first frame")}</div>
            )}
          </div>
        </div>

        <div className="msgBoxActions">
          <button className="msgBoxSecondary" onClick={handlePreview} disabled={previewBusy || !datasetId}>
            {previewBusy ? "Rendering…" : "Preview frame"}
          </button>
          <button className="msgBoxSecondary" onClick={onCancel}>Cancel</button>
          <button className="msgBoxOk" onClick={handleSubmit} disabled={!datasetId || endUs <= startUs}>Render</button>
        </div>
      </div>
    </div>
  );
}
