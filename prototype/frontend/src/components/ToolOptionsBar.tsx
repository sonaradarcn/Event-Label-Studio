import { BoxSelect, Eraser, FileDown, FlipHorizontal2, FlipVertical2, FolderOpen, Hand, Redo2, RotateCw, Save, Undo2, SquareDashedBottom, Circle, Spline, Wand2, ScanLine } from "lucide-react";
import { useRef } from "react";
import { useI18n } from "../i18n/I18nContext";
import type { ToolMode } from "../pointcloud/PointCloudView";

type Props = {
  onFileUpload: (file: File) => void;
  onLoad: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onExport: () => void;
  onClear: () => void;
  onPaintRange: () => void;
  onSelectRanges: () => void;
  viewMode: "3d" | "xy";
  onViewMode: (mode: "3d" | "xy") => void;
  pointSizeScale: number;
  setPointSizeScale: (v: number) => void;
  datasetId: string;
  toolMode: ToolMode;
  brushRadius: number;
  selectThrough: boolean;
  setToolMode: (mode: ToolMode) => void;
  setBrushRadius: (r: number) => void;
  setSelectThrough: (v: boolean) => void;
  xyRotation: number;
  rotateXyView: () => void;
  xyFlipY: boolean;
  toggleXyFlipY: () => void;
  xyFlipX: boolean;
  toggleXyFlipX: () => void;
  wandVoxelXy: number;
  wandVoxelTms: number;
  wandMinPts: number;
  wandHasSeed: boolean;
  applyWandParams: () => void;
  setWandVoxelXy: (v: number) => void;
  setWandVoxelTms: (v: number) => void;
  setWandMinPts: (v: number) => void;
};

export function ToolOptionsBar({
  onFileUpload, onLoad, onUndo, onRedo, onExport, onClear,
  onPaintRange, onSelectRanges, viewMode, onViewMode, pointSizeScale, setPointSizeScale, datasetId,
  toolMode, brushRadius, selectThrough, setToolMode, setBrushRadius, setSelectThrough,
  xyRotation, rotateXyView, xyFlipY, toggleXyFlipY, xyFlipX, toggleXyFlipX,
  wandVoxelXy, setWandVoxelXy, wandVoxelTms, setWandVoxelTms,
  wandMinPts, setWandMinPts,
  wandHasSeed, applyWandParams,
}: Props) {
  const { t } = useI18n();
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      onFileUpload(file);
      e.target.value = "";
    }
  }

  return (
    <div className="toolOptionsBar">
      <input ref={fileInputRef} type="file" style={{ display: "none" }} accept=".aedat,.aedat4,.raw,.h5,.hdf5,.es,.bin" onChange={handleFileChange} />

      <button onClick={() => fileInputRef.current?.click()} title={t("menu.file.open")}><FolderOpen size={14} /> {t("toolbar.open")}</button>
      <button onClick={onUndo} title="Ctrl+Z"><Undo2 size={14} /> {t("toolbar.undo")}</button>
      <button onClick={onRedo} title="Ctrl+Y"><Redo2 size={14} /> {t("toolbar.redo")}</button>
      <button onClick={onExport} title={t("toolbar.export")}><FileDown size={14} /> {t("toolbar.export")}</button>
      <button onClick={onClear} title="Del"><Eraser size={14} /> {t("toolbar.clear")}</button>

      <span style={{ color: "var(--text-muted)", fontSize: 11 }}>|</span>

      {/* Pan first, then the Blender-style selection tools. */}
      <button className={toolMode === "pan" ? "active" : ""} onClick={() => setToolMode("pan")} title={t("tool.pan")}>
        <Hand size={14} /> {t("tool.pan")}
      </button>
      <button className={toolMode === "box" ? "active" : ""} onClick={() => setToolMode("box")} title={`${t("tool.box")} (B)`}>
        <SquareDashedBottom size={14} /> {t("tool.box")}
      </button>
      {/* Circle is a 2D brush — only useful in the XY view. */}
      {viewMode !== "3d" && (
        <button className={toolMode === "circle" ? "active" : ""} onClick={() => setToolMode("circle")} title={`${t("tool.circle")} (C)`}>
          <Circle size={14} /> {t("tool.circle")}
        </button>
      )}
      <button className={toolMode === "lasso" ? "active" : ""} onClick={() => setToolMode("lasso")} title={`${t("tool.lasso")} (L)`}>
        <Spline size={14} /> {t("tool.lasso")}
      </button>
      {/* Wand (spatiotemporal connected-components) — only meaningful in 3D. */}
      {viewMode === "3d" && (
        <button className={toolMode === "wand" ? "active" : ""} onClick={() => setToolMode("wand")} title={`${t("tool.wand")} (W)`}>
          <Wand2 size={14} /> {t("tool.wand")}
        </button>
      )}
      {viewMode === "3d" && (
        <button className={toolMode === "tscrub" ? "active" : ""} onClick={() => setToolMode(toolMode === "tscrub" ? "pan" : "tscrub")} title={t("tool.tscrub.hint")}>
          <ScanLine size={14} /> {t("tool.tscrub")}
        </button>
      )}

      {/* Circle brush radius */}
      {toolMode === "circle" && (
        <label style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--text-secondary)", margin: 0 }}>
          {t("tool.brushRadius")}
          <input type="range" min={5} max={120} value={brushRadius}
            onChange={(e) => setBrushRadius(Number(e.target.value))}
            style={{ width: 80, height: 20 }} />
          <span>{brushRadius}px</span>
        </label>
      )}

      {/* Wand = voxel connected-components. Three always-on knobs size the voxel
          "box" (width px / duration ms) and its occupancy threshold (min/voxel).
          Changes take effect on the next point click, or via Apply. */}
      {toolMode === "wand" && (
        <>
          <span style={{ fontSize: 11, color: "var(--text-secondary)" }} title={t("tool.wand.voxelxy.hint")}>
            {t("tool.wand.voxelxy")}: {wandVoxelXy}px
          </span>
          <input type="range" min={2} max={40} step={1} value={wandVoxelXy}
            onChange={(e) => setWandVoxelXy(Number(e.target.value))}
            title={t("tool.wand.voxelxy.hint")} style={{ width: 80, height: 20 }} />
          <span style={{ fontSize: 11, color: "var(--text-secondary)" }} title={t("tool.wand.voxelt.hint")}>
            {t("tool.wand.voxelt")}: {wandVoxelTms}ms
          </span>
          <input type="range" min={5} max={500} step={5} value={wandVoxelTms}
            onChange={(e) => setWandVoxelTms(Number(e.target.value))}
            title={t("tool.wand.voxelt.hint")} style={{ width: 80, height: 20 }} />
          <span style={{ fontSize: 11, color: "var(--text-secondary)" }} title={t("tool.wand.minpts.hint")}>
            {t("tool.wand.minpts")}: {wandMinPts}
          </span>
          <input type="range" min={1} max={20} step={1} value={wandMinPts}
            onChange={(e) => setWandMinPts(Number(e.target.value))}
            title={t("tool.wand.minpts.hint")} style={{ width: 80, height: 20 }} />
          {/* Manual apply — parameter changes don't auto-apply; click to re-run
              on the current selection (a new point click uses current params). */}
          <button onClick={applyWandParams} disabled={!wandHasSeed}
            title={t("tool.wand.apply.hint")}>{t("tool.wand.apply")}</button>
        </>
      )}

      {/* Through / Visible-only toggle */}
      {(toolMode === "box" || toolMode === "circle" || toolMode === "lasso") && (
        <label style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--text-secondary)", margin: 0, cursor: "pointer" }}>
          <input type="checkbox" checked={selectThrough} onChange={(e) => setSelectThrough(e.target.checked)}
            style={{ width: "auto" }} />
          {t("tool.through")}
        </label>
      )}

      <span style={{ color: "var(--text-muted)", fontSize: 11 }}>|</span>

      {(["3d", "xy"] as const).map((mode) => (
        <button key={mode} className={viewMode === mode ? "active" : ""} onClick={() => onViewMode(mode)}>
          {t(`view.${mode}`)}
        </button>
      ))}
      {viewMode === "xy" && (
        <>
          <button onClick={rotateXyView} title={`${t("toolbar.rotate")} 90° (${xyRotation}°)`}>
            <RotateCw size={14} /> {t("toolbar.rotate")} {xyRotation}°
          </button>
          <button
            className={xyFlipY ? "active" : ""}
            onClick={toggleXyFlipY}
            title={t("toolbar.flipY")}
          >
            <FlipVertical2 size={14} /> {t("toolbar.flipY")}
          </button>
          <button
            className={xyFlipX ? "active" : ""}
            onClick={toggleXyFlipX}
            title={t("toolbar.flipX")}
          >
            <FlipHorizontal2 size={14} /> {t("toolbar.flipX")}
          </button>
        </>
      )}

      <span style={{ color: "var(--text-muted)", fontSize: 11 }}>|</span>

      {/* Point size aid (colour mode is fixed to polarity — see Settings → Display). */}
      <label style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--text-secondary)", margin: 0 }}
        title={t("display.pointSize")}>
        {t("display.pointSize")}
        <input type="range" min={0.05} max={2} step={0.05} value={pointSizeScale}
          onChange={(e) => setPointSizeScale(Number(e.target.value))}
          style={{ width: 80, height: 20 }} />
        <span>{pointSizeScale.toFixed(2)}×</span>
      </label>
    </div>
  );
}
