import { useEffect, useState } from "react";
import { GpuSelectModal } from "../gpu-select/GpuSelectModal";
import { useI18n } from "../i18n/I18nContext";
import { MessageBox } from "./MessageBox";
import { NameProjectModal } from "./NameProjectModal";
import { PrintDialog } from "./PrintDialog";
import { SettingsModal } from "./SettingsModal";
import { RenderVideoModal } from "./RenderVideoModal";
import { ExportModal } from "./ExportModal";
import type { ExportParams, RenderVideoParams } from "../api/client";
import { useGpuPreference } from "../gpu-select/useGpuPreference";
import type { ImportPromptState, ProgressState, StatsData } from "../hooks/useAppState";
import { LeftPanel } from "./LeftPanel";
import { MainViewport } from "./MainViewport";
import { MenuBar } from "./MenuBar";
import { RightPanel } from "./RightPanel";
import { StatusBar } from "./StatusBar";
import { ToolOptionsBar } from "./ToolOptionsBar";
import type { ChunkManifest, DatasetMeta, LabelClass, PointPayload, Recording } from "../api/types";
import type { ToolMode, FilterPreview } from "../pointcloud/PointCloudView";

type Props = {
  recordings: Recording[];
  datasetId: string;
  meta: DatasetMeta | null;
  labels: LabelClass[];
  activeLabel: number;
  payload: PointPayload | null;
  manifest: ChunkManifest | null;
  chunksMap: Map<number, PointPayload>;
  visibleChunkIds: number[];
  viewMode: "3d" | "xy";
  polarity: string;
  sample: number;
  startUs: number;
  endUs: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  filterPreview: FilterPreview | null;
  setFilterPreview: (p: FilterPreview | null) => void;
  selectedIds: Set<number>;
  labelPatch: { seq: number; perChunk: Map<number, number[]> } | null;
  colorMode: "polarity" | "time";
  pointSizeScale: number;
  setColorMode: (m: "polarity" | "time") => void;
  setPointSizeScale: (v: number) => void;
  track: import("../hooks/useAppState").TrackUiState | null;
  trackCreate: () => void;
  trackPropagate: () => void;
  trackUseSelectionAsSeed: () => void;
  trackPreviewCandidate: (index: number) => void;
  trackConfirmCandidate: () => void;
  trackCancel: () => void;
  trackSetDirection: (d: "forward" | "backward") => void;
  status: string;
  progress: ProgressState;
  stats: StatsData;
  newLabelName: string;
  histogram: Map<number, number>;
  visibleEventCount: number;
  assignBusy: boolean;
  timeRange: number;
  timeSliderMin: number;
  timeSliderMax: number;
  setActiveLabel: (id: number) => void;
  setViewMode: (mode: "3d" | "xy") => void;
  setPolarity: (v: string) => void;
  setSample: (v: number) => void;
  setStartUs: (v: number) => void;
  setEndUs: (v: number) => void;
  setXMin: (v: number) => void;
  setXMax: (v: number) => void;
  setYMin: (v: number) => void;
  setYMax: (v: number) => void;
  setNewLabelName: (name: string) => void;
  handleOpen: (path: string) => void;
  handleFileUpload: (file: File) => void;
  loadPoints: () => void;
  selectByRanges: () => void;
  selectAll: () => void;
  assign: (op: "assign" | "clear") => void;
  assignSingle: (eventId: number, labelId: number, op: "assign" | "clear") => void;
  assignByFilter: () => void;
  clearByFilter: () => void;
  runCommand: (name: "undo" | "redo") => void;
  exportVisible: boolean;
  openExport: () => void;
  closeExport: () => void;
  startExport: (params: ExportParams) => void;
  addLabel: () => void;
  removeLabel: (id: number) => void;
  updateLabel: (id: number, changes: { name?: string; color?: string }) => void;
  clearAllLabels: () => void;
  toolMode: ToolMode;
  brushRadius: number;
  selectThrough: boolean;
  xyRotation: number;
  rotateXyView: () => void;
  xyFlipY: boolean;
  toggleXyFlipY: () => void;
  xyFlipX: boolean;
  toggleXyFlipX: () => void;
  recents: { path: string; name: string; project?: string }[];
  refreshRecents: () => void;
  closeProject: () => void;
  importPrompt: ImportPromptState;
  confirmImport: (name: string) => void;
  cancelImport: () => void;
  screenshotView: () => void;
  printVisible: boolean;
  openPrint: () => void;
  cancelPrint: () => void;
  confirmPrint: (orientation: "landscape" | "portrait") => void;
  messageBox: {
    visible: boolean; title: string; body: string;
    kind: "info" | "success" | "error";
    actions?: { label: string; onClick: () => void; primary?: boolean }[];
  };
  hideMessage: () => void;
  settingsVisible: boolean;
  settingsSection: "display" | "general" | "cache" | "shortcuts" | "about";
  openSettings: (section?: "display" | "general" | "cache" | "shortcuts" | "about") => void;
  closeSettings: () => void;
  unlabeledColor: string;
  setUnlabeledColor: (hex: string) => void;
  polarityContrast: number;
  setPolarityContrast: (v: number) => void;
  renderVideoVisible: boolean;
  openRenderVideo: () => void;
  closeRenderVideo: () => void;
  startRenderVideo: (params: RenderVideoParams) => void;
  setToolMode: (mode: ToolMode) => void;
  setBrushRadius: (r: number) => void;
  setSelectThrough: (v: boolean) => void;
  addToSelection: (ids: number[]) => void;
  removeFromSelection: (ids: number[]) => void;
  setSelectedIds: (ids: Set<number>) => void;
  selectComponent: (seedEventId: number, op: import("../pointcloud/PointCloudView").SelectionOp) => void;
  wandVoxelXy: number;
  wandVoxelTms: number;
  wandMinPts: number;
  wandHasSeed: boolean;
  applyWandParams: () => void;
  setWandVoxelXy: (v: number) => void;
  setWandVoxelTms: (v: number) => void;
  setWandMinPts: (v: number) => void;
  // Playback
  currentFrameUs: number;
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
};

export function AppShell(props: Props) {
  const { preference: gpuPreference, gpuName, showModal: showGpuModal, options: gpuOptions, select: selectGpu, openModal: openGpuModal } = useGpuPreference();
  const { t } = useI18n();
  // Collapsible side panels (persisted).
  const [leftCollapsed, setLeftCollapsed] = useState(() => localStorage.getItem("ui.leftCollapsed") === "1");
  const [rightCollapsed, setRightCollapsed] = useState(() => localStorage.getItem("ui.rightCollapsed") === "1");
  useEffect(() => { localStorage.setItem("ui.leftCollapsed", leftCollapsed ? "1" : "0"); }, [leftCollapsed]);
  useEffect(() => { localStorage.setItem("ui.rightCollapsed", rightCollapsed ? "1" : "0"); }, [rightCollapsed]);

  return (
    <div className="app">
      <MenuBar
        onOpen={props.handleOpen}
        onFileUpload={props.handleFileUpload}
        recents={props.recents}
        onOpenMenu={props.refreshRecents}
        onOpenProjects={() => props.openSettings("cache")}
        closeProject={props.closeProject}
        onUndo={() => props.runCommand("undo")}
        onRedo={() => props.runCommand("redo")}
        onExport={props.openExport}
        onRenderVideo={props.openRenderVideo}
        onScreenshot={props.screenshotView}
        onPrint={props.openPrint}
        onSelectAll={props.selectAll}
        onClear={() => props.assign("clear")}
        onOpenSettings={props.openSettings}
        datasetId={props.datasetId}
      />
      <ToolOptionsBar
        onFileUpload={props.handleFileUpload}
        onLoad={props.loadPoints}
        onUndo={() => props.runCommand("undo")}
        onRedo={() => props.runCommand("redo")}
        onExport={props.openExport}
        onClear={() => props.assign("clear")}
        onPaintRange={props.assignByFilter}
        onSelectRanges={props.selectByRanges}
        viewMode={props.viewMode}
        onViewMode={props.setViewMode}
        pointSizeScale={props.pointSizeScale}
        setPointSizeScale={props.setPointSizeScale}
        datasetId={props.datasetId}
        toolMode={props.toolMode}
        brushRadius={props.brushRadius}
        selectThrough={props.selectThrough}
        setToolMode={props.setToolMode}
        setBrushRadius={props.setBrushRadius}
        setSelectThrough={props.setSelectThrough}
        xyRotation={props.xyRotation}
        rotateXyView={props.rotateXyView}
        xyFlipY={props.xyFlipY}
        toggleXyFlipY={props.toggleXyFlipY}
        xyFlipX={props.xyFlipX}
        toggleXyFlipX={props.toggleXyFlipX}
        wandVoxelXy={props.wandVoxelXy}
        wandVoxelTms={props.wandVoxelTms}
        wandMinPts={props.wandMinPts}
        wandHasSeed={props.wandHasSeed}
        applyWandParams={props.applyWandParams}
        setWandVoxelXy={props.setWandVoxelXy}
        setWandVoxelTms={props.setWandVoxelTms}
        setWandMinPts={props.setWandMinPts}
      />
      <main
        className="workspace"
        style={{ gridTemplateColumns: `${leftCollapsed ? 0 : 220}px minmax(0, 1fr) ${rightCollapsed ? 0 : 260}px` }}
      >
        <button
          className="panelToggle left"
          style={{ left: leftCollapsed ? 0 : 220 }}
          title={leftCollapsed ? t("panel.show") : t("panel.hide")}
          onClick={() => setLeftCollapsed((v) => !v)}
        >{leftCollapsed ? "›" : "‹"}</button>
        <button
          className="panelToggle right"
          style={{ right: rightCollapsed ? 0 : 260 }}
          title={rightCollapsed ? t("panel.show") : t("panel.hide")}
          onClick={() => setRightCollapsed((v) => !v)}
        >{rightCollapsed ? "‹" : "›"}</button>
        {leftCollapsed ? <div className="panelCollapsed" /> : <LeftPanel
          meta={props.meta}
          datasetId={props.datasetId}
          stats={props.stats}
          labels={props.labels}
          activeLabel={props.activeLabel}
          newLabelName={props.newLabelName}
          setActiveLabel={props.setActiveLabel}
          setNewLabelName={props.setNewLabelName}
          addLabel={props.addLabel}
          removeLabel={props.removeLabel}
          updateLabel={props.updateLabel}
          assign={props.assign}
          assignBusy={props.assignBusy}
          selectedIds={props.selectedIds}
        />}
        <MainViewport
          payload={props.payload}
          manifest={props.manifest}
          chunksMap={props.chunksMap}
          visibleChunkIds={props.visibleChunkIds}
          viewMode={props.viewMode}
          selectedIds={props.selectedIds}
          labelPatch={props.labelPatch}
          colorMode={props.colorMode}
          pointSizeScale={props.pointSizeScale}
          unlabeledColor={props.unlabeledColor}
          polarityContrast={props.polarityContrast}
          xMin={props.xMin}
          xMax={props.xMax}
          yMin={props.yMin}
          yMax={props.yMax}
          polarity={props.polarity}
          filterPreview={props.filterPreview}
          gpuPreference={gpuPreference}
          labels={props.labels}
          meta={props.meta ?? undefined}
          datasetId={props.datasetId}
          progress={props.progress}
          assignSingle={props.assignSingle}
          toolMode={props.toolMode}
          brushRadius={props.brushRadius}
          selectThrough={props.selectThrough}
          xyRotation={props.xyRotation}
          xyFlipY={props.xyFlipY}
          xyFlipX={props.xyFlipX}
          addToSelection={props.addToSelection}
          removeFromSelection={props.removeFromSelection}
          setSelectedIds={props.setSelectedIds}
          setBrushRadius={props.setBrushRadius}
          selectComponent={props.selectComponent}
          currentFrameUs={props.currentFrameUs}
          startUs={props.startUs}
          endUs={props.endUs}
          playing={props.playing}
          playbackSpeed={props.playbackSpeed}
          frameWindowUs={props.frameWindowUs}
          togglePlayback={props.togglePlayback}
          stepForward={props.stepForward}
          stepBackward={props.stepBackward}
          goToStart={props.goToStart}
          goToEnd={props.goToEnd}
          setCurrentFrameUs={props.setCurrentFrameUs}
          setPlaybackSpeed={props.setPlaybackSpeed}
          setFrameWindowUs={props.setFrameWindowUs}
          track={props.track}
          trackCreate={props.trackCreate}
          trackPropagate={props.trackPropagate}
          trackUseSelectionAsSeed={props.trackUseSelectionAsSeed}
          trackPreviewCandidate={props.trackPreviewCandidate}
          trackConfirmCandidate={props.trackConfirmCandidate}
          trackCancel={props.trackCancel}
          trackSetDirection={props.trackSetDirection}
        />
        {rightCollapsed ? <div className="panelCollapsed" /> : <RightPanel
          payload={props.payload}
          visibleEventCount={props.visibleEventCount}
          selectedIds={props.selectedIds}
          histogram={props.histogram}
          labels={props.labels}
          meta={props.meta}
          startUs={props.startUs}
          endUs={props.endUs}
          xMin={props.xMin}
          xMax={props.xMax}
          yMin={props.yMin}
          yMax={props.yMax}
          polarity={props.polarity}
          sample={props.sample}
          timeSliderMin={props.timeSliderMin}
          timeSliderMax={props.timeSliderMax}
          timeRange={props.timeRange}
          setStartUs={props.setStartUs}
          setEndUs={props.setEndUs}
          setXMin={props.setXMin}
          setXMax={props.setXMax}
          setYMin={props.setYMin}
          setYMax={props.setYMax}
          setPolarity={props.setPolarity}
          setSample={props.setSample}
          setFilterPreview={props.setFilterPreview}
        />}
      </main>
      <StatusBar
        status={props.status}
        payloadCount={props.visibleEventCount}
        selectedCount={props.selectedIds.size}
        gpuName={gpuName}
      />
      <GpuSelectModal options={gpuOptions} onSelect={selectGpu} visible={showGpuModal} />
      <NameProjectModal
        visible={props.importPrompt.visible}
        fileName={props.importPrompt.fileName}
        suggested={props.importPrompt.suggested}
        existingNames={props.importPrompt.existingNames}
        onConfirm={props.confirmImport}
        onCancel={props.cancelImport}
      />
      <PrintDialog
        visible={props.printVisible}
        onChoose={props.confirmPrint}
        onCancel={props.cancelPrint}
      />
      <MessageBox
        visible={props.messageBox.visible}
        title={props.messageBox.title}
        body={props.messageBox.body}
        kind={props.messageBox.kind}
        actions={props.messageBox.actions}
        onClose={props.hideMessage}
      />
      <RenderVideoModal
        visible={props.renderVideoVisible}
        datasetId={props.datasetId}
        meta={props.meta}
        defaultStartUs={props.startUs}
        defaultEndUs={props.endUs}
        unlabeledColor={props.unlabeledColor}
        polarityContrast={props.polarityContrast}
        onCancel={props.closeRenderVideo}
        onSubmit={props.startRenderVideo}
      />
      <ExportModal
        visible={props.exportVisible}
        datasetId={props.datasetId}
        meta={props.meta}
        stats={props.stats}
        onCancel={props.closeExport}
        onSubmit={props.startExport}
      />
      <SettingsModal
        visible={props.settingsVisible}
        unlabeledColor={props.unlabeledColor}
        polarityContrast={props.polarityContrast}
        onUnlabeledColor={props.setUnlabeledColor}
        onPolarityContrast={props.setPolarityContrast}
        onClose={props.closeSettings}
        gpuName={gpuName}
        onOpenGpuModal={openGpuModal}
        onOpenDataset={props.handleOpen}
        initialSection={props.settingsSection}
      />
    </div>
  );
}
