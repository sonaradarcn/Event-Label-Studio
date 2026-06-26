import { useI18n } from "../i18n/I18nContext";
import { useFps } from "../hooks/useFps";

type Props = {
  status: string;
  payloadCount: number;
  selectedCount: number;
  gpuName: string;
};

export function StatusBar({ status, payloadCount, selectedCount, gpuName }: Props) {
  const { locale } = useI18n();
  const fps = useFps();
  // Colour hint: green = smooth, amber = sluggish, red = janky.
  const fpsColor = fps >= 50 ? "var(--success, #4caf50)" : fps >= 25 ? "var(--warning, #e0a000)" : "var(--danger, #e05050)";

  return (
    <div className="statusBar">
      <span className="statusText">{status}</span>
      <span className="statusRight">
        <span style={{ color: fpsColor }} title="Render frames per second">{fps} FPS</span>
        <span>{payloadCount.toLocaleString()} pts</span>
        <span>Selected: {selectedCount.toLocaleString()}</span>
        <span>GPU: {gpuName}</span>
        <span>{locale}</span>
      </span>
    </div>
  );
}
