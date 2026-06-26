import { useI18n } from "../i18n/I18nContext";
import type { GpuOption } from "./useGpuPreference";

type Props = {
  options: GpuOption[];
  onSelect: (opt: GpuOption) => void;
  visible: boolean;
};

export function GpuSelectModal({ options, onSelect, visible }: Props) {
  const { t } = useI18n();
  if (!visible) return null;

  const singleGpu = options.length === 1;
  const noDiscrete = !options.some((o) => o.preference === "high-performance" && !o.name.includes("High-performance"));

  return (
    <div className="gpuModalOverlay">
      <div className="gpuModal">
        <h2>{t("gpu.title")}</h2>
        {(singleGpu || noDiscrete) && (
          <div className="gpuWarning">
            <strong>{t("gpu.warning.title")}</strong>
            <p>{t("gpu.warning.text")}</p>
            <ol>
              <li>Open <strong>Windows Settings → System → Display → Graphics</strong></li>
              <li>Click <strong>"Add"</strong> (or "Browse") and find your browser executable</li>
              <li>Click it, set to <strong>"High performance"</strong> (your discrete GPU)</li>
              <li><strong>Restart the browser</strong> and refresh this page</li>
            </ol>
            <p style={{ fontSize: 12, color: "var(--text-muted)" }}>{t("gpu.warning.hint")}</p>
          </div>
        )}
        <div className="gpuOptions">
          {options.map((opt) => (
            <button key={opt.preference} className="gpuOption" onClick={() => onSelect(opt)}>
              <span className="gpuOptionName">{opt.name}</span>
              {opt.vendor && <span className="gpuOptionVendor">{opt.vendor}</span>}
              <span className="gpuOptionTag">
                {opt.preference === "high-performance" ? t("gpu.discrete") : opt.preference === "low-power" ? t("gpu.integrated") : t("gpu.default")}
              </span>
            </button>
          ))}
        </div>
        {!singleGpu && (
          <p className="gpuModalDesc">{t("gpu.changeLater")}</p>
        )}
      </div>
    </div>
  );
}
