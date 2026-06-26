import { Download } from "lucide-react";
import { useEffect, useState } from "react";
import type { ExportParams } from "../api/client";
import type { DatasetMeta } from "../api/types";
import type { StatsData } from "../hooks/useAppState";
import { useI18n } from "../i18n/I18nContext";

type Props = {
  visible: boolean;
  datasetId: string;
  meta: DatasetMeta | null;
  stats: StatsData;
  onCancel: () => void;
  onSubmit: (params: ExportParams) => void;
};

const SAMPLE_SIZES = [50_000, 100_000, 200_000, 500_000, 1_000_000];

export function ExportModal({ visible, datasetId, meta, stats, onCancel, onSubmit }: Props) {
  const { t } = useI18n();
  const [sample, setSample] = useState(false);
  const [sampleSize, setSampleSize] = useState(200_000);
  const [onlyLabelled, setOnlyLabelled] = useState(false);
  const [includeNpz, setIncludeNpz] = useState(true);
  const [includePly, setIncludePly] = useState(true);
  const [includeCsv, setIncludeCsv] = useState(true);

  // Reset to defaults whenever the dialog reopens.
  useEffect(() => {
    if (!visible) return;
    setSample(false);
    setSampleSize(200_000);
    setOnlyLabelled(false);
    setIncludeNpz(true);
    setIncludePly(true);
    setIncludeCsv(true);
  }, [visible]);

  useEffect(() => {
    if (!visible) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onCancel(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible, onCancel]);

  if (!visible) return null;

  const total = meta?.event_count ?? stats?.total ?? 0;
  const labelled = stats ? stats.total - stats.unlabelled : 0;
  const exportSet = onlyLabelled ? labelled : total;
  const vizCount = sample ? Math.min(sampleSize, exportSet) : exportSet;
  const nothingSelected = !includeNpz && !includePly && !includeCsv;
  const emptyExport = onlyLabelled && labelled === 0;

  const handleSubmit = () => {
    onSubmit({
      sample,
      sample_size: sampleSize,
      only_labelled: onlyLabelled,
      include_npz: includeNpz,
      include_ply: includePly,
      include_csv: includeCsv,
    });
  };

  return (
    <div className="msgBoxOverlay" onMouseDown={onCancel}>
      <div className="msgBox exportModal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="msgBoxHeader">
          <span className="msgBoxIcon" style={{ background: "var(--accent)" }}><Download size={13} /></span>
          <span className="msgBoxTitle">{t("export.title")}</span>
        </div>

        <div className="exportBody">
          <div className="exportField">
            <label className="exportCheck">
              <input type="checkbox" checked={sample} onChange={(e) => setSample(e.target.checked)} />
              {t("export.sample")}
            </label>
            <span className="exportHint">{t("export.sample.hint")}</span>
            {sample && (
              <div className="exportSampleSize">
                <label>{t("export.sampleSize")}</label>
                <select value={sampleSize} onChange={(e) => setSampleSize(Number(e.target.value))}>
                  {SAMPLE_SIZES.map((s) => (
                    <option key={s} value={s}>{s.toLocaleString()}</option>
                  ))}
                </select>
              </div>
            )}
          </div>

          <div className="exportField">
            <label className="exportCheck">
              <input type="checkbox" checked={onlyLabelled} onChange={(e) => setOnlyLabelled(e.target.checked)} />
              {t("export.onlyLabelled")}
            </label>
            <span className="exportHint">{t("export.onlyLabelled.hint")}</span>
          </div>

          <div className="exportField">
            <label className="exportFieldLabel">{t("export.include")}</label>
            <div className="exportIncludeRow">
              <label className="exportCheck">
                <input type="checkbox" checked={includeNpz} onChange={(e) => setIncludeNpz(e.target.checked)} />
                {t("export.include.npz")}
              </label>
              <label className="exportCheck">
                <input type="checkbox" checked={includePly} onChange={(e) => setIncludePly(e.target.checked)} />
                {t("export.include.ply")}
              </label>
              <label className="exportCheck">
                <input type="checkbox" checked={includeCsv} onChange={(e) => setIncludeCsv(e.target.checked)} />
                {t("export.include.csv")}
              </label>
            </div>
          </div>

          <div className="exportSummary">
            <span>{t("export.summary.set")}: <b>{exportSet.toLocaleString()}</b></span>
            <span>{t("export.summary.viz")}: <b>{vizCount.toLocaleString()}</b></span>
            <span>{sample ? t("export.summary.naming.preview") : t("export.summary.naming.full")}</span>
          </div>
          {emptyExport && <p className="exportWarn">{t("export.warn.empty")}</p>}
        </div>

        <div className="msgBoxActions">
          <button className="msgBoxSecondary" onClick={onCancel}>{t("export.cancel")}</button>
          <button className="msgBoxOk" onClick={handleSubmit} disabled={!datasetId || nothingSelected || emptyExport}>
            {t("export.run")}
          </button>
        </div>
      </div>
    </div>
  );
}
