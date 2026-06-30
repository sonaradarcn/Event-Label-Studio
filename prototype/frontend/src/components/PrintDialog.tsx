import { useEffect } from "react";
import { useI18n } from "../i18n/I18nContext";

// Asks the user for page orientation before printing the current view.
// Styled like the confirm / message modals.
type Props = {
  visible: boolean;
  onChoose: (orientation: "landscape" | "portrait") => void;
  onCancel: () => void;
};

export function PrintDialog({ visible, onChoose, onCancel }: Props) {
  const { t } = useI18n();

  useEffect(() => {
    if (!visible) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onCancel(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible, onCancel]);

  if (!visible) return null;

  return (
    <div className="confirmOverlay" onMouseDown={onCancel}>
      <div className="msgBox" onMouseDown={(e) => e.stopPropagation()}>
        <div className="msgBoxHeader">
          <span className="msgBoxIcon" style={{ background: "var(--accent)" }}>⎙</span>
          <span className="msgBoxTitle">{t("print.title")}</span>
        </div>
        <div className="npBody">
          <p className="npHint">{t("print.hint")}</p>
        </div>
        <div className="msgBoxActions">
          <button className="msgBoxSecondary" onClick={onCancel}>{t("confirm.cancel")}</button>
          <button className="msgBoxOk" onClick={() => onChoose("portrait")}>{t("print.portrait")}</button>
          <button className="msgBoxOk" onClick={() => onChoose("landscape")}>{t("print.landscape")}</button>
        </div>
      </div>
    </div>
  );
}
