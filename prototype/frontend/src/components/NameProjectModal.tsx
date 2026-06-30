import { useEffect, useRef, useState } from "react";
import { useI18n } from "../i18n/I18nContext";
import "./NameProjectModal.css";

// Shown when importing a recording file. The name the user enters becomes the
// project's cache-folder id, so the same recording can be imported several times
// as independent projects. Styled like the confirm / message modals.
type Props = {
  visible: boolean;
  fileName: string;
  suggested: string;
  existingNames: string[];
  onConfirm: (name: string) => void;
  onCancel: () => void;
};

export function NameProjectModal({ visible, fileName, suggested, existingNames, onConfirm, onCancel }: Props) {
  const { t } = useI18n();
  const [name, setName] = useState(suggested);
  const inputRef = useRef<HTMLInputElement>(null);

  // Reset + focus the field each time the dialog opens for a new file.
  useEffect(() => {
    if (!visible) return;
    setName(suggested);
    const id = window.setTimeout(() => { inputRef.current?.focus(); inputRef.current?.select(); }, 0);
    return () => window.clearTimeout(id);
  }, [visible, suggested]);

  const trimmed = name.trim();
  const valid = trimmed.length > 0;
  const collides = valid && existingNames.includes(trimmed);

  const submit = () => { if (valid) onConfirm(trimmed); };

  useEffect(() => {
    if (!visible) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
      else if (e.key === "Enter") submit();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible, onCancel, trimmed, valid]);

  if (!visible) return null;

  return (
    <div className="confirmOverlay" onMouseDown={onCancel}>
      <div className="msgBox" onMouseDown={(e) => e.stopPropagation()}>
        <div className="msgBoxHeader">
          <span className="msgBoxIcon" style={{ background: "var(--accent)" }}>+</span>
          <span className="msgBoxTitle">{t("import.title")}</span>
        </div>
        <div className="npBody">
          <p className="npHint">{t("import.hint")}</p>
          <div className="npField">
            <span className="npLabel">{t("import.sourceLabel")}</span>
            <div className="npSource" title={fileName}>{fileName}</div>
          </div>
          <div className="npField">
            <label className="npLabel" htmlFor="npName">{t("import.nameLabel")}</label>
            <input id="npName" ref={inputRef} className="npInput" value={name}
              onChange={(e) => setName(e.target.value)} spellCheck={false} autoComplete="off" />
          </div>
          {collides && <div className="npWarn">{t("import.exists")}</div>}
        </div>
        <div className="msgBoxActions">
          <button className="msgBoxSecondary" onClick={onCancel}>{t("confirm.cancel")}</button>
          <button className="msgBoxOk" onClick={submit} disabled={!valid}>{t("import.confirm")}</button>
        </div>
      </div>
    </div>
  );
}
