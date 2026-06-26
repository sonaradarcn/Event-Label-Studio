import { useEffect } from "react";

export type MessageKind = "info" | "success" | "error";

type Action = { label: string; onClick: () => void; primary?: boolean };

type Props = {
  visible: boolean;
  title: string;
  body: string;
  kind?: MessageKind;
  okLabel?: string;
  onClose: () => void;
  actions?: Action[]; // Extra buttons shown left of OK.
};

export function MessageBox({ visible, title, body, kind = "info", okLabel = "OK", onClose, actions }: Props) {
  useEffect(() => {
    if (!visible) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.key === "Enter") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible, onClose]);

  if (!visible) return null;

  const accent = kind === "error" ? "#e74c3c" : kind === "success" ? "#2ecc71" : "var(--accent)";
  const icon = kind === "error" ? "✕" : kind === "success" ? "✓" : "i";

  return (
    <div className="msgBoxOverlay" onMouseDown={onClose}>
      <div className="msgBox" onMouseDown={(e) => e.stopPropagation()}>
        <div className="msgBoxHeader">
          <span className="msgBoxIcon" style={{ background: accent }}>{icon}</span>
          <span className="msgBoxTitle">{title}</span>
        </div>
        <pre className="msgBoxBody">{body}</pre>
        <div className="msgBoxActions">
          {actions?.map((a, i) => (
            <button
              key={i}
              className={a.primary ? "msgBoxOk" : "msgBoxSecondary"}
              onClick={a.onClick}
            >
              {a.label}
            </button>
          ))}
          <button className="msgBoxOk" onClick={onClose} autoFocus>{okLabel}</button>
        </div>
      </div>
    </div>
  );
}
