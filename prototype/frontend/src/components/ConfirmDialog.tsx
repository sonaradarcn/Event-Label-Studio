import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { useI18n } from "../i18n/I18nContext";

// A promise-based confirm dialog, styled like the export / video-render modals.
// Replaces window.confirm everywhere: `const confirm = useConfirm(); if (await
// confirm({ message, danger:true })) { ... }`.
export type ConfirmOptions = {
  title?: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean; // red icon + red confirm button for destructive actions
};

type Confirm = (opts: ConfirmOptions) => Promise<boolean>;

const ConfirmContext = createContext<Confirm>(async () => false);

export function useConfirm(): Confirm {
  return useContext(ConfirmContext);
}

export function ConfirmProvider({ children }: { children: ReactNode }) {
  const [req, setReq] = useState<{ opts: ConfirmOptions; resolve: (v: boolean) => void } | null>(null);

  const confirm = useCallback<Confirm>(
    (opts) => new Promise<boolean>((resolve) => setReq({ opts, resolve })),
    [],
  );
  const finish = useCallback((result: boolean) => {
    setReq((cur) => { cur?.resolve(result); return null; });
  }, []);

  return (
    <ConfirmContext.Provider value={confirm}>
      {children}
      <ConfirmDialog visible={!!req} opts={req?.opts} onConfirm={() => finish(true)} onCancel={() => finish(false)} />
    </ConfirmContext.Provider>
  );
}

function ConfirmDialog({ visible, opts, onConfirm, onCancel }: {
  visible: boolean;
  opts?: ConfirmOptions;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const { t } = useI18n();

  useEffect(() => {
    if (!visible) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
      // Enter confirms only for non-destructive prompts (avoids accidental deletes).
      else if (e.key === "Enter" && !opts?.danger) onConfirm();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible, opts, onConfirm, onCancel]);

  if (!visible || !opts) return null;
  const danger = !!opts.danger;

  return (
    <div className="confirmOverlay" onMouseDown={onCancel}>
      <div className="msgBox" onMouseDown={(e) => e.stopPropagation()}>
        <div className="msgBoxHeader">
          <span className="msgBoxIcon" style={{ background: danger ? "#e74c3c" : "var(--accent)" }}>{danger ? "!" : "?"}</span>
          <span className="msgBoxTitle">{opts.title ?? t("confirm.title")}</span>
        </div>
        <pre className="msgBoxBody">{opts.message}</pre>
        <div className="msgBoxActions">
          <button className="msgBoxSecondary" onClick={onCancel} autoFocus>{opts.cancelLabel ?? t("confirm.cancel")}</button>
          <button className={danger ? "msgBoxOk msgBoxDanger" : "msgBoxOk"} onClick={onConfirm}>
            {opts.confirmLabel ?? t("confirm.ok")}
          </button>
        </div>
      </div>
    </div>
  );
}
