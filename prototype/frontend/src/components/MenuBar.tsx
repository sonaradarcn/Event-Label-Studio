import { useCallback, useEffect, useRef, useState } from "react";
import { useI18n } from "../i18n/I18nContext";

type Props = {
  onOpen: (path: string, name?: string) => void;
  onFileUpload: (file: File) => void;
  recents: { path: string; name: string; project?: string }[];
  onOpenMenu?: () => void;
  onOpenProjects: () => void;
  closeProject: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onExport: () => void;
  onRenderVideo: () => void;
  onScreenshot: () => void;
  onPrint: () => void;
  onSelectAll: () => void;
  onClear: () => void;
  onOpenSettings: () => void;
  datasetId: string;
};

export function MenuBar({ onOpen, onFileUpload, recents, onOpenMenu, onOpenProjects, closeProject, onUndo, onRedo, onExport, onRenderVideo, onScreenshot, onPrint, onSelectAll, onClear, onOpenSettings, datasetId }: Props) {
  const { t } = useI18n();
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleClickOutside = useCallback((e: MouseEvent) => {
    if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
      setOpenMenu(null);
    }
  }, []);

  useEffect(() => {
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [handleClickOutside]);

  function item(label: string, action: () => void, shortcut?: string) {
    return (
      <div className="menuDropdownItem" key={label} onClick={(e) => { e.stopPropagation(); setOpenMenu(null); action(); }}>
        {label}
        {shortcut && <span className="shortcut">{shortcut}</span>}
      </div>
    );
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      onFileUpload(file);
      e.target.value = "";
    }
    setOpenMenu(null);
  }

  return (
    <div className="menuBar" ref={menuRef}>
      <input ref={fileInputRef} type="file" style={{ display: "none" }} accept=".aedat,.aedat4,.raw,.h5,.hdf5,.es,.bin" onChange={handleFileChange} />

      {/* File */}
      <div className="menuItem" onClick={() => { const next = openMenu === "file" ? null : "file"; setOpenMenu(next); if (next === "file") onOpenMenu?.(); }}>
        {t("menu.file")}
        {openMenu === "file" && (
          <div className="menuDropdown">
            {item(t("menu.file.open"), () => fileInputRef.current?.click(), "Ctrl+O")}

            {/* Recent files submenu */}
            <div className="menuDropdownItem hasSubmenu">
              {t("menu.file.recent")}
              <span className="shortcut">▸</span>
              <div className="menuSubmenu">
                {recents.length === 0 ? (
                  <div className="menuDropdownItem" style={{ color: "var(--text-muted)", cursor: "default" }}>
                    {t("menu.file.recent.empty")}
                  </div>
                ) : (
                  recents.map((r) => (
                    <div key={`${r.path}::${r.project ?? ""}`} className="menuDropdownItem"
                      onClick={(e) => { e.stopPropagation(); setOpenMenu(null); onOpen(r.path, r.project); }}
                      title={r.project ? `${r.name} — ${r.path}` : r.path}>
                      {r.name}
                    </div>
                  ))
                )}
              </div>
            </div>
            {item(t("menu.file.projects"), onOpenProjects)}

            <div className="menuSeparator" />
            {item(t("menu.file.close"), closeProject, "Ctrl+W")}
            <div className="menuSeparator" />
            {item(t("menu.file.export"), onExport, "Ctrl+E")}
            {item(t("menu.file.renderVideo"), onRenderVideo)}
            <div className="menuSeparator" />
            {item(t("menu.file.screenshot"), onScreenshot)}
            {item(t("menu.file.print"), onPrint)}
          </div>
        )}
      </div>

      {/* Edit */}
      <div className="menuItem" onClick={() => setOpenMenu(openMenu === "edit" ? null : "edit")}>
        {t("menu.edit")}
        {openMenu === "edit" && (
          <div className="menuDropdown">
            {item(t("menu.edit.undo"), onUndo, "Ctrl+Z")}
            {item(t("menu.edit.redo"), onRedo, "Ctrl+Y")}
            <div className="menuSeparator" />
            {item(t("menu.edit.selectAll"), onSelectAll, "Ctrl+A")}
            {item(t("menu.edit.clear"), onClear, "Del")}
          </div>
        )}
      </div>

      {/* Settings (was View; opens the settings dialog directly — view modes,
          language/theme/GPU, cache, shortcuts and about all live inside it) */}
      <div className="menuItem" onClick={() => { setOpenMenu(null); onOpenSettings(); }}>
        {t("menu.settings")}
      </div>
    </div>
  );
}
