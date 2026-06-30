import { useEffect, useState } from "react";
import { BookOpen, Github } from "lucide-react";
import { useI18n } from "../i18n/I18nContext";
import { useTheme } from "../theme/ThemeContext";
import { listCache, deleteDataset, renameDataset, revealDataset, exportDatasetDefault, type CacheEntry } from "../api/client";
import { useConfirm } from "./ConfirmDialog";
import eggMask from "../assets/easter-egg.png";

const GITHUB_URL = "https://github.com/sonaradarcn/Event-Label-Studio";

type Props = {
  visible: boolean;
  unlabeledColor: string;
  polarityContrast: number;
  onUnlabeledColor: (hex: string) => void;
  onPolarityContrast: (v: number) => void;
  onClose: () => void;
  gpuName: string;
  onOpenGpuModal: () => void;
  onOpenDataset?: (sourcePath: string, name?: string) => void;
  initialSection?: "display" | "general" | "cache" | "shortcuts" | "about";
};

// Human-readable byte size (e.g. 12.3 MB). Binary (1024) units.
function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const v = bytes / Math.pow(1024, i);
  return `${i === 0 ? v : v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`;
}

// Parse a "#rrggbb" hex string into [r, g, b] (0..255). Falls back to mid-gray.
function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([0-9a-fA-F]{6})$/.exec(hex.trim());
  if (!m) return [136, 136, 136];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}

function rgbToHex(r: number, g: number, b: number): string {
  const clamp = (c: number) => Math.max(0, Math.min(255, Math.round(c)));
  return "#" + [r, g, b].map((c) => clamp(c).toString(16).padStart(2, "0")).join("");
}

// ON: brighten base toward white by contrast k (c -> c + (255 - c) * k).
function brighten(hex: string, k: number): string {
  const [r, g, b] = hexToRgb(hex);
  return rgbToHex(r + (255 - r) * k, g + (255 - g) * k, b + (255 - b) * k);
}

// OFF: darken base toward black by contrast k (c -> c * (1 - k)).
function darken(hex: string, k: number): string {
  const [r, g, b] = hexToRgb(hex);
  return rgbToHex(r * (1 - k), g * (1 - k), b * (1 - k));
}

export function SettingsModal({
  visible,
  unlabeledColor,
  polarityContrast,
  onUnlabeledColor,
  onPolarityContrast,
  onClose,
  gpuName,
  onOpenGpuModal,
  onOpenDataset,
  initialSection,
}: Props) {
  const { t, locale, setLocale, locales } = useI18n();
  const { theme, toggleTheme } = useTheme();
  const confirm = useConfirm();
  const [section, setSection] = useState<"display" | "general" | "cache" | "shortcuts" | "about">("general");

  // Cache section state.
  const [cacheEntries, setCacheEntries] = useState<CacheEntry[]>([]);
  const [cacheLoading, setCacheLoading] = useState(false);
  // Inline project rename.
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [renameError, setRenameError] = useState<string | null>(null);

  // Open at the requested section (e.g. File → Project Manager jumps to "cache").
  useEffect(() => {
    if (visible) setSection(initialSection ?? "general");
  }, [visible, initialSection]);

  const refreshCache = () => {
    setCacheLoading(true);
    listCache()
      .then((entries) => setCacheEntries(entries))
      .catch(() => setCacheEntries([]))
      .finally(() => setCacheLoading(false));
  };

  // Esc closes.
  useEffect(() => {
    if (!visible) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible, onClose]);

  // Fetch the cache listing when the cache section becomes visible.
  useEffect(() => {
    if (!visible || section !== "cache") return;
    refreshCache();
  }, [visible, section]);

  if (!visible) return null;

  const onSwatch = brighten(unlabeledColor, polarityContrast);
  const offSwatch = darken(unlabeledColor, polarityContrast);

  const totalCacheBytes = cacheEntries.reduce((sum, e) => sum + (e.size_bytes || 0), 0);

  const handleOpen = (entry: CacheEntry) => {
    // Open the EXISTING cache by its id (not re-derived from the file stem), so a
    // custom-named project re-opens correctly and never re-decodes.
    onOpenDataset?.(entry.source_path, entry.dataset_id);
    onClose?.();
  };

  const handleReveal = (id: string) => {
    revealDataset(id).catch(() => { /* best-effort, server-side reveal */ });
  };

  const handleExport = (id: string) => {
    exportDatasetDefault(id).catch(() => { /* best-effort */ });
  };

  const startRename = (entry: CacheEntry) => {
    setRenamingId(entry.dataset_id);
    setRenameValue(entry.dataset_id);
    setRenameError(null);
  };
  const cancelRename = () => { setRenamingId(null); setRenameError(null); };
  const confirmRename = async (oldId: string) => {
    const name = renameValue.trim();
    if (!name || name === oldId) { cancelRename(); return; }
    const res = await renameDataset(oldId, name);
    if (!res.ok) { setRenameError(res.error || "Rename failed"); return; }
    setRenamingId(null);
    setRenameError(null);
    refreshCache();
  };

  const handleDelete = async (id: string) => {
    if (!(await confirm({ message: t("settings.cache.deleteConfirm"), confirmLabel: t("confirm.delete"), danger: true }))) return;
    deleteDataset(id).then(() => refreshCache()).catch(() => refreshCache());
  };

  const handleClearAll = async () => {
    if (cacheEntries.length === 0) return;
    if (!(await confirm({ message: t("settings.cache.clearAllConfirm"), confirmLabel: t("confirm.delete"), danger: true }))) return;
    Promise.allSettled(cacheEntries.map((e) => deleteDataset(e.dataset_id))).then(() => refreshCache());
  };

  return (
    <div className="settingsOverlay" onMouseDown={onClose}>
      <div className="settingsModal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="settingsHeader">
          <span className="settingsTitle">{t("settings.title")}</span>
          <button className="settingsClose" onClick={onClose} aria-label={t("tooltip.close")}>
            ×
          </button>
        </div>

        <div className="settingsBody">
          <nav className="settingsNav">
            <button className={`settingsNavItem ${section === "general" ? "active" : ""}`} onClick={() => setSection("general")}>
              {t("settings.nav.general")}
            </button>
            <button className={`settingsNavItem ${section === "cache" ? "active" : ""}`} onClick={() => setSection("cache")}>
              {t("settings.nav.cache")}
            </button>
            <button className={`settingsNavItem ${section === "display" ? "active" : ""}`} onClick={() => setSection("display")}>
              {t("settings.nav.display")}
            </button>
            <button className={`settingsNavItem ${section === "shortcuts" ? "active" : ""}`} onClick={() => setSection("shortcuts")}>
              {t("settings.nav.shortcuts")}
            </button>
            <button className={`settingsNavItem ${section === "about" ? "active" : ""}`} onClick={() => setSection("about")}>
              {t("settings.nav.about")}
            </button>
          </nav>

          {section === "display" && (
            <div className="settingsContent">
              <div className="settingsSectionTitle">{t("settings.section.display.title")}</div>

              <div className="settingsRow">
                <label>{t("settings.unlabeledColor")}</label>
                <div className="settingsControl">
                  <input
                    type="color"
                    className="settingsColorInput"
                    value={unlabeledColor}
                    onChange={(e) => onUnlabeledColor(e.target.value)}
                  />
                </div>
              </div>

              <div className="settingsRow">
                <label>{t("settings.contrast")}</label>
                <div className="settingsControl settingsContrastControl">
                  <div className="settingsRangeRow">
                    <input
                      type="range"
                      className="settingsRange"
                      min={0}
                      max={1}
                      step={0.05}
                      value={polarityContrast}
                      onChange={(e) => onPolarityContrast(Number(e.target.value))}
                    />
                    <span className="settingsRangeValue">{polarityContrast.toFixed(2)}</span>
                  </div>
                  <div className="settingsPreview">
                    <span className="settingsSwatch" style={{ background: onSwatch }}>
                      {t("panel.inspector.polarity.on")}
                    </span>
                    <span className="settingsSwatch" style={{ background: offSwatch }}>
                      {t("panel.inspector.polarity.off")}
                    </span>
                  </div>
                  <div className="settingsHint">{t("settings.contrast.hint")}</div>
                </div>
              </div>
            </div>
          )}

          {section === "general" && (
            <div className="settingsContent">
              <div className="settingsSectionTitle">{t("settings.section.general.title")}</div>

              <div className="settingsRow">
                <label>{t("menu.view.language")}</label>
                <div className="settingsControl">
                  <select className="settingsSelect" value={locale} onChange={(e) => setLocale(e.target.value)}>
                    {locales.map((l) => (
                      <option key={l.code} value={l.code}>{l.name}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="settingsRow">
                <label>{t("menu.view.theme")}</label>
                <div className="settingsControl settingsSegment">
                  <button
                    className={theme === "dark" ? "active" : ""}
                    onClick={() => { if (theme !== "dark") toggleTheme(); }}
                  >☽ {t("settings.theme.dark")}</button>
                  <button
                    className={theme === "light" ? "active" : ""}
                    onClick={() => { if (theme !== "light") toggleTheme(); }}
                  >☀ {t("settings.theme.light")}</button>
                </div>
              </div>

              <div className="settingsRow">
                <label>{t("toolbar.gpu")}</label>
                <div className="settingsControl settingsGpuControl">
                  <span className="settingsGpuName" title={gpuName}>{gpuName}</span>
                  <button onClick={onOpenGpuModal}>{t("settings.gpu.change")}</button>
                </div>
              </div>
            </div>
          )}

          {section === "cache" && (
            <div className="settingsContent">
              <div className="settingsSectionTitle">{t("settings.section.cache.title")}</div>

              <div className="cacheHeader">
                <span className="cacheTotal">
                  {t("settings.cache.total")}: <strong>{formatBytes(totalCacheBytes)}</strong>
                </span>
                <span className="cacheHeaderActions">
                  <button className="cacheBtn" onClick={refreshCache} disabled={cacheLoading}>
                    {t("settings.cache.refresh")}
                  </button>
                  <button
                    className="cacheBtn cacheBtnDanger"
                    onClick={handleClearAll}
                    disabled={cacheLoading || cacheEntries.length === 0}
                  >
                    {t("settings.cache.clearAll")}
                  </button>
                </span>
              </div>

              {cacheLoading && cacheEntries.length === 0 ? (
                <div className="cacheEmpty">{t("settings.cache.refresh")}…</div>
              ) : cacheEntries.length === 0 ? (
                <div className="cacheEmpty">{t("settings.cache.empty")}</div>
              ) : (
                <div className="cacheTable">
                  {cacheEntries.map((entry) => (
                    <div className="cacheRow" key={entry.dataset_id}>
                      <div className="cacheRowMain">
                        {renamingId === entry.dataset_id ? (
                          <>
                            <div className="cacheRename">
                              <input className="cacheRenameInput" value={renameValue} autoFocus
                                onChange={(e) => setRenameValue(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === "Enter") confirmRename(entry.dataset_id);
                                  else if (e.key === "Escape") cancelRename();
                                }} />
                              <button className="cacheBtn" onClick={() => confirmRename(entry.dataset_id)}>{t("confirm.ok")}</button>
                              <button className="cacheBtn" onClick={cancelRename}>{t("confirm.cancel")}</button>
                            </div>
                            {renameError && <div className="cacheRenameError">{renameError}</div>}
                          </>
                        ) : (
                          <div className="cacheName">{entry.dataset_id}</div>
                        )}
                      </div>
                      <div className="cacheMeta">
                        <span className="cacheMetaItem">{formatBytes(entry.size_bytes)}</span>
                        <span className="cacheMetaSep">·</span>
                        <span className="cacheMetaItem">
                          {entry.event_count.toLocaleString()} {t("settings.cache.events")}
                        </span>
                        <span className="cacheMetaSep">·</span>
                        <span className="cacheMetaItem">{entry.width}×{entry.height}</span>
                      </div>
                      <div className="cacheActions">
                        <button className="cacheBtn" onClick={() => handleOpen(entry)}>
                          {t("settings.cache.open")}
                        </button>
                        <button className="cacheBtn" onClick={() => startRename(entry)}>
                          {t("settings.cache.rename")}
                        </button>
                        <button className="cacheBtn" onClick={() => handleReveal(entry.dataset_id)}>
                          {t("settings.cache.reveal")}
                        </button>
                        <button className="cacheBtn" onClick={() => handleExport(entry.dataset_id)}>
                          {t("settings.cache.export")}
                        </button>
                        <button className="cacheBtn cacheBtnDanger" onClick={() => handleDelete(entry.dataset_id)}>
                          {t("settings.cache.delete")}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {section === "shortcuts" && (
            <div className="settingsContent">
              <div className="settingsSectionTitle">{t("settings.section.shortcuts.title")}</div>
              <dl className="settingsShortcuts">
                {SHORTCUTS.map(([keys, labelKey]) => (
                  <div className="settingsShortcutRow" key={keys}>
                    <kbd>{keys}</kbd>
                    <span>{t(labelKey)}</span>
                  </div>
                ))}
              </dl>
            </div>
          )}

          {section === "about" && (
            <div className="settingsContent">
              <div className="settingsSectionTitle">{t("settings.section.about.title")}</div>
              <div className="aboutWrap">
                {/* Hero: logo + name + version + tagline */}
                <div className="aboutHero">
                  <div className="aboutLogo" aria-hidden="true">
                    <svg viewBox="0 0 48 48" fill="none" stroke="#fff" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round">
                      <rect x="15" y="15" width="18" height="18" rx="4" />
                      <circle cx="24" cy="24" r="4.5" />
                      <circle cx="24" cy="24" r="1.4" fill="#fff" stroke="none" />
                      <path d="M19 15V9 M24 15V9 M29 15V9 M19 33v6 M24 33v6 M29 33v6 M15 19H9 M15 24H9 M15 29H9 M33 19h6 M33 24h6 M33 29h6" />
                    </svg>
                  </div>
                  <div className="aboutHeroText">
                    <div className="aboutTitle">Event Label Studio</div>
                    <span className="aboutVersionPill">v{__APP_VERSION__}</span>
                    <p className="aboutTagline">{t("settings.about.desc")}</p>
                  </div>
                </div>

                {/* Quick links */}
                <div className="aboutCards">
                  <a className="aboutCard" href={`${GITHUB_URL}#readme`} target="_blank" rel="noreferrer">
                    <span className="aboutCardIcon docs"><BookOpen size={20} /></span>
                    <div className="aboutCardTitle">{t("settings.about.docs.title")}</div>
                    <div className="aboutCardDesc">{t("settings.about.docs.desc")}</div>
                  </a>
                  <a className="aboutCard" href={GITHUB_URL} target="_blank" rel="noreferrer">
                    <span className="aboutCardIcon gh"><Github size={20} /></span>
                    <div className="aboutCardTitle">{t("settings.about.github.title")}</div>
                    <div className="aboutCardDesc">{t("settings.about.github.desc")}</div>
                  </a>
                </div>

                {/* Easter egg: grey line-art figure (pink-red on hover) that opens a hidden page. */}
                <div
                  className="aboutEgg"
                  style={{ WebkitMaskImage: `url(${eggMask})`, maskImage: `url(${eggMask})` }}
                  onClick={() => { window.location.href = "/egg.html"; }}
                  aria-hidden="true"
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Keyboard shortcuts shown in the Settings → Shortcuts section (moved here from
// the old Help menu). [keys, i18n label key].
const SHORTCUTS: [string, string][] = [
  ["Ctrl+Z", "menu.edit.undo"],
  ["Ctrl+Y", "menu.edit.redo"],
  ["Ctrl+S", "toolbar.load"],
  ["1 / 2 / 3", "panel.labels"],
  ["Enter", "panel.labels.assignSelected"],
  ["Delete", "menu.edit.clear"],
  ["F", "toolbar.selectRanges"],
  ["E", "toolbar.clear"],
];
