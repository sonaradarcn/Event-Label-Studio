# Changelog

## v0.1.2

- **Projects**: name each project on import — the same recording can be imported
  as several independent projects. New projects start with no preset labels.
- **Project Manager** (Settings → renamed from "Cache"): rename projects; the
  original file path is no longer shown. New entry under File → Project Manager.
- **Recent Projects** now auto-scans the cache, so any project shows up there.
- Terminology: "Open" → **Import**, "Recent" → **Recent Projects**.
- **Screenshot** (copies the current 3D / XY view to the clipboard) and **Print**
  (transparent background, page-orientation prompt; axes & labels printed as
  crisp vector SVG in the PDF) in the File menu.
- **Select All** (Ctrl+A) and **Esc** to deselect; the point-size setting is now
  remembered across sessions.
- **Collapsible** left / right panels.
- Removed the Import / Export buttons from the toolbar (still in the File menu).

## v0.1.1

- Added an animated startup splash screen (theme-adaptive, light/dark).
- Redesigned the **Settings → About** page.
- Removed the experimental desktop (pywebview) build — the app runs in the browser.
- Fixed the Windows launcher (`run.bat`) to reliably use the project's conda environment.

## v0.1.0

- Initial release: event-camera `.es` loading, 3D / XY views, selection tools
  (pan / box / circle / lasso / wand), ranges & filters, label management,
  Time Scroll, semi-automatic propagation tracking, video & data export,
  FPS readout, English/Chinese UI, and light/dark themes.
