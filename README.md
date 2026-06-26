<p align="center">
	<h1 align="center">Event Label Studio</h1>
</p>

<p align="center">An interactive labelling tool for event-camera (neuromorphic) point-cloud data.</p>

<p align="center">
	<a href="#"><img src="https://img.shields.io/badge/version-v0.1.0-blue.svg"></a>
	<a href="#"><img src="https://img.shields.io/badge/python-3.10-3776AB.svg?logo=python&logoColor=white"></a>
	<a href="#"><img src="https://img.shields.io/badge/React-19-61DAFB.svg?logo=react&logoColor=white"></a>
	<a href="#"><img src="https://img.shields.io/badge/Three.js-0.182-000000.svg?logo=three.js&logoColor=white"></a>
	<a href="#"><img src="https://img.shields.io/badge/Flask-backend-000000.svg?logo=flask&logoColor=white"></a>
	<a href="#"><img src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>

## Introduction

Event Label Studio is an interactive tool for **labelling event-camera (neuromorphic) point-cloud data**. Event cameras do not produce frames — each pixel asynchronously emits an event `(x, y, t, polarity)` whenever its brightness changes, so a few seconds of recording is a **point cloud of tens of millions of events** in `(x, y, t)` space. You cannot draw boxes on it like an image, and general-purpose point-cloud tools are not built for the time axis, the polarity channel, or this scale.

This tool addresses that gap with a complete workflow — **load, visualise, filter and efficiently label** — backed by smart assists (a connected-components "wand" and semi-automatic propagation tracking) that do the heavy lifting and leave only the fine-tuning to the human.

* Decoupled front/back end: frontend `React 19 + TypeScript + Vite + Three.js`, backend `Flask + NumPy/SciPy`.
* One page, three ways to open it: a browser, a built-in desktop window (`pywebview`, Windows/macOS/Linux), or a one-click script.
* Built for scale: multi-chunk GPU rendering + LOD + time paging — **handles 40M+ events on a single machine**.
* Smart assists: voxel connected-components wand and semi-automatic propagation tracking (classical algorithms, human-in-the-loop).

## Features

1. **Data loading** — open `.es` (event-stream / DVS) recordings; decoded once into a compressed cache so reopening is near-instant; optional event decimation (low-memory cap).
2. **3D / XY views** — a 3D `(x, y, t)` point-cloud view plus an XY top-down view; XY supports rotation (0/90/180/270°), horizontal / vertical mirroring and frame-by-frame playback (0.25–8× speed, single-frame stepping).
3. **Per-axis stretch** — scale the X / Y / T axes independently in 3D to stretch the "time tube" for easier identification; zoom limits adapt to resolution / duration.
4. **Selection tools** — Pan, Box, Circle (XY), Lasso, and the **Wand** (voxel connected-components); tools auto-hide where they don't apply to the current view.
5. **Wand (adjustable voxel box)** — three always-on knobs directly size the "box": **voxel size (px) / voxel time (ms) / min events per voxel (min_pts)**; the `min_pts` threshold drops sparse noise bridges so dense data is not welded into one blob; parameter changes do not auto-apply — click *Apply*, or simply click a new point (which uses the current parameters).
6. **Ranges & filters** — dual-slider filtering on time / X / Y / polarity, with a live yellow preview box in 3D and a Load-to-commit / Reset flow.
7. **Label management** — create / rename / recolour / delete labels, single-label-per-point model, a live per-class histogram, and a confirmation dialog for destructive actions.
8. **Polarity colours & contrast** — a configurable base colour for unlabelled events, with Polarity Contrast shading ON/OFF polarity; preferences persist.
9. **Time Scroll** — scroll along the T axis to instantly preview the XY frame at that moment, to understand what the events represent.
10. **Propagate Track** — after selecting a target in XY, propagate the label slice-by-slice via constant-velocity prediction + a local ROI + identity scoring; it pauses on object crossings / sudden size jumps and hands control back to the user (no LLM / deep learning).
11. **Two-stage annotation workflow** — propagate in the XY view to grab the bulk, then refine missed points with the lasso in the 3D view, iterating until everything is labelled.
12. **Video export** — export the labelled XY animation to MP4 / WebM with configurable colour mode, background, frame window and frame rate, plus a first-frame preview.
13. **Data export** — export `NPZ` (lossless, full), `PLY` / `CSV` (preview and full), and a reloadable `project.json` project file.
14. **Cache & project management** — manage opened recordings (export, delete, reveal in folder) for auto-save and project management.
15. **Settings center** — General / Cache / Display / Shortcuts / About; central switches for language, theme, GPU and point size.
16. **Cross-platform desktop** — a `pywebview` native window; config / recent files are shared across browsers and the window (server-side shared prefs).
17. **Performance & large-scale rendering** — multi-chunk `THREE.Points` + Uint8 colour buffers + async chunked builds + LOD + time paging for smooth interaction at tens of millions of events.
18. **Usability** — English/Chinese i18n, light/dark theme, keyboard shortcuts, undo / redo, foolproof disabled states, a unified confirm dialog, and a **live FPS readout** in the bottom-right status bar.

## Architecture

| Layer | Stack |
|---|---|
| Frontend | React 19 · TypeScript · Vite · Three.js · in-house i18n / theming / binary point-cloud protocol |
| Backend | Python 3.10 · Flask · NumPy · SciPy · event-stream (decode) · imageio + ffmpeg (video) |
| Desktop | pywebview (Windows WebView2 / macOS WKWebView / Linux WebKitGTK) |

## Requirements

* Python 3.10 (recommended: create the conda env `eventcamera-blender` from `environment.yml`)
* Node.js ≥ 18 (with npm)
* Desktop mode also needs `pywebview` (Linux additionally needs the system WebKitGTK packages — see `prototype/backend/requirements-desktop.txt`)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/sonaradarcn/Event-Label-Studio.git && cd Event-Label-Studio

# 2. Create and activate the backend environment
conda env create -f environment.yml
conda activate eventcamera-blender

# 3. Start the backend (defaults to http://127.0.0.1:5050)
python prototype/backend/app.py

# 4. Start the frontend (dev mode, defaults to http://127.0.0.1:5173)
cd prototype/frontend
npm install
npm run dev
```

**Production / single-window** — build the frontend first and the backend will serve the static page:

```bash
cd prototype/frontend && npm run build      # outputs to dist/
python prototype/backend/app.py             # open http://127.0.0.1:5050

# Or open as a native desktop window (pip install -r prototype/backend/requirements-desktop.txt first)
python prototype/backend/desktop.py
```

**Windows one-click** — from the repo root, run `run.bat` (browser mode).

## Project Structure

```
Event-Label-Studio/
├─ prototype/
│  ├─ frontend/                # React + Three.js frontend
│  │  └─ src/{components, pointcloud, hooks, api, i18n}
│  └─ backend/                 # Flask backend
│     ├─ app.py                # REST API + static hosting (port 5050)
│     ├─ cache_store.py        # decode cache / LOD / chunking
│     ├─ labels.py             # label read/write / undo-redo
│     ├─ tracks.py             # semi-automatic propagation tracking
│     ├─ video_render.py       # video / single-frame rendering
│     └─ desktop.py            # pywebview desktop window
├─ environment.yml             # conda environment definition
├─ run.bat                     # Windows one-click launcher
└─ README.md
```

## License

Released under the [MIT](LICENSE) license.

## Acknowledgements

This project is an MSc dissertation for COMP66060 at the University of Manchester. Thanks to my supervisor, Dr. Alex Marcireau, for guidance on event-camera data labelling.
