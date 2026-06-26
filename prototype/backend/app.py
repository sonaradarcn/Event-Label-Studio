from __future__ import annotations

import collections
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import zipfile
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np

from binary_api import iter_pack_arrays, pack_arrays  # noqa: F401
from cache_store import CacheStore
from config import CACHE_DIR, DEFAULT_SAMPLE, OUTPUT_DIR, PROTOTYPE_ROOT, RECORDINGS_DIR
from export import colors_for, export_outputs
from labels import LabelManager
from sampling import coordinate_positions, filter_indices, time_stratified_indices
from tracks import propagate_track
from video_render import render_event_frame, render_event_video


app = Flask(__name__)
store = CacheStore()
labels = LabelManager(store)
tasks: dict[str, dict] = {}

ALLOWED_ROOTS = [RECORDINGS_DIR.resolve(), CACHE_DIR.resolve(), OUTPUT_DIR.resolve()]

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# ─────────── Serve the built frontend (single-origin / desktop window) ───────────
# In dev the Vite server (:5173) proxies /api to here. For the packaged desktop
# app there is no Vite — Flask serves the built SPA from frontend/dist so the
# whole UI + API live on one origin. /api/* keeps its own (more specific) rules;
# this catch-all only returns static assets or the SPA index fallback.
FRONTEND_DIST = PROTOTYPE_ROOT / "frontend" / "dist"

# Windows maps .js -> text/plain in the registry, which Python's mimetypes picks
# up; Chromium/WebView2 then refuses to run the ES-module bundle (strict MIME)
# and the window stays blank/black. Force correct types for the static assets.
for _ext, _type in (
    (".js", "text/javascript"),
    (".mjs", "text/javascript"),
    (".css", "text/css"),
    (".json", "application/json"),
    (".svg", "image/svg+xml"),
    (".wasm", "application/wasm"),
):
    mimetypes.add_type(_type, _ext)


@app.get("/")
def _serve_index():
    return send_from_directory(FRONTEND_DIST, "index.html")


@app.get("/<path:path>")
def _serve_spa(path: str):
    if path.startswith("api/"):
        abort(404)  # never shadow the API
    if (FRONTEND_DIST / path).is_file():
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, "index.html")


# ─────────── Shared client prefs (recents + UI config) ───────────
# localStorage is per-origin, so the dev browser (:5173) and the desktop window
# (127.0.0.1:<random-port>) cannot see each other's recents/settings. The
# backend is the one thing every client shares, so we persist those prefs here
# as a simple JSON key→value bag. The frontend seeds localStorage from this on
# boot and mirrors writes back, so config is the same wherever it's opened.
PREFS_FILE = PROTOTYPE_ROOT / "app_prefs.json"
_prefs_lock = threading.Lock()


def _read_prefs() -> dict:
    try:
        return json.loads(PREFS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}


@app.get("/api/prefs")
def get_prefs():
    with _prefs_lock:
        return jsonify(_read_prefs())


@app.put("/api/prefs")
def put_prefs():
    patch = request.get_json(silent=True)
    if not isinstance(patch, dict):
        return jsonify({"error": "expected a JSON object"}), 400
    with _prefs_lock:
        prefs = _read_prefs()
        for key, value in patch.items():
            if value is None:
                prefs.pop(key, None)  # null = delete (e.g. clear recents)
            else:
                prefs[key] = value
        PREFS_FILE.write_text(json.dumps(prefs, indent=2, ensure_ascii=False), encoding="utf-8")
        return jsonify(prefs)


# ─────────────────── Browser dev-log telemetry ───────────────────
DEVLOG_FILE = CACHE_DIR / "devlog.jsonl"
DEVLOG_FILE.parent.mkdir(parents=True, exist_ok=True)
_devlog_lock = threading.Lock()


@app.post("/api/devlog")
def devlog_append():
    """Receive a batch of browser-side log entries and append them to a local
    jsonl file. Used to capture console errors, unhandled rejections, memory
    snapshots, etc. while the user interacts with the page."""
    import json as _json
    payload = request.get_json(force=True, silent=True) or []
    if not isinstance(payload, list):
        payload = [payload]
    with _devlog_lock:
        with DEVLOG_FILE.open("a", encoding="utf-8") as f:
            for entry in payload:
                if not isinstance(entry, dict):
                    entry = {"raw": entry}
                entry.setdefault("server_t", time.time())
                f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    return jsonify({"ok": True, "count": len(payload)})


@app.post("/api/devlog/clear")
def devlog_clear():
    with _devlog_lock:
        if DEVLOG_FILE.exists():
            DEVLOG_FILE.unlink()
    return jsonify({"ok": True})


def _validate_path(path: Path) -> Path:
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise ValueError(f"Path {resolved} is outside allowed directories")
    return resolved


def _parse_client_path(raw: str) -> Path:
    """Parse a path string sent by the browser.

    Recents persisted by an earlier WSL/Linux session arrive as
    "/mnt/<drive>/...", which Windows would otherwise resolve to
    "<cwd drive>:\\mnt\\..." and reject as outside the allowed roots.
    """
    m = re.fullmatch(r"/mnt/([a-zA-Z])/(.*)", raw)
    if m:
        raw = f"{m.group(1).upper()}:/{m.group(2)}"
    path = Path(raw)
    if not path.is_absolute():
        path = (RECORDINGS_DIR / path).resolve()
    return _validate_path(path)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "http://127.0.0.1:5173"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return response


@app.route("/api/<path:_path>", methods=["OPTIONS"])
def options(_path):
    return ("", 204)


@app.errorhandler(ValueError)
def handle_value_error(exc):
    return jsonify({"error": str(exc)}), 400


@app.errorhandler(FileNotFoundError)
def handle_not_found(exc):
    return jsonify({"error": str(exc)}), 404


@app.get("/api/recordings")
def recordings():
    files = [{"name": p.name, "path": str(p)} for p in sorted(RECORDINGS_DIR.glob("*.es"))]
    return jsonify(files)


@app.post("/api/datasets/upload")
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    dest = RECORDINGS_DIR / secure_filename(f.filename)
    f.save(str(dest))
    return jsonify({"path": str(dest)})


@app.get("/api/datasets")
def datasets():
    return jsonify(store.list_datasets())


def _folder_size_and_mtime(folder: Path) -> tuple[int, float]:
    """Sum st_size of every file in `folder` (recursive) and find the most
    recent file mtime. Falls back to the folder's own mtime when it has no
    files. Returns (size_bytes, modified_epoch_seconds)."""
    total = 0
    latest = 0.0
    for path in folder.rglob("*"):
        try:
            st = path.stat()
        except OSError:
            continue
        if path.is_file():
            total += st.st_size
        if st.st_mtime > latest:
            latest = st.st_mtime
    if latest == 0.0:
        try:
            latest = folder.stat().st_mtime
        except OSError:
            latest = 0.0
    return total, latest


@app.get("/api/cache")
def cache_index():
    """List every cached dataset, augmenting each meta.json with the dataset's
    on-disk footprint (size_bytes) and most-recent modification time (modified).
    Returns a JSON array of CacheEntry objects."""
    entries = []
    for meta in store.list_datasets():
        dataset_id = str(meta.get("dataset_id", ""))
        folder = store.dataset_dir(dataset_id)
        size_bytes, modified = _folder_size_and_mtime(folder)
        entries.append({
            "dataset_id": dataset_id,
            "source_path": str(meta.get("source_path", "")),
            "width": int(meta.get("width", 0) or 0),
            "height": int(meta.get("height", 0) or 0),
            "event_count": int(meta.get("event_count", 0) or 0),
            "t_min_us": int(meta.get("t_min_us", 0) or 0),
            "t_max_us": int(meta.get("t_max_us", 0) or 0),
            "size_bytes": int(size_bytes),
            "modified": float(modified),
        })
    return jsonify(entries)


@app.delete("/api/datasets/<dataset_id>")
def delete_dataset(dataset_id: str):
    """Delete a dataset's cache folder. The resolved path is verified to live
    inside CACHE_DIR before anything is removed."""
    folder = store.dataset_dir(dataset_id).resolve()
    cache_root = CACHE_DIR.resolve()
    if folder == cache_root or not folder.is_relative_to(cache_root):
        return jsonify({"ok": False, "error": "Path is outside the cache directory"}), 400
    if not folder.exists():
        return jsonify({"ok": False, "error": "Dataset not found"}), 404
    try:
        shutil.rmtree(folder)
    except PermissionError as exc:
        # Windows holds a lock on memmapped files while the dataset is open.
        return jsonify({"ok": False, "error": f"Cannot delete while in use: {exc}"}), 409
    return jsonify({"ok": True})


@app.post("/api/datasets/<dataset_id>/reveal")
def reveal_dataset(dataset_id: str):
    """Open the dataset's cache folder in the OS file manager. The backend runs
    locally, so this opens the explorer on the same machine."""
    folder = store.dataset_dir(dataset_id).resolve()
    if not folder.exists():
        return jsonify({"ok": False, "error": "Dataset not found"}), 404
    path = str(folder)
    if sys.platform.startswith("win"):
        try:
            os.startfile(path)  # noqa: S606 — local-only file manager open
        except OSError:
            subprocess.Popen(["explorer", path])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
    return jsonify({"ok": True})


@app.post("/api/datasets/open")
def open_dataset():
    body = request.get_json(force=True)
    path = _parse_client_path(str(body["path"]))
    meta = store.open_dataset(path)
    return jsonify({"dataset_id": meta["dataset_id"], "status": "ready", "meta": meta})


@app.post("/api/datasets/open_async")
def open_dataset_async():
    body = request.get_json(force=True)
    path = _parse_client_path(str(body["path"]))
    task_id = uuid.uuid4().hex
    tasks[task_id] = {"task_id": task_id, "status": "running", "stage": "queued", "progress": 0.0, "message": "Queued", "started_at": time.time()}

    def progress(stage: str, value: float, message: str) -> None:
        tasks[task_id].update({"stage": stage, "progress": max(0.0, min(1.0, value)), "message": message, "updated_at": time.time()})

    def worker() -> None:
        try:
            meta = store.open_dataset(path, progress=progress)
            tasks[task_id].update({"status": "ready", "stage": "ready", "progress": 1.0, "message": "Dataset ready", "dataset_id": meta["dataset_id"], "meta": meta, "finished_at": time.time()})
        except Exception as exc:
            tasks[task_id].update({"status": "failed", "stage": "error", "progress": 1.0, "message": str(exc), "finished_at": time.time()})

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"task_id": task_id, "status": "running"})


@app.get("/api/tasks/<task_id>")
def task_status(task_id: str):
    task = tasks.get(task_id)
    if task is None:
        return jsonify({"error": "unknown task"}), 404
    return jsonify(task)


@app.get("/api/datasets/<dataset_id>/meta")
def meta(dataset_id: str):
    return jsonify(store.meta(dataset_id))


@app.get("/api/datasets/<dataset_id>/labels/schema")
def get_schema(dataset_id: str):
    return jsonify(store.schema(dataset_id))


@app.put("/api/datasets/<dataset_id>/labels/schema")
def put_schema(dataset_id: str):
    schema = request.get_json(force=True)
    if "labels" not in schema or not isinstance(schema["labels"], list):
        return jsonify({"error": "Schema must contain a 'labels' list"}), 400
    store.save_schema(dataset_id, schema)
    return jsonify(schema)


@app.post("/api/datasets/<dataset_id>/labels/update")
def update_labels(dataset_id: str):
    body = request.get_json(force=True)
    if len(body.get("event_ids", [])) > 50_000:
        return jsonify({"error": "JSON label update is capped at 50000 IDs. Use /labels/update_by_filter for large selections."}), 413
    return jsonify(labels.update(dataset_id, body.get("event_ids", []), int(body.get("label", -1)), body.get("operation", "assign")))


@app.post("/api/datasets/<dataset_id>/labels/update_by_filter")
def update_labels_by_filter(dataset_id: str):
    body = request.get_json(force=True)
    return jsonify(labels.update_by_filter(
        dataset_id,
        label=int(body.get("label", -1)),
        operation=body.get("operation", "assign"),
        start_us=body.get("start_us"),
        end_us=body.get("end_us"),
        polarity=body.get("polarity", "all"),
        x_min=body.get("x_min"),
        x_max=body.get("x_max"),
        y_min=body.get("y_min"),
        y_max=body.get("y_max"),
    ))


@app.post("/api/datasets/<dataset_id>/labels/update.bin")
def update_labels_bin(dataset_id: str):
    """Single binary label update — body is a little-endian uint32 array of
    event ids. One request, one lock, one in-place memmap write (no 50k JSON
    batching, no whole-array re-compress). Returns updated count, new
    label_version and a stats_delta the client folds into its counts."""
    label = request.args.get("label", default=-1, type=int)
    operation = request.args.get("operation", default="assign", type=str)
    raw = request.get_data(cache=False, as_text=False)
    if len(raw) % 4 != 0:
        return jsonify({"error": "Body must be little-endian uint32 IDs"}), 400
    ids = np.frombuffer(raw, dtype="<u4").astype(np.int64, copy=False)
    return jsonify(labels.update_array(dataset_id, ids, label, operation))


@app.post("/api/datasets/<dataset_id>/labels/undo")
def undo(dataset_id: str):
    return jsonify(labels.undo(dataset_id))


@app.post("/api/datasets/<dataset_id>/labels/redo")
def redo(dataset_id: str):
    return jsonify(labels.redo(dataset_id))


@app.get("/api/datasets/<dataset_id>/points.bin")
def points_bin(dataset_id: str):
    m = store.meta(dataset_id)
    # `sample` is optional. Omit / 0 / negative → full detail (no subsampling).
    sample_raw = request.args.get("sample", default=None, type=int)
    sample: int | None = sample_raw if sample_raw is not None and sample_raw > 0 else None

    use_lod = request.args.get("lod", default="false") == "true"
    lod_arrays = None
    if use_lod:
        # LOD path always needs a positive sample target.
        lod_arrays = store.lod_arrays(dataset_id, sample or DEFAULT_SAMPLE)

    if lod_arrays is not None:
        arrays = lod_arrays
        schema = store.schema(dataset_id)["labels"]
        positions = coordinate_positions(arrays, np.arange(len(arrays["event_id"]), dtype=np.uint64), m["width"], m["height"], m["t_min_us"], m["t_max_us"])
        colors = colors_for(arrays, np.arange(len(arrays["event_id"]), dtype=np.uint64), schema)
        all_idx = np.arange(len(arrays["event_id"]), dtype=np.uint64)
        pairs = [
            ("positions", positions.astype("float32")),
            ("colors", colors.astype("uint8")),
            ("event_id", arrays["event_id"][all_idx.astype(np.int64)].astype("uint64")),
            ("t_us", arrays["t_us"][all_idx.astype(np.int64)].astype("uint64")),
            ("polarity", arrays["polarity"][all_idx.astype(np.int64)].astype("uint8")),
            ("label", arrays["label"][all_idx.astype(np.int64)].astype("int16")),
        ]
        return Response(iter_pack_arrays(pairs), mimetype="application/octet-stream")

    arrays = store.arrays(dataset_id)
    start_us = request.args.get("start_us", type=int)
    end_us = request.args.get("end_us", type=int)
    polarity = request.args.get("polarity", default="all")
    indices = filter_indices(arrays, start_us, end_us, polarity, sample)
    positions = coordinate_positions(arrays, indices, m["width"], m["height"], m["t_min_us"], m["t_max_us"])
    schema = store.schema(dataset_id)["labels"]
    colors = colors_for(arrays, indices, schema)
    idx = indices.astype("int64")
    pairs = [
        ("positions", positions.astype("float32")),
        ("colors", colors.astype("uint8")),
        ("event_id", arrays["event_id"][idx].astype("uint64")),
        ("t_us", arrays["t_us"][idx].astype("uint64")),
        ("polarity", arrays["polarity"][idx].astype("uint8")),
        ("label", arrays["label"][idx].astype("int16")),
    ]
    # Stream the payload to the client so neither side has to hold the whole
    # ~500 MB blob in RAM at once.
    return Response(iter_pack_arrays(pairs), mimetype="application/octet-stream")


# ─────────────────── Chunked dataset access ───────────────────
# Time-binned chunk index lets the frontend page only the visible window into
# memory instead of materialising the whole 16M-event payload at once.
DEFAULT_CHUNK_DURATION_US = 500_000  # 0.5 second per chunk

# Manifest is a pure function of the (immutable) event timestamps + chunk size,
# so cache it per (dataset_id, chunk_us). Without this, every single chunk.bin
# request re-read the full t_us array and re-ran searchsorted. Label edits do
# NOT invalidate this — the manifest carries no label data.
_manifest_cache: dict[tuple[str, int], dict] = {}


def _chunk_manifest(dataset_id: str, chunk_us: int = DEFAULT_CHUNK_DURATION_US) -> dict:
    cache_key = (dataset_id, int(chunk_us))
    cached = _manifest_cache.get(cache_key)
    if cached is not None:
        return cached
    m = store.meta(dataset_id)
    arr_t = store.arrays(dataset_id)["t_us"]
    t_min = int(m["t_min_us"])
    t_max = int(m["t_max_us"])
    duration_us = max(0, t_max - t_min)
    if duration_us == 0:
        empty = {"dataset_id": dataset_id, "total_events": int(len(arr_t)),
                 "duration_us": 0, "t_min_us": t_min, "t_max_us": t_max,
                 "chunk_duration_us": chunk_us, "chunks": []}
        _manifest_cache[cache_key] = empty
        return empty
    n_chunks = (duration_us + chunk_us - 1) // chunk_us
    edges = t_min + np.arange(n_chunks + 1, dtype=np.int64) * chunk_us
    edges[-1] = max(edges[-1], t_max + 1)  # ensure last chunk includes t_max
    # Vectorised: searchsorted twice gives [start_index, end_index) per chunk.
    starts = np.searchsorted(arr_t, edges[:-1], side="left")
    ends = np.searchsorted(arr_t, edges[1:], side="left")
    chunks = []
    for i in range(int(n_chunks)):
        cs, ce = int(starts[i]), int(ends[i])
        if cs == ce:
            continue  # skip empty chunks (dead air with no events)
        chunks.append({
            "chunk_id": int(i),
            "start_us": int(edges[i]),
            "end_us": int(edges[i + 1]),
            "event_start_index": cs,
            "event_count": ce - cs,
        })
    result = {
        "dataset_id": dataset_id,
        "total_events": int(len(arr_t)),
        "duration_us": int(duration_us),
        "t_min_us": t_min,
        "t_max_us": t_max,
        "chunk_duration_us": int(chunk_us),
        "n_chunks": len(chunks),
        "width": int(m["width"]),
        "height": int(m["height"]),
        "chunks": chunks,
    }
    _manifest_cache[cache_key] = result
    return result


def _adaptive_chunk_us(dataset_id: str, target_events: int = 250_000) -> int:
    """Pick a chunk duration that yields ≈target_events per chunk based on the
    dataset's average event rate. Bounded to [10ms, 2s]."""
    m = store.meta(dataset_id)
    duration_us = max(1, int(m["t_max_us"]) - int(m["t_min_us"]))
    n_events = max(1, int(m["event_count"]))
    rate = n_events / duration_us  # events per microsecond
    chunk_us = int(target_events / max(rate, 1e-9))
    return max(10_000, min(chunk_us, 2_000_000))


@app.get("/api/datasets/<dataset_id>/manifest")
def chunk_manifest(dataset_id: str):
    raw = request.args.get("chunk_us", type=int)
    if raw is None:
        chunk_us = _adaptive_chunk_us(dataset_id)
    else:
        chunk_us = max(10_000, min(int(raw), 5_000_000))  # clamp 10ms..5s
    return jsonify(_chunk_manifest(dataset_id, chunk_us))


@app.get("/api/datasets/<dataset_id>/chunks/<int:chunk_id>.bin")
def chunk_bin(dataset_id: str, chunk_id: int):
    """Streaming binary payload for a single time chunk. Same wire format as
    /points.bin so the frontend can use the existing parser."""
    chunk_us = request.args.get("chunk_us", default=DEFAULT_CHUNK_DURATION_US, type=int)
    chunk_us = max(50_000, min(int(chunk_us), 5_000_000))
    manifest = _chunk_manifest(dataset_id, chunk_us)
    chunk = next((c for c in manifest["chunks"] if c["chunk_id"] == chunk_id), None)
    if chunk is None:
        return ("Chunk not found", 404)

    m = store.meta(dataset_id)
    arrays = store.arrays(dataset_id)
    cs, ce = chunk["event_start_index"], chunk["event_start_index"] + chunk["event_count"]
    indices = np.arange(cs, ce, dtype=np.uint64)
    # Global time range → chunks stack correctly along the time (Z) axis.
    positions = coordinate_positions(arrays, indices, int(m["width"]), int(m["height"]), m["t_min_us"], m["t_max_us"])
    idx = indices.astype("int64")
    # `colors` is intentionally omitted — the frontend recomputes colours on the
    # GPU side from label/polarity and skips this field on parse, so sending it
    # was 3 B/event of pure waste plus a colors_for() pass per chunk.
    pairs = [
        ("positions", positions.astype("float32")),
        ("event_id", arrays["event_id"][idx].astype("uint64")),
        ("t_us", arrays["t_us"][idx].astype("uint64")),
        ("polarity", arrays["polarity"][idx].astype("uint8")),
        ("label", arrays["label"][idx].astype("int16")),
    ]
    return Response(iter_pack_arrays(pairs), mimetype="application/octet-stream")


@app.post("/api/datasets/<dataset_id>/tracks/propagate")
def tracks_propagate(dataset_id: str):
    """Propagation tracking: follow a seed target across time slices with a
    constant-velocity + local-ROI baseline, pausing at the first low-confidence
    slice. Stateless in v1 — the client owns the track session and freezes the
    accepted events via the normal binary label-update path."""
    body = request.get_json(force=True) or {}
    seed = np.asarray(body.get("seed_event_ids", []), dtype=np.int64)
    arrays = store.arrays(dataset_id)
    result = propagate_track(
        arrays, seed,
        step_us=int(body.get("step_us", 33_333)),
        direction=str(body.get("direction", "forward")),
        max_slices=int(body.get("max_slices", 60)),
        stop_on_low=bool(body.get("stop_on_low", True)),
        conf_threshold=float(body.get("conf_threshold", 0.40)),
    )
    return jsonify(result)


# ── Wand voxel segmentation (min-events-per-voxel occupancy) ───────────────────
# A voxel counts as "occupied" only if it holds >= min_pts events. On dense
# recordings (rain/snow/rocket) the old ">=1 event" rule let sparse single-event
# voxels bridge everything into one giant component; requiring a few events per
# voxel drops those noise bridges so distinct objects separate (e.g. rain's
# largest component falls 71%->26% at min_pts=2). Computing it over the full
# (16-41M-event) cloud takes ~1-2s, so memoise per (dataset, voxel, min_pts) —
# the geometry never changes, so label edits don't invalidate it.
_WAND_VOXEL_CACHE: dict = {}
_WAND_SHIFT_X, _WAND_SHIFT_Y, _WAND_MASK20 = 40, 20, (1 << 20) - 1


def _wand_voxel_struct(dataset_id, x, y, t, vxy, vt_us, min_pts):
    """Return (comp_per_event int32 [-1 = sparse voxel], dense_per_event bool).

    comp_per_event: connected-component id (26-connectivity over occupied voxels)
    so the wand can return the seed's whole object in O(1) after the first call.
    dense_per_event: whether each event sits in an occupied (>= min_pts) voxel,
    used as a density gate by the radius BFS.
    """
    key = (dataset_id, int(vxy), int(vt_us), int(min_pts))
    cached = _WAND_VOXEL_CACHE.get(key)
    if cached is not None:
        return cached
    vx = (x.astype(np.int64) // int(vxy))
    vy = (y.astype(np.int64) // int(vxy))
    vt = ((t.astype(np.int64) - int(t.min())) // int(vt_us))
    keys = (vx << _WAND_SHIFT_X) | (vy << _WAND_SHIFT_Y) | vt
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    occ = counts >= int(min_pts)
    m = int(uniq.size)
    uniq_list = uniq.tolist()
    k2i = {int(k): i for i, k in enumerate(uniq_list)}
    comp = np.full(m, -1, dtype=np.int64)
    offs = [(dx, dy, dt) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dt in (-1, 0, 1) if not (dx == 0 and dy == 0 and dt == 0)]
    cid = 0
    for s in range(m):
        if not occ[s] or comp[s] != -1:
            continue
        comp[s] = cid
        stack = [s]
        while stack:
            kk = uniq_list[stack.pop()]
            cvx, cvy, cvt = kk >> _WAND_SHIFT_X, (kk >> _WAND_SHIFT_Y) & _WAND_MASK20, kk & _WAND_MASK20
            for dx, dy, dt in offs:
                nk = ((cvx + dx) << _WAND_SHIFT_X) | ((cvy + dy) << _WAND_SHIFT_Y) | (cvt + dt)
                j = k2i.get(nk)
                if j is not None and occ[j] and comp[j] == -1:
                    comp[j] = cid
                    stack.append(j)
        cid += 1
    comp_per_event = comp[inv].astype(np.int32)
    dense_per_event = np.asarray(occ)[inv]
    if len(_WAND_VOXEL_CACHE) >= 3:   # keep memory bounded (a few datasets/params)
        _WAND_VOXEL_CACHE.clear()
    res = (comp_per_event, dense_per_event)
    _WAND_VOXEL_CACHE[key] = res
    return res


@app.post("/api/datasets/<dataset_id>/select_component")
def select_component(dataset_id: str):
    """Spatiotemporal connected-component "wand" selection.

    Click one event → BFS over the proximity graph (events are connected
    if their pixel distance ≤ r_xy AND their time gap ≤ r_t_us). Returns
    every event reachable from the seed, capped by `max_size` and a
    wall-clock budget (the radii control whether a click pulls in a
    coherent blob, a whole motion trail, or a runaway region).

    Inputs:
      seed_event_id    — clicked event's global id.
      r_xy             — pixel-space neighbour radius. -1 = no spatial cap.
      r_t_us           — temporal neighbour radius (microseconds). -1 = no
                         temporal cap.
      polarity_match   — if true, only walk events with the seed's polarity.
      max_size         — cap on component size.
      time_budget_s    — wall-clock cap; large radii can balloon BFS into
                         the millions of events on a single-threaded Flask.
    """
    body = request.get_json(force=True) or {}
    seed = int(body["seed_event_id"])
    auto = bool(body.get("auto", False))
    # Min events per voxel for a voxel to count as "occupied" (1 = legacy
    # behaviour; >1 drops sparse noise bridges on dense data). Plus the voxel
    # size used by the density gate / unlabelled segmentation.
    min_pts = max(1, int(body.get("min_pts", 2)))
    voxel_xy = max(1, int(body.get("voxel_xy", 15)))
    voxel_t_us = max(1, int(body.get("voxel_t_us", 100_000)))

    arrays = store.arrays(dataset_id)
    t = arrays["t_us"]
    x = arrays["x"]
    y = arrays["y"]
    p = arrays["polarity"]
    label = arrays["label"]
    n = len(t)
    if seed < 0 or seed >= n:
        return jsonify({"error": "seed out of range"}), 400

    started = time.monotonic()

    # ── Foolproof "auto" mode ──
    # When the seed has a label, the user means "select this 3D blob".
    # Strategy: iterative AABB growth restricted to same-label events.
    # Start with a small box around the seed; each round, snap the box to
    # the min/max of the events of the same label currently inside it,
    # plus a small buffer that bridges typical sensor gaps. Converges in
    # a handful of iterations and naturally stops at the blob boundary —
    # so two separate same-label blobs (each well-isolated in space/time)
    # do NOT merge.
    if auto:
        seed_label = int(label[seed])
        if seed_label >= 0:
            sl_idx = np.flatnonzero(label == seed_label)
            if sl_idx.size == 0:
                return jsonify({"event_ids": [int(arrays["event_id"][seed])],
                                "size": 1, "truncated": False, "elapsed_s": 0.0})
            # ── Voxel BFS on same-label events ──
            # Bin events into a (voxel_xy × voxel_xy × voxel_t_us) grid; a voxel
            # is "occupied" if it holds >= min_pts same-label events. BFS over
            # 26-neighbour adjacency from the seed's voxel collects the connected
            # blob. The voxel size is the largest internal gap still considered
            # "connected"; min_pts drops sparse voxels so noise doesn't bridge.
            # All three are user-set (defaults 15 px / 100 ms / min_pts=2).
            VXY = voxel_xy
            VT = voxel_t_us
            t0 = int(t.min())   # relative time so vt fits the bit budget
            sl_vx = (x[sl_idx].astype(np.int64) // VXY)
            sl_vy = (y[sl_idx].astype(np.int64) // VXY)
            sl_vt = ((t[sl_idx].astype(np.int64) - t0) // VT)
            # Pack (vx, vy, vt) into one int64 key — 20 bits each is ample for
            # small voxels on long recordings.
            SHIFT_X, SHIFT_Y = 40, 20
            MASK20 = (1 << 20) - 1
            sl_keys = (sl_vx << SHIFT_X) | (sl_vy << SHIFT_Y) | sl_vt
            if min_pts > 1:
                _uk, _uc = np.unique(sl_keys, return_counts=True)
                occupied = set(_uk[_uc >= min_pts].tolist())
            else:
                occupied = set(sl_keys.tolist())
            seed_vx = int(x[seed]) // VXY
            seed_vy = int(y[seed]) // VXY
            seed_vt = (int(t[seed]) - t0) // VT
            seed_key = (seed_vx << SHIFT_X) | (seed_vy << SHIFT_Y) | seed_vt
            visited: set[int] = {seed_key}
            queue: list[int] = [seed_key]
            qi = 0
            while qi < len(queue):
                k = queue[qi]
                qi += 1
                cx_v = k >> SHIFT_X
                cy_v = (k >> SHIFT_Y) & MASK20
                ct_v = k & MASK20
                for ddx in (-1, 0, 1):
                    nx = cx_v + ddx
                    for ddy in (-1, 0, 1):
                        ny = cy_v + ddy
                        for ddt in (-1, 0, 1):
                            if ddx == 0 and ddy == 0 and ddt == 0:
                                continue
                            nt = ct_v + ddt
                            nk = (nx << SHIFT_X) | (ny << SHIFT_Y) | nt
                            if nk in occupied and nk not in visited:
                                visited.add(nk)
                                queue.append(nk)
            # Map visited voxels back to event indices: any sl_idx event
            # whose voxel is in `visited` is selected.
            visited_arr = np.fromiter(visited, dtype=np.int64, count=len(visited))
            in_mask = np.isin(sl_keys, visited_arr)
            selected = sl_idx[in_mask]
            ids = arrays["event_id"][selected].astype(np.int64).tolist()
            return jsonify({
                "event_ids": ids,
                "size": int(selected.size),
                "truncated": False,
                "truncated_reason": None,
                "elapsed_s": time.monotonic() - started,
                "mode": "auto_voxel_bfs",
                "voxels_visited": len(visited),
                "voxels_total": len(occupied),
                "seed_label": seed_label,
                "min_pts": min_pts,
                "voxel_xy": voxel_xy,
                "voxel_t_us": voxel_t_us,
            })
        # ── Unlabelled seed: voxel connected-components with a min_pts gate ──
        # The old radius BFS ballooned on dense data (a click could engulf the
        # whole frame). Segment the cloud into voxel components where a voxel is
        # occupied only with >= min_pts events, then return the seed's component.
        # O(1) per click after the (cached) first segmentation.
        max_size = 2_000_000
        comp_pe, _dense = _wand_voxel_struct(dataset_id, x, y, t, voxel_xy, voxel_t_us, min_pts)
        c = int(comp_pe[seed])
        if c < 0:
            # Seed sits in a too-sparse voxel — return just the seed's voxel mates.
            seed_keys = ((int(x[seed]) // voxel_xy) << _WAND_SHIFT_X) | \
                        ((int(y[seed]) // voxel_xy) << _WAND_SHIFT_Y) | \
                        ((int(t[seed]) - int(t.min())) // voxel_t_us)
            vx = (x.astype(np.int64) // voxel_xy)
            vy = (y.astype(np.int64) // voxel_xy)
            vt = ((t.astype(np.int64) - int(t.min())) // voxel_t_us)
            sel = np.flatnonzero(((vx << _WAND_SHIFT_X) | (vy << _WAND_SHIFT_Y) | vt) == seed_keys)
        else:
            sel = np.flatnonzero(comp_pe == c)
        truncated = sel.size > max_size
        if truncated:
            sel = sel[:max_size]
        ids = arrays["event_id"][sel].astype(np.int64).tolist()
        return jsonify({
            "event_ids": ids, "size": int(sel.size), "truncated": truncated,
            "truncated_reason": "max_size" if truncated else None,
            "elapsed_s": time.monotonic() - started, "mode": "auto_voxel_seg",
            "min_pts": min_pts, "voxel_xy": voxel_xy, "voxel_t_us": voxel_t_us,
            "n_clusters": int(comp_pe.max()) + 1 if comp_pe.max() >= 0 else 0,
        })
    else:
        r_xy = float(body.get("r_xy", 5.0))
        r_t_us = int(body.get("r_t_us", 30_000))
        polarity_match = bool(body.get("polarity_match", False))
        label_match = bool(body.get("label_match", False))
        max_size = int(body.get("max_size", 500_000))
        time_budget_s = float(body.get("time_budget_s", 3.0))
    unlimited_xy = r_xy < 0
    unlimited_t = r_t_us < 0

    target_pol = int(p[seed]) if polarity_match else None
    target_label = int(label[seed]) if label_match else None
    r_xy_sq = r_xy * r_xy

    # Density gate: when min_pts > 1, only flood through events that sit in an
    # occupied (>= min_pts) voxel, so the BFS can't bridge across sparse noise.
    # min_pts == 1 leaves `dense` all-True → identical to the legacy behaviour.
    dense_gate = None
    if min_pts > 1:
        _comp, dense_gate = _wand_voxel_struct(dataset_id, x, y, t, voxel_xy, voxel_t_us, min_pts)

    visited = np.zeros(n, dtype=bool)
    visited[seed] = True
    component_arr = np.empty(max_size, dtype=np.int64)
    component_arr[0] = seed
    component_len = 1
    queue: collections.deque[int] = collections.deque([seed])

    truncated_reason: str | None = None

    while queue:
        if component_len >= max_size:
            truncated_reason = "max_size"
            break
        if time.monotonic() - started > time_budget_s:
            truncated_reason = "time_budget"
            break
        i = queue.popleft()
        ti = int(t[i])
        if unlimited_t:
            lo, hi = 0, n
        else:
            lo = int(np.searchsorted(t, ti - r_t_us, side="left"))
            hi = int(np.searchsorted(t, ti + r_t_us, side="right"))
        if lo >= hi:
            continue
        v_slice = visited[lo:hi]
        if v_slice.all():
            continue
        not_visited_local = np.where(~v_slice)[0]
        if not_visited_local.size == 0:
            continue
        cand = not_visited_local + lo
        if unlimited_xy:
            mask = np.ones(cand.shape, dtype=bool)
        else:
            dx = x[cand].astype(np.int32) - int(x[i])
            dy = y[cand].astype(np.int32) - int(y[i])
            mask = (dx * dx + dy * dy) <= r_xy_sq
        if polarity_match:
            mask &= (p[cand] == target_pol)
        if label_match:
            mask &= (label[cand] == target_label)
        if dense_gate is not None:
            mask &= dense_gate[cand]
        new_idx = cand[mask]
        if new_idx.size == 0:
            continue
        remaining = max_size - component_len
        if new_idx.size > remaining:
            new_idx = new_idx[:remaining]
        visited[new_idx] = True
        component_arr[component_len:component_len + new_idx.size] = new_idx
        component_len += int(new_idx.size)
        queue.extend(new_idx.tolist())

    component_view = component_arr[:component_len]
    ids = arrays["event_id"][component_view].astype(np.int64).tolist()
    truncated = truncated_reason is not None
    return jsonify({
        "event_ids": ids,
        "size": component_len,
        "truncated": truncated,
        "truncated_reason": truncated_reason,
        "elapsed_s": time.monotonic() - started,
        "params_used": {
            "r_xy": r_xy, "r_t_us": r_t_us,
            "label_match": label_match, "polarity_match": polarity_match,
            "max_size": max_size,
        },
    })


@app.get("/api/datasets/<dataset_id>/stats")
def dataset_stats(dataset_id: str):
    m = store.meta(dataset_id)
    arrays = store.arrays(dataset_id)
    label = arrays["label"]
    schema = store.schema(dataset_id)["labels"]
    total = len(label)
    unlabelled = int(np.sum(label == -1))
    per_class = {}
    for cls in schema:
        cid = cls["id"]
        per_class[cls["name"]] = int(np.sum(label == cid))
    return jsonify({"total": total, "unlabelled": unlabelled, "per_class": per_class})


@app.post("/api/datasets/<dataset_id>/export")
def export_dataset(dataset_id: str):
    body = request.get_json(force=True, silent=True) or {}
    do_sample = bool(body.get("sample", False))
    sample_size = int(body.get("sample_size", DEFAULT_SAMPLE))
    only_labelled = bool(body.get("only_labelled", False))
    include_npz = bool(body.get("include_npz", True))
    include_ply = bool(body.get("include_ply", True))
    include_csv = bool(body.get("include_csv", True))

    m = store.meta(dataset_id)
    arrays = store.arrays(dataset_id)
    n = len(arrays["event_id"])
    schema = store.schema(dataset_id)["labels"]

    # Authoritative export set: every event, or only the labelled ones.
    if only_labelled:
        npz_indices = np.flatnonzero(arrays["label"] != -1).astype(np.uint64)
    else:
        npz_indices = np.arange(n, dtype=np.uint64)

    # Visualisation set for .ply/.csv: the export set, optionally sub-sampled.
    if do_sample and sample_size > 0 and len(npz_indices) > sample_size:
        sub = time_stratified_indices(arrays["t_us"][npz_indices.astype(np.int64)], sample_size)
        viz_indices = npz_indices[sub]
        sampled = True
    else:
        viz_indices = npz_indices
        sampled = False

    # coordinate_positions divides by the time span, so guard the empty case
    # (e.g. only_labelled with nothing labelled yet).
    if len(viz_indices) > 0:
        positions = coordinate_positions(arrays, viz_indices, m["width"], m["height"], m["t_min_us"], m["t_max_us"])
        colors = colors_for(arrays, viz_indices, schema)
    else:
        positions = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)

    result = export_outputs(
        OUTPUT_DIR, dataset_id, m, arrays, positions, colors, viz_indices, npz_indices, schema,
        sampled=sampled, only_labelled=only_labelled,
        include_npz=include_npz, include_ply=include_ply, include_csv=include_csv,
    )
    # Bundle the dataset's output folder into a single zip for one-click download.
    target_dir = OUTPUT_DIR / dataset_id
    zip_path = OUTPUT_DIR / f"{dataset_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(target_dir.iterdir()):
            if file.is_file():
                zf.write(file, arcname=file.name)
    result["zip"] = str(zip_path)
    result["zip_bytes"] = zip_path.stat().st_size
    result["zip_url"] = f"/api/datasets/{dataset_id}/export.zip"
    return jsonify(result)


@app.get("/api/datasets/<dataset_id>/export.zip")
def export_zip(dataset_id: str):
    zip_path = OUTPUT_DIR / f"{dataset_id}.zip"
    if not zip_path.exists():
        return ("Export zip not found. Run export first.", 404)
    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{dataset_id}_labels.zip",
    )


# ─────────────────── Video rendering ───────────────────

def _video_path(dataset_id: str, fmt: str) -> Path:
    suffix = "mp4" if fmt == "mp4" else "webm"
    return OUTPUT_DIR / f"{dataset_id}_video.{suffix}"


@app.post("/api/datasets/<dataset_id>/render_video")
def render_video_async(dataset_id: str):
    body = request.get_json(force=True) or {}
    m = store.meta(dataset_id)
    arrays = store.arrays(dataset_id)
    schema = store.schema(dataset_id)["labels"]
    width = int(m["width"])
    height = int(m["height"])
    fmt = body.get("format", "mp4")
    out_path = _video_path(dataset_id, fmt)

    params = {
        "start_us": int(body.get("start_us", m.get("t_min_us", 0))),
        "end_us": int(body.get("end_us", m.get("t_max_us", m.get("t_min_us", 0) + 1_000_000))),
        "fps": int(body.get("fps", 30)),
        "frame_window_us": int(body.get("frame_window_us", 33_333)),
        "color_mode": str(body.get("color_mode", "label_polarity")),
        "background": str(body.get("background", "black")),
        "unlabeled_color": str(body.get("unlabeled_color", "#888888")),
        "polarity_contrast": float(body.get("polarity_contrast", 0.5)),
        "fmt": fmt,
    }

    task_id = uuid.uuid4().hex
    tasks[task_id] = {
        "task_id": task_id, "status": "running", "stage": "queued", "progress": 0.0,
        "message": "Queued", "started_at": time.time(),
    }

    def progress(value: float, message: str) -> None:
        tasks[task_id].update({"stage": "rendering", "progress": value, "message": message, "updated_at": time.time()})

    def worker() -> None:
        try:
            result = render_event_video(
                arrays, schema, width, height, out_path, progress=progress, **params,
            )
            tasks[task_id].update({
                "status": "ready", "stage": "ready", "progress": 1.0,
                "message": "Video ready", "result": {**result, "url": f"/api/datasets/{dataset_id}/render_video.{fmt}"},
                "finished_at": time.time(),
            })
        except Exception as exc:
            tasks[task_id].update({"status": "failed", "stage": "error", "progress": 1.0, "message": str(exc), "finished_at": time.time()})

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"task_id": task_id, "status": "running"})


@app.post("/api/datasets/<dataset_id>/render_video/preview")
def render_video_preview(dataset_id: str):
    body = request.get_json(force=True) or {}
    m = store.meta(dataset_id)
    arrays = store.arrays(dataset_id)
    schema = store.schema(dataset_id)["labels"]
    frame = render_event_frame(
        arrays, schema, int(m["width"]), int(m["height"]),
        t_us=int(body.get("t_us", m.get("t_min_us", 0))),
        frame_window_us=int(body.get("frame_window_us", 33_333)),
        color_mode=str(body.get("color_mode", "label_polarity")),
        background=str(body.get("background", "black")),
        unlabeled_color=str(body.get("unlabeled_color", "#888888")),
        polarity_contrast=float(body.get("polarity_contrast", 0.5)),
    )
    # Encode as PNG in memory for the dialog preview.
    import io
    import imageio.v2 as imageio
    buf = io.BytesIO()
    imageio.imwrite(buf, frame, format="png")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.get("/api/datasets/<dataset_id>/render_video.<fmt>")
def render_video_download(dataset_id: str, fmt: str):
    if fmt not in ("mp4", "webm"):
        return ("Unsupported format", 400)
    path = _video_path(dataset_id, fmt)
    if not path.exists():
        return ("Video not found. Render first.", 404)
    return send_file(
        path,
        mimetype="video/mp4" if fmt == "mp4" else "video/webm",
        as_attachment=True,
        download_name=f"{dataset_id}.{fmt}",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
