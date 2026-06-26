from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from config import CACHE_DIR, DEFAULT_LABELS, DEFAULT_SAMPLE, DEFAULT_WINDOW_US
from event_io import build_event_arrays, load_events_npz, read_es
from sampling import time_stratified_indices


class CacheStore:
    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._label_locks: dict[str, threading.RLock] = {}

    def _lock(self, dataset_id: str) -> threading.RLock:
        if dataset_id not in self._label_locks:
            self._label_locks[dataset_id] = threading.RLock()
        return self._label_locks[dataset_id]

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.cache_dir / dataset_id

    def open_dataset(self, source: Path, progress=None) -> dict:
        progress = progress or (lambda *_args, **_kwargs: None)
        dataset_id = source.stem
        target = self.dataset_dir(dataset_id)
        target.mkdir(parents=True, exist_ok=True)
        meta_path = target / "meta.json"
        events_path = target / "events.npz"
        schema_path = target / "label_schema.json"
        if not events_path.exists() or not meta_path.exists():
            progress("decode", 0.08, f"Decoding {source.name}")
            events, width, height = read_es(source)
            progress("arrays", 0.42, f"Building arrays for {len(events):,} events")
            arrays = build_event_arrays(events)
            progress("cache", 0.58, "Writing compressed events.npz")
            np.savez_compressed(events_path, **arrays)
            meta = {
                "dataset_id": dataset_id,
                "source_path": str(source),
                "width": width,
                "height": height,
                "event_count": int(len(events)),
                "t_min_us": int(arrays["t_us"][0]) if len(events) else 0,
                "t_max_us": int(arrays["t_us"][-1]) if len(events) else 0,
                "cache_version": 1,
                "default_window_us": DEFAULT_WINDOW_US,
                "default_sampling": "time_stratified",
            }
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            progress("lod_50k", 0.72, "Writing 50k LOD cache")
            self._write_lod(target, arrays, 50_000)
            progress("lod_200k", 0.84, "Writing 200k LOD cache")
            self._write_lod(target, arrays, DEFAULT_SAMPLE)
        else:
            progress("cache_hit", 0.7, "Using existing cache")
        if not schema_path.exists():
            progress("schema", 0.9, "Writing default label schema")
            schema_path.write_text(json.dumps({"labels": DEFAULT_LABELS}, indent=2, ensure_ascii=False), encoding="utf-8")
        progress("labels", 0.95, "Preparing label store")
        # Creates labels.npy (migrating an old labels.snapshot.npz if present,
        # else a fresh all-unlabelled array).
        self.ensure_labels_npy(dataset_id)
        log_path = target / "labels.log.jsonl"
        if not log_path.exists():
            log_path.touch()
        else:
            self._replay_log(dataset_id)
        progress("ready", 1.0, "Dataset ready")
        return self.meta(dataset_id)

    def _replay_log(self, dataset_id: str) -> None:
        log_path = self.dataset_dir(dataset_id) / "labels.log.jsonl"
        marker_path = self.dataset_dir(dataset_id) / "replay_marker.json"
        if not log_path.exists():
            return
        self.ensure_labels_npy(dataset_id)
        last_replayed = 0
        if marker_path.exists():
            try:
                last_replayed = json.loads(marker_path.read_text(encoding="utf-8")).get("last_line", 0)
            except (json.JSONDecodeError, ValueError):
                last_replayed = 0
        labels = np.load(self.labels_npy_path(dataset_id), mmap_mode="r+")
        final_line = last_replayed
        needs_flush = False
        with log_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                final_line = line_no
                if line_no <= last_replayed:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    rtype = record.get("type")
                    if rtype == "update":
                        ids = np.asarray(record["event_ids"], dtype=np.int64)
                        valid = (ids >= 0) & (ids < len(labels))
                        labels[ids[valid]] = record.get("label", -1)
                        needs_flush = True
                    elif rtype == "update_by_filter":
                        pass  # filter ops persist to labels.npy immediately; skip during replay
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        if needs_flush:
            labels.flush()
        del labels
        marker_path.write_text(json.dumps({"last_line": final_line}), encoding="utf-8")

    def _write_lod(self, target: Path, arrays: dict[str, np.ndarray], size: int) -> None:
        indices = time_stratified_indices(arrays["t_us"], min(size, len(arrays["t_us"])))
        np.savez_compressed(target / f"lod_{size // 1000}k.npz", indices=indices)

    def lod_arrays(self, dataset_id: str, size: int = DEFAULT_SAMPLE) -> dict[str, np.ndarray] | None:
        lod_path = self.dataset_dir(dataset_id) / f"lod_{size // 1000}k.npz"
        if not lod_path.exists():
            return None
        lod_data = load_events_npz(lod_path)
        if "indices" in lod_data:
            full_arrays = load_events_npz(self.dataset_dir(dataset_id) / "events.npz")
            snapshot = self.dataset_dir(dataset_id) / "labels.snapshot.npz"
            if snapshot.exists():
                labels = np.load(snapshot)["label"].astype(np.int16)
                if len(labels) == len(full_arrays["label"]):
                    full_arrays["label"] = labels
            idx = lod_data["indices"].astype(np.int64)
            return {k: v[idx] for k, v in full_arrays.items()}
        return lod_data

    def meta(self, dataset_id: str) -> dict:
        return json.loads((self.dataset_dir(dataset_id) / "meta.json").read_text(encoding="utf-8"))

    # ── Mutable labels: uncompressed .npy, memory-mapped for O(edit) writes ──
    # Compressed labels.snapshot.npz forced a full re-compress on every edit
    # (O(N) per label change). labels.npy is a flat int16 array we can open as
    # a memmap and patch in place — a 200k-event assign touches only those
    # rows instead of rewriting the whole array. snapshot.npz is migrated on
    # first access and then ignored.
    def labels_npy_path(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "labels.npy"

    def ensure_labels_npy(self, dataset_id: str) -> None:
        with self._lock(dataset_id):
            path = self.labels_npy_path(dataset_id)
            if path.exists():
                return
            snapshot = self.dataset_dir(dataset_id) / "labels.snapshot.npz"
            if snapshot.exists():
                with np.load(snapshot) as data:
                    labels = data["label"].astype(np.int16, copy=True)
            else:
                n = int(self.meta(dataset_id)["event_count"])
                labels = np.full(n, -1, dtype=np.int16)
            np.save(path, labels, allow_pickle=False)

    def labels_rw(self, dataset_id: str) -> np.memmap:
        """Read/write memmap of the label array — patch slices in place + flush."""
        self.ensure_labels_npy(dataset_id)
        return np.load(self.labels_npy_path(dataset_id), mmap_mode="r+")

    def labels_ro(self, dataset_id: str) -> np.memmap:
        self.ensure_labels_npy(dataset_id)
        return np.load(self.labels_npy_path(dataset_id), mmap_mode="r")

    def arrays(self, dataset_id: str) -> dict[str, np.ndarray]:
        arrays = load_events_npz(self.dataset_dir(dataset_id) / "events.npz")
        self.ensure_labels_npy(dataset_id)
        labels = np.load(self.labels_npy_path(dataset_id), mmap_mode="r")
        if len(labels) == len(arrays["label"]):
            arrays["label"] = np.asarray(labels, dtype=np.int16)  # copy out of the memmap
        return arrays

    def save_labels(self, dataset_id: str, labels: np.ndarray) -> None:
        """Persist a whole label array. Writes in place when the .npy already
        exists at the same shape (no os.replace → safe while memmaps are open
        on Windows); otherwise creates it."""
        with self._lock(dataset_id):
            path = self.labels_npy_path(dataset_id)
            labels = labels.astype(np.int16, copy=False)
            if path.exists():
                mm = np.load(path, mmap_mode="r+")
                if mm.shape == labels.shape:
                    mm[:] = labels
                    mm.flush()
                    del mm
                    return
                del mm
            np.save(path, labels, allow_pickle=False)

    def schema(self, dataset_id: str) -> dict:
        return json.loads((self.dataset_dir(dataset_id) / "label_schema.json").read_text(encoding="utf-8"))

    def save_schema(self, dataset_id: str, schema: dict) -> None:
        (self.dataset_dir(dataset_id) / "label_schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_datasets(self) -> list[dict]:
        datasets = []
        for meta_path in sorted(self.cache_dir.glob("*/meta.json")):
            datasets.append(json.loads(meta_path.read_text(encoding="utf-8")))
        return datasets
