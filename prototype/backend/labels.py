from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cache_store import CacheStore
from sampling import filter_indices


@dataclass
class LabelManager:
    store: CacheStore
    undo_stack: dict[str, list[dict]] = field(default_factory=dict)
    redo_stack: dict[str, list[dict]] = field(default_factory=dict)
    # Monotonic per-dataset version, bumped on every mutation. The client uses
    # it to detect when its optimistic local state has diverged from the server.
    versions: dict[str, int] = field(default_factory=dict)

    def _bump_version(self, dataset_id: str) -> int:
        self.versions[dataset_id] = self.versions.get(dataset_id, 0) + 1
        return self.versions[dataset_id]

    @staticmethod
    def _stats_delta(previous: np.ndarray, new_value: int, n_ids: int) -> dict[str, int]:
        """Count change per label id caused by setting `n_ids` events (whose old
        values are `previous`) to `new_value`. Keyed by label id as a string
        ("-1" == unlabelled). Lets the client patch its stats without an O(N) scan."""
        delta: dict[str, int] = {}
        if previous.size:
            old_vals, old_counts = np.unique(previous, return_counts=True)
            for ov, oc in zip(old_vals, old_counts):
                k = str(int(ov))
                delta[k] = delta.get(k, 0) - int(oc)
        k = str(int(new_value))
        delta[k] = delta.get(k, 0) + int(n_ids)
        return {k: v for k, v in delta.items() if v != 0}

    def update_array(self, dataset_id: str, event_ids: np.ndarray, label: int, operation: str = "assign") -> dict:
        """Core label write: patch a memmap in place (O(selected)) instead of
        re-compressing the whole label array. Accepts a NumPy id array so the
        binary endpoint can pass `np.frombuffer` results with no Python list."""
        with self.store._lock(dataset_id):
            labels = self.store.labels_rw(dataset_id)
            ids = np.asarray(event_ids, dtype=np.int64)
            ids = np.unique(ids[(ids >= 0) & (ids < labels.shape[0])])
            previous = np.asarray(labels[ids], dtype=np.int16).copy()
            new_value = -1 if operation == "clear" else int(label)
            labels[ids] = np.int16(new_value)
            labels.flush()
            del labels  # release the memmap handle promptly
            record = {
                "type": "update",
                "event_ids": ids.tolist(),
                "previous": previous.astype(int).tolist(),
                "new_value": int(new_value),
                "label": int(new_value),
                "operation": operation,
            }
            self.undo_stack.setdefault(dataset_id, []).append(record)
            self.redo_stack.setdefault(dataset_id, []).clear()
            self._append_log(dataset_id, record)
            return {
                "updated": int(ids.size),
                "label": int(new_value),
                "label_version": self._bump_version(dataset_id),
                "stats_delta": self._stats_delta(previous, int(new_value), int(ids.size)),
            }

    def update(self, dataset_id: str, event_ids: list[int], label: int, operation: str = "assign") -> dict:
        # JSON-compatibility wrapper around update_array.
        return self.update_array(dataset_id, np.asarray(event_ids, dtype=np.int64), int(label), operation)

    def update_by_filter(
        self,
        dataset_id: str,
        label: int,
        operation: str = "assign",
        start_us: int | None = None,
        end_us: int | None = None,
        polarity: str = "all",
        x_min: int | None = None,
        x_max: int | None = None,
        y_min: int | None = None,
        y_max: int | None = None,
    ) -> dict:
        with self.store._lock(dataset_id):
            arrays = self.store.arrays(dataset_id)
            mask = np.ones(len(arrays["t_us"]), dtype=bool)
            if start_us is not None:
                mask &= arrays["t_us"] >= start_us
            if end_us is not None:
                mask &= arrays["t_us"] <= end_us
            if polarity == "on":
                mask &= arrays["polarity"] == 1
            elif polarity == "off":
                mask &= arrays["polarity"] == 0
            if x_min is not None:
                mask &= arrays["x"] >= x_min
            if x_max is not None:
                mask &= arrays["x"] <= x_max
            if y_min is not None:
                mask &= arrays["y"] >= y_min
            if y_max is not None:
                mask &= arrays["y"] <= y_max
            ids = np.flatnonzero(mask)
            if len(ids) == 0:
                return {"updated": 0, "label": label, "label_version": self.versions.get(dataset_id, 0), "stats_delta": {}}
            op_id = uuid.uuid4().hex[:12]
            labels = self.store.labels_rw(dataset_id)
            previous = np.asarray(labels[ids], dtype=np.int16).copy()
            snapshot_path = self.store.dataset_dir(dataset_id) / f"pre_filter_{op_id}.npz"
            np.savez_compressed(snapshot_path, ids=ids, previous=previous)
            new_value = -1 if operation == "clear" else int(label)
            labels[ids] = np.int16(new_value)
            labels.flush()
            del labels
            record = {
                "type": "update_by_filter",
                "count": int(len(ids)),
                "filter": {
                    "start_us": start_us, "end_us": end_us,
                    "polarity": polarity,
                    "x_min": x_min, "x_max": x_max,
                    "y_min": y_min, "y_max": y_max,
                },
                "new_value": int(new_value),
                "label": int(new_value),
                "operation": operation,
                "snapshot_file": snapshot_path.name,
            }
            self.undo_stack.setdefault(dataset_id, []).append(record)
            self.redo_stack.setdefault(dataset_id, []).clear()
            self._append_log(dataset_id, record)
            return {
                "updated": int(len(ids)),
                "label": int(new_value),
                "label_version": self._bump_version(dataset_id),
                "stats_delta": self._stats_delta(previous, int(new_value), int(len(ids))),
            }

    def undo(self, dataset_id: str) -> dict:
        with self.store._lock(dataset_id):
            stack = self.undo_stack.setdefault(dataset_id, [])
            if not stack:
                return {"updated": 0, "label_version": self.versions.get(dataset_id, 0)}
            record = stack.pop()
            labels = self.store.labels_rw(dataset_id)
            n = labels.shape[0]
            if record.get("type") == "update_by_filter":
                snapshot_file = record.get("snapshot_file")
                snapshot_path = self.store.dataset_dir(dataset_id) / snapshot_file if snapshot_file else None
                if snapshot_path and snapshot_path.exists():
                    snap = np.load(snapshot_path)
                    ids = snap["ids"].astype(np.int64)
                    previous = snap["previous"].astype(np.int16)
                    valid = (ids >= 0) & (ids < n)
                    labels[ids[valid]] = previous[valid]
                else:
                    ids = self._filter_ids(self.store.arrays(dataset_id), record["filter"])
                    labels[ids] = np.int16(-1)
            else:
                ids = np.asarray(record["event_ids"], dtype=np.int64)
                labels[ids] = np.asarray(record["previous"], dtype=np.int16)
            labels.flush()
            del labels
            self.redo_stack.setdefault(dataset_id, []).append(record)
            self._append_log(dataset_id, {"type": "undo", "count": int(len(ids))})
            return {"updated": int(len(ids)), "label_version": self._bump_version(dataset_id)}

    def redo(self, dataset_id: str) -> dict:
        with self.store._lock(dataset_id):
            stack = self.redo_stack.setdefault(dataset_id, [])
            if not stack:
                return {"updated": 0, "label_version": self.versions.get(dataset_id, 0)}
            record = stack.pop()
            labels = self.store.labels_rw(dataset_id)
            if record.get("type") == "update_by_filter":
                ids = self._filter_ids(self.store.arrays(dataset_id), record["filter"])
                labels[ids] = np.int16(int(record["label"]))
            else:
                ids = np.asarray(record["event_ids"], dtype=np.int64)
                labels[ids] = np.int16(int(record["label"]))
            labels.flush()
            del labels
            self.undo_stack.setdefault(dataset_id, []).append(record)
            self._append_log(dataset_id, {"type": "redo", "count": int(len(ids))})
            return {"updated": int(len(ids)), "label_version": self._bump_version(dataset_id)}

    def _filter_ids(self, arrays: dict, filt: dict) -> np.ndarray:
        mask = np.ones(len(arrays["t_us"]), dtype=bool)
        if filt.get("start_us") is not None:
            mask &= arrays["t_us"] >= filt["start_us"]
        if filt.get("end_us") is not None:
            mask &= arrays["t_us"] <= filt["end_us"]
        if filt.get("polarity") == "on":
            mask &= arrays["polarity"] == 1
        elif filt.get("polarity") == "off":
            mask &= arrays["polarity"] == 0
        if filt.get("x_min") is not None:
            mask &= arrays["x"] >= filt["x_min"]
        if filt.get("x_max") is not None:
            mask &= arrays["x"] <= filt["x_max"]
        if filt.get("y_min") is not None:
            mask &= arrays["y"] >= filt["y_min"]
        if filt.get("y_max") is not None:
            mask &= arrays["y"] <= filt["y_max"]
        return np.flatnonzero(mask)

    def _append_log(self, dataset_id: str, record: dict) -> None:
        path = self.store.dataset_dir(dataset_id) / "labels.log.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
