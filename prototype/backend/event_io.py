from __future__ import annotations

from pathlib import Path

import event_stream
import numpy as np


def read_es(path: Path) -> tuple[np.ndarray, int, int]:
    chunks: list[np.ndarray] = []
    with event_stream.Decoder(str(path)) as decoder:
        if decoder.type != "dvs":
            raise ValueError(f"Only DVS streams are supported, got {decoder.type!r}")
        width, height = decoder.width, decoder.height
        for chunk in decoder:
            if len(chunk):
                chunks.append(chunk.copy())
    events = np.concatenate(chunks) if chunks else np.empty(0, dtype=event_stream.dvs_dtype)
    return events, width, height


def build_event_arrays(events: np.ndarray) -> dict[str, np.ndarray]:
    count = len(events)
    return {
        "event_id": np.arange(count, dtype=np.uint64),
        "t_us": events["t"].astype(np.uint64),
        "x": events["x"].astype(np.uint16),
        "y": events["y"].astype(np.uint16),
        "polarity": events["on"].astype(np.uint8),
        "label": np.full(count, -1, dtype=np.int16),
    }


def load_events_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}

