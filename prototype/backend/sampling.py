from __future__ import annotations

import numpy as np


def time_stratified_indices(t_us: np.ndarray, max_count: int, bins: int = 256) -> np.ndarray:
    total = len(t_us)
    if total <= max_count:
        return np.arange(total, dtype=np.uint64)
    max_count = max(1, int(max_count))
    bins = max(1, min(bins, max_count, total))
    edges = np.linspace(0, total, bins + 1, dtype=np.int64)
    per_bin = max(1, max_count // bins)
    selected: list[np.ndarray] = []
    remaining = max_count
    for start, end in zip(edges[:-1], edges[1:]):
        if remaining <= 0 or end <= start:
            continue
        take = min(per_bin, end - start, remaining)
        selected.append(np.linspace(start, end - 1, take, dtype=np.uint64))
        remaining -= take
    if remaining > 0:
        used = np.concatenate(selected) if selected else np.empty(0, dtype=np.uint64)
        mask = np.ones(total, dtype=bool)
        mask[used.astype(np.int64)] = False
        extra = np.flatnonzero(mask)[:remaining].astype(np.uint64)
        selected.append(extra)
    return np.sort(np.concatenate(selected)[:max_count]).astype(np.uint64)


def filter_indices(
    arrays: dict[str, np.ndarray],
    start_us: int | None,
    end_us: int | None,
    polarity: str,
    sample: int | None = None,
    x_min: int | None = None,
    x_max: int | None = None,
    y_min: int | None = None,
    y_max: int | None = None,
) -> np.ndarray:
    """Return indices matching the filters. ``sample`` is optional — pass
    ``None`` (or ``<= 0``) for no sub-sampling (full-detail load)."""
    t_us = arrays["t_us"]
    mask = np.ones(len(t_us), dtype=bool)
    if start_us is not None:
        mask &= t_us >= start_us
    if end_us is not None:
        mask &= t_us <= end_us
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
    indices = np.flatnonzero(mask)
    if sample is not None and sample > 0 and len(indices) > sample:
        local = time_stratified_indices(t_us[indices], sample)
        indices = indices[local.astype(np.int64)]
    return indices.astype(np.uint64)


def coordinate_positions(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    width: int,
    height: int,
    t_min_us: float | None = None,
    t_max_us: float | None = None,
) -> np.ndarray:
    """Map (x, y, t) events to centred 3D positions.

    X → centred pixel x; Y → centred, flipped pixel y; Z → time.

    When `t_min_us`/`t_max_us` are given, Z is normalised over that GLOBAL range
    and centred to [-width/2, +width/2]. This is essential for the chunked
    renderer: without a global range each chunk was normalised over its OWN time
    span (all centred at z≈0), collapsing the whole recording into one slab.
    With the global range, chunks stack correctly along time. Larger t → larger
    z, and the 3D camera's up = +Z, so the latest events sit at the top.
    """
    idx = indices.astype(np.int64)
    x = arrays["x"][idx].astype(np.float32)
    y = arrays["y"][idx].astype(np.float32)
    t = arrays["t_us"][idx].astype(np.float64)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    positions = np.empty((len(idx), 3), dtype=np.float32)
    positions[:, 0] = (x - center_x)
    positions[:, 1] = (center_y - y)
    if t_min_us is not None and t_max_us is not None and float(t_max_us) > float(t_min_us):
        tmn = float(t_min_us)
        span = float(t_max_us) - tmn
        z = ((t - tmn) / span) * float(width)        # → [0, width] over the global range
        positions[:, 2] = (z - float(width) / 2.0).astype(np.float32)
    else:
        # Legacy fallback: normalise over this subset's own range (centred).
        duration = max(float(t.max() - t.min()) / 1_000_000.0, 1e-6) if len(t) else 1.0
        time_scale = width / duration
        z = ((t - t.min()) / 1_000_000.0 * time_scale).astype(np.float32)
        positions[:, 2] = z - (float(z.max() + z.min()) / 2.0 if len(z) else 0.0)
    return positions
