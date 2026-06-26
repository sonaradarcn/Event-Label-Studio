"""Render labelled events to a 2-D video file.

Each output frame accumulates the events whose timestamp falls inside a fixed
window (``frame_window_us``). Each event is splatted to its (x, y) pixel and
coloured by its label (or polarity if unlabelled). Frames are streamed to an
``imageio`` writer that pipes raw RGB into a bundled ffmpeg.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import numpy as np


ProgressFn = Callable[[float, str], None]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _build_label_colors(schema: list[dict]) -> dict[int, tuple[int, int, int]]:
    return {int(s["id"]): _hex_to_rgb(s["color"]) for s in schema}


def _build_label_lut(schema: list[dict]):
    """Dense label-id → RGB lookup + validity mask, for vectorised colouring.
    Ids absent from the schema (e.g. orphan ids from a deleted label) fall back
    to the unlabelled colour."""
    label_colors = _build_label_colors(schema)
    cn = max((max((int(s["id"]) for s in schema), default=-1)) + 1, 1)
    lut = np.zeros((cn, 3), dtype=np.float32)
    has = np.zeros(cn, dtype=bool)
    for lid, rgb in label_colors.items():
        if 0 <= lid < cn:
            lut[lid] = rgb
            has[lid] = True
    return lut, has, cn


def _compute_colors(labels, pols, color_mode, lut, has, cn, unl, k, bg_color):
    """Per-frame point colours matching the app's display: base = label colour
    (or the configurable unlabelled colour `unl`) modulated by polarity — ON
    brighter (c+(255-c)k), OFF darker (c(1-k)). label_only = flat label colours
    (unlabelled → background); polarity_only ignores labels."""
    n = labels.shape[0]
    # Events whose label id is a real, in-range schema colour. Orphan / unknown
    # ids (e.g. left over from a deleted label) fall through to the unlabelled
    # colour (or background for label_only) — and must NOT index the LUT.
    sel = (labels >= 0) & (labels < cn)
    if sel.any():
        sel[sel] = has[labels[sel]]
    if color_mode == "label_only":
        colors = np.empty((n, 3), dtype=np.uint8)
        colors[:] = bg_color
        if sel.any():
            colors[sel] = lut[labels[sel]].astype(np.uint8)
        return colors
    base = np.empty((n, 3), dtype=np.float32)
    base[:] = unl
    if color_mode != "polarity_only" and sel.any():  # label_polarity → labelled events use label colour
        base[sel] = lut[labels[sel]]
    pf = pols[:, None]
    mod = np.where(pf, base + (255.0 - base) * k, base * (1.0 - k))
    return np.clip(mod, 0.0, 255.0).astype(np.uint8)


def _ensure_even(n: int) -> int:
    """libx264 + yuv420p needs even dims; round up to nearest even."""
    return n + (n & 1)


def render_event_video(
    arrays: dict[str, np.ndarray],
    schema: list[dict],
    width: int,
    height: int,
    output_path: Path,
    *,
    start_us: int,
    end_us: int,
    fps: int = 30,
    frame_window_us: int = 33_333,
    color_mode: str = "label_polarity",  # label_polarity | polarity_only | label_only
    background: str = "black",            # black | white
    unlabeled_color: str = "#888888",     # base colour for unlabelled events
    polarity_contrast: float = 0.5,       # ON brighter / OFF darker strength (0..1)
    fmt: str = "mp4",
    progress: ProgressFn | None = None,
) -> dict:
    if end_us <= start_us:
        raise ValueError("end_us must be greater than start_us")
    if width <= 0 or height <= 0:
        raise ValueError("invalid dataset dimensions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.01, "Preparing event index")

    t = arrays["t_us"]
    # Pre-slice once to the time range — every frame intersects this window.
    rng_lo = np.searchsorted(t, start_us, side="left")
    rng_hi = np.searchsorted(t, end_us, side="right")
    t_view = t[rng_lo:rng_hi]
    x_view = arrays["x"][rng_lo:rng_hi]
    y_view = arrays["y"][rng_lo:rng_hi]
    label_view = arrays["label"][rng_lo:rng_hi]
    pol_view = arrays["polarity"][rng_lo:rng_hi]

    lut, has_lc, cn = _build_label_lut(schema)
    unl = np.array(_hex_to_rgb(unlabeled_color), dtype=np.float32)
    k = float(polarity_contrast)

    duration_us = end_us - start_us
    n_frames = max(1, int(round(duration_us * fps / 1_000_000)))

    # Output dims must be even for yuv420p; render at requested size, pad if odd.
    out_w = _ensure_even(width)
    out_h = _ensure_even(height)
    pad_x = out_w - width
    pad_y = out_h - height

    bg_color = (0, 0, 0) if background == "black" else (255, 255, 255)

    codec = "libx264" if fmt == "mp4" else "libvpx-vp9"
    if codec == "libx264":
        # -preset only trades encode speed for file size at a FIXED quality
        # (the quality= setting), so the visual result is unchanged while
        # encoding runs several times faster than libx264's "medium" default.
        # -threads 0 lets x264 use every core.
        ff_params = ["-pix_fmt", "yuv420p", "-preset", "veryfast", "-threads", "0"]
    else:
        # libvpx-vp9 is extremely slow by default; row multithreading + a higher
        # cpu-used (speed) level accelerate it a lot without changing the quality.
        ff_params = ["-pix_fmt", "yuv420p", "-row-mt", "1", "-cpu-used", "5", "-deadline", "good", "-threads", "0"]
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec=codec,
        quality=8,
        macro_block_size=1,
        ffmpeg_params=ff_params,
    )

    # Reuse one frame buffer across all frames — append_data consumes it
    # synchronously, so we only memset (not reallocate) per frame.
    frame = np.empty((out_h, out_w, 3), dtype=np.uint8)
    try:
        for fi in range(n_frames):
            t0 = start_us + (fi * 1_000_000) // fps
            t1 = t0 + frame_window_us
            lo = np.searchsorted(t_view, t0, side="left")
            hi = np.searchsorted(t_view, t1, side="right")
            xs = x_view[lo:hi]
            ys = y_view[lo:hi]
            labels = label_view[lo:hi]
            pols = pol_view[lo:hi].astype(bool)

            frame[:] = bg_color

            if xs.size:
                colors = _compute_colors(labels, pols, color_mode, lut, has_lc, cn, unl, k, bg_color)

                # Clamp to camera resolution; ignore stray indices.
                valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
                if not valid.all():
                    xs = xs[valid]; ys = ys[valid]; colors = colors[valid]

                # Splat — last write wins for same-pixel collisions in this frame.
                frame[ys.astype(np.int64), xs.astype(np.int64)] = colors

            writer.append_data(frame)
            if progress and (fi % 8 == 0 or fi == n_frames - 1):
                progress(min(0.99, (fi + 1) / n_frames), f"Frame {fi + 1}/{n_frames}")
    finally:
        writer.close()

    return {
        "path": str(output_path),
        "frames": n_frames,
        "fps": fps,
        "width": out_w,
        "height": out_h,
        "padding": {"x": pad_x, "y": pad_y},
        "duration_us": duration_us,
        "bytes": output_path.stat().st_size,
    }


def render_event_frame(
    arrays: dict[str, np.ndarray],
    schema: list[dict],
    width: int,
    height: int,
    *,
    t_us: int,
    frame_window_us: int = 33_333,
    color_mode: str = "label_polarity",
    background: str = "black",
    unlabeled_color: str = "#888888",
    polarity_contrast: float = 0.5,
) -> np.ndarray:
    """Render a single PNG-style frame for the dialog preview button."""
    t = arrays["t_us"]
    lo = np.searchsorted(t, t_us, side="left")
    hi = np.searchsorted(t, t_us + frame_window_us, side="right")
    xs = arrays["x"][lo:hi]
    ys = arrays["y"][lo:hi]
    labels = arrays["label"][lo:hi]
    pols = arrays["polarity"][lo:hi].astype(bool)

    lut, has_lc, cn = _build_label_lut(schema)
    unl = np.array(_hex_to_rgb(unlabeled_color), dtype=np.float32)
    k = float(polarity_contrast)
    bg_color = (0, 0, 0) if background == "black" else (255, 255, 255)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = bg_color
    if xs.size:
        colors = _compute_colors(labels, pols, color_mode, lut, has_lc, cn, unl, k, bg_color)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if not valid.all():
            xs = xs[valid]; ys = ys[valid]; colors = colors[valid]
        frame[ys.astype(np.int64), xs.astype(np.int64)] = colors
    return frame
