from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from config import DEFAULT_LABELS, OFF_COLOR, ON_COLOR


def label_color_map(schema: list[dict]) -> dict[int, tuple[int, int, int]]:
    colors: dict[int, tuple[int, int, int]] = {}
    for item in schema:
        value = item["color"].lstrip("#")
        colors[int(item["id"])] = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    return colors


def colors_for(arrays: dict[str, np.ndarray], indices: np.ndarray, schema: list[dict] | None = None) -> np.ndarray:
    schema = schema or DEFAULT_LABELS
    idx = indices.astype(np.int64)
    colors = np.empty((len(idx), 3), dtype=np.uint8)
    polarity = arrays["polarity"][idx].astype(bool)
    colors[polarity] = ON_COLOR
    colors[~polarity] = OFF_COLOR
    labels = arrays["label"][idx]
    cmap = label_color_map(schema)
    for label, color in cmap.items():
        colors[labels == label] = color
    return colors


def write_binary_ply(path: Path, positions: np.ndarray, colors: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # red/green/blue/alpha is Blender's native "Byte Color" vertex-colour
    # layout — the importer maps it straight onto a colour attribute. alpha is
    # constant 255 (opaque). `label` is kept as a per-point scalar so tools
    # like CloudCompare can colour/filter by class; Blender ignores it harmlessly.
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
            ("label", "<i2"),
        ]
    )
    vertices = np.empty(len(positions), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    vertices["alpha"] = 255
    vertices["label"] = labels.astype(np.int16)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"
        "property short label\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


# Export artifacts this module may write — cleaned before each run so a new
# export never leaves stale files from a previous one in the bundled zip.
_EXPORT_ARTIFACTS = (
    "labels.npz",
    "labels.csv",
    "labels_preview.csv",
    "labelled.ply",
    "labelled_preview.ply",
    "project.json",
)


def export_outputs(
    output_dir: Path,
    dataset_id: str,
    meta: dict,
    arrays: dict[str, np.ndarray],
    positions: np.ndarray,
    colors: np.ndarray,
    viz_indices: np.ndarray,
    npz_indices: np.ndarray,
    schema: list[dict],
    *,
    sampled: bool,
    only_labelled: bool,
    include_npz: bool = True,
    include_ply: bool = True,
    include_csv: bool = True,
) -> dict:
    # `npz_indices` is the authoritative export set (all events, or only the
    # labelled ones). `viz_indices` is what the .ply/.csv actually contain —
    # equal to npz_indices when not sampling, or a time-stratified subset when
    # sampling. `positions`/`colors` are aligned to `viz_indices`.
    target = output_dir / dataset_id
    target.mkdir(parents=True, exist_ok=True)
    for name in _EXPORT_ARTIFACTS:
        stale = target / name
        if stale.exists():
            # Best-effort: a file held open elsewhere (e.g. Blender, a viewer)
            # must not abort the whole export. Files we re-write this run get
            # overwritten anyway; one we skip stays as-is.
            try:
                stale.unlink()
            except OSError:
                pass

    result: dict[str, str] = {}
    files: dict[str, str] = {}

    if include_npz:
        ni = npz_indices.astype(np.int64)
        npz_path = target / "labels.npz"
        np.savez_compressed(
            npz_path,
            event_id=arrays["event_id"][ni],
            t_us=arrays["t_us"][ni],
            x=arrays["x"][ni],
            y=arrays["y"][ni],
            polarity=arrays["polarity"][ni],
            label=arrays["label"][ni],
        )
        result["labels"] = str(npz_path)
        files["labels"] = npz_path.name

    if include_csv:
        csv_path = target / ("labels_preview.csv" if sampled else "labels.csv")
        ci = viz_indices.astype(np.int64)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["event_id", "t_us", "x", "y", "polarity", "label"])
            for i in ci:
                writer.writerow([int(arrays["event_id"][i]), int(arrays["t_us"][i]), int(arrays["x"][i]), int(arrays["y"][i]), int(arrays["polarity"][i]), int(arrays["label"][i])])
        result["csv"] = str(csv_path)
        files["csv"] = csv_path.name

    if include_ply:
        ply_path = target / ("labelled_preview.ply" if sampled else "labelled.ply")
        write_binary_ply(ply_path, positions, colors, arrays["label"][viz_indices.astype(np.int64)])
        result["ply"] = str(ply_path)
        files["ply"] = ply_path.name

    project_path = target / "project.json"
    project = {
        "dataset_id": dataset_id,
        "meta": meta,
        "label_schema": schema,
        "export": {
            "sampled": sampled,
            "only_labelled": only_labelled,
            "exported_events": int(len(npz_indices)),
            "ply_csv_events": int(len(viz_indices)),
        },
        "files": files,
    }
    project_path.write_text(json.dumps(project, indent=2, ensure_ascii=False), encoding="utf-8")
    result["project"] = str(project_path)
    return result

