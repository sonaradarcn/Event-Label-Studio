from __future__ import annotations

import json
import struct
from pathlib import Path

from app import app
from config import CACHE_DIR, RECORDINGS_DIR


def test_smoke() -> None:
    client = app.test_client()
    source = RECORDINGS_DIR / "stars_0.es"
    opened = client.post("/api/datasets/open", json={"path": str(source)})
    assert opened.status_code == 200, opened.text
    dataset_id = opened.get_json()["dataset_id"]
    dataset_dir = CACHE_DIR / dataset_id
    assert (dataset_dir / "meta.json").exists()
    assert (dataset_dir / "events.npz").exists()
    assert (dataset_dir / "labels.npy").exists()
    response = client.get(f"/api/datasets/{dataset_id}/points.bin?sample=50000")
    assert response.status_code == 200
    data = response.data
    header_len = struct.unpack("<I", data[:4])[0]
    header = json.loads(data[4 : 4 + header_len].decode("utf-8"))
    payload_len = len(data) - 4 - header_len
    assert payload_len == sum(item["bytes"] for item in header["arrays"])
    assert header["count"] <= 50_000
    update = client.post(f"/api/datasets/{dataset_id}/labels/update", json={"event_ids": [0, 1, 2], "label": 1, "operation": "assign"})
    assert update.status_code == 200
    assert update.get_json()["updated"] == 3
    assert client.post(f"/api/datasets/{dataset_id}/labels/undo").get_json()["updated"] == 3
    assert client.post(f"/api/datasets/{dataset_id}/labels/redo").get_json()["updated"] == 3
    exported = client.post(f"/api/datasets/{dataset_id}/export")
    assert exported.status_code == 200
    files = exported.get_json()
    # Response mixes file paths with zip_bytes (int) / zip_url (route) —
    # only the path-valued entries can be existence-checked.
    for key, value in files.items():
        if key in ("zip_bytes", "zip_url"):
            continue
        assert Path(value).exists(), f"{key}: {value}"


def test_export_options() -> None:
    import numpy as np

    client = app.test_client()
    source = RECORDINGS_DIR / "stars_0.es"
    dataset_id = client.post("/api/datasets/open", json={"path": str(source)}).get_json()["dataset_id"]
    # Guarantee at least some labelled events.
    client.post(f"/api/datasets/{dataset_id}/labels/update", json={"event_ids": [0, 1, 2, 3, 4], "label": 1, "operation": "assign"})

    stats = client.get(f"/api/datasets/{dataset_id}/stats").get_json()
    labelled = stats["total"] - stats["unlabelled"]
    assert labelled > 0

    # only-labelled, npz only, no sampling — npz must hold exactly the labelled set.
    r = client.post(f"/api/datasets/{dataset_id}/export", json={
        "only_labelled": True, "sample": False,
        "include_npz": True, "include_ply": False, "include_csv": False,
    })
    assert r.status_code == 200, r.text
    files = r.get_json()
    assert "ply" not in files and "csv" not in files          # excluded files absent
    with np.load(files["labels"]) as data:                    # `with` closes the npz handle
        assert len(data["event_id"]) == labelled              # only labelled exported
        assert int((data["label"] == -1).sum()) == 0          # no unlabelled leaked

    # sampled full-dataset export → preview-named ply/csv.
    r2 = client.post(f"/api/datasets/{dataset_id}/export", json={
        "sample": True, "sample_size": 1000, "only_labelled": False,
        "include_npz": False, "include_ply": True, "include_csv": True,
    })
    f2 = r2.get_json()
    assert f2["ply"].endswith("labelled_preview.ply"), f2["ply"]
    assert f2["csv"].endswith("labels_preview.csv"), f2["csv"]

    # full (unsampled) export → plain names.
    f3 = client.post(f"/api/datasets/{dataset_id}/export", json={"sample": False}).get_json()
    assert f3["ply"].endswith("labelled.ply"), f3["ply"]
    assert f3["csv"].endswith("labels.csv"), f3["csv"]


def test_binary_update_and_stats_delta() -> None:
    import numpy as np

    client = app.test_client()
    source = RECORDINGS_DIR / "stars_0.es"
    dataset_id = client.post("/api/datasets/open", json={"path": str(source)}).get_json()["dataset_id"]
    # Mutable labels now live in an uncompressed, memmap-friendly labels.npy.
    labels_npy = CACHE_DIR / dataset_id / "labels.npy"
    assert labels_npy.exists()

    ids = np.array([100, 101, 102, 103, 104], dtype="<u4")
    # Clear first so the assign delta is deterministic (all previously -1).
    client.post(f"/api/datasets/{dataset_id}/labels/update.bin?operation=clear",
                data=ids.tobytes(), content_type="application/octet-stream")
    r = client.post(f"/api/datasets/{dataset_id}/labels/update.bin?label=2&operation=assign",
                    data=ids.tobytes(), content_type="application/octet-stream")
    assert r.status_code == 200, r.text
    j = r.get_json()
    assert j["updated"] == 5
    assert j["label"] == 2
    assert j.get("label_version", 0) >= 1
    assert j["stats_delta"].get("2") == 5          # 5 events gained label 2
    assert j["stats_delta"].get("-1") == -5        # ...lost from unlabelled
    assert sum(j["stats_delta"].values()) == 0     # counts are conserved

    # Persisted in-place to labels.npy.
    labels = np.load(labels_npy, mmap_mode="r")
    assert int(labels[100]) == 2 and int(labels[104]) == 2
    del labels

    # Odd-length body is rejected.
    bad = client.post(f"/api/datasets/{dataset_id}/labels/update.bin",
                      data=b"\x01\x02\x03", content_type="application/octet-stream")
    assert bad.status_code == 400

    # Undo restores the cleared (-1) state.
    client.post(f"/api/datasets/{dataset_id}/labels/undo")
    labels2 = np.load(labels_npy, mmap_mode="r")
    assert int(labels2[100]) == -1
    del labels2


def test_track_propagation() -> None:
    import numpy as np
    from tracks import propagate_track

    step = 33_333
    rng = np.random.default_rng(0)
    xs, ys, ts = [], [], []
    for k in range(12):  # a blob moving diagonally, present for 12 slices
        cx, cy = 100 + k * 9, 80 + k * 6
        xs.append(cx + rng.normal(0, 6, 200)); ys.append(cy + rng.normal(0, 6, 200))
        ts.append(k * step + rng.uniform(0, step, 200))
    xs.append(rng.uniform(0, 640, 3000)); ys.append(rng.uniform(0, 480, 3000))
    ts.append(rng.uniform(0, 20 * step, 3000))  # adversarial background noise
    x = np.concatenate(xs); y = np.concatenate(ys); t = np.concatenate(ts)
    o = np.argsort(t, kind="stable")
    x = x[o].astype(np.uint16); y = y[o].astype(np.uint16); t = t[o].astype(np.uint64)
    arrays = {"t_us": t, "x": x, "y": y, "polarity": np.ones(len(t), np.uint8), "event_id": np.arange(len(t))}
    seed = np.nonzero((t < step) & (np.abs(x.astype(int) - 100) < 20) & (np.abs(y.astype(int) - 80) < 20))[0]
    assert seed.size > 50

    res = propagate_track(arrays, seed, step_us=step, direction="forward", max_slices=30)
    ok = [s for s in res["slices"] if s["status"] == "ok"]
    assert len(ok) >= 10                      # followed the target for ~12 slices
    assert res["stop_reason"] == "low_confidence"   # paused when it vanished
    # The tracked centre actually moved along the target's diagonal path.
    assert ok[-1]["cx"] > ok[0]["cx"] + 50 and ok[-1]["cy"] > ok[0]["cy"] + 30
    # Empty seed is handled.
    assert propagate_track(arrays, np.array([], dtype=np.int64))["stop_reason"] == "no_seed"


def test_track_crossing_pauses_instead_of_switching() -> None:
    """Two equal blobs cross head-on. The tracker must NOT silently follow the
    wrong one — it should pause (ambiguous) around the crossing, having tracked
    the target rightward up to that point."""
    import numpy as np
    from tracks import propagate_track

    step = 33_333
    rng = np.random.default_rng(1)
    xs, ys, ts = [], [], []
    for k in range(16):
        # target: left→right along y=200
        tx, ty = 60 + k * 14, 200
        xs.append(tx + rng.normal(0, 5, 200)); ys.append(ty + rng.normal(0, 5, 200))
        ts.append(k * step + rng.uniform(0, step, 200))
        # distractor: right→left along the SAME line → they cross ~k=9
        dx, dy = 320 - k * 14, 200
        xs.append(dx + rng.normal(0, 5, 200)); ys.append(dy + rng.normal(0, 5, 200))
        ts.append(k * step + rng.uniform(0, step, 200))
    x = np.concatenate(xs); y = np.concatenate(ys); t = np.concatenate(ts)
    o = np.argsort(t, kind="stable")
    x = x[o].astype(np.uint16); y = y[o].astype(np.uint16); t = t[o].astype(np.uint64)
    arrays = {"t_us": t, "x": x, "y": y, "polarity": np.ones(len(t), np.uint8), "event_id": np.arange(len(t))}
    seed = np.nonzero((t < step) & (np.abs(x.astype(int) - 60) < 18) & (np.abs(y.astype(int) - 200) < 18))[0]
    assert seed.size > 50

    res = propagate_track(arrays, seed, step_us=step, direction="forward", max_slices=30)
    # Paused at the crossing rather than running to the end on the wrong target.
    assert res["stop_reason"] == "ambiguous", res["stop_reason"]
    assert res["candidates"], "ambiguous pause should expose candidates"
    ok = [s for s in res["slices"] if s["status"] == "ok"]
    assert len(ok) >= 4
    # Tracked the target rightward the whole time — never the leftward distractor.
    cxs = [s["cx"] for s in ok]
    assert all(b >= a - 1 for a, b in zip(cxs, cxs[1:])), cxs   # monotonic right
    assert ok[-1]["cx"] < 210                                   # didn't jump to the distractor side


def test_track_parallel_distractor_keeps_target() -> None:
    """A second object travels parallel and nearby. The tracker must stay on the
    original target (identity continuity), not drift onto the distractor."""
    import numpy as np
    from tracks import propagate_track

    step = 33_333
    rng = np.random.default_rng(2)
    xs, ys, ts = [], [], []
    for k in range(16):
        tx, ty = 70 + k * 13, 200          # target lane
        xs.append(tx + rng.normal(0, 4, 200)); ys.append(ty + rng.normal(0, 4, 200))
        ts.append(k * step + rng.uniform(0, step, 200))
        dx, dy = 70 + k * 13, 250          # distractor in a separate parallel lane
        xs.append(dx + rng.normal(0, 4, 200)); ys.append(dy + rng.normal(0, 4, 200))
        ts.append(k * step + rng.uniform(0, step, 200))
    x = np.concatenate(xs); y = np.concatenate(ys); t = np.concatenate(ts)
    o = np.argsort(t, kind="stable")
    x = x[o].astype(np.uint16); y = y[o].astype(np.uint16); t = t[o].astype(np.uint64)
    arrays = {"t_us": t, "x": x, "y": y, "polarity": np.ones(len(t), np.uint8), "event_id": np.arange(len(t))}
    seed = np.nonzero((t < step) & (np.abs(x.astype(int) - 70) < 16) & (np.abs(y.astype(int) - 200) < 16))[0]
    assert seed.size > 50

    res = propagate_track(arrays, seed, step_us=step, direction="forward", max_slices=14)
    ok = [s for s in res["slices"] if s["status"] == "ok"]
    assert len(ok) >= 10                                   # followed the target, not derailed
    assert all(185 <= s["cy"] <= 215 for s in ok), [round(s["cy"]) for s in ok]  # stayed in the target lane
    assert ok[-1]["cx"] > ok[0]["cx"] + 80                 # moved along the target's path


def test_track_merges_object_fragments() -> None:
    """An object whose events split into two separated fragments each slice (as
    event-camera edges do) must stay MOSTLY selected — not just one fragment."""
    import numpy as np
    from tracks import propagate_track

    step = 33_333
    rng = np.random.default_rng(3)
    xs, ys, ts = [], [], []
    for k in range(12):
        cx, cy = 60 + 12 * k, 200
        for dxf in (-35, 35):  # two fragments of one wide object, ~52px gap apart
            xs.append(cx + dxf + rng.normal(0, 3, 100)); ys.append(cy + rng.normal(0, 3, 100))
            ts.append(k * step + rng.uniform(0, step, 100))
    x = np.concatenate(xs); y = np.concatenate(ys); t = np.concatenate(ts)
    o = np.argsort(t, kind="stable")
    x = x[o].astype(np.uint16); y = y[o].astype(np.uint16); t = t[o].astype(np.uint64)
    arrays = {"t_us": t, "x": x, "y": y, "polarity": np.ones(len(t), np.uint8), "event_id": np.arange(len(t))}
    seed = np.nonzero((t < step) & (np.abs(x.astype(int) - 60) < 48) & (np.abs(y.astype(int) - 200) < 12))[0]
    assert seed.size > 150  # seed spans BOTH fragments (the whole wide object)

    res = propagate_track(arrays, seed, step_us=step, direction="forward", max_slices=20)
    ok = [s for s in res["slices"] if s["status"] == "ok"]
    assert len(ok) >= 8
    # Each tracked slice keeps most of the object (~200 events = both fragments),
    # not just one fragment (~100). This is the "only part got tracked" fix.
    assert all(s["count"] >= 150 for s in ok), [s["count"] for s in ok]


def test_track_excludes_adjacent_object() -> None:
    """A dense object (bus) with a SEPARATE object (person) beside it. Seeding
    the bus must not sweep in the person — the over-merge the user hit."""
    import numpy as np
    from tracks import propagate_track

    step = 33_333
    rng = np.random.default_rng(4)
    xs, ys, ts = [], [], []
    for k in range(12):
        bx, by = 100 + 10 * k, 200
        xs.append(rng.uniform(bx - 30, bx + 30, 300)); ys.append(rng.uniform(by - 12, by + 12, 300))
        ts.append(k * step + rng.uniform(0, step, 300))
        px = bx + 55  # a person just beyond the bus's right edge, moving with it
        xs.append(px + rng.normal(0, 4, 100)); ys.append(200 + rng.normal(0, 8, 100))
        ts.append(k * step + rng.uniform(0, step, 100))
    x = np.concatenate(xs); y = np.concatenate(ys); t = np.concatenate(ts)
    o = np.argsort(t, kind="stable")
    x = x[o].astype(np.uint16); y = y[o].astype(np.uint16); t = t[o].astype(np.uint64)
    arrays = {"t_us": t, "x": x, "y": y, "polarity": np.ones(len(t), np.uint8), "event_id": np.arange(len(t))}
    seed = np.nonzero((t < step) & (np.abs(x.astype(int) - 100) < 32) & (np.abs(y.astype(int) - 200) < 14))[0]
    assert seed.size > 200  # the bus only

    res = propagate_track(arrays, seed, step_us=step, direction="forward", max_slices=20)
    ok = [s for s in res["slices"] if s["status"] == "ok"]
    assert len(ok) >= 6
    assert all(s["count"] >= 220 for s in ok), [s["count"] for s in ok]   # kept the bus
    assert all(s["count"] <= 380 for s in ok), [s["count"] for s in ok]   # but NOT the +100 person


def test_open_wsl_style_path() -> None:
    # Recents saved by an earlier WSL/Linux session send "/mnt/d/..." paths;
    # they must map back to the Windows drive path instead of being rejected.
    client = app.test_client()
    source = RECORDINGS_DIR / "stars_0.es"
    drive = source.drive[0].lower()
    wsl_path = "/mnt/" + drive + source.as_posix()[2:]
    r = client.post("/api/datasets/open", json={"path": wsl_path})
    assert r.status_code == 200, r.text
    assert r.get_json()["dataset_id"] == "stars_0"


def test_path_validation() -> None:
    client = app.test_client()
    r = client.post("/api/datasets/open", json={"path": "/etc/passwd"})
    assert r.status_code == 400
    assert "outside" in r.get_json()["error"]


def test_schema_validation() -> None:
    client = app.test_client()
    r = client.put("/api/datasets/stars_0/labels/schema", json={"bad": "data"})
    assert r.status_code == 400
    r = client.put("/api/datasets/stars_0/labels/schema", json={"labels": [{"id": 0, "name": "bg", "color": "#808080"}]})
    assert r.status_code == 200


def test_stats() -> None:
    client = app.test_client()
    r = client.get("/api/datasets/stars_0/stats")
    assert r.status_code == 200
    data = r.get_json()
    assert data["total"] == 308948
    assert "unlabelled" in data
    assert "per_class" in data


def test_filter_label_undo_redo() -> None:
    client = app.test_client()
    r = client.post("/api/datasets/stars_0/labels/update_by_filter", json={
        "label": 1, "operation": "assign", "start_us": 0, "end_us": 5000,
    })
    assert r.status_code == 200
    count = r.get_json()["updated"]
    assert count > 0

    r = client.post("/api/datasets/stars_0/labels/undo")
    assert r.status_code == 200
    assert r.get_json()["updated"] == count

    r = client.post("/api/datasets/stars_0/labels/redo")
    assert r.status_code == 200
    assert r.get_json()["updated"] == count

    r = client.post("/api/datasets/stars_0/labels/undo")
    assert r.status_code == 200
    assert r.get_json()["updated"] == count


def test_lod_points() -> None:
    client = app.test_client()
    r = client.get("/api/datasets/stars_0/points.bin?lod=true&sample=50000")
    assert r.status_code == 200
    assert len(r.data) > 1000
    header_len = struct.unpack("<I", r.data[:4])[0]
    header = json.loads(r.data[4 : 4 + header_len].decode("utf-8"))
    assert header["count"] > 0


def test_recordings_list() -> None:
    client = app.test_client()
    r = client.get("/api/recordings")
    assert r.status_code == 200
    data = r.get_json()
    assert len(data) >= 6
    assert all("name" in rec and "path" in rec for rec in data)


def test_chunked_label_update() -> None:
    client = app.test_client()
    ids = list(range(100))
    r = client.post("/api/datasets/stars_0/labels/update", json={
        "event_ids": ids, "label": 0, "operation": "assign",
    })
    assert r.status_code == 200
    assert r.get_json()["updated"] == 100
    client.post("/api/datasets/stars_0/labels/undo")


if __name__ == "__main__":
    test_smoke()
    print("1/7 smoke: OK")
    test_path_validation()
    print("2/7 path_validation: OK")
    test_export_options()
    print("2.4 export_options: OK")
    test_binary_update_and_stats_delta()
    print("2.45 binary_update_and_stats_delta: OK")
    test_track_propagation()
    print("2.47 track_propagation: OK")
    test_open_wsl_style_path()
    print("2.5 open_wsl_style_path: OK")
    test_schema_validation()
    print("3/7 schema_validation: OK")
    test_stats()
    print("4/7 stats: OK")
    test_filter_label_undo_redo()
    print("5/7 filter_label_undo_redo: OK")
    test_lod_points()
    print("6/7 lod_points: OK")
    test_recordings_list()
    print("7/7 recordings_list: OK")
    test_chunked_label_update()
    print("8/8 chunked_label_update: OK")
    print("\nAll tests passed!")
