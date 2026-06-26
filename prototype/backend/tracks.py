"""Propagation tracking — follow one target across time, robust to nearby objects.

This is NOT a global connected-component (that's the wand). It follows a *single*
target slice-by-slice with a constant-velocity prediction and a local
region-of-interest, but inside that ROI it now:

  1. clusters the events into candidate blobs (so a second object entering the
     ROI is a *separate* candidate, not absorbed into one inflated centroid);
  2. scores each candidate against a carried identity model (position, count,
     size, polarity) + the motion prediction, and picks the best;
  3. detects target crossing by the best-vs-second-best score margin — when two
     candidates are comparably good it PAUSES (status "ambiguous") instead of
     silently switching identity.

This directly fixes the "nearby object breaks propagation" failure: when the
distractor is separate, the target still wins on identity; when they actually
overlap/cross, the margin collapses and we hand control back to the user.

No Kalman / optical-flow / deep model yet — a small, explainable, CPU-only
baseline that the correction loop is built around. Coordinates are sensor pixels
(x, y); time is microseconds; event ids are global array indices. Time ascends,
so per-slice windows come from two binary searches.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label as _nd_label, binary_dilation as _nd_dilate

# Caps so a single (synchronous) request stays bounded even on 16M-event sets.
DEFAULT_STEP_US = 33_333          # ~30 fps slice; independent of the view window
MAX_SLICES_CAP = 240
PER_SLICE_EVENT_CAP = 120_000     # raised: a dense car can be tens of k / slice;
TOTAL_EVENT_CAP = 1_500_000       # the old 20k/400k caps thinned the selection
CONF_THRESHOLD = 0.40             # below this a slice is "low" → auto-pause
ROI_MARGIN = 1.7                  # ROI half-size = prev half-extent × margin
MIN_HALF_EXTENT = 10.0            # px floor so tiny seeds still have a search box

# Candidate clustering / crossing detection.
CLUSTER_CELL = 4.0                # px per occupancy-grid cell (coarse → fast)
CLUSTER_CLOSE = 1                 # grid-dilation iters (≈4px). Kept SMALL so
                                  # clustering does NOT bridge the target to a
                                  # nearby object (that snowballed into the bus
                                  # swallowing the person). Reuniting the target's
                                  # OWN scattered fragments is instead handled by
                                  # the footprint-overlap merge below.
MARGIN_MIN = 0.12                 # best-vs-second score gap below this = crossing
AMBIG_SECOND_FLOOR = 0.40         # ...only if the runner-up is itself plausible
OVERLAP_MERGE = 0.25              # a blob is a fragment of the SAME object (merge)
                                  # if ≥ this fraction of it overlaps the object's
                                  # previous footprint (motion-compensated). This
                                  # reunites the object's scattered pieces but
                                  # EXCLUDES a separate nearby object (e.g. a
                                  # person beside the bus) that doesn't overlap
                                  # the object's past — a distance radius can't
                                  # tell those apart, footprint overlap can.
GROW_CAP = 1.3                    # a fragment may only join if it keeps the merged
                                  # extent ≤ object_extent×this — stops a nearby
                                  # object (person) being absorbed + snowballing.
CAND_EVENT_CAP = 5_000            # ids returned per candidate on an ambiguous pause
_NB8 = np.ones((3, 3), dtype=np.int32)  # 8-connectivity structuring element

# Identity-match score weights (sum = 1). Footprint IoU (spatial continuity with
# the previous slice, motion-compensated) is the strongest cue for picking the
# SAME object out of several nearby blobs; position/count/size/polarity refine it.
IOU_CELL = 4.0                    # px per cell for the footprint-overlap grid
FOOTPRINT_DILATE = 3              # dilate the object footprint by this many cells
                                  # (≈12px) so it represents the object REGION,
                                  # not just its sparse event edges — a moving
                                  # object's new edges then still land inside it
W_IOU, W_POS, W_COUNT, W_SIZE, W_POL = 0.35, 0.25, 0.15, 0.15, 0.10


def _half_extent(vals: np.ndarray) -> float:
    if vals.size == 0:
        return MIN_HALF_EXTENT
    return max(MIN_HALF_EXTENT, float(vals.max() - vals.min()) / 2.0)


def _cluster_blobs(xs: np.ndarray, ys: np.ndarray, cell: float) -> list[np.ndarray]:
    """8-connected components on a coarse occupancy grid.

    Returns a list of arrays of LOCAL indices (into xs/ys), one per blob. The
    grid is bounded by the ROI box, so it stays tiny regardless of event count.
    """
    n = xs.size
    if n == 0:
        return []
    x0, y0 = float(xs.min()), float(ys.min())
    gx = ((xs - x0) / cell).astype(np.int32)
    gy = ((ys - y0) / cell).astype(np.int32)
    w = int(gx.max()) + 1
    h = int(gy.max()) + 1
    grid = np.zeros((h, w), dtype=np.uint8)
    grid[gy, gx] = 1
    # Dilate before labelling so internal sparsity (event cameras fire on edges,
    # so one object is many disconnected fragments) doesn't split the target into
    # pieces — that was the "only part of the target got selected" bug. We label
    # the dilated grid but keep the ORIGINAL events' component membership, so
    # measurements (centroid/size/IoU) stay on the real points.
    if CLUSTER_CLOSE > 0:
        grid = _nd_dilate(grid, structure=_NB8, iterations=CLUSTER_CLOSE).astype(np.uint8)
    lab, nlab = _nd_label(grid, structure=_NB8)
    if nlab <= 1:
        return [np.arange(n, dtype=np.int64)]
    ev_lab = lab[gy, gx]
    blobs: list[np.ndarray] = []
    for li in range(1, nlab + 1):
        sel = np.nonzero(ev_lab == li)[0]
        if sel.size:
            blobs.append(sel.astype(np.int64))
    return blobs


def _cell_keys(xs: np.ndarray, ys: np.ndarray, cell: float) -> np.ndarray:
    """Unique occupancy-grid cell keys for a blob's footprint (for IoU)."""
    gx = np.floor(xs / cell).astype(np.int64)
    gy = np.floor(ys / cell).astype(np.int64)
    return np.unique(gx * 100003 + gy)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two sets of cell keys."""
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = int(np.intersect1d(a, b, assume_unique=True).size)
    union = a.size + b.size - inter
    return inter / union if union else 0.0


def _dilate_keys(keys: np.ndarray, r: int) -> np.ndarray:
    """Grow a set of cell keys by ±r cells in both axes → a filled region mask.
    Keys are gx*100003 + gy with gx, gy ≥ 0 (called before any velocity shift)."""
    if keys.size == 0 or r <= 0:
        return keys
    gx = keys // 100003
    gy = keys % 100003
    outs = []
    for dx in range(-r, r + 1):
        base = (gx + dx) * 100003
        for dy in range(-r, r + 1):
            outs.append(base + (gy + dy))
    return np.unique(np.concatenate(outs))


def propagate_track(
    arrays: dict[str, np.ndarray],
    seed_event_ids: np.ndarray,
    *,
    step_us: int = DEFAULT_STEP_US,
    direction: str = "forward",
    max_slices: int = 60,
    stop_on_low: bool = True,
    conf_threshold: float = CONF_THRESHOLD,
) -> dict:
    """Follow the seed target across `max_slices` time slices in `direction`.

    Returns dict with:
      slices: [{t_start, t_end, t_center, cx, cy, count, confidence, status,
                event_ids}] — status ∈ {ok, low, ambiguous, lost}
      frontier_event_ids: events of the last OK slice (seed for a "continue")
      pause_time_us: t_center of the slice the UI should jump to, or None
      candidates: on an ambiguous pause, the competing blobs
                  [{cx, cy, count, score, event_ids}] so the UI can let the user
                  pick which one to keep tracking; [] otherwise
      stop_reason: max_slices | low_confidence | ambiguous | no_events | edge | total_cap
    """
    t = arrays["t_us"]
    x = arrays["x"]
    y = arrays["y"]
    pol = arrays.get("polarity")
    n = int(len(t))
    seed = np.asarray(seed_event_ids, dtype=np.int64)
    seed = seed[(seed >= 0) & (seed < n)]
    if seed.size == 0:
        return {"slices": [], "frontier_event_ids": [], "pause_time_us": None,
                "candidates": [], "stop_reason": "no_seed"}

    sgn = -1 if direction == "backward" else 1
    step = max(1, int(step_us))
    max_slices = min(int(max_slices), MAX_SLICES_CAP)

    sx = x[seed].astype(np.float64)
    sy = y[seed].astype(np.float64)
    stime = t[seed].astype(np.float64)
    # Keyframe anchor: just past the seed in the propagation direction.
    cur_t = float(stime.max()) if sgn > 0 else float(stime.min())
    prev_cx, prev_cy = float(sx.mean()), float(sy.mean())
    seed_hx, seed_hy = _half_extent(sx), _half_extent(sy)
    prev_hx, prev_hy = seed_hx, seed_hy
    ref_count = float(seed.size)   # slow baseline → catches gradual target loss
    ref_hx, ref_hy = seed_hx, seed_hy  # slow size baseline → catches sudden merges
    ref_pol = float((pol[seed] == 1).mean()) if pol is not None else 0.5
    prev_keys = _dilate_keys(_cell_keys(sx, sy, IOU_CELL), FOOTPRINT_DILATE)  # filled object footprint
    vx = vy = 0.0  # per-step centroid displacement (constant-velocity model)

    t_lo_all = float(t[0])
    t_hi_all = float(t[n - 1])

    slices: list[dict] = []
    frontier: list[int] = []
    pause_time = None
    candidates: list[dict] = []
    diag: list[dict] = []
    stop_reason = "max_slices"
    total_events = 0

    def _emit(g: np.ndarray) -> list[int]:
        ev = g.tolist()
        if len(ev) > PER_SLICE_EVENT_CAP:
            stride = int(np.ceil(len(ev) / PER_SLICE_EVENT_CAP))
            ev = ev[::stride]
        return ev

    for k in range(1, max_slices + 1):
        if sgn > 0:
            w0 = cur_t + (k - 1) * step
            w1 = w0 + step
        else:
            w1 = cur_t - (k - 1) * step
            w0 = w1 - step
        t_center = (w0 + w1) / 2.0

        # Ran off the end of the recording.
        if w1 <= t_lo_all or w0 >= t_hi_all:
            stop_reason = "edge"
            break

        # Predicted target centre this slice (constant velocity).
        pred_cx = prev_cx + vx
        pred_cy = prev_cy + vy
        roi_hx = max(prev_hx * ROI_MARGIN, 12.0)
        roi_hy = max(prev_hy * ROI_MARGIN, 12.0)
        roi_diag = float(np.hypot(roi_hx, roi_hy)) or 1.0

        lo = int(np.searchsorted(t, w0, side="left"))
        hi = int(np.searchsorted(t, w1, side="left"))
        if hi <= lo:
            cand = np.empty(0, dtype=np.int64)
        else:
            xs = x[lo:hi]
            ys = y[lo:hi]
            in_roi = (
                (xs >= pred_cx - roi_hx) & (xs <= pred_cx + roi_hx)
                & (ys >= pred_cy - roi_hy) & (ys <= pred_cy + roi_hy)
            )
            cand = (np.nonzero(in_roi)[0] + lo).astype(np.int64)

        if cand.size == 0:
            # No support in the predicted box → target lost / temporarily no events.
            slices.append({
                "t_start": int(w0), "t_end": int(w1), "t_center": int(t_center),
                "cx": float(pred_cx), "cy": float(pred_cy), "count": 0,
                "confidence": 0.0, "status": "lost", "event_ids": [],
            })
            pause_time = int(t_center)
            stop_reason = "no_events"
            break

        # --- Step 1: cluster the ROI into candidate blobs -------------------
        xc = x[cand].astype(np.float64)
        yc = y[cand].astype(np.float64)
        blobs_local = _cluster_blobs(xc, yc, CLUSTER_CELL)

        # Motion-compensate the previous footprint so a moving target still
        # overlaps itself (shift the prev cells by the predicted velocity).
        shift = int(round(vx / IOU_CELL)) * 100003 + int(round(vy / IOU_CELL))
        prev_keys_shifted = prev_keys + shift

        # --- Step 2: score each candidate against the identity model --------
        scored: list[dict] = []
        for bl in blobs_local:
            g = cand[bl]
            bcx = float(x[g].mean())
            bcy = float(y[g].mean())
            bhx = _half_extent(x[g].astype(np.float64))
            bhy = _half_extent(y[g].astype(np.float64))
            bcount = int(g.size)
            bpol = float((pol[g] == 1).mean()) if pol is not None else ref_pol
            bkeys = _cell_keys(x[g].astype(np.float64), y[g].astype(np.float64), IOU_CELL)
            inter = int(np.intersect1d(bkeys, prev_keys_shifted, assume_unique=True).size)
            union = bkeys.size + prev_keys_shifted.size - inter
            iou = inter / union if union else 0.0
            ovl = inter / max(1, bkeys.size)   # fraction of THIS blob inside the prev footprint
            dist = float(np.hypot(bcx - pred_cx, bcy - pred_cy))
            pos_sim = 1.0 - min(1.0, dist / roi_diag)
            count_ratio = min(bcount, ref_count) / max(bcount, ref_count, 1.0)
            size_ratio = (
                min(bhx, prev_hx) / max(bhx, prev_hx, 1e-6)
                * min(bhy, prev_hy) / max(bhy, prev_hy, 1e-6)
            )
            pol_sim = 1.0 - min(1.0, abs(bpol - ref_pol) * 2.0)
            score = (W_IOU * iou + W_POS * pos_sim + W_COUNT * count_ratio
                     + W_SIZE * size_ratio + W_POL * pol_sim)
            scored.append({
                "g": g, "cx": bcx, "cy": bcy, "hx": bhx, "hy": bhy,
                "count": bcount, "pol": bpol, "keys": bkeys, "score": score, "ovl": ovl,
            })
        scored.sort(key=lambda d: d["score"], reverse=True)
        best = scored[0]

        # --- Step 2.5: main blob + compatible fragments --------------------
        # Event-camera targets are sparse edge fragments, so the SAME object is
        # usually several blobs. Re-unite them: any other candidate whose centroid
        # falls within the object's own extent (around the main blob) is a
        # fragment → merge it; candidates farther out are separate competitors
        # (used only for crossing detection). This is what stops "only part of
        # the object got tracked".
        # A blob joins the object only if (a) most of it overlaps the object's
        # previous (motion-compensated) footprint AND (b) adding it keeps the
        # object's extent bounded. (a) reunites the target's scattered fragments;
        # (b) stops a nearby object (e.g. a person beside the bus) from being
        # absorbed and then snowballing. Everything else is a separate competitor
        # (used for crossing detection). Process by overlap so fragments win.
        # Cap against the STABLE seed extent (not prev_hx, which can ratchet up
        # once a stray blob inflates it and then never lets go).
        max_hx = seed_hx * GROW_CAP + 8.0
        max_hy = seed_hy * GROW_CAP + 8.0
        merged = [best]
        bxmin, bxmax = best["cx"] - best["hx"], best["cx"] + best["hx"]
        bymin, bymax = best["cy"] - best["hy"], best["cy"] + best["hy"]
        competitors: list[dict] = []
        for s in sorted(scored[1:], key=lambda d: d["ovl"], reverse=True):
            nxmin, nxmax = min(bxmin, s["cx"] - s["hx"]), max(bxmax, s["cx"] + s["hx"])
            nymin, nymax = min(bymin, s["cy"] - s["hy"]), max(bymax, s["cy"] + s["hy"])
            if s["ovl"] >= OVERLAP_MERGE and (nxmax - nxmin) / 2 <= max_hx and (nymax - nymin) / 2 <= max_hy:
                merged.append(s)
                bxmin, bxmax, bymin, bymax = nxmin, nxmax, nymin, nymax
            else:
                competitors.append(s)
        g_all = np.concatenate([s["g"] for s in merged]) if len(merged) > 1 else best["g"]
        cx = float(x[g_all].mean())
        cy = float(y[g_all].mean())
        hx = _half_extent(x[g_all].astype(np.float64))
        hy = _half_extent(y[g_all].astype(np.float64))
        count = int(g_all.size)
        bpol = float((pol[g_all] == 1).mean()) if pol is not None else ref_pol
        keys_all = _cell_keys(x[g_all].astype(np.float64), y[g_all].astype(np.float64), IOU_CELL)

        # --- Step 3: confidence + crossing detection -----------------------
        # Score the MERGED object (not the single best fragment, which looks too
        # small/sparse on its own and would falsely read as low confidence).
        m_inter = int(np.intersect1d(keys_all, prev_keys_shifted, assume_unique=True).size)
        m_union = keys_all.size + prev_keys_shifted.size - m_inter
        m_iou = m_inter / m_union if m_union else 0.0
        m_pos = 1.0 - min(1.0, float(np.hypot(cx - pred_cx, cy - pred_cy)) / roi_diag)
        m_count_ratio = min(count, ref_count) / max(count, ref_count, 1.0)
        m_size_ratio = (min(hx, prev_hx) / max(hx, prev_hx, 1e-6)) * (min(hy, prev_hy) / max(hy, prev_hy, 1e-6))
        m_pol = 1.0 - min(1.0, abs(bpol - ref_pol) * 2.0)
        merged_score = W_IOU * m_iou + W_POS * m_pos + W_COUNT * m_count_ratio + W_SIZE * m_size_ratio + W_POL * m_pol
        # support gates everything (target vanished → count collapses → low).
        second = competitors[0]["score"] if competitors else 0.0
        support = min(1.0, count / max(1.0, 0.5 * ref_count))
        confidence = max(0.0, min(1.0, support * merged_score))
        # Two reasons to suspect a nearby-object collision and hand back control:
        #  (a) a comparably-good SEPARATE competitor near the prediction = a
        #      crossing; (b) the merged object's count/size suddenly jumps vs the
        #      slow baseline = it absorbed another object (a merge). Either way,
        #      pause rather than risk a silent identity switch (the priority rule).
        crossing = bool(competitors) and second >= AMBIG_SECOND_FLOOR and (merged_score - second) < MARGIN_MIN
        size_jump = k > 2 and (
            count > ref_count * 1.9 or hx > ref_hx * 1.9 or hy > ref_hy * 1.9
        )
        ambiguous = crossing or size_jump
        if ambiguous:
            status = "ambiguous"
        elif confidence < conf_threshold:
            status = "low"
        else:
            status = "ok"

        ev = _emit(g_all)
        # Observability: how much of the ROI did we actually keep this slice?
        diag.append({
            "roi": int(cand.size), "blobs": len(blobs_local),
            "merged_blobs": len(merged), "competitors": len(competitors),
            "best_count": int(best["count"]), "kept": int(count),
            "kept_frac_roi": round(count / max(1, int(cand.size)), 3),
            "status": status,
        })
        slices.append({
            "t_start": int(w0), "t_end": int(w1), "t_center": int(t_center),
            "cx": cx, "cy": cy, "count": count,
            "confidence": round(confidence, 3), "status": status, "event_ids": ev,
        })

        if status == "ok":
            frontier = ev
        total_events += len(ev)

        if ambiguous:
            pause_time = int(t_center)
            stop_reason = "ambiguous"
            # Candidate 1 = the merged main object; then the separate competitors.
            def _cap(e: list[int]) -> list[int]:
                return e[::int(np.ceil(len(e) / CAND_EVENT_CAP))] if len(e) > CAND_EVENT_CAP else e
            candidates = [{"cx": cx, "cy": cy, "count": count, "score": round(best["score"], 3), "event_ids": _cap(g_all.tolist())}]
            candidates += [
                {"cx": s["cx"], "cy": s["cy"], "count": s["count"], "score": round(s["score"], 3), "event_ids": _cap(s["g"].tolist())}
                for s in competitors[:3]
            ]
            break
        if status == "low" and stop_on_low:
            pause_time = int(t_center)
            stop_reason = "low_confidence"
            break
        if total_events >= TOTAL_EVENT_CAP:
            stop_reason = "total_cap"
            break

        # Constant-velocity update (EMA so a single noisy slice doesn't whip the
        # predictor around) — all from the MERGED object.
        vx = 0.6 * vx + 0.4 * (cx - prev_cx)
        vy = 0.6 * vy + 0.4 * (cy - prev_cy)
        # Clamp the carried blob size to a band around the seed so a noisy slice
        # can't inflate the ROI and snowball.
        prev_hx = min(max(hx, seed_hx * 0.6), seed_hx * 2.0)
        prev_hy = min(max(hy, seed_hy * 0.6), seed_hy * 2.0)
        prev_cx, prev_cy = cx, cy
        prev_keys = _dilate_keys(keys_all, FOOTPRINT_DILATE)  # carry the filled merged footprint
        ref_count = 0.8 * ref_count + 0.2 * count   # slow baselines
        ref_hx = 0.8 * ref_hx + 0.2 * hx
        ref_hy = 0.8 * ref_hy + 0.2 * hy
        ref_pol = 0.8 * ref_pol + 0.2 * bpol

    return {
        "slices": slices,
        "frontier_event_ids": frontier,
        "pause_time_us": pause_time,
        "candidates": candidates,
        "diag": diag[:80],
        "stop_reason": stop_reason,
        "step_us": step,
        "direction": "backward" if sgn < 0 else "forward",
    }
