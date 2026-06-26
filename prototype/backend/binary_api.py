from __future__ import annotations

import json
import struct

import numpy as np


def pack_arrays(arrays: list[tuple[str, np.ndarray]]) -> bytes:
    """Non-streaming variant — kept for callers that want a single bytes blob."""
    return b"".join(iter_pack_arrays(arrays))


def iter_pack_arrays(arrays: list[tuple[str, np.ndarray]], chunk_size: int = 1 << 20):
    """Yield the same bytes as pack_arrays() but in chunks suitable for
    streaming responses. Lets the frontend allocate final typed-arrays from
    the header and copy each chunk straight into them — no need to hold the
    whole 500-MB+ payload in JS memory."""
    manifest = {"version": 2, "count": int(len(arrays[0][1])) if arrays else 0, "arrays": []}
    contiguous_arrays: list[tuple[np.ndarray, int]] = []  # (array, leading_pad)

    cursor = 0
    for name, array in arrays:
        contiguous = np.ascontiguousarray(array)
        align = max(int(contiguous.dtype.itemsize), 1)
        pad = (-cursor) % align
        cursor += pad
        manifest["arrays"].append({
            "name": name,
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "bytes": int(contiguous.nbytes),
            "offset": cursor,
        })
        contiguous_arrays.append((contiguous, pad))
        cursor += int(contiguous.nbytes)

    header = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    total = 4 + len(header)
    pad_count = (-total) % 8
    if pad_count:
        header = header + b" " * pad_count

    yield struct.pack("<I", len(header)) + header

    for contiguous, pad in contiguous_arrays:
        if pad:
            yield b"\x00" * pad
        # Stream from the numpy buffer in chunk_size pieces — avoids a single
        # giant tobytes() allocation (~190 MB for positions on 16 M events).
        mv = memoryview(contiguous).cast("B")
        n = len(mv)
        for off in range(0, n, chunk_size):
            yield bytes(mv[off:off + chunk_size])

