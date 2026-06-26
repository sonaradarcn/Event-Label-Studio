from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOTYPE_ROOT = ROOT / "prototype"
RECORDINGS_DIR = ROOT / "event_camera_get_started" / "recordings"
CACHE_DIR = PROTOTYPE_ROOT / "cache"
OUTPUT_DIR = PROTOTYPE_ROOT / "output"

DEFAULT_LABELS = [
    {"id": 0, "name": "target", "color": "#ff6430"},
    {"id": 1, "name": "noise", "color": "#3c8cff"},
]

ON_COLOR = (255, 100, 40)
OFF_COLOR = (40, 140, 255)
SELECTED_COLOR = (255, 255, 255)
DEFAULT_WINDOW_US = 1_000_000
DEFAULT_SAMPLE = 200_000

