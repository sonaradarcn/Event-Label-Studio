"""Desktop window for Event Label Studio (pywebview).

Runs the existing Flask app on a free local port in a background thread, then
opens a native OS window pointing at it — the whole UI + API on one origin, so
the same page you see in the browser shows up as a desktop window on
Windows / Ubuntu / macOS.

  Windows : pywebview uses WebView2 (Chromium) — same WebGL as Chrome.
  macOS   : WKWebView (WebKit).
  Linux   : WebKitGTK  (install the system webkit2gtk packages + pywebview[gtk]).

Run:  python prototype/backend/desktop.py   (frontend must be built first:
      cd prototype/frontend && npm run build)
"""
from __future__ import annotations

import socket
import sys
import threading
import time

from app import app
from config import PROTOTYPE_ROOT

import webview  # type: ignore


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_until_up(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.3):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def main() -> None:
    dist = PROTOTYPE_ROOT / "frontend" / "dist"
    if not (dist / "index.html").exists():
        print("[desktop] Frontend not built. Run:\n"
              "  cd prototype/frontend && npm run build", file=sys.stderr)
        sys.exit(1)

    port = _free_port()
    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False),
        daemon=True,
    ).start()
    if not _wait_until_up(port):
        print("[desktop] Backend failed to start.", file=sys.stderr)
        sys.exit(1)

    webview.create_window(
        "Event Label Studio",
        f"http://127.0.0.1:{port}",
        width=1480,
        height=920,
        min_size=(1000, 640),
    )
    # gui auto-detects: Windows -> edgechromium (WebView2), macOS -> cocoa,
    # Linux -> gtk. WebView2 gives Chromium-grade WebGL on Windows.
    webview.start()


if __name__ == "__main__":
    main()
