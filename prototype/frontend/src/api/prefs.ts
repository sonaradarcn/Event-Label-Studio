// Shared client preferences (recents + UI config).
//
// localStorage is per-origin, so the dev browser, the pywebview desktop window
// (which runs on a different — and changing — port), and any other browser each
// have their own isolated copy. The Python backend is the single shared point,
// so we mirror a whitelist of pref keys to it: on boot we pull the server copy
// into localStorage *before React renders* (so the existing localStorage-based
// state initializers read the shared values), and we mirror future writes back.
//
// Everything degrades gracefully: if the server is unreachable the app just
// keeps using local-only storage as before.

const API = "/api";

// Keys whose values should be shared across every client.
const SHARED_KEYS = new Set<string>([
  "recent_files",            // recently opened files
  "display.unlabeledColor",  // settings → display
  "display.polarityContrast",
  "gpu_preference",          // GPU select
  "gpu_name",
  "locale",                  // language
  "theme",                   // dark / light
]);

let pending: Record<string, string | null> = {};
let flushTimer: ReturnType<typeof setTimeout> | null = null;

function flush(): void {
  flushTimer = null;
  const body = pending;
  pending = {};
  if (Object.keys(body).length === 0) return;
  fetch(`${API}/prefs`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).catch(() => { /* offline / static host — keep local-only */ });
}

function queue(key: string, value: string | null): void {
  pending[key] = value;
  if (flushTimer != null) clearTimeout(flushTimer);
  flushTimer = setTimeout(flush, 300);
}

// Pull shared prefs from the server into localStorage, then start mirroring
// writes back. Resolves even on failure so the app always renders.
export async function bootstrapPrefs(): Promise<void> {
  const rawSet = localStorage.setItem.bind(localStorage);
  let server: Record<string, unknown> = {};
  let reachable = false;
  try {
    const res = await fetch(`${API}/prefs`);
    if (res.ok) { server = (await res.json()) as Record<string, unknown>; reachable = true; }
  } catch {
    /* server unreachable — fall back to whatever is already in localStorage */
  }

  if (reachable) {
    // Server is the shared source of truth → seed localStorage from it.
    for (const [k, v] of Object.entries(server)) {
      if (SHARED_KEYS.has(k) && v != null) {
        try { rawSet(k, String(v)); } catch { /* ignore quota */ }
      }
    }
    // First-run migration: upload any pref this client already had locally but
    // the server doesn't know yet (so existing recents/config aren't lost and
    // become visible to the desktop window / other browsers).
    const migrate: Record<string, string> = {};
    for (const k of SHARED_KEYS) {
      if (server[k] == null) {
        const local = localStorage.getItem(k);
        if (local != null) migrate[k] = local;
      }
    }
    if (Object.keys(migrate).length > 0) {
      fetch(`${API}/prefs`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(migrate),
      }).catch(() => { /* best-effort */ });
    }
  }

  // Wrap setItem/removeItem so future writes of shared keys reach the server.
  // Installed *after* seeding so the seed writes above don't echo back.
  const rawRemove = localStorage.removeItem.bind(localStorage);
  localStorage.setItem = (key: string, value: string) => {
    rawSet(key, value);
    if (SHARED_KEYS.has(key)) queue(key, value);
  };
  localStorage.removeItem = (key: string) => {
    rawRemove(key);
    if (SHARED_KEYS.has(key)) queue(key, null);
  };
}
