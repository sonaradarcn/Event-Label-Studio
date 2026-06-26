// Browser-side telemetry: hooks console, window errors, unhandled rejections
// and periodic memory snapshots. Posts batches to /api/devlog so the dev can
// `tail -f prototype/cache/devlog.jsonl` and watch what the page is doing in
// real time. Lightweight: ~200 LOC, no extra deps.

type LogEntry = {
  level: "log" | "info" | "warn" | "error" | "memory" | "tag" | "perf";
  msg: string;
  args?: unknown[];
  stack?: string;
  url?: string;
  t: number;
  jsHeapMB?: number;
  totalHeapMB?: number;
  limitHeapMB?: number;
};

const queue: LogEntry[] = [];
let flushTimer: number | null = null;
let initialized = false;
let memTimer: number | null = null;

function safeStringify(arg: unknown): string {
  if (arg instanceof Error) return arg.stack || `${arg.name}: ${arg.message}`;
  if (typeof arg === "object" && arg !== null) {
    try { return JSON.stringify(arg); } catch { return Object.prototype.toString.call(arg); }
  }
  return String(arg);
}

function send(): void {
  if (queue.length === 0) return;
  const batch = queue.splice(0, queue.length);
  // keepalive lets us flush on unload too.
  fetch("/api/devlog", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(batch),
    keepalive: true,
  }).catch(() => { /* swallow — telemetry must never throw */ });
}

function scheduleFlush(): void {
  if (flushTimer != null) return;
  flushTimer = window.setTimeout(() => {
    flushTimer = null;
    send();
  }, 250);
}

function record(entry: Omit<LogEntry, "t">): void {
  queue.push({ ...entry, t: Date.now() });
  // Flush sooner if errors pile up.
  if (entry.level === "error" || queue.length >= 32) send();
  else scheduleFlush();
}

function snapshotMemory(): void {
  const m = (performance as unknown as {
    memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number };
  }).memory;
  if (!m) return;
  const used = m.usedJSHeapSize / 1048576;
  const total = m.totalJSHeapSize / 1048576;
  const limit = m.jsHeapSizeLimit / 1048576;
  record({
    level: "memory",
    msg: `js=${used.toFixed(0)}MB total=${total.toFixed(0)}MB limit=${limit.toFixed(0)}MB`,
    jsHeapMB: used,
    totalHeapMB: total,
    limitHeapMB: limit,
  });

  // Try the newer cross-origin-isolated API for total per-process memory.
  // Includes typed-array external memory which `performance.memory` misses.
  // Throttled — only request once per 5 s to avoid the 100 ms call cost.
  const perf = performance as unknown as { measureUserAgentSpecificMemory?: () => Promise<{ bytes: number; breakdown: unknown[] }> };
  if (perf.measureUserAgentSpecificMemory && Date.now() - lastUaMemAt > 5000) {
    lastUaMemAt = Date.now();
    perf.measureUserAgentSpecificMemory().then((r) => {
      record({
        level: "memory",
        msg: `ua-total=${(r.bytes / 1048576).toFixed(0)}MB (incl. typed arrays / external)`,
        jsHeapMB: r.bytes / 1048576,
      });
    }).catch(() => { /* requires cross-origin isolation; ignore otherwise */ });
  }
}

let lastUaMemAt = 0;

/** Explicit allocation tracker — call before / after big typed-array allocs to
 *  surface external-memory pressure that performance.memory does not report. */
export function trackAlloc(label: string, bytes: number): void {
  record({
    level: "memory",
    msg: `alloc ${label} +${(bytes / 1048576).toFixed(1)}MB external`,
  });
  // Also force an immediate JS-heap sample so we see the timing.
  snapshotMemory();
}

export function tag(msg: string): void {
  record({ level: "tag", msg });
  send();
}

export function clearDevLog(): void {
  fetch("/api/devlog/clear", { method: "POST" }).catch(() => { /* ignore */ });
}

export function initDevLog(): void {
  if (initialized) return;
  initialized = true;

  // Wrap console methods.
  const origs: Record<string, (...a: unknown[]) => void> = {
    log: console.log.bind(console),
    info: console.info.bind(console),
    warn: console.warn.bind(console),
    error: console.error.bind(console),
  };
  (["log", "info", "warn", "error"] as const).forEach((level) => {
    (console[level] as (...a: unknown[]) => void) = (...args: unknown[]) => {
      try {
        record({ level, msg: args.map(safeStringify).join(" ") });
      } catch { /* never break console */ }
      origs[level](...args);
    };
  });

  // Uncaught synchronous errors.
  window.addEventListener("error", (e) => {
    record({
      level: "error",
      msg: e.message || "uncaught error",
      stack: (e.error as Error | undefined)?.stack,
      url: `${e.filename}:${e.lineno}:${e.colno}`,
    });
  });

  // Unhandled promise rejections.
  window.addEventListener("unhandledrejection", (e) => {
    const reason = e.reason;
    record({
      level: "error",
      msg: "unhandledrejection: " + safeStringify(reason),
      stack: (reason as Error | undefined)?.stack,
    });
  });

  // Memory snapshot every 2 seconds. Cheap (few µs).
  memTimer = window.setInterval(snapshotMemory, 2000);

  // Final flush before unload.
  window.addEventListener("beforeunload", () => {
    snapshotMemory();
    send();
  });

  // Expose tag()/clear() to the console for manual session marking.
  (window as unknown as { _dev?: object })._dev = {
    tag,
    clear: clearDevLog,
    snapshotMemory,
    flush: send,
  };

  // Boot marker.
  record({ level: "tag", msg: `devlog started ua=${navigator.userAgent}` });
  snapshotMemory();
}

// Optional convenience: clean up if HMR ever re-imports this module.
if ((import.meta as unknown as { hot?: { dispose: (cb: () => void) => void } }).hot) {
  (import.meta as unknown as { hot: { dispose: (cb: () => void) => void } }).hot.dispose(() => {
    if (memTimer != null) window.clearInterval(memTimer);
  });
}
