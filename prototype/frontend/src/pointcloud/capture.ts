// Imperative escape hatch so the File menu can capture the live WebGL view
// (3D or XY — whichever is active) without threading refs through the tree.
// PointCloudView registers a capture fn on mount; menu handlers call captureView.

export type CaptureOpts = { transparent?: boolean };
type CaptureFn = (opts?: CaptureOpts) => string | null;
type SvgFn = () => string | null;

let _fn: CaptureFn | null = null;
let _svg: SvgFn | null = null;

export function registerCapture(fn: CaptureFn | null): void {
  _fn = fn;
}

/** Returns a PNG data URL of the current view, or null if nothing is mounted. */
export function captureView(opts?: CaptureOpts): string | null {
  return _fn ? _fn(opts) : null;
}

/** Register the SVG export (points as a high-res image layer, axes/labels as
 *  true vector). Used for the vector PDF print. */
export function registerCaptureSvg(fn: SvgFn | null): void {
  _svg = fn;
}

/** Returns an SVG document string of the current view, or null. */
export function captureSvgView(): string | null {
  return _svg ? _svg() : null;
}

/** Print an inline SVG document via a hidden iframe at the chosen orientation.
 *  The SVG keeps axes/text crisp as vectors in the resulting PDF. */
export function printSvgMarkup(svg: string, orientation: "landscape" | "portrait"): void {
  const iframe = document.createElement("iframe");
  iframe.setAttribute("aria-hidden", "true");
  iframe.style.cssText = "position:fixed;right:0;bottom:0;width:0;height:0;border:0;";
  document.body.appendChild(iframe);
  const cw = iframe.contentWindow;
  if (!cw) { iframe.remove(); return; }
  const doc = cw.document;
  doc.open();
  doc.write(
    `<!doctype html><html><head><meta charset="utf-8"><style>` +
    `@page { size: ${orientation}; margin: 0; }` +
    `html,body { margin:0; padding:0; background:transparent; }` +
    `svg { width:100%; height:100vh; display:block; }` +
    `</style></head><body>${svg}</body></html>`,
  );
  doc.close();
  const go = () => { cw.focus(); cw.print(); setTimeout(() => iframe.remove(), 1000); };
  // Give the embedded raster <image> a tick to decode before printing.
  setTimeout(go, 120);
}

/** Copy a PNG data URL to the system clipboard as an image (image/png).
 *  Requires a secure context — 127.0.0.1 / localhost qualifies. Must be called
 *  from a user gesture (the menu click). */
export async function copyDataUrlToClipboard(dataUrl: string): Promise<void> {
  const comma = dataUrl.indexOf(",");
  const head = dataUrl.slice(0, comma);
  const mime = /:(.*?);/.exec(head)?.[1] || "image/png";
  const bin = atob(dataUrl.slice(comma + 1));
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  const blob = new Blob([arr], { type: mime });
  await navigator.clipboard.write([new ClipboardItem({ [mime]: blob })]);
}

/** Print a PNG data URL via a hidden iframe at the chosen page orientation.
 *  Transparent PNGs print as white paper (no dark viewport rectangle). */
export function printDataUrl(dataUrl: string, orientation: "landscape" | "portrait"): void {
  const iframe = document.createElement("iframe");
  iframe.setAttribute("aria-hidden", "true");
  iframe.style.cssText = "position:fixed;right:0;bottom:0;width:0;height:0;border:0;";
  document.body.appendChild(iframe);
  const cw = iframe.contentWindow;
  if (!cw) { iframe.remove(); return; }
  const doc = cw.document;
  doc.open();
  doc.write(
    `<!doctype html><html><head><meta charset="utf-8"><style>` +
    `@page { size: ${orientation}; margin: 0; }` +
    `html,body { margin:0; padding:0; background:transparent; }` +
    `img { width:100%; height:100vh; object-fit:contain; display:block; }` +
    `</style></head><body><img src="${dataUrl}"></body></html>`,
  );
  doc.close();
  const cleanup = () => setTimeout(() => iframe.remove(), 1000);
  const go = () => { cw.focus(); cw.print(); cleanup(); };
  const img = doc.querySelector("img") as HTMLImageElement | null;
  if (img && !img.complete) { img.onload = go; img.onerror = go; }
  else setTimeout(go, 50);
}
