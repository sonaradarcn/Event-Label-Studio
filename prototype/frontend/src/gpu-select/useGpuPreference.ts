import { useCallback, useEffect, useState } from "react";

export type GpuOption = {
  name: string;
  vendor: string;
  preference: "high-performance" | "low-power" | "default";
};

const STORAGE_KEY = "gpu_preference";
const STORAGE_NAME_KEY = "gpu_name";

function probeWebGLRenderer(powerPreference: "high-performance" | "low-power" | "default"): { renderer: string; vendor: string } | null {
  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 1;
  const gl = canvas.getContext("webgl2", { powerPreference }) || canvas.getContext("webgl", { powerPreference });
  if (!gl) return null;
  const ext = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  const vendor = ext ? gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR);
  const loseCtx = gl.getExtension("WEBGL_lose_context");
  if (loseCtx) loseCtx.loseContext();
  return { renderer: String(renderer), vendor: String(vendor) };
}

export async function detectGpus(): Promise<GpuOption[]> {
  const options: GpuOption[] = [];
  const seen = new Set<string>();

  // Probe each powerPreference to discover actual GPUs
  for (const pref of ["high-performance", "low-power", "default"] as const) {
    const info = probeWebGLRenderer(pref);
    if (!info) continue;
    // Deduplicate by renderer name (same GPU may appear for multiple prefs)
    if (seen.has(info.renderer)) continue;
    seen.add(info.renderer);
    options.push({ name: info.renderer, vendor: info.vendor, preference: pref });
  }

  // Fallback: if nothing detected
  if (options.length === 0) {
    options.push(
      { name: "High-performance GPU", vendor: "", preference: "high-performance" },
      { name: "Power-saving GPU", vendor: "", preference: "low-power" },
      { name: "System default", vendor: "", preference: "default" },
    );
  }

  return options;
}

export function useGpuPreference() {
  const [preference, setPreference] = useState<"high-performance" | "low-power" | "default">(() => {
    return (localStorage.getItem(STORAGE_KEY) as "high-performance" | "low-power" | "default") || "default";
  });
  const [gpuName, setGpuName] = useState(() => localStorage.getItem(STORAGE_NAME_KEY) || "Default");
  const [showModal, setShowModal] = useState(() => !localStorage.getItem(STORAGE_KEY));
  const [options, setOptions] = useState<GpuOption[]>([]);

  useEffect(() => {
    detectGpus().then(setOptions);
  }, []);

  const select = useCallback((opt: GpuOption) => {
    localStorage.setItem(STORAGE_KEY, opt.preference);
    localStorage.setItem(STORAGE_NAME_KEY, opt.name);
    setPreference(opt.preference);
    setGpuName(opt.name);
    setShowModal(false);
  }, []);

  const openModal = useCallback(() => setShowModal(true), []);

  return { preference, gpuName, showModal, options, select, openModal, closeModal: useCallback(() => { if (localStorage.getItem(STORAGE_KEY)) setShowModal(false); }, []) };
}
