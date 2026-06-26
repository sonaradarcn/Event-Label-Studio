import { useEffect, useRef, useState } from "react";

/**
 * Live frames-per-second meter, sampled every `sampleMs` (default 500 ms).
 *
 * Runs its own requestAnimationFrame loop on the main thread — the same thread
 * the Three.js renderer uses — so when 3D rendering is heavy the callbacks lag
 * and the reported FPS drops. It therefore doubles as a practical render-load
 * indicator (the same technique stats.js uses), without having to hook into the
 * render loop itself.
 */
export function useFps(sampleMs = 500): number {
  const [fps, setFps] = useState(0);
  const framesRef = useRef(0);
  const lastRef = useRef(0);

  useEffect(() => {
    let raf = 0;
    lastRef.current = performance.now();
    framesRef.current = 0;
    const tick = (now: number) => {
      framesRef.current += 1;
      const dt = now - lastRef.current;
      if (dt >= sampleMs) {
        setFps(Math.round((framesRef.current * 1000) / dt));
        framesRef.current = 0;
        lastRef.current = now;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [sampleMs]);

  return fps;
}
