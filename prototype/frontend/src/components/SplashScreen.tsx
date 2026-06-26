import { useEffect, useState } from "react";
import "./SplashScreen.css";

const TITLE = "Event Label Studio";
const TAGLINE = "Event-camera point-cloud labelling";
const HOLD_MS = 2200;   // time the splash stays before fading out
const FADE_MS = 650;    // must match the CSS opacity transition
const NBSP = " ";

/** Animated startup splash shown on first load; auto-dismisses, click to skip. */
export function SplashScreen() {
  const [leaving, setLeaving] = useState(false);
  const [gone, setGone] = useState(false);

  useEffect(() => {
    const t = window.setTimeout(() => setLeaving(true), HOLD_MS);
    return () => window.clearTimeout(t);
  }, []);

  useEffect(() => {
    if (!leaving) return;
    const t = window.setTimeout(() => setGone(true), FADE_MS);
    return () => window.clearTimeout(t);
  }, [leaving]);

  if (gone) return null;

  return (
    <div
      className={`splash ${leaving ? "splashLeaving" : ""}`}
      onClick={() => setLeaving(true)}
      role="presentation"
    >
      <div className="splashGrid" />
      <div className="splashGlow" />

      <div className="splashDots">
        {Array.from({ length: 30 }).map((_, i) => (
          <span
            key={i}
            className={`splashDot ${i % 2 ? "off" : "on"}`}
            style={{
              left: `${(i * 37) % 100}%`,
              top: `${(i * 53) % 100}%`,
              animationDelay: `${(i % 12) * 160}ms`,
            }}
          />
        ))}
      </div>

      <div className="splashContent">
        <div className="splashTitle">
          {Array.from(TITLE).map((ch, i) => (
            <span key={i} className="splashChar" style={{ animationDelay: `${i * 45}ms` }}>
              {ch === " " ? NBSP : ch}
            </span>
          ))}
        </div>
        <div className="splashLine" />
        <div className="splashTagline">{TAGLINE}</div>
      </div>
    </div>
  );
}
