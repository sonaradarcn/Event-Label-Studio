import { useRef } from "react";
import { Crosshair, Play, X, MapPin, Check } from "lucide-react";
import type { TrackUiState } from "../hooks/useAppState";
import { useI18n } from "../i18n/I18nContext";

type Props = {
  track: TrackUiState | null;
  selectedCount: number;
  trackCreate: () => void;
  trackPropagate: () => void;
  trackUseSelectionAsSeed: () => void;
  trackPreviewCandidate: (index: number) => void;
  trackConfirmCandidate: () => void;
  trackCancel: () => void;
  trackSetDirection: (d: "forward" | "backward") => void;
  currentFrameUs: number;
  onSeek: (us: number) => void;
};

// Draggable progress bar over the track's covered time range, synced with the
// bottom playback bar (both drive currentFrameUs).
function TrackProgress({ startUs, endUs, currentUs, onSeek }: {
  startUs: number; endUs: number; currentUs: number; onSeek: (us: number) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const span = Math.max(1, endUs - startUs);
  const frac = Math.min(1, Math.max(0, (currentUs - startUs) / span));
  const scrubAt = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    const f = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
    onSeek(startUs + f * span);
  };
  return (
    <div className="trackProgress">
      <div className="trackProgressLabels">
        <span>{(startUs / 1e6).toFixed(2)}s</span>
        <span className="trackProgressCur">{(currentUs / 1e6).toFixed(2)}s</span>
        <span>{(endUs / 1e6).toFixed(2)}s</span>
      </div>
      <div
        ref={ref}
        className="trackProgressBar"
        onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); scrubAt(e.clientX); }}
        onPointerMove={(e) => { if (e.buttons === 1) scrubAt(e.clientX); }}
      >
        <div className="trackProgressFill" style={{ width: `${frac * 100}%` }} />
        <div className="trackProgressHandle" style={{ left: `${frac * 100}%` }} />
      </div>
    </div>
  );
}

export function TrackPanel({
  track, selectedCount, trackCreate, trackPropagate, trackUseSelectionAsSeed,
  trackPreviewCandidate, trackConfirmCandidate, trackCancel, trackSetDirection,
  currentFrameUs, onSeek,
}: Props) {
  const { t } = useI18n();
  const busy = track?.status === "propagating";

  // Idle: a single affordance to start a track from the current selection.
  if (!track) {
    return (
      <div className="trackPanel">
        <div className="trackHeader"><MapPin size={13} /> {t("track.title")}</div>
        <button className="trackPrimary" onClick={trackCreate} disabled={selectedCount === 0}>
          <Crosshair size={13} /> {t("track.create")}
        </button>
        <p className="trackHint">{t("track.create.hint")}</p>
      </div>
    );
  }

  const conf = track.lastConfidence;
  const confPct = conf == null ? null : Math.round(conf * 100);
  const reasonLabel = track.lastReason ? (t(`track.reason.${track.lastReason}`) || track.lastReason) : null;

  return (
    <div className="trackPanel">
      <div className="trackHeader"><MapPin size={13} /> {t("track.title")}</div>

      <div className="trackDir">
        <span className="trackDirLabel">{t("track.direction")}</span>
        <button className={track.direction === "backward" ? "active" : ""} onClick={() => trackSetDirection("backward")} title={t("track.dir.backward")}>◀</button>
        <button className={track.direction === "forward" ? "active" : ""} onClick={() => trackSetDirection("forward")} title={t("track.dir.forward")}>▶</button>
      </div>

      <div className="trackStats">
        <span>{t("track.events")}: <b>{track.acceptedCount.toLocaleString()}</b></span>
        <span>{t("track.slices")}: <b>{track.slices}</b></span>
        <span>{t("track.span")}: <b>{track.spanMs.toFixed(0)} ms</b></span>
      </div>

      {track.coverStartUs != null && track.coverEndUs != null && track.coverEndUs > track.coverStartUs && (
        <TrackProgress startUs={track.coverStartUs} endUs={track.coverEndUs} currentUs={currentFrameUs} onSeek={onSeek} />
      )}

      {track.status === "paused" && (
        <div className={`trackStatus ${conf != null && conf < 0.4 ? "low" : "ok"}`}>
          {confPct != null && <span>{t("track.confidence")}: {confPct}%</span>}
          {reasonLabel && <span className="trackReason">{t("track.paused")}: {reasonLabel}</span>}
        </div>
      )}

      {track.candidates && track.candidates.length > 1 && (
        <div className="trackCandidates">
          <span className="trackCandHint">{t("track.candidates")}</span>
          {track.candidates.map((c, i) => (
            <button
              key={i}
              className={track.previewCand === i ? "trackCandidate active" : "trackCandidate"}
              onClick={() => trackPreviewCandidate(i)}
            >
              <span>{t("track.candidate")} {i + 1}</span>
              <span className="trackCandMeta">{Math.round(c.score * 100)}% · {c.count.toLocaleString()}</span>
            </button>
          ))}
          <button className="trackPrimary" onClick={trackConfirmCandidate} disabled={track.previewCand == null}>
            <Check size={13} /> {t("track.candidate.confirm")}
          </button>
        </div>
      )}

      <button className="trackPrimary" onClick={trackPropagate} disabled={busy}>
        <Play size={13} /> {busy ? t("track.propagating") : t("track.propagate")}
      </button>
      <button className="trackSecondary" onClick={trackUseSelectionAsSeed} disabled={busy} title={t("track.correct.hint")}>
        <Crosshair size={13} /> {t("track.correct")}
      </button>
      <button className="trackCancel" onClick={trackCancel} disabled={busy}>
        <X size={13} /> {t("track.cancel")}
      </button>
    </div>
  );
}
