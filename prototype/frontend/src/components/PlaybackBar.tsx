import { ChevronFirst, ChevronLast, Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useI18n } from "../i18n/I18nContext";

type Props = {
  currentFrameUs: number;
  startUs: number;
  endUs: number;
  playing: boolean;
  playbackSpeed: number;
  frameWindowUs: number;
  onTogglePlayback: () => void;
  onStepForward: () => void;
  onStepBackward: () => void;
  onGoToStart: () => void;
  onGoToEnd: () => void;
  onSeek: (us: number) => void;
  onSetSpeed: (speed: number) => void;
  onSetFrameWindow: (us: number) => void;
};

function formatTime(us: number): string {
  const totalSec = us / 1_000_000;
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${sec.toFixed(3).padStart(6, "0")}`;
}

export function PlaybackBar({
  currentFrameUs, startUs, endUs, playing, playbackSpeed, frameWindowUs,
  onTogglePlayback, onStepForward, onStepBackward, onGoToStart, onGoToEnd,
  onSeek, onSetSpeed, onSetFrameWindow,
}: Props) {
  const { t } = useI18n();
  const range = endUs - startUs;
  const pos = range > 0 ? ((currentFrameUs - startUs) / range) * 100 : 0;

  return (
    <div className="playbackBar">
      <div className="playbackControls">
        <button onClick={onGoToStart} title={t("playback.skipStart")}><ChevronFirst size={16} /></button>
        <button onClick={onStepBackward} title={t("playback.stepBack")}><SkipBack size={16} /></button>
        <button className="playBtn" onClick={onTogglePlayback} title={playing ? t("playback.pause") : t("playback.play")}>
          {playing ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <button onClick={onStepForward} title={t("playback.stepForward")}><SkipForward size={16} /></button>
        <button onClick={onGoToEnd} title={t("playback.skipEnd")}><ChevronLast size={16} /></button>
      </div>

      <div className="playbackSpeed">
        <select value={playbackSpeed} onChange={(e) => onSetSpeed(Number(e.target.value))}>
          {[0.25, 0.5, 1, 2, 4, 8].map(s => <option key={s} value={s}>{s}x</option>)}
        </select>
      </div>

      <div className="playbackScrubber">
        <input type="range" min={0} max={100} step={0.01} value={pos}
          onChange={(e) => onSeek(startUs + (Number(e.target.value) / 100) * range)} />
      </div>

      <div className="playbackTime">
        <span>{formatTime(currentFrameUs)}</span>
        <span className="playbackTimeSep">/</span>
        <span>{formatTime(endUs)}</span>
      </div>

      <div className="playbackWindow">
        <label>
          {t("playback.window")}
          <select value={frameWindowUs} onChange={(e) => onSetFrameWindow(Number(e.target.value))}>
            <option value={1000}>1ms</option>
            <option value={5000}>5ms</option>
            <option value={10000}>10ms</option>
            <option value={20000}>20ms</option>
            <option value={50000}>50ms</option>
            <option value={100000}>100ms</option>
          </select>
        </label>
      </div>
    </div>
  );
}
