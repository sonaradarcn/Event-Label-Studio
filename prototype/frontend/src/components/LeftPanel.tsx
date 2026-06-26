import { ChevronDown, ChevronRight, Plus, Trash2 } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import type { DatasetMeta, LabelClass } from "../api/types";
import type { StatsData } from "../hooks/useAppState";
import { useI18n } from "../i18n/I18nContext";
import { useConfirm } from "./ConfirmDialog";

type Props = {
  meta: DatasetMeta | null;
  datasetId: string;
  stats: StatsData;
  labels: LabelClass[];
  activeLabel: number;
  newLabelName: string;
  setActiveLabel: (id: number) => void;
  setNewLabelName: (name: string) => void;
  addLabel: () => void;
  removeLabel: (id: number) => void;
  updateLabel: (id: number, changes: { name?: string; color?: string }) => void;
  assign: (op: "assign" | "clear") => void;
  assignBusy: boolean;
  selectedIds: Set<number>;
};

function Section({ title, defaultOpen, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen ?? true);
  return (
    <div style={{ marginBottom: 4 }}>
      <button className="sectionToggle" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {title}
      </button>
      {open && <div className="sectionContent">{children}</div>}
    </div>
  );
}

function LabelItem({ label, isActive, onSelect, onRemove, onUpdate }: {
  label: LabelClass;
  isActive: boolean;
  onSelect: () => void;
  onRemove: () => void;
  onUpdate: (changes: { name?: string; color?: string }) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState(label.name);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing && inputRef.current) { inputRef.current.focus(); inputRef.current.select(); }
  }, [editing]);

  useEffect(() => { setEditName(label.name); }, [label.name]);

  function commitName() {
    onUpdate({ name: editName.trim() || label.name });
    setEditing(false);
  }

  // Row click selects; clicking an already-active label's name renames it (no
  // separate Edit button); clicking the swatch recolours it directly.
  return (
    <div className={isActive ? "label active" : "label"} title={label.name}
      onClick={() => { if (!editing) onSelect(); }}>
      <input type="color" className="labelColorEdit" value={label.color} title={label.color}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => onUpdate({ color: e.target.value })} />
      {editing ? (
        <input className="labelNameEdit" value={editName} ref={inputRef}
          onClick={(e) => e.stopPropagation()}
          onChange={(e) => setEditName(e.target.value)}
          onBlur={commitName}
          onKeyDown={(e) => {
            if (e.key === "Enter") commitName();
            if (e.key === "Escape") { setEditName(label.name); setEditing(false); }
          }} />
      ) : (
        <span className="labelName"
          onClick={(e) => { if (isActive) { e.stopPropagation(); setEditing(true); } }}>
          {label.name}
        </span>
      )}
      {!editing && (
        <span className="labelActions">
          <span className="labelIconBtn labelIconDelete" title="Delete" onClick={(e) => { e.stopPropagation(); onRemove(); }}>
            <Trash2 size={11} />
          </span>
        </span>
      )}
    </div>
  );
}

export function LeftPanel({ meta, datasetId, stats, labels, activeLabel, newLabelName, setActiveLabel, setNewLabelName, addLabel, removeLabel, updateLabel, assign, assignBusy, selectedIds }: Props) {
  const { t } = useI18n();
  const confirm = useConfirm();
  // Foolproofing: no label operations until a dataset is open.
  const noData = !meta;

  return (
    <aside className="panel">
      <Section title={t("panel.dataset")}>
        {meta ? (
          <dl>
            <dt>{t("panel.dataset.id")}</dt><dd>{datasetId}</dd>
            <dt>{t("panel.dataset.events")}</dt><dd>{meta.event_count.toLocaleString()}</dd>
            <dt>{t("panel.dataset.sensor")}</dt><dd>{meta.width} x {meta.height}</dd>
            <dt>{t("panel.dataset.timeRange")}</dt><dd>{(meta.t_max_us - meta.t_min_us).toLocaleString()} us</dd>
          </dl>
        ) : <p>{t("panel.dataset.noData")}</p>}
      </Section>

      {stats && (
        <Section title={t("panel.stats")}>
          <p>{t("panel.stats.total")}: {stats.total.toLocaleString()}</p>
          <p>{t("panel.stats.unlabelled")}: {stats.unlabelled.toLocaleString()} ({((stats.unlabelled / stats.total) * 100).toFixed(1)}%)</p>
          {Object.entries(stats.per_class).map(([name, count]) => <p key={name}>{name}: {Number(count).toLocaleString()}</p>)}
        </Section>
      )}

      <Section title={t("panel.labels")}>
        <div style={noData ? { opacity: 0.5, pointerEvents: "none" } : undefined} aria-disabled={noData}>
          {labels.map((label) => (
            <LabelItem
              key={label.id}
              label={label}
              isActive={activeLabel === label.id}
              onSelect={() => setActiveLabel(label.id)}
              onRemove={async () => {
                const ok = await confirm({
                  message: t("panel.labels.deleteConfirm").replace("{name}", label.name),
                  confirmLabel: t("confirm.delete"),
                  danger: true,
                });
                if (ok) removeLabel(label.id);
              }}
              onUpdate={(changes) => updateLabel(label.id, changes)}
            />
          ))}
          <div className="label labelNew">
            <Plus size={14} className="labelNewIcon" />
            <input className="labelNameEdit" placeholder={t("panel.labels.newPlaceholder")}
              value={newLabelName} onChange={(e) => setNewLabelName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addLabel()} />
          </div>
          <div>
            <button
              className={selectedIds.size > 0 && !assignBusy ? "applyBtn dirty" : "applyBtn"}
              style={{ cursor: assignBusy ? "wait" : undefined }}
              disabled={assignBusy || selectedIds.size === 0}
              onClick={() => assign("assign")}
            >
              {assignBusy
                ? t("panel.labels.assigning")
                : `${t("panel.labels.assignSelected")} (${selectedIds.size.toLocaleString()})`}
            </button>
          </div>
          <button
            className="clearLabelsBtn"
            disabled={assignBusy || selectedIds.size === 0}
            onClick={() => assign("clear")}
          >
            {t("menu.edit.clear")} ({selectedIds.size.toLocaleString()})
          </button>
        </div>
      </Section>
    </aside>
  );
}
