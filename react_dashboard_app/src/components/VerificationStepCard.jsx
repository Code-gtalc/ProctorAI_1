import React from "react";
import { CheckCircle2, Clock3 } from "lucide-react";
import StatusBadge from "./StatusBadge";

export default function VerificationStepCard({
  title,
  description,
  completed,
  buttonLabel,
  onAction,
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h4 className="text-base font-semibold text-slate-900">{title}</h4>
          <p className="mt-1 text-sm text-slate-600">{description}</p>
        </div>
        <StatusBadge completed={completed} />
      </div>

      <div className="mt-4 flex items-center justify-between">
        <div className={`inline-flex items-center gap-2 text-sm ${completed ? "text-emerald-600" : "text-amber-600"}`}>
          {completed ? <CheckCircle2 size={16} /> : <Clock3 size={16} />}
          <span>{completed ? "Ready" : "Awaiting action"}</span>
        </div>
        <button
          type="button"
          onClick={onAction}
          disabled={completed}
          className="rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {completed ? "Completed" : buttonLabel}
        </button>
      </div>
    </div>
  );
}
