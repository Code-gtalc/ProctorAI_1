import React from "react";

const styles = {
  completed: "bg-emerald-100 text-emerald-700 border-emerald-200",
  pending: "bg-amber-100 text-amber-700 border-amber-200",
};

export default function StatusBadge({ completed }) {
  const key = completed ? "completed" : "pending";
  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide ${styles[key]}`}
    >
      {completed ? "Completed" : "Pending"}
    </span>
  );
}
