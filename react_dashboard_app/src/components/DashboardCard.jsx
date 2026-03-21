import React from "react";

export default function DashboardCard({ title, subtitle, children, className = "" }) {
  return (
    <section className={`rounded-2xl border border-slate-200 bg-white p-6 shadow-sm ${className}`}>
      {title ? <h3 className="text-lg font-semibold text-slate-900">{title}</h3> : null}
      {subtitle ? <p className="mt-1 text-sm text-slate-500">{subtitle}</p> : null}
      <div className={title || subtitle ? "mt-4" : ""}>{children}</div>
    </section>
  );
}
