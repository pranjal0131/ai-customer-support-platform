import { BrainCircuit, Database, FlaskConical } from 'lucide-react'
import { formatLabel, percent, shortDate } from '../format'
import type { ModelMetrics as Metrics } from '../types'

export function ModelMetrics({ data }: { data: Metrics }) {
  return (
    <div className="space-y-6">
      <header><p className="text-sm text-slate-500">Evidence from completed local evaluation runs</p><h1 className="mt-1 text-2xl font-semibold text-slate-950">Model metrics</h1></header>
      {data.demo_mode && <div className="flex gap-3 border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900"><FlaskConical className="shrink-0" size={19} /><div><strong>Demo inference is active.</strong><p className="mt-0.5 text-xs leading-5">Train and mount model artifacts to replace heuristics. The table below never contains estimated or placeholder scores.</p></div></div>}
      {!data.runs.length ? (
        <section className="panel px-6 py-16 text-center"><BrainCircuit className="mx-auto text-slate-300" size={36} /><h2 className="mt-4 font-semibold text-slate-800">No executed evaluation runs</h2><p className="mx-auto mt-2 max-w-lg text-sm leading-6 text-slate-500">{data.message} Start with <code className="bg-slate-100 px-1.5 py-0.5 text-xs">make train-baselines</code>, then refresh this page.</p></section>
      ) : (
        <section className="panel overflow-x-auto">
          <table className="w-full min-w-[850px] text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500"><tr><th className="px-5 py-3">Task / model</th><th className="px-4 py-3">Dataset</th><th className="px-4 py-3">Accuracy</th><th className="px-4 py-3">Macro precision</th><th className="px-4 py-3">Macro recall</th><th className="px-4 py-3">Macro F1</th><th className="px-5 py-3">Run</th></tr></thead>
            <tbody className="divide-y divide-line">{data.runs.map((run, index) => <tr key={`${run.task}-${run.model}-${index}`}><td className="px-5 py-4"><p className="font-medium text-slate-900">{run.model}</p><p className="text-xs text-slate-500">{formatLabel(run.task)}</p></td><td className="max-w-xs px-4 py-4"><div className="flex items-center gap-2"><Database size={14} className="shrink-0 text-slate-400" /><span className="truncate">{run.dataset}</span></div><p className="mt-1 text-xs text-slate-400">{run.split}</p></td><Metric value={run.metrics.accuracy} /><Metric value={run.metrics.macro_precision} /><Metric value={run.metrics.macro_recall} /><Metric value={run.metrics.macro_f1} strong /><td className="px-5 py-4 text-xs text-slate-500">{run.created_at ? shortDate(run.created_at) : '—'}</td></tr>)}</tbody>
          </table>
        </section>
      )}
      <section className="border border-blue-200 bg-blue-50 p-4 text-xs leading-5 text-blue-900"><strong>Evaluation note:</strong> urgency scores measure agreement with deterministic weak labels, not human-reviewed ground truth. They should not be compared directly with intent or sentiment benchmarks.</section>
    </div>
  )
}

function Metric({ value, strong = false }: { value: number | undefined; strong?: boolean }) {
  return <td className={`px-4 py-4 ${strong ? 'font-semibold text-blue-700' : 'text-slate-700'}`}>{value === undefined ? '—' : percent(value)}</td>
}
