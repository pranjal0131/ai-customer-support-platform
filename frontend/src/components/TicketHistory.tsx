import { Search, SlidersHorizontal } from 'lucide-react'
import { useMemo, useState } from 'react'
import { formatLabel, percent, shortDate } from '../format'
import type { Ticket } from '../types'

interface TicketHistoryProps {
  tickets: Ticket[]
  loading: boolean
}

export function TicketHistory({ tickets, loading }: TicketHistoryProps) {
  const [search, setSearch] = useState('')
  const [urgency, setUrgency] = useState('all')
  const [sentiment, setSentiment] = useState('all')
  const [selected, setSelected] = useState<Ticket | null>(null)
  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return tickets.filter(
      (ticket) =>
        (!needle || `${ticket.subject ?? ''} ${ticket.text} ${ticket.customer_name ?? ''}`.toLowerCase().includes(needle)) &&
        (urgency === 'all' || ticket.urgency === urgency) &&
        (sentiment === 'all' || ticket.sentiment === sentiment),
    )
  }, [tickets, search, urgency, sentiment])

  return (
    <div className="space-y-6">
      <header>
        <p className="text-sm text-slate-500">Search and inspect analyzed support conversations</p>
        <h1 className="mt-1 text-2xl font-semibold text-slate-950">Ticket history</h1>
      </header>
      <section className="panel">
        <div className="grid gap-3 border-b border-line p-4 md:grid-cols-[1fr_180px_180px]">
          <label className="relative block">
            <span className="sr-only">Search tickets</span>
            <Search className="absolute left-3 top-2.5 text-slate-400" size={18} />
            <input className="field pl-10" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search subject, message, or customer" />
          </label>
          <label>
            <span className="sr-only">Filter by urgency</span>
            <select className="field" value={urgency} onChange={(event) => setUrgency(event.target.value)}>
              <option value="all">All urgencies</option>
              <option value="critical">Critical</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option>
            </select>
          </label>
          <label>
            <span className="sr-only">Filter by sentiment</span>
            <select className="field" value={sentiment} onChange={(event) => setSentiment(event.target.value)}>
              <option value="all">All sentiments</option><option value="negative">Negative</option><option value="neutral">Neutral</option><option value="positive">Positive</option>
            </select>
          </label>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[800px] text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
              <tr><th className="px-5 py-3 font-semibold">Ticket</th><th className="px-4 py-3 font-semibold">Category</th><th className="px-4 py-3 font-semibold">Sentiment</th><th className="px-4 py-3 font-semibold">Urgency</th><th className="px-4 py-3 font-semibold">Status</th><th className="px-5 py-3 text-right font-semibold">Received</th></tr>
            </thead>
            <tbody className="divide-y divide-line">
              {filtered.map((ticket) => (
                <tr key={ticket.id} className="cursor-pointer hover:bg-slate-50" tabIndex={0} onClick={() => setSelected(ticket)} onKeyDown={(event) => { if (event.key === 'Enter') setSelected(ticket) }}>
                  <td className="max-w-xs px-5 py-4"><p className="truncate font-medium text-slate-900">{ticket.subject ?? ticket.summary}</p><p className="mt-1 truncate text-xs text-slate-500">{ticket.customer_name ?? 'Unknown customer'} · {ticket.channel}</p></td>
                  <td className="px-4 py-4"><span className="bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700">{formatLabel(ticket.category)}</span><p className="mt-1 text-[10px] text-slate-400">{percent(ticket.intent_confidence)}</p></td>
                  <td className="px-4 py-4 capitalize text-slate-700">{ticket.sentiment}</td>
                  <td className="px-4 py-4"><UrgencyBadge value={ticket.urgency} /></td>
                  <td className="px-4 py-4 capitalize text-slate-700">{ticket.status}</td>
                  <td className="px-5 py-4 text-right text-xs text-slate-500">{shortDate(ticket.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {loading && <div className="p-10 text-center text-sm text-slate-500">Loading ticket history…</div>}
        {!loading && !filtered.length && <div className="p-10 text-center"><SlidersHorizontal className="mx-auto text-slate-300" size={26} /><p className="mt-3 text-sm font-medium text-slate-700">No tickets match these filters</p><p className="mt-1 text-xs text-slate-500">Try a broader search or clear a filter.</p></div>}
        <div className="border-t border-line px-5 py-3 text-xs text-slate-500">Showing {filtered.length} of {tickets.length} tickets</div>
      </section>
      {selected && <TicketDetail ticket={selected} onClose={() => setSelected(null)} />}
    </div>
  )
}

function UrgencyBadge({ value }: { value: string }) {
  const color = value === 'critical' ? 'bg-red-50 text-red-700' : value === 'high' ? 'bg-amber-50 text-amber-700' : 'bg-slate-100 text-slate-700'
  return <span className={`px-2 py-1 text-xs font-semibold ${color}`}>{formatLabel(value)}</span>
}

function TicketDetail({ ticket, onClose }: { ticket: Ticket; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/30" role="dialog" aria-modal="true" aria-label="Ticket details" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <div className="h-full w-full max-w-xl overflow-y-auto bg-white p-6 shadow-2xl">
        <div className="flex items-start justify-between gap-5 border-b border-line pb-5"><div><p className="text-xs uppercase tracking-wide text-slate-500">Ticket {ticket.id.slice(0, 8)}</p><h2 className="mt-1 text-xl font-semibold text-slate-950">{ticket.subject ?? 'Untitled ticket'}</h2></div><button onClick={onClose} className="border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50">Close</button></div>
        <dl className="grid grid-cols-2 gap-4 border-b border-line py-5 text-sm"><div><dt className="text-xs text-slate-500">Customer</dt><dd className="mt-1 font-medium">{ticket.customer_name ?? 'Unknown'}</dd></div><div><dt className="text-xs text-slate-500">Channel</dt><dd className="mt-1 capitalize">{ticket.channel}</dd></div><div><dt className="text-xs text-slate-500">Category</dt><dd className="mt-1">{formatLabel(ticket.category)}</dd></div><div><dt className="text-xs text-slate-500">Urgency</dt><dd className="mt-1"><UrgencyBadge value={ticket.urgency} /></dd></div></dl>
        <section className="py-5"><h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Original message</h3><p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-700">{ticket.text}</p></section>
        <section className="border-t border-line py-5"><h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Summary</h3><p className="mt-2 text-sm leading-6 text-slate-700">{ticket.summary}</p></section>
        <section className="border-t border-line py-5"><h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">AI suggested response · review required</h3><p className="mt-2 whitespace-pre-wrap bg-slate-50 p-4 text-sm leading-6 text-slate-700">{ticket.suggested_response}</p></section>
      </div>
    </div>
  )
}
