import { AlertTriangle, ArrowRight, Gauge, Inbox, Radio } from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { formatLabel, percent, shortDate } from '../format'
import type { Analytics, Ticket } from '../types'

const SENTIMENT_COLORS: Record<string, string> = {
  negative: '#ef4444',
  neutral: '#64748b',
  positive: '#16a34a',
}

interface OverviewProps {
  analytics: Analytics
  tickets: Ticket[]
  onAnalyze: () => void
  onTickets: () => void
}

export function Overview({ analytics, tickets, onAnalyze, onTickets }: OverviewProps) {
  const stats = [
    { label: 'Total tickets', value: analytics.total_tickets, icon: Inbox, note: 'All recorded' },
    { label: 'Open queue', value: analytics.open_tickets, icon: Radio, note: 'Needs action' },
    {
      label: 'Critical priority',
      value: analytics.critical_tickets,
      icon: AlertTriangle,
      note: 'Review immediately',
    },
    {
      label: 'Avg. confidence',
      value: percent(analytics.average_confidence),
      icon: Gauge,
      note: 'Intent predictions',
    },
  ]

  return (
    <div className="space-y-6">
      <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
        <div>
          <p className="text-sm text-slate-500">Queue health and model-assisted triage</p>
          <h1 className="mt-1 text-2xl font-semibold text-slate-950">Operations overview</h1>
        </div>
        <button
          onClick={onAnalyze}
          className="inline-flex items-center justify-center gap-2 bg-accent px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700"
        >
          Analyze new ticket <ArrowRight size={16} />
        </button>
      </div>

      <section aria-label="Ticket summary" className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {stats.map(({ label, value, icon: Icon, note }) => (
          <article className="panel p-4" key={label}>
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
                <p className="mt-2 text-2xl font-semibold text-slate-950">{value}</p>
              </div>
              <span className="bg-slate-100 p-2 text-slate-600">
                <Icon size={18} aria-hidden="true" />
              </span>
            </div>
            <p className="mt-3 text-xs text-slate-500">{note}</p>
          </article>
        ))}
      </section>

      <section className="grid gap-4 xl:grid-cols-5">
        <article className="panel p-5 xl:col-span-3">
          <div className="mb-5">
            <h2 className="font-semibold text-slate-900">Tickets by category</h2>
            <p className="text-xs text-slate-500">Current distribution across the queue</p>
          </div>
          <div className="h-72">
            {analytics.categories.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analytics.categories.slice(0, 8)} margin={{ left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="name" tickFormatter={(value: string) => formatLabel(value).split(' ')[0]} tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip formatter={(value) => [value, 'Tickets']} labelFormatter={(label) => formatLabel(String(label))} />
                  <Bar dataKey="value" fill="#2563eb" maxBarSize={44} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChart />
            )}
          </div>
        </article>

        <article className="panel p-5 xl:col-span-2">
          <h2 className="font-semibold text-slate-900">Sentiment mix</h2>
          <p className="text-xs text-slate-500">Customer tone at intake</p>
          <div className="h-56">
            {analytics.sentiments.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={analytics.sentiments} dataKey="value" nameKey="name" innerRadius={55} outerRadius={83} paddingAngle={2}>
                    {analytics.sentiments.map((item) => (
                      <Cell key={item.name} fill={SENTIMENT_COLORS[item.name] ?? '#2563eb'} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChart />
            )}
          </div>
          <div className="flex flex-wrap justify-center gap-4">
            {analytics.sentiments.map((item) => (
              <div key={item.name} className="flex items-center gap-2 text-xs text-slate-600">
                <span className="h-2 w-2" style={{ backgroundColor: SENTIMENT_COLORS[item.name] }} />
                {formatLabel(item.name)} <strong>{item.value}</strong>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="panel">
        <div className="flex items-center justify-between border-b border-line px-5 py-4">
          <div>
            <h2 className="font-semibold text-slate-900">Recent tickets</h2>
            <p className="text-xs text-slate-500">Latest analyzed conversations</p>
          </div>
          <button onClick={onTickets} className="text-sm font-medium text-accent hover:text-blue-800">
            View all
          </button>
        </div>
        {tickets.length ? (
          <div className="divide-y divide-line">
            {tickets.slice(0, 5).map((ticket) => (
              <div className="grid gap-3 px-5 py-4 sm:grid-cols-[1fr_auto] sm:items-center" key={ticket.id}>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-slate-900">{ticket.subject ?? ticket.summary}</p>
                  <p className="mt-1 truncate text-xs text-slate-500">{ticket.customer_name ?? 'Unknown customer'} · {shortDate(ticket.created_at)}</p>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <Badge value={ticket.category} />
                  <Badge value={ticket.urgency} urgency />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-10 text-center text-sm text-slate-500">No tickets yet. Analyze one to start the queue.</div>
        )}
      </section>
    </div>
  )
}

function EmptyChart() {
  return <div className="grid h-full place-items-center text-sm text-slate-400">No data available</div>
}

function Badge({ value, urgency = false }: { value: string; urgency?: boolean }) {
  const urgentColor =
    value === 'critical'
      ? 'bg-red-50 text-red-700'
      : value === 'high'
        ? 'bg-amber-50 text-amber-700'
        : 'bg-slate-100 text-slate-600'
  return (
    <span className={`px-2 py-1 font-medium ${urgency ? urgentColor : 'bg-blue-50 text-blue-700'}`}>
      {formatLabel(value)}
    </span>
  )
}
