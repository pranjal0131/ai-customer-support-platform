import { AlertCircle, Bell, Menu, RefreshCw } from 'lucide-react'
import { lazy, Suspense, useCallback, useEffect, useState } from 'react'
import { getAnalytics, getModelMetrics, getTickets } from './api'
import { Sidebar, type View } from './components/Sidebar'
import type { Analytics, ModelMetrics as Metrics, Ticket } from './types'

const emptyAnalytics: Analytics = { total_tickets: 0, open_tickets: 0, critical_tickets: 0, average_confidence: 0, categories: [], sentiments: [], urgencies: [] }
const emptyMetrics: Metrics = { demo_mode: true, message: 'No metrics loaded.', runs: [] }
const AnalysisWorkspace = lazy(() => import('./components/AnalysisWorkspace').then((module) => ({ default: module.AnalysisWorkspace })))
const ModelMetrics = lazy(() => import('./components/ModelMetrics').then((module) => ({ default: module.ModelMetrics })))
const Overview = lazy(() => import('./components/Overview').then((module) => ({ default: module.Overview })))
const TicketHistory = lazy(() => import('./components/TicketHistory').then((module) => ({ default: module.TicketHistory })))

export default function App() {
  const [view, setView] = useState<View>('overview')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [analytics, setAnalytics] = useState<Analytics>(emptyAnalytics)
  const [tickets, setTickets] = useState<Ticket[]>([])
  const [metrics, setMetrics] = useState<Metrics>(emptyMetrics)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const loadData = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const [analyticsData, ticketData, metricData] = await Promise.all([getAnalytics(), getTickets('limit=100'), getModelMetrics()])
      setAnalytics(analyticsData)
      setTickets(ticketData.items)
      setMetrics(metricData)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not connect to the SupportIQ API.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void loadData() }, [loadData])

  return (
    <div className="min-h-screen bg-canvas">
      <Sidebar view={view} onChange={setView} open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <div className="lg:pl-64">
        <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-line bg-white/95 px-4 backdrop-blur sm:px-6 lg:px-8">
          <div className="flex items-center gap-3"><button onClick={() => setSidebarOpen(true)} className="p-1 text-slate-600 lg:hidden" aria-label="Open navigation"><Menu size={22} /></button><div className="hidden text-sm text-slate-500 sm:block">Customer support workspace</div></div>
          <div className="flex items-center gap-3"><span className="hidden items-center gap-2 border border-slate-200 px-2.5 py-1.5 text-xs text-slate-600 sm:flex"><span className={`h-2 w-2 ${error ? 'bg-red-500' : 'bg-emerald-500'}`} />{error ? 'API unavailable' : metrics.demo_mode ? 'Demo models' : 'Trained models'}</span><button onClick={() => void loadData()} aria-label="Refresh dashboard" className="p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-900"><RefreshCw size={18} className={loading ? 'animate-spin' : ''} /></button><button aria-label="Notifications" className="p-2 text-slate-500 hover:bg-slate-100"><Bell size={18} /></button><span className="grid h-8 w-8 place-items-center bg-slate-800 text-xs font-semibold text-white">OP</span></div>
        </header>
        <main className="mx-auto max-w-[1500px] p-4 sm:p-6 lg:p-8">
          {error && <div role="alert" className="mb-5 flex items-center justify-between gap-4 border border-red-200 bg-red-50 p-3 text-sm text-red-800"><span className="flex items-center gap-2"><AlertCircle size={17} />{error}</span><button className="font-semibold underline" onClick={() => void loadData()}>Retry</button></div>}
          <Suspense fallback={<PageSkeleton />}>
            {loading && !tickets.length && view !== 'analyze' ? <PageSkeleton /> : view === 'overview' ? <Overview analytics={analytics} tickets={tickets} onAnalyze={() => setView('analyze')} onTickets={() => setView('tickets')} /> : view === 'analyze' ? <AnalysisWorkspace onSaved={() => void loadData()} /> : view === 'tickets' ? <TicketHistory tickets={tickets} loading={loading} /> : <ModelMetrics data={metrics} />}
          </Suspense>
        </main>
      </div>
    </div>
  )
}

function PageSkeleton() {
  return <div className="animate-pulse space-y-6" aria-label="Loading dashboard"><div className="h-14 w-72 bg-slate-200" /><div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{[1, 2, 3, 4].map((item) => <div className="h-32 bg-slate-200" key={item} />)}</div><div className="h-80 bg-slate-200" /></div>
}
