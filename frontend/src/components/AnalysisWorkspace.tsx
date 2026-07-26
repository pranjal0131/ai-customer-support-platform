import { AlertCircle, Check, Copy, LoaderCircle, Save, ShieldCheck, Sparkles } from 'lucide-react'
import { useState } from 'react'
import { analyzeTicket, createTicket } from '../api'
import { formatLabel, percent } from '../format'
import type { Analysis } from '../types'

interface AnalysisWorkspaceProps {
  onSaved: () => void
}

export function AnalysisWorkspace({ onSaved }: AnalysisWorkspaceProps) {
  const [text, setText] = useState('')
  const [subject, setSubject] = useState('')
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [response, setResponse] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [copied, setCopied] = useState(false)

  async function handleAnalyze() {
    if (text.trim().length < 3) {
      setError('Enter at least three characters of ticket text.')
      return
    }
    setError('')
    setSaved(false)
    setLoading(true)
    try {
      const result = await analyzeTicket(text.trim())
      setAnalysis(result)
      setResponse(result.suggested_response)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Analysis failed. Check the API connection.')
    } finally {
      setLoading(false)
    }
  }

  async function handleSave() {
    if (!analysis || saved) return
    setSaving(true)
    setError('')
    try {
      await createTicket({ text: text.trim(), subject: subject.trim() || undefined, channel: 'web' })
      setSaved(true)
      onSaved()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not save this ticket.')
    } finally {
      setSaving(false)
    }
  }

  async function copyResponse() {
    await navigator.clipboard.writeText(response)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1600)
  }

  return (
    <div className="space-y-6">
      <header>
        <p className="text-sm text-slate-500">Triage a conversation before adding it to the queue</p>
        <h1 className="mt-1 text-2xl font-semibold text-slate-950">Analyze ticket</h1>
      </header>

      {error && (
        <div role="alert" className="flex gap-3 border border-red-200 bg-red-50 p-3 text-sm text-red-800">
          <AlertCircle className="mt-0.5 shrink-0" size={17} /> {error}
        </div>
      )}

      <section className="panel p-5">
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px]">
          <div>
            <label className="label" htmlFor="ticket-text">Ticket conversation</label>
            <textarea
              id="ticket-text"
              className="field min-h-44 resize-y"
              value={text}
              maxLength={10000}
              onChange={(event) => setText(event.target.value)}
              placeholder="Paste the customer's message or conversation here…"
            />
            <div className="mt-1.5 flex justify-between text-xs text-slate-400">
              <span>Emails and phone numbers are redacted during preprocessing.</span>
              <span>{text.length.toLocaleString()} / 10,000</span>
            </div>
          </div>
          <div className="space-y-4">
            <div>
              <label className="label" htmlFor="ticket-subject">Subject (optional)</label>
              <input id="ticket-subject" className="field" value={subject} maxLength={240} onChange={(event) => setSubject(event.target.value)} placeholder="Short ticket title" />
            </div>
            <div className="border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-600">
              <div className="mb-1 flex items-center gap-2 font-semibold text-slate-800">
                <ShieldCheck size={15} /> Human review policy
              </div>
              Predictions assist triage. Review category, urgency, summary, and response before acting.
            </div>
            <button
              onClick={handleAnalyze}
              disabled={loading || text.trim().length < 3}
              className="flex w-full items-center justify-center gap-2 bg-accent px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {loading ? <LoaderCircle className="animate-spin" size={17} /> : <Sparkles size={17} />}
              {loading ? 'Analyzing…' : 'Analyze ticket'}
            </button>
          </div>
        </div>
      </section>

      {loading && <AnalysisSkeleton />}

      {analysis && !loading && (
        <div className="space-y-4" aria-live="polite">
          {analysis.demo_mode && (
            <div className="border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-900">
              Demo mode is active. These transparent heuristic predictions will be replaced when trained artifacts are mounted.
            </div>
          )}
          <section className="grid gap-3 sm:grid-cols-3">
            <PredictionCard title="Category" value={analysis.category.label} confidence={analysis.category.confidence} color="blue" />
            <PredictionCard title="Sentiment" value={analysis.sentiment.label} confidence={analysis.sentiment.confidence} color={analysis.sentiment.label === 'negative' ? 'red' : 'green'} />
            <PredictionCard title="Urgency" value={analysis.urgency.label} confidence={analysis.urgency.confidence} color={analysis.urgency.label === 'critical' || analysis.urgency.label === 'high' ? 'red' : 'amber'} />
          </section>

          <section className="grid gap-4 xl:grid-cols-2">
            <article className="panel p-5">
              <h2 className="text-sm font-semibold text-slate-900">Ticket summary</h2>
              <p className="mt-3 text-sm leading-6 text-slate-700">{analysis.summary}</p>
            </article>
            <article className="panel p-5">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold text-slate-900">Suggested response</h2>
                  <p className="text-xs text-slate-500">AI draft · human review required</p>
                </div>
                <button className="p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-900" onClick={copyResponse} aria-label="Copy suggested response">
                  {copied ? <Check size={17} /> : <Copy size={17} />}
                </button>
              </div>
              <label className="sr-only" htmlFor="suggested-response">Editable suggested response</label>
              <textarea id="suggested-response" className="field mt-3 min-h-36 resize-y leading-6" value={response} onChange={(event) => setResponse(event.target.value)} />
            </article>
          </section>

          <section className="panel">
            <div className="border-b border-line px-5 py-4">
              <h2 className="text-sm font-semibold text-slate-900">Similar historical tickets</h2>
              <p className="text-xs text-slate-500">Top semantic matches from recorded conversations</p>
            </div>
            {analysis.similar_tickets.length ? (
              <div className="divide-y divide-line">
                {analysis.similar_tickets.map((ticket) => (
                  <article className="px-5 py-4" key={ticket.id}>
                    <div className="flex justify-between gap-4">
                      <p className="text-sm font-medium text-slate-900">{ticket.subject ?? formatLabel(ticket.category)}</p>
                      <span className="shrink-0 text-xs font-semibold text-blue-700">{percent(ticket.similarity)} match</span>
                    </div>
                    <p className="mt-1 line-clamp-2 text-xs leading-5 text-slate-500">{ticket.text}</p>
                  </article>
                ))}
              </div>
            ) : (
              <p className="px-5 py-8 text-center text-sm text-slate-500">No historical matches yet.</p>
            )}
          </section>
          <div className="flex justify-end">
            <button onClick={handleSave} disabled={saving || saved} className="inline-flex items-center gap-2 bg-navy px-5 py-2.5 text-sm font-semibold text-white hover:bg-slate-800 disabled:bg-slate-400">
              {saved ? <Check size={17} /> : saving ? <LoaderCircle className="animate-spin" size={17} /> : <Save size={17} />}
              {saved ? 'Saved to history' : saving ? 'Saving…' : 'Save ticket'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function PredictionCard({ title, value, confidence, color }: { title: string; value: string; confidence: number; color: 'blue' | 'green' | 'red' | 'amber' }) {
  const colors = { blue: 'bg-blue-600', green: 'bg-emerald-600', red: 'bg-red-600', amber: 'bg-amber-500' }
  return (
    <article className="panel p-4">
      <div className="flex items-start justify-between gap-3">
        <div><p className="text-xs font-medium uppercase tracking-wide text-slate-500">{title}</p><p className="mt-1.5 font-semibold text-slate-950">{formatLabel(value)}</p></div>
        <span className="text-sm font-semibold text-slate-700">{percent(confidence)}</span>
      </div>
      <div className="mt-4 h-1.5 bg-slate-100"><div className={`h-full ${colors[color]}`} style={{ width: percent(confidence) }} /></div>
    </article>
  )
}

function AnalysisSkeleton() {
  return <div className="grid animate-pulse gap-3 sm:grid-cols-3" aria-label="Loading predictions">{[1, 2, 3].map((item) => <div className="panel h-28 bg-slate-100" key={item} />)}</div>
}
