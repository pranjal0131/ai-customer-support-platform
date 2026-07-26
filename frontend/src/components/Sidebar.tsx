import {
  BarChart3,
  BrainCircuit,
  Headphones,
  Inbox,
  PanelLeftClose,
  Sparkles,
} from 'lucide-react'

export type View = 'overview' | 'analyze' | 'tickets' | 'models'

const navigation: Array<{ id: View; label: string; icon: typeof BarChart3 }> = [
  { id: 'overview', label: 'Overview', icon: BarChart3 },
  { id: 'analyze', label: 'Analyze ticket', icon: Sparkles },
  { id: 'tickets', label: 'Ticket history', icon: Inbox },
  { id: 'models', label: 'Model metrics', icon: BrainCircuit },
]

interface SidebarProps {
  view: View
  onChange: (view: View) => void
  open: boolean
  onClose: () => void
}

export function Sidebar({ view, onChange, open, onClose }: SidebarProps) {
  return (
    <>
      {open && (
        <button
          aria-label="Close navigation"
          className="fixed inset-0 z-30 bg-slate-950/30 lg:hidden"
          onClick={onClose}
        />
      )}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-64 flex-col bg-navy text-white transition-transform lg:translate-x-0 ${open ? 'translate-x-0' : '-translate-x-full'}`}
      >
        <div className="flex h-16 items-center justify-between border-b border-white/10 px-5">
          <button
            className="flex items-center gap-3 text-left"
            onClick={() => onChange('overview')}
          >
            <span className="grid h-8 w-8 place-items-center bg-blue-500">
              <Headphones aria-hidden="true" size={18} />
            </span>
            <span>
              <span className="block text-sm font-bold tracking-wide">SupportIQ</span>
              <span className="block text-[10px] uppercase tracking-[0.14em] text-slate-400">
                Operations
              </span>
            </span>
          </button>
          <button
            className="p-1 text-slate-400 hover:text-white lg:hidden"
            onClick={onClose}
            aria-label="Close navigation"
          >
            <PanelLeftClose size={19} />
          </button>
        </div>
        <nav aria-label="Primary" className="flex-1 px-3 py-5">
          <p className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
            Workspace
          </p>
          <ul className="space-y-1">
            {navigation.map((item) => {
              const Icon = item.icon
              const selected = item.id === view
              return (
                <li key={item.id}>
                  <button
                    className={`flex w-full items-center gap-3 border-l-2 px-3 py-2.5 text-sm transition-colors ${
                      selected
                        ? 'border-blue-400 bg-white/10 font-medium text-white'
                        : 'border-transparent text-slate-300 hover:bg-white/5 hover:text-white'
                    }`}
                    aria-current={selected ? 'page' : undefined}
                    onClick={() => {
                      onChange(item.id)
                      onClose()
                    }}
                  >
                    <Icon aria-hidden="true" size={18} />
                    {item.label}
                  </button>
                </li>
              )
            })}
          </ul>
        </nav>
        <div className="border-t border-white/10 p-4">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="h-2 w-2 bg-emerald-400" /> API connection monitored
          </div>
        </div>
      </aside>
    </>
  )
}
