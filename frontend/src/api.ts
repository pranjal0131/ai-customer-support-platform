import type { Analysis, Analytics, ModelMetrics, Ticket, TicketList } from './types'

const API_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message)
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  })
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as {
      detail?: string
      error?: string
    }
    throw new ApiError(body.detail ?? body.error ?? 'Request failed', response.status)
  }
  return response.json() as Promise<T>
}

export function analyzeTicket(text: string): Promise<Analysis> {
  return request('/api/tickets/analyze', {
    method: 'POST',
    body: JSON.stringify({ text }),
  })
}

export function createTicket(input: {
  text: string
  subject?: string
  channel?: string
}): Promise<Ticket> {
  return request('/api/tickets', { method: 'POST', body: JSON.stringify(input) })
}

export function getTickets(query = ''): Promise<TicketList> {
  return request(`/api/tickets${query ? `?${query}` : ''}`)
}

export function getAnalytics(): Promise<Analytics> {
  return request('/api/analytics/overview')
}

export function getModelMetrics(): Promise<ModelMetrics> {
  return request('/api/models/metrics')
}
