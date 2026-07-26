export interface Prediction {
  label: string
  confidence: number
}

export interface SimilarTicket {
  id: string
  subject: string | null
  text: string
  category: string
  status: string
  similarity: number
}

export interface Analysis {
  category: Prediction
  sentiment: Prediction
  urgency: Prediction
  summary: string
  suggested_response: string
  similar_tickets: SimilarTicket[]
  ai_suggestion_requires_review: boolean
  demo_mode: boolean
}

export interface Ticket {
  id: string
  subject: string | null
  text: string
  customer_name: string | null
  channel: string
  status: string
  category: string
  intent_confidence: number
  sentiment: string
  sentiment_confidence: number
  urgency: string
  urgency_confidence: number
  summary: string
  suggested_response: string
  created_at: string
  updated_at: string
}

export interface TicketList {
  items: Ticket[]
  total: number
  limit: number
  offset: number
}

export interface CountDatum {
  name: string
  value: number
}

export interface Analytics {
  total_tickets: number
  open_tickets: number
  critical_tickets: number
  average_confidence: number
  categories: CountDatum[]
  sentiments: CountDatum[]
  urgencies: CountDatum[]
}

export interface ModelRun {
  task: string
  model: string
  dataset: string
  split: string
  metrics: Record<string, number>
  created_at: string | null
  artifact_path: string | null
}

export interface ModelMetrics {
  demo_mode: boolean
  message: string
  runs: ModelRun[]
}
