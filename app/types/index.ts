export interface TickerInfo {
  ticker: string
  name: string
  sector: string
}

export interface SourceChunk {
  chunk_id: string
  text: string
  score: number
}

export interface AnswerResponse {
  ticker: string
  question: string
  reasoning: string
  answer: string
  sources: SourceChunk[]
  filing_date: string
  filing_url: string
}

export interface AnswerRequest {
  ticker: string
  question: string
}
