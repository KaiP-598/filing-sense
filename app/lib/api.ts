import { TickerInfo, AnswerResponse, AnswerRequest } from "@/types"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export async function getTickers(): Promise<TickerInfo[]> {
  const res = await fetch(`${API_URL}/tickers`, { cache: "force-cache" })
  if (!res.ok) throw new Error("Failed to fetch tickers")
  return res.json()
}

export async function getAnswer(request: AnswerRequest): Promise<AnswerResponse> {
  const res = await fetch(`${API_URL}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Request failed" }))
    throw new Error(error.detail || "Request failed")
  }
  return res.json()
}
