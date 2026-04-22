import { TickerInfo, AnswerResponse, AnswerRequest, StreamCallbacks } from "@/types"

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

export async function streamAnswer(
  request: AnswerRequest,
  callbacks: StreamCallbacks
): Promise<void> {
  const res = await fetch(`${API_URL}/answer/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  })

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Request failed" }))
    callbacks.onError(error.detail || "Request failed")
    return
  }

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split("\n")
    buffer = lines.pop() ?? ""

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue
      try {
        const event = JSON.parse(line.slice(6))
        if (event.type === "token") callbacks.onToken(event.text)
        else if (event.type === "sources") callbacks.onSources(event.data)
        else if (event.type === "error") callbacks.onError(event.detail)
      } catch {
        // malformed line — skip
      }
    }
  }
}
