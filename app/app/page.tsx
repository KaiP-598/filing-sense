"use client"

import { useState, useEffect } from "react"
import { TickerInfo, AnswerResponse } from "@/types"
import { getTickers, streamAnswer } from "@/lib/api"
import CompanySelector from "@/components/CompanySelector"
import QuestionInput from "@/components/QuestionInput"
import AnswerDisplay from "@/components/AnswerDisplay"
import Navbar from "@/components/Navbar"

export default function Home() {
  const [tickers, setTickers] = useState<TickerInfo[]>([])
  const [selectedTicker, setSelectedTicker] = useState("NVDA")
  const [question, setQuestion] = useState("")
  const [result, setResult] = useState<AnswerResponse | null>(null)
  const [streamingText, setStreamingText] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    getTickers()
      .then(setTickers)
      .catch(() => setError("Failed to load companies. Is the backend running?"))
  }, [])

  async function handleSubmit() {
    if (!question.trim() || loading) return
    setLoading(true)
    setError(null)
    setResult(null)
    setStreamingText("")
    try {
      await streamAnswer(
        { ticker: selectedTicker, question },
        {
          onToken: (text) => setStreamingText((prev) => prev + text),
          onSources: (data) => {
            setResult(data)
            setStreamingText("")
          },
          onError: (detail) => setError(detail),
        }
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong")
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <Navbar />
      <main className="min-h-screen bg-gradient-to-b from-white to-zinc-50/80">
        <div className="mx-auto max-w-3xl px-6 pb-24 pt-32">

          {/* Hero */}
          <div className="mb-10 text-center">
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-zinc-200 bg-white px-3.5 py-1.5 text-xs font-medium text-zinc-500 shadow-sm">
              <span className="h-1.5 w-1.5 rounded-full bg-indigo-500" />
              30 companies · Latest 10-Ks · Powered by GRPO
            </div>
            <h1 className="mb-4 text-5xl font-bold tracking-tight text-zinc-900">
              Ask anything about a<br />
              <span className="text-indigo-600">company&apos;s financials</span>
            </h1>
            <p className="mx-auto max-w-md text-base text-zinc-500">
              Answers grounded in SEC 10-K filings using hybrid BM25 + FAISS
              retrieval and a GRPO-trained reasoning model.
            </p>
          </div>

          {/* Query card */}
          <div className="mb-4 rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
            <div className="mb-3 flex items-center gap-2">
              <div className="flex h-5 w-5 items-center justify-center rounded-full bg-zinc-900 text-white">
                <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth={2.5}>
                  <circle cx="11" cy="11" r="8" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35" />
                </svg>
              </div>
              <span className="text-xs font-semibold uppercase tracking-widest text-zinc-400">
                Query
              </span>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
              {mounted && (
                <CompanySelector
                  tickers={tickers}
                  selected={selectedTicker}
                  onChange={(t) => {
                    setSelectedTicker(t)
                    setResult(null)
                    setError(null)
                  }}
                />
              )}
              <QuestionInput
                value={question}
                onChange={setQuestion}
                onSubmit={handleSubmit}
                loading={loading}
              />
            </div>
          </div>

          {/* Performance notice */}
          <div className="mb-8 rounded-2xl border border-zinc-200 bg-zinc-50 px-5 py-4">
            <p className="text-sm font-semibold text-zinc-700 mb-1">⚡ Powered by GPU — answers in ~15 seconds</p>
            <p className="text-sm text-zinc-500">
              This demo runs a 3B-parameter Qwen2.5 model fine-tuned with GRPO on an RTX 3070 GPU.
              Answers stream in real-time as the model generates them.
            </p>
          </div>

          {/* Results */}
          <AnswerDisplay result={result} streamingText={streamingText} loading={loading} error={error} />

          {/* Footer */}
          <div className="mt-20 flex items-center justify-center gap-6 text-xs text-zinc-400">
            <span>FilingSense</span>
            <span>·</span>
            <a href="https://github.com/KaiP-598/filing-sense" target="_blank" rel="noopener noreferrer" className="hover:text-zinc-700">GitHub</a>
            <span>·</span>
            <a href="https://huggingface.co/kaiwu598" target="_blank" rel="noopener noreferrer" className="hover:text-zinc-700">HuggingFace</a>
            <span>·</span>
            <a href="https://arxiv.org/abs/2109.00122" target="_blank" rel="noopener noreferrer" className="hover:text-zinc-700">FinQA Paper</a>
          </div>
        </div>
      </main>
    </>
  )
}
