"use client"

import { useState } from "react"
import { AnswerResponse } from "@/types"

interface Props {
  result: AnswerResponse | null
  loading: boolean
  error: string | null
}

export default function AnswerDisplay({ result, loading, error }: Props) {
  const [sourcesOpen, setSourcesOpen] = useState(false)

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-600">
        {error}
      </div>
    )
  }

  if (loading) {
    return <LoadingSkeleton />
  }

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl border border-zinc-200 bg-zinc-50 shadow-sm">
          <span className="text-xl">◎</span>
        </div>
        <p className="text-sm font-medium text-zinc-500">
          Select a company and ask a question
        </p>
        <p className="mt-1 text-xs text-zinc-400">
          Answers are grounded in the company&apos;s latest SEC 10-K filing
        </p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Reasoning */}
      {result.reasoning && (
        <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-5 shadow-sm">
          <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-zinc-400">
            Reasoning
          </p>
          <p className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-zinc-600">
            {result.reasoning}
          </p>
        </div>
      )}

      {/* Answer — indigo left border, Apple style */}
      <div className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm">
        <div className="flex gap-4">
          <div className="w-0.5 shrink-0 rounded-full bg-indigo-500" />
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-indigo-500">
              Answer
            </p>
            <p className="text-lg font-medium leading-relaxed text-zinc-900">
              {result.answer}
            </p>
          </div>
        </div>
      </div>

      {/* Sources */}
      <div className="rounded-xl border border-zinc-200 bg-white shadow-sm">
        <button
          onClick={() => setSourcesOpen(!sourcesOpen)}
          className="flex w-full items-center justify-between px-5 py-3.5 text-left"
        >
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold uppercase tracking-widest text-zinc-400">
              Sources
            </span>
            <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-xs font-medium text-zinc-500">
              {result.sources.length} chunks
            </span>
            <span className="text-xs text-zinc-400">
              {result.ticker} · {result.filing_date}
            </span>
          </div>
          <ChevronIcon open={sourcesOpen} />
        </button>

        {sourcesOpen && (
          <div className="border-t border-zinc-100 px-5 pb-5 pt-4">
            <div className="flex flex-col gap-3">
              {result.sources.map((s, i) => (
                <div
                  key={s.chunk_id}
                  className="rounded-lg border border-zinc-100 bg-zinc-50 p-4"
                >
                  <div className="mb-2 flex items-center justify-between">
                    <span className="font-mono text-xs text-zinc-400">
                      chunk {i + 1}
                    </span>
                    <span className="text-xs text-zinc-400">
                      score {s.score.toFixed(3)}
                    </span>
                  </div>
                  <p className="text-xs leading-relaxed text-zinc-500">{s.text}</p>
                </div>
              ))}
            </div>
            <a
              href={result.filing_url}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-4 flex items-center gap-1 text-xs font-medium text-indigo-600 hover:text-indigo-500"
            >
              View full filing on SEC EDGAR →
            </a>
          </div>
        )}
      </div>
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="flex flex-col gap-4 animate-pulse">
      <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-5 shadow-sm">
        <div className="mb-3 h-2.5 w-20 rounded-full bg-zinc-200" />
        <div className="space-y-2.5">
          <div className="h-3 w-full rounded-full bg-zinc-200" />
          <div className="h-3 w-4/5 rounded-full bg-zinc-200" />
          <div className="h-3 w-3/5 rounded-full bg-zinc-200" />
        </div>
      </div>
      <div className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm">
        <div className="flex gap-4">
          <div className="w-0.5 rounded-full bg-indigo-200" />
          <div className="flex-1">
            <div className="mb-3 h-2.5 w-14 rounded-full bg-indigo-100" />
            <div className="h-5 w-2/3 rounded-full bg-zinc-200" />
          </div>
        </div>
      </div>
    </div>
  )
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={`h-4 w-4 text-zinc-400 transition-transform ${open ? "rotate-180" : ""}`}
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  )
}
