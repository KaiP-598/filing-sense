"use client"

import { KeyboardEvent } from "react"

const EXAMPLE_QUESTIONS = [
  "What was revenue growth year over year?",
  "How much did the company spend on R&D?",
  "What are the biggest risk factors?",
  "What is the gross margin?",
]

interface Props {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  loading: boolean
}

export default function QuestionInput({ value, onChange, onSubmit, loading }: Props) {
  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !loading && value.trim()) {
      onSubmit()
    }
  }

  return (
    <div className="flex flex-1 flex-col gap-3">
      <div className="flex items-center gap-2">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about this company's 10-K filing..."
          disabled={loading}
          className="flex-1 rounded-xl border border-zinc-200 bg-white px-4 py-3 text-sm text-zinc-900 shadow-sm placeholder-zinc-400 transition-all focus:border-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-900 focus:ring-offset-1 disabled:opacity-50"
        />
        <button
          onClick={onSubmit}
          disabled={loading || !value.trim()}
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-zinc-900 text-white shadow-sm transition-all hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-30"
        >
          {loading ? <Spinner /> : <ArrowIcon />}
        </button>
      </div>

      {/* Example chips */}
      <div className="flex flex-wrap gap-2">
        {EXAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => onChange(q)}
            disabled={loading}
            className="rounded-full border border-zinc-200 bg-white px-3 py-1 text-xs text-zinc-500 shadow-sm transition-all hover:border-zinc-300 hover:text-zinc-800 hover:shadow disabled:opacity-40"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

function ArrowIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14M12 5l7 7-7 7" />
    </svg>
  )
}

function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  )
}
