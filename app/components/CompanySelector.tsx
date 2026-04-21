"use client"

import { useState, useRef, useEffect } from "react"
import { TickerInfo } from "@/types"

interface Props {
  tickers: TickerInfo[]
  selected: string
  onChange: (ticker: string) => void
}

export default function CompanySelector({ tickers, selected, onChange }: Props) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState("")
  const ref = useRef<HTMLDivElement>(null)

  const selectedTicker = tickers.find((t) => t.ticker === selected)

  const filtered = tickers.filter(
    (t) =>
      t.ticker.toLowerCase().includes(search.toLowerCase()) ||
      t.name.toLowerCase().includes(search.toLowerCase()) ||
      t.sector.toLowerCase().includes(search.toLowerCase())
  )

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
        setSearch("")
      }
    }
    document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [])

  return (
    <div ref={ref} className="relative w-56 shrink-0">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between rounded-xl border border-zinc-200 bg-white px-4 py-3 text-sm text-zinc-900 shadow-sm transition-all hover:border-zinc-300 hover:shadow focus:outline-none focus:ring-2 focus:ring-zinc-900 focus:ring-offset-1"
      >
        <div className="flex items-center gap-2.5">
          {selectedTicker ? (
            <>
              <span className="font-mono text-xs font-bold text-zinc-400">
                {selectedTicker.ticker}
              </span>
              <span className="font-medium text-zinc-800">{selectedTicker.name}</span>
            </>
          ) : (
            <span className="text-zinc-400">Select company</span>
          )}
        </div>
        <ChevronIcon open={open} />
      </button>

      {open && (
        <div className="absolute left-0 top-full z-50 mt-1.5 w-80 overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-xl shadow-zinc-200/60">
          <div className="border-b border-zinc-100 p-2">
            <input
              autoFocus
              type="text"
              placeholder="Search companies..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full rounded-lg bg-zinc-50 px-3 py-2 text-sm text-zinc-900 placeholder-zinc-400 focus:outline-none"
            />
          </div>
          <div className="max-h-72 overflow-y-auto">
            {filtered.length === 0 ? (
              <div className="px-4 py-3 text-sm text-zinc-400">No results</div>
            ) : (
              filtered.map((t) => (
                <button
                  key={t.ticker}
                  onClick={() => {
                    onChange(t.ticker)
                    setOpen(false)
                    setSearch("")
                  }}
                  className={`flex w-full items-center justify-between px-4 py-2.5 text-left text-sm transition-colors hover:bg-zinc-50 ${
                    t.ticker === selected ? "bg-zinc-50" : ""
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className="w-12 font-mono text-xs font-bold text-zinc-400">
                      {t.ticker}
                    </span>
                    <span className="font-medium text-zinc-800">{t.name}</span>
                  </div>
                  <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-xs text-zinc-500">
                    {t.sector}
                  </span>
                </button>
              ))
            )}
          </div>
        </div>
      )}
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
