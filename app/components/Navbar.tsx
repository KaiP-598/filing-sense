"use client"

import Link from "next/link"

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-zinc-200/80 bg-white/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-4xl items-center justify-between px-6">
        <span className="text-sm font-semibold tracking-tight text-zinc-900">
          FilingSense
        </span>
        <div className="flex items-center gap-5">
          <Link
            href="https://github.com/KaiP-598/filing-sense"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs text-zinc-500 transition-colors hover:text-zinc-900"
          >
            <GitHubIcon />
            GitHub
          </Link>
          <Link
            href="https://huggingface.co/kaiwu598"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs text-zinc-500 transition-colors hover:text-zinc-900"
          >
            <HuggingFaceIcon />
            Models
          </Link>
        </div>
      </div>
    </nav>
  )
}

function GitHubIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 fill-current">
      <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2z" />
    </svg>
  )
}

function HuggingFaceIcon() {
  return (
    <svg viewBox="0 0 95 88" className="h-3.5 w-3.5 fill-current">
      <path d="M47.327 0C21.238 0 0 19.876 0 44.389c0 24.514 21.238 44.39 47.327 44.39s47.327-19.876 47.327-44.39C94.654 19.876 73.416 0 47.327 0zm-.68 12.604c2.42 0 4.382 1.962 4.382 4.382s-1.962 4.382-4.382 4.382-4.382-1.962-4.382-4.382 1.962-4.382 4.382-4.382zm13.825 0c2.42 0 4.382 1.962 4.382 4.382s-1.962 4.382-4.382 4.382-4.382-1.962-4.382-4.382 1.962-4.382 4.382-4.382zM28.01 34.036c4.074 0 7.376 3.302 7.376 7.376s-3.302 7.376-7.376 7.376-7.376-3.302-7.376-7.376 3.302-7.376 7.376-7.376zm38.633 0c4.074 0 7.376 3.302 7.376 7.376s-3.302 7.376-7.376 7.376-7.376-3.302-7.376-7.376 3.302-7.376 7.376-7.376zm-19.316 9.834c8.787 0 16.614 4.42 21.18 11.12a26.3 26.3 0 0 1-21.18 10.637 26.3 26.3 0 0 1-21.18-10.637c4.566-6.7 12.393-11.12 21.18-11.12z" />
    </svg>
  )
}
