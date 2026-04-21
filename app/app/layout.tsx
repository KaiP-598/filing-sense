import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" })

export const metadata: Metadata = {
  title: "FilingSense — Ask anything about SEC 10-K filings",
  description:
    "AI-powered financial Q&A from SEC 10-K filings. Powered by hybrid BM25+FAISS retrieval and a GRPO-trained Qwen2.5-3B model.",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} antialiased`}>
      <body className="bg-white font-sans text-zinc-900">{children}</body>
    </html>
  )
}
