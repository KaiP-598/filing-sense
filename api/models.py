"""Pydantic request/response models for the FilingSense API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. 'AAPL'")
    question: str = Field(..., description="Natural language question about the company")


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float


class AnswerResponse(BaseModel):
    ticker: str
    question: str
    reasoning: str
    answer: str
    sources: list[SourceChunk]
    filing_date: str
    filing_url: str


class TickerInfo(BaseModel):
    ticker: str
    name: str
    sector: str


class HealthResponse(BaseModel):
    status: str
    tickers_loaded: int
    model_loaded: bool
