"""FilingSense FastAPI backend.

Serves the RAG pipeline as a REST API.
The pipeline loads once at startup and stays warm for all requests.

Endpoints:
    GET  /health    — liveness check, confirms model + indexes are loaded
    GET  /tickers   — list of supported tickers with company names
    POST /answer    — answer a question about a company's 10-K filing
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.config import TICKERS, TICKER_SET
from api.models import (
    AnswerRequest,
    AnswerResponse,
    HealthResponse,
    TickerInfo,
)
from api.pipeline import FilingSensePipeline


# ---------------------------------------------------------------------------
# App lifecycle — load pipeline once at startup
# ---------------------------------------------------------------------------

pipeline = FilingSensePipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load()
    yield


app = FastAPI(
    title="FilingSense API",
    description="Answer financial questions from SEC 10-K filings using RAG + GRPO.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tightened in production via ALLOWED_ORIGINS env var
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        tickers_loaded=pipeline.tickers_loaded,
        model_loaded=pipeline.model_loaded,
    )


@app.get("/tickers", response_model=list[TickerInfo])
def tickers():
    return [TickerInfo(**t) for t in TICKERS]


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest):
    ticker = request.ticker.upper().strip()
    question = request.question.strip()

    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    if ticker not in TICKER_SET:
        raise HTTPException(status_code=400, detail=f"Ticker '{ticker}' not supported")
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if len(question) > 500:
        raise HTTPException(status_code=400, detail="question must be under 500 characters")

    try:
        return pipeline.answer(ticker, question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
