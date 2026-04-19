"""Agentic RAG pipeline using LangGraph.

Improves on single-pass retrieval by adding:
    1. Query decomposition — breaks complex questions into sub-queries
    2. Multi-query retrieval — searches for each sub-query independently
    3. Relevance grading — OpenAI API checks if chunks are useful (with retry)
    4. Query rewriting — rewrites failed queries and retries retrieval

The state machine:

    START
      │
      ▼
    [decompose] ── splits question into sub-queries
      │
      ▼
    [retrieve] ── hybrid BM25+FAISS per sub-query
      │
      ▼
    [grade] ── OpenAI grades each chunk for relevance
      │
      ├── relevant chunks found → [generate] → END
      ├── no relevant + retries < 2 → [rewrite] → [retrieve]
      └── retries exhausted → [generate] → END

Usage:
    from src.agent import build_agent, AgentState
    from src.indexing import build_index
    from src.retrieval import load_reranker

    agent = build_agent(index, reranker, model, tokenizer, openai_api_key)
    result = agent.invoke({"query": "What was the change in revenue from 2010 to 2011?"})
    print(result["answer"])
"""

from __future__ import annotations

import os
from typing import Literal

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from src.retrieval import RetrievalResult, retrieve
from src.generation import (
    GenerationResult,
    build_prompt,
    generate_answer,
    SYSTEM_PROMPT,
)
from src.indexing import DualIndex


# ---------------------------------------------------------------------------
# State — the dict that flows through every node
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """State that accumulates as the query flows through the agent."""
    # Input
    query: str

    # Decomposition
    sub_queries: list[str]

    # Retrieval
    retrieved_chunks: list[RetrievalResult]

    # Grading
    relevant_chunks: list[RetrievalResult]

    # Control flow
    retries: int

    # Output
    answer: str
    generation_result: GenerationResult


# ---------------------------------------------------------------------------
# Node 1: Query Decomposition
# ---------------------------------------------------------------------------

def make_decompose_node(openai_api_key: str, model_name: str = "gpt-4o-mini"):
    """Create a node that decomposes complex questions into sub-queries.

    Uses OpenAI to identify what specific pieces of data we need to search for.
    Example: "What was the change from 2010 to 2011?" → ["revenue 2010", "revenue 2011"]
    """
    import openai

    client = openai.OpenAI(api_key=openai_api_key)

    def decompose(state: AgentState) -> dict:
        query = state["query"]

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You decompose financial questions into specific search queries. "
                        "Each query should target ONE piece of data needed to answer the question.\n\n"
                        "Rules:\n"
                        "- Output one query per line, nothing else\n"
                        "- Include specific terms (company names, metrics, years)\n"
                        "- For comparison questions, create separate queries for each item\n"
                        "- For simple questions that need only one lookup, return just that query\n"
                        "- Maximum 3 sub-queries\n\n"
                        "Examples:\n"
                        'Q: "What was the change in operating income from 2010 to 2011?"\n'
                        "operating income 2010\noperating income 2011\n\n"
                        'Q: "What percentage of total revenue came from product sales in 2013?"\n'
                        "product sales revenue 2013\ntotal revenue 2013\n\n"
                        'Q: "What was the gross margin in 2012?"\n'
                        "gross margin 2012"
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=150,
        )

        raw = response.choices[0].message.content.strip()
        # Strip bullet prefixes (-, *, numbers) and surrounding quotes
        sub_queries = []
        for q in raw.split("\n"):
            q = q.strip().lstrip("-*•123456789. ").strip().strip('"').strip("'")
            if q:
                sub_queries.append(q)

        # Always include the original query — it may work better than decomposed ones
        if query not in sub_queries:
            sub_queries.insert(0, query)

        # Cap at 4 queries (original + 3 decomposed)
        sub_queries = sub_queries[:4]

        return {"sub_queries": sub_queries, "retries": state.get("retries", 0)}

    return decompose


# ---------------------------------------------------------------------------
# Node 2: Multi-Query Retrieval
# ---------------------------------------------------------------------------

def make_retrieve_node(index: DualIndex, reranker=None, top_k: int = 5):
    """Create a node that retrieves chunks for each sub-query.

    Runs our existing hybrid retrieval (BM25 + FAISS + reranker) per sub-query,
    then deduplicates by chunk_id, keeping the highest-scoring result.
    """

    def retrieve_node(state: AgentState) -> dict:
        sub_queries = state.get("sub_queries", [state["query"]])
        all_results: dict[str, RetrievalResult] = {}  # chunk_id → best result

        for sq in sub_queries:
            results = retrieve(
                index,
                sq,
                top_k_per_source=20,
                top_n_final=top_k,
                reranker=reranker,
                use_reranker=reranker is not None,
            )
            for r in results:
                cid = r.chunk.chunk_id
                if cid not in all_results or r.score > all_results[cid].score:
                    all_results[cid] = r

        # Sort by score descending, take top results
        sorted_results = sorted(all_results.values(), key=lambda r: r.score, reverse=True)
        return {"retrieved_chunks": sorted_results[: top_k * 2]}  # keep more for grading

    return retrieve_node


# ---------------------------------------------------------------------------
# Node 3: Relevance Grading (OpenAI)
# ---------------------------------------------------------------------------

def make_grade_node(openai_api_key: str, model_name: str = "gpt-4o-mini"):
    """Create a node that grades retrieved chunks for relevance in a single batch call.

    Sends all chunks in one prompt instead of one call per chunk — faster, cheaper,
    and allows the grader to compare chunks against each other.
    Uses a generous grading bar: any chunk topically related or containing relevant
    numbers/terms passes, to avoid false negatives.
    """
    import openai

    client = openai.OpenAI(api_key=openai_api_key)

    def grade(state: AgentState) -> dict:
        query = state["query"]
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return {"relevant_chunks": []}

        # Build batch prompt — all chunks in one API call
        chunks_text = ""
        for i, r in enumerate(chunks, 1):
            chunks_text += f"Chunk {i}:\n{r.chunk.text[:800]}\n\n"

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a relevance grader for financial document retrieval. "
                        "Given a question and multiple document chunks from SEC 10-K filings, "
                        "identify which chunks are topically related to the question and could "
                        "help answer it, even partially.\n\n"
                        "Be generous — if a chunk contains any numbers, metrics, or terms "
                        "related to the question topic, mark it as relevant.\n\n"
                        "Reply with ONLY a comma-separated list of relevant chunk numbers. "
                        "Example: '1,3,5' or '2,4' or 'none'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"{chunks_text}"
                        "Which chunk numbers are relevant to answering this question?"
                    ),
                    },
                ],
                temperature=0.0,
                max_tokens=30,
            )

        raw = response.choices[0].message.content.strip().lower()

        # Parse "1,3,5" or "none"
        relevant = []
        if raw != "none":
            for part in raw.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1  # convert to 0-based
                    if 0 <= idx < len(chunks):
                        relevant.append(chunks[idx])

        return {"relevant_chunks": relevant}

    return grade


# ---------------------------------------------------------------------------
# Node 4: Query Rewriting
# ---------------------------------------------------------------------------

def make_rewrite_node(openai_api_key: str, model_name: str = "gpt-4o-mini"):
    """Create a node that rewrites the query when retrieval fails.

    Generates alternative search terms — especially useful for financial data
    where terminology varies (e.g., "revenue" vs "net sales" vs "total sales").
    """
    import openai

    client = openai.OpenAI(api_key=openai_api_key)

    def rewrite(state: AgentState) -> dict:
        query = state["query"]
        retries = state.get("retries", 0)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite financial search queries using alternative terminology. "
                        "SEC filings use inconsistent terms for the same concepts.\n\n"
                        "Common synonyms:\n"
                        "- revenue / net sales / total sales / net revenue\n"
                        "- operating income / income from operations / operating profit\n"
                        "- net income / net earnings / net profit\n"
                        "- total debt / long-term debt + short-term debt\n"
                        "- R&D / research and development expenses\n\n"
                        "Rewrite the query using different but equivalent financial terms. "
                        "Output one rewritten query per line, max 2 queries."
                    ),
                },
                {"role": "user", "content": f"Original query: {query}"},
            ],
            temperature=0.3,
            max_tokens=150,
        )

        raw = response.choices[0].message.content.strip()
        rewritten = [q.strip() for q in raw.split("\n") if q.strip()]

        # Include original + rewritten
        sub_queries = [query] + rewritten[:2]

        return {"sub_queries": sub_queries, "retries": retries + 1}

    return rewrite


# ---------------------------------------------------------------------------
# Node 5: Answer Generation
# ---------------------------------------------------------------------------

def make_generate_node(model=None, tokenizer=None, model_name: str = "qwen2.5-3b"):
    """Create a node that generates the final answer from relevant chunks.

    Uses our local GRPO-trained model — the same model from the non-agent pipeline.
    Falls back to all retrieved chunks if no relevant chunks survived grading.
    """

    def generate(state: AgentState) -> dict:
        query = state["query"]

        # Use graded chunks if available, otherwise fall back to all retrieved
        chunks = state.get("relevant_chunks", [])
        if not chunks:
            chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return {
                "answer": "Insufficient context.",
                "generation_result": GenerationResult(
                    answer="Insufficient context.",
                    prompt="",
                    model_name=model_name,
                    num_chunks_used=0,
                ),
            }

        # Take top 5 for the prompt
        chunks = chunks[:5]

        result = generate_answer(
            query=query,
            results=chunks,
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
        )

        return {"answer": result.answer, "generation_result": result}

    return generate


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def should_retry_or_generate(state: AgentState) -> Literal["rewrite", "generate"]:
    """Decide whether to rewrite the query and retry, or proceed to answer.

    Routes to rewrite if:
        - No relevant chunks found after grading
        - AND we haven't exhausted retries (max 2)
    Otherwise, proceed to generate an answer with whatever we have.
    """
    relevant = state.get("relevant_chunks", [])
    retries = state.get("retries", 0)

    if len(relevant) == 0 and retries < 2:
        return "rewrite"
    return "generate"


# ---------------------------------------------------------------------------
# Build the agent graph
# ---------------------------------------------------------------------------

def build_agent(
    index: DualIndex,
    reranker=None,
    model=None,
    tokenizer=None,
    openai_api_key: str | None = None,
    model_name: str = "qwen2.5-3b",
    grading_model: str = "gpt-4o-mini",
    retrieval_top_k: int = 5,
) -> StateGraph:
    """Build the LangGraph agentic RAG pipeline.

    Args:
        index: DualIndex with BM25 + FAISS indexes built from FinQA chunks
        reranker: cross-encoder reranker (optional but recommended)
        model: local LLM for answer generation (transformers model)
        tokenizer: tokenizer for the local model
        openai_api_key: for query decomposition, grading, and rewriting
        model_name: name of the local model (for logging)
        grading_model: OpenAI model for grading/decomposition
        retrieval_top_k: number of chunks to retrieve per sub-query

    Returns:
        Compiled LangGraph that can be invoked with {"query": "..."}
    """
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    # Create nodes
    decompose = make_decompose_node(api_key, model_name=grading_model)
    retrieve_node = make_retrieve_node(index, reranker, top_k=retrieval_top_k)
    grade = make_grade_node(api_key, model_name=grading_model)
    rewrite = make_rewrite_node(api_key, model_name=grading_model)
    generate = make_generate_node(model, tokenizer, model_name)

    # Build graph
    workflow = StateGraph(AgentState)

    workflow.add_node("decompose", decompose)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)

    # Edges
    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", should_retry_or_generate)
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()
