"""Generation module: take retrieved chunks + query → LLM → answer.

This is the "G" in RAG. The retrieval pipeline finds the relevant chunks,
this module builds a prompt and asks the LLM to answer the question.

Supports two backends:
    1. Local model (Qwen2.5-3B via transformers) — for benchmarking and fine-tuning
    2. API model (any OpenAI-compatible API) — for quick prototyping
"""

from __future__ import annotations

from dataclasses import dataclass

from src.retrieval import RetrievalResult


SYSTEM_PROMPT = """You are a financial analyst. Answer the question based ONLY on the provided context from SEC 10-K filings.

Rules:
- If the answer is a number, compute it step by step and give the final numerical answer.
- If the answer requires a percentage, show your calculation.
- If the context doesn't contain enough information, say "Insufficient context."
- Always show your reasoning before the final answer.

Format your response as:
Reasoning: <your step-by-step calculation>
Answer: <final answer>"""


def build_prompt(query: str, results: list[RetrievalResult]) -> str:
    """Build a prompt from the query and retrieved chunks.

    Format:
        Context 1: [chunk text]
        Context 2: [chunk text]
        ...
        Question: [query]
    """
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"Context {i} (source: {r.chunk.chunk_id}):\n{r.chunk.text}")

    context_block = "\n\n---\n\n".join(context_parts)

    return f"""{context_block}

---

Question: {query}"""


@dataclass
class GenerationResult:
    """The LLM's response plus metadata."""
    answer: str           # the full response text
    prompt: str           # the prompt that was sent
    model_name: str       # which model generated this
    num_chunks_used: int  # how many chunks were in the prompt


def generate_answer(
    query: str,
    results: list[RetrievalResult],
    model=None,
    tokenizer=None,
    model_name: str = "qwen2.5-3b",
    max_new_tokens: int = 512,
) -> GenerationResult:
    """Generate an answer using a local HuggingFace model.

    Args:
        query: the user's question
        results: retrieved chunks from the retrieval pipeline
        model: a loaded transformers model (AutoModelForCausalLM)
        tokenizer: the model's tokenizer
        model_name: name for logging
        max_new_tokens: max tokens to generate
    """
    user_prompt = build_prompt(query, results)

    # Build chat-format messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Use the tokenizer's chat template to format
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,        # greedy for reproducibility
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the new tokens (skip the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return GenerationResult(
        answer=answer.strip(),
        prompt=user_prompt,
        model_name=model_name,
        num_chunks_used=len(results),
    )


def generate_answer_api(
    query: str,
    results: list[RetrievalResult],
    api_key: str | None = None,
    model_name: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    max_tokens: int = 512,
) -> GenerationResult:
    """Generate an answer using an OpenAI-compatible API.

    Useful for quick prototyping before fine-tuning your own model.
    Works with OpenAI, Together, Groq, or any compatible endpoint.
    """
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    user_prompt = build_prompt(query, results)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )

    answer = response.choices[0].message.content

    return GenerationResult(
        answer=answer.strip(),
        prompt=user_prompt,
        model_name=model_name,
        num_chunks_used=len(results),
    )
