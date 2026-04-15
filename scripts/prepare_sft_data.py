"""Convert FinQA dataset into chat-format SFT training data.

Each FinQA example becomes a (system, user, assistant) conversation:
  - user:      gold context (pre_text + table + post_text) + question
  - assistant:  step-by-step reasoning (from program field) + final answer

Why gold context (not retrieved chunks)?
  We want to teach the model HOW to reason over financial tables,
  not how to handle bad retrieval. At eval time, the RAG pipeline
  feeds whatever it finds, and the fine-tuned model applies its skills.

Usage:
    python scripts/prepare_sft_data.py --output data/sft_train.jsonl
    python scripts/prepare_sft_data.py --output data/sft_train.jsonl --preview 5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


# Same system prompt as generation.py — model learns to follow this format
SYSTEM_PROMPT = """You are a financial analyst. Answer the question based ONLY on the provided context from SEC 10-K filings.

Rules:
- If the answer is a number, compute it step by step and give the final numerical answer.
- If the answer requires a percentage, show your calculation.
- If the context doesn't contain enough information, say "Insufficient context."
- Always show your reasoning before the final answer.

Format your response as:
Reasoning: <your step-by-step calculation>
Answer: <final answer>"""


# FinQA program DSL operations
OPS = {
    "add": "+",
    "subtract": "-",
    "multiply": "×",
    "divide": "÷",
    "greater": ">",
    "exp": "^",
}

CONSTANTS = {
    "const_100": "100",
    "const_1000": "1000",
    "const_1000000": "1000000",
    "const_1": "1",
    "const_2": "2",
    "const_3": "3",
    "const_4": "4",
    "const_10": "10",
    "const_0.01": "0.01",
}


def _resolve_arg(arg: str, step_results: dict[int, str]) -> str:
    """Resolve a program argument: #0 → result of step 0, const_100 → 100, else literal."""
    arg = arg.strip()
    if arg.startswith("#"):
        step_idx = int(arg[1:])
        return step_results.get(step_idx, arg)
    if arg in CONSTANTS:
        return CONSTANTS[arg]
    # Clean up percentage notation
    if arg.endswith("%"):
        return arg
    return arg


def program_to_reasoning(program: str, final_answer) -> str:
    """Convert a FinQA program string into natural-language reasoning.

    Example:
        program: "subtract(313.45, const_100), divide(#0, const_100)"
        exe_ans: 2.1345
        →
        "Step 1: 313.45 - 100 = 213.45
         Step 2: 213.45 ÷ 100 = 2.1345
         Answer: 2.1345"
    """
    if not program:
        return f"Answer: {final_answer}"

    # Parse steps: "op(arg1, arg2), op(arg1, arg2)" → list of (op, args)
    # Pattern: word(stuff) separated by commas at the top level
    steps = []
    depth = 0
    current = []
    for char in program:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
            if depth == 0:
                steps.append("".join(current).strip())
                current = []
        elif char == "," and depth == 0:
            # Skip commas between steps
            if current:
                steps.append("".join(current).strip())
                current = []
        else:
            current.append(char)
    if current:
        step_str = "".join(current).strip()
        if step_str:
            steps.append(step_str)

    # Filter out empty strings
    steps = [s.strip().strip(",").strip() for s in steps if s.strip().strip(",")]

    step_results = {}
    reasoning_lines = []

    for i, step_str in enumerate(steps):
        # Parse "op(arg1, arg2)"
        match = re.match(r"(\w+)\((.+)\)", step_str)
        if not match:
            continue

        op_name = match.group(1)
        args_str = match.group(2)

        # Split args (careful: nested parens shouldn't exist here, but be safe)
        args = [a.strip() for a in args_str.split(",")]
        resolved = [_resolve_arg(a, step_results) for a in args]

        op_symbol = OPS.get(op_name, op_name)

        # Compute result for intermediate steps
        try:
            nums = []
            for r in resolved:
                cleaned = r.replace("%", "").replace(",", "")
                nums.append(float(cleaned))

            if op_name == "add":
                result = nums[0] + nums[1]
            elif op_name == "subtract":
                result = nums[0] - nums[1]
            elif op_name == "multiply":
                # Handle percentage: multiply(700, 4.750%) means 700 * 0.04750
                if "%" in resolved[1]:
                    result = nums[0] * (nums[1] / 100)
                elif "%" in resolved[0]:
                    result = (nums[0] / 100) * nums[1]
                else:
                    result = nums[0] * nums[1]
            elif op_name == "divide":
                result = nums[0] / nums[1] if nums[1] != 0 else 0
            elif op_name == "greater":
                result = "yes" if nums[0] > nums[1] else "no"
            elif op_name == "exp":
                result = nums[0] ** nums[1]
            else:
                result = nums[0]  # fallback

            if isinstance(result, float):
                result_str = f"{result:.5f}".rstrip("0").rstrip(".")
            else:
                result_str = str(result)

            step_results[i] = result_str
        except (ValueError, ZeroDivisionError, IndexError):
            result_str = str(final_answer) if i == len(steps) - 1 else "?"
            step_results[i] = result_str

        # Build readable line
        if op_name == "greater":
            reasoning_lines.append(
                f"Step {i+1}: Compare {resolved[0]} and {resolved[1]} → "
                f"{resolved[0]} {'>' if result_str == 'yes' else '<'} {resolved[1]} → {result_str}"
            )
        else:
            reasoning_lines.append(
                f"Step {i+1}: {resolved[0]} {op_symbol} {resolved[1]} = {result_str}"
            )

    reasoning = "\n".join(reasoning_lines) if reasoning_lines else "Direct lookup from the table."
    return f"Reasoning: {reasoning}\nAnswer: {final_answer}"


def _safe_join(raw) -> str:
    """Join a list of strings or return string as-is.

    The wandb mirror stores pre_text/post_text as stringified Python lists,
    e.g. "['sentence one', 'sentence two']". We parse those back.
    """
    if isinstance(raw, list):
        return " ".join(str(s) for s in raw)
    if isinstance(raw, str):
        # Try to parse stringified list
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                import ast
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return " ".join(str(item) for item in parsed)
            except (ValueError, SyntaxError):
                pass
        return s
    return str(raw) if raw else ""


def _table_to_text(raw_table) -> str:
    """Convert table to readable text."""
    if not raw_table:
        return ""
    if isinstance(raw_table, str):
        return raw_table
    if isinstance(raw_table, list):
        lines = []
        for row in raw_table:
            if isinstance(row, list):
                lines.append(" | ".join(str(cell) for cell in row))
            else:
                lines.append(str(row))
        return "\n".join(lines)
    return str(raw_table)


def build_user_message(example: dict) -> str:
    """Build the user message from gold context + question."""
    pre = _safe_join(example.get("pre_text", ""))
    table = _table_to_text(example.get("table", ""))
    post = _safe_join(example.get("post_text", ""))

    context = f"{pre}\n\n{table}\n\n{post}".strip()
    query = example.get("query", "")

    return f"{context}\n\n---\n\nQuestion: {query}"


def convert_example(example: dict) -> dict:
    """Convert one FinQA example into chat-format SFT data."""
    user_msg = build_user_message(example)
    program = example.get("program", "")
    exe_ans = example.get("exe_ans", "")
    assistant_msg = program_to_reasoning(program, exe_ans)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "id": example.get("id", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare FinQA SFT training data")
    parser.add_argument("--output", type=str, default="data/sft_train.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to convert")
    parser.add_argument("--preview", type=int, default=0,
                        help="Print N examples and exit (don't write file)")
    args = parser.parse_args()

    print("Loading FinQA dataset...")
    ds = load_dataset("wandb/finqa-data-processed", split=args.split)
    print(f"  {len(ds)} examples in '{args.split}' split")

    if args.preview > 0:
        print(f"\n{'='*60}")
        print(f"Preview of {args.preview} examples:")
        print(f"{'='*60}")
        for i in range(min(args.preview, len(ds))):
            converted = convert_example(ds[i])
            print(f"\n--- Example {i} ---")
            print(f"User: {converted['messages'][1]['content'][:200]}...")
            print(f"\nAssistant: {converted['messages'][2]['content']}")
            print()
        return

    # Convert all examples
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = []
    skipped = 0
    for i, ex in enumerate(ds):
        try:
            item = convert_example(ex)
            converted.append(item)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Warning: skipped example {i}: {e}")

    # Write JSONL
    with open(output_path, "w") as f:
        for item in converted:
            f.write(json.dumps(item) + "\n")

    print(f"\nWrote {len(converted)} examples to {output_path}")
    if skipped:
        print(f"  Skipped {skipped} examples due to errors")

    # Stats
    assistant_lengths = [
        len(item["messages"][2]["content"]) for item in converted
    ]
    print(f"  Assistant response length: "
          f"min={min(assistant_lengths)}, "
          f"max={max(assistant_lengths)}, "
          f"avg={sum(assistant_lengths)/len(assistant_lengths):.0f} chars")


if __name__ == "__main__":
    main()
