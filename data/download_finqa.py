"""Download and explore the FinQA dataset from HuggingFace.

FinQA (Chen et al., EMNLP 2021): 8,281 Q&A pairs grounded in SEC 10-K
annual reports from S&P 500 companies, 1999-2019.

Uses the wandb/finqa-data-processed mirror (pre-processed, no custom script).

Each example contains:
    - query: natural language question
    - context: combined pre_text + table + post_text
    - output / exe_ans: ground-truth answer (string or number)
    - program: step-by-step calculation (e.g., subtract(265.6, 229.2), divide(#0, 229.2))
    - pre_text / post_text: paragraphs surrounding the financial table
    - table: the financial table
"""

from datasets import load_dataset


DATASET_ID = "wandb/finqa-data-processed"


def download():
    """Download FinQA and print basic stats."""
    print("Downloading FinQA from HuggingFace...")
    ds = load_dataset(DATASET_ID)

    for split in ds:
        print(f"  {split}: {len(ds[split])} examples")

    # Show a few examples
    for i in range(3):
        ex = ds["train"][i]
        print(f"\n--- Example {i} ---")
        print(f"Query:   {ex['query']}")
        print(f"Answer:  {ex['exe_ans']}")
        print(f"Program: {ex['program']}")

    return ds


if __name__ == "__main__":
    download()
