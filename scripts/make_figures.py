"""
Generate the FilingSense Sankey figure for README and the LinkedIn post.

Reads:
    - results/error_taxonomy.json

Writes:
    - results/figures/pipeline_sankey.png
"""
from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go


REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "results" / "figures"
TAXONOMY_PATH = REPO_ROOT / "results" / "error_taxonomy.json"


def chart_sankey_funnel(model: str = "LoRA SFT"):
    with open(TAXONOMY_PATH) as f:
        data = json.load(f)

    cats = data[model]["categories"]
    n_total = data[model]["total"]
    n_correct = cats["correct"]["count"]
    n_retrieval_miss = cats["retrieval_miss"]["count"]
    n_wrong_extract = cats["wrong_extraction"]["count"]
    n_wrong_op = cats["wrong_operation"]["count"]
    n_loop = cats["reasoning_loop"]["count"]
    n_format = cats["format_error"]["count"]

    n_after_retrieval = n_total - n_retrieval_miss
    n_after_extract = n_after_retrieval - n_wrong_extract
    n_after_compute = n_after_extract - n_wrong_op
    n_format_loop = n_loop + n_format

    PIPELINE_C = "#1d6fb8"
    LLM_C = "#7a4baa"
    FAIL_C = "#d0021b"
    SUCCESS_C = "#52a661"
    QUERY_C = "#444444"

    PASS_RIBBON = "rgba(29, 111, 184, 0.40)"
    LLM_RIBBON = "rgba(122, 75, 170, 0.40)"
    FAIL_RIBBON = "rgba(208, 2, 27, 0.50)"
    SUCCESS_RIBBON = "rgba(82, 166, 97, 0.55)"

    def pct(n):
        return f"{n / n_total * 100:.1f}%"

    nodes = [
        dict(label=f"Query<br>n = {n_total}",                                  color=QUERY_C,    x=0.001, y=0.30),
        dict(label="<b>Hybrid Retrieval</b><br>BM25 · FAISS · RRF",            color=PIPELINE_C, x=0.16,  y=0.30),
        dict(label="<b>Cross-encoder<br>Rerank</b>",                           color=PIPELINE_C, x=0.31,  y=0.30),
        dict(label="<b>LLM: Extract</b><br>pull the numbers<br>→ 111, 89",                  color=LLM_C,      x=0.47,  y=0.30),
        dict(label="<b>LLM: Compute</b><br>run the math<br>(111−89)/89 → 0.247",             color=LLM_C,      x=0.62,  y=0.30),
        dict(label="<b>LLM: Format</b><br>match expected form<br>e.g. 0.247 → 24.7%",         color=LLM_C,      x=0.77,  y=0.30),
        dict(label=f"<b>Correct</b><br>{n_correct} ({pct(n_correct)})",        color=SUCCESS_C,  x=0.999, y=0.20),
        dict(label=f"Retrieval miss<br>{n_retrieval_miss} ({pct(n_retrieval_miss)})",  color=FAIL_C, x=0.30, y=0.85),
        dict(label=f"Picked wrong number<br>{n_wrong_extract} ({pct(n_wrong_extract)})",  color=FAIL_C, x=0.60, y=0.85),
        dict(label=f"Wrong operation<br>{n_wrong_op} ({pct(n_wrong_op)})",     color=FAIL_C,    x=0.75, y=0.85),
        dict(label=f"Wrong sign or scale<br>{n_format_loop} ({pct(n_format_loop)})", color=FAIL_C,    x=0.999, y=0.85),
    ]

    links = [
        dict(source=0, target=1, value=n_total,           color=PASS_RIBBON),
        dict(source=1, target=2, value=n_after_retrieval, color=PASS_RIBBON),
        dict(source=1, target=7, value=n_retrieval_miss,  color=FAIL_RIBBON),
        dict(source=2, target=3, value=n_after_retrieval, color=LLM_RIBBON),
        dict(source=3, target=4, value=n_after_extract,   color=LLM_RIBBON),
        dict(source=3, target=8, value=n_wrong_extract,   color=FAIL_RIBBON),
        dict(source=4, target=5, value=n_after_compute,   color=LLM_RIBBON),
        dict(source=4, target=9, value=n_wrong_op,        color=FAIL_RIBBON),
        dict(source=5, target=6, value=n_correct,         color=SUCCESS_RIBBON),
        dict(source=5, target=10, value=n_format_loop,    color=FAIL_RIBBON),
    ]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18,
            thickness=22,
            line=dict(color="white", width=0.5),
            label=[n["label"] for n in nodes],
            color=[n["color"] for n in nodes],
            x=[n["x"] for n in nodes],
            y=[n["y"] for n in nodes],
        ),
        link=dict(
            source=[l["source"] for l in links],
            target=[l["target"] for l in links],
            value=[l["value"] for l in links],
            color=[l["color"] for l in links],
        ),
        textfont=dict(family="Helvetica Neue, Helvetica, Arial", size=12, color="#1a1a1a"),
    )])

    fig.update_layout(
        title=dict(
            text=(
                f"<span style='font-size:13px; color:#1d6fb8; letter-spacing:3px; font-weight:700;'>"
                f"ERROR TAXONOMY  ·  END-TO-END FAILURE ANALYSIS"
                f"</span><br>"
                f"<span style='font-size:10px;'>&nbsp;</span><br>"
                f"<b style='font-size:26px;'>Retrieval misses cause more failures than reasoning errors</b><br>"
                f"<span style='font-size:13px; color:#555;'>"
                f"{model} on FinQA · <b>{n_correct} correct ({pct(n_correct)})</b> "
                f"— vs. base Qwen2.5-3B + RAG, no fine-tuning: <b>23 correct (11.5%)</b>"
                f"</span>"
            ),
            x=0.02, y=0.97, xanchor="left", yanchor="top",
        ),
        annotations=[
            dict(
                text=(
                    f"<i>{n_retrieval_miss / n_total * 100:.0f}% of failures happen before the LLM even sees the right context — "
                    f"a better-trained model can't fix retrieval misses.</i>"
                ),
                xref="paper", yref="paper",
                x=0.5, y=-0.14,
                showarrow=False,
                font=dict(size=12, color="#444"),
                xanchor="center",
            ),
        ],
        font=dict(family="Helvetica Neue, Helvetica, Arial", size=12, color="#1a1a1a"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=1500,
        height=900,
        margin=dict(l=20, r=20, t=180, b=140),
    )

    out = FIG_DIR / "pipeline_sankey.png"
    fig.write_image(str(out), width=1500, height=900, scale=2)
    print(f"  wrote {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating chart: sankey funnel...")
    chart_sankey_funnel()
    print("Done.")


if __name__ == "__main__":
    main()
