"""Generate financial-analyst QA pairs from a 10-K text chunk using Claude.

Given a chunk of cleaned 10-K text (e.g. one produced by parser.py), prompt
Claude to act as a financial analyst and generate 2 highly specific
question/answer pairs grounded ONLY in that chunk. Output is constrained to a
Pydantic schema via the Anthropic structured-outputs API, so the result is
guaranteed-valid JSON.

Run directly to demo against the first chunk of a parser.py output file:
    python src/generator.py
(Requires ANTHROPIC_API_KEY in the environment / .env.)
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Literal

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# Use the most capable current Claude model. (The prompt mentioned
# claude-3-5-sonnet, but that model was retired; gpt-4o is a non-Anthropic
# model and this pipeline is built on the Anthropic SDK.)
MODEL = "claude-opus-4-8"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"

# Single shared async client (resolves ANTHROPIC_API_KEY from the environment).
client = AsyncAnthropic()


# --- Output schema -----------------------------------------------------------

QuestionType = Literal[
    "Literal",
    "Fact extraction",
    "Numeric calculation",
    "Comparison",
    "Multi-step reasoning",
]
Difficulty = Literal["Easy", "Medium", "Hard"]


class QAPair(BaseModel):
    """A single question/answer pair grounded in the source chunk."""

    Question: str = Field(description="A specific question answerable from the chunk alone.")
    Ground_Truth_Answer: str = Field(description="The correct, complete answer.")
    Source_Passage: str = Field(
        description="The exact verbatim quote from the chunk that supports the answer."
    )
    Question_Type: QuestionType
    Difficulty: Difficulty


class QAGenerationResult(BaseModel):
    """The structured result returned for one chunk: exactly 2 QA pairs."""

    qa_pairs: list[QAPair]


SYSTEM_PROMPT = (
    "You are a meticulous financial analyst building an evaluation dataset from "
    "SEC 10-K filings. You generate precise question/answer pairs that test deep "
    "understanding of the source text."
)

USER_PROMPT_TEMPLATE = """\
Below is an excerpt from a company's 10-K annual report. Acting as a financial \
analyst, generate exactly 2 HIGHLY SPECIFIC question/answer pairs based ONLY on \
the information in this excerpt.

Strict rules:
- Both the question and the answer must be fully answerable from the excerpt alone. \
Do not use outside knowledge or make assumptions beyond the text.
- `Source_Passage` must be an EXACT, verbatim quote copied from the excerpt that \
supports the answer (no paraphrasing).
- Make the two pairs distinct in focus, and prefer different `Question_Type` and \
`Difficulty` values where the text supports it.
- Questions should be specific and non-trivial (avoid yes/no questions).

Excerpt:
\"\"\"
{chunk}
\"\"\"
"""


# --- Core generation function ------------------------------------------------

async def generate_qa_pairs(chunk_text: str) -> QAGenerationResult:
    """Generate 2 grounded QA pairs for one text chunk, as a structured object."""
    response = await client.messages.parse(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(chunk=chunk_text)}
        ],
        output_format=QAGenerationResult,
    )
    return response.parsed_output


# --- Demo / orchestration ----------------------------------------------------

async def _demo(ticker: str = "MSFT", chunk_index: int = 0) -> None:
    chunks_path = INPUT_DIR / f"{ticker}_10K_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"{chunks_path.relative_to(PROJECT_ROOT)} not found — run src/parser.py first."
        )

    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunk = data["chunks"][chunk_index]["text"]

    print(f"Generating QA pairs for {ticker} chunk #{chunk_index} ...")
    result = await generate_qa_pairs(chunk)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY (in .env) to run the demo.")
    asyncio.run(_demo())
