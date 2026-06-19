"""Verify a generated QA pair against its source to catch hallucinations.

A QA pair (from generator.py) is valid only if BOTH hold:
  1. The Source_Passage actually contains/supports the Ground_Truth_Answer.
  2. The Source_Passage is an exact substring of the original chunk.

Check (2) is a deterministic string operation, so it is done in Python — the
authoritative gate, since LLMs are unreliable at exact-substring matching.
Check (1) is a semantic judgment, so it goes to Claude. The LLM is also asked
to confirm both (per spec), but the Python substring result is what decides (2).
The function returns True only if every check passes, else False.

Run directly for a self-contained demo (the substring logic runs without a key;
the LLM check requires ANTHROPIC_API_KEY in .env):
    python src/verifier.py
"""

from __future__ import annotations

import asyncio
import os

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "claude-opus-4-8"

client = AsyncAnthropic()


# --- Structured verdict from the LLM -----------------------------------------

class LLMVerdict(BaseModel):
    """The LLM's strict assessment of a QA pair."""

    answer_supported_by_passage: bool = Field(
        description="True only if the Source_Passage actually contains/supports the answer."
    )
    passage_is_exact_substring: bool = Field(
        description="True only if the Source_Passage appears verbatim in the original chunk."
    )
    reasoning: str = Field(description="One sentence explaining the verdict.")


SYSTEM_PROMPT = (
    "You are a strict fact-checker for a financial QA dataset. You catch any "
    "hallucination or mismatch. You are conservative: if you are not certain a "
    "condition holds, you answer false."
)

USER_PROMPT_TEMPLATE = """\
Strictly verify the following question/answer pair against its source. Answer two \
independent checks, each true ONLY if you are certain:

CHECK 1 — Does the Source_Passage actually contain and support the \
Ground_Truth_Answer? The answer must be derivable from the passage alone, with no \
outside knowledge and no contradiction. Set `answer_supported_by_passage`.

CHECK 2 — Is the Source_Passage an EXACT, verbatim substring of the Original Chunk \
(character-for-character, not paraphrased or summarized)? Set \
`passage_is_exact_substring`.

If there is any hallucination, fabrication, paraphrase, or mismatch, the \
corresponding check is false.

Question:
{question}

Ground_Truth_Answer:
{answer}

Source_Passage:
{passage}

Original Chunk:
\"\"\"
{chunk}
\"\"\"
"""


# --- Deterministic substring check -------------------------------------------

def is_exact_substring(passage: str, chunk: str) -> bool:
    """Authoritative check for condition (2): exact verbatim substring."""
    return passage in chunk


# --- Verification API --------------------------------------------------------

async def verify_qa_pair_detailed(
    question: str,
    ground_truth_answer: str,
    source_passage: str,
    original_chunk: str,
) -> tuple[bool, LLMVerdict, bool]:
    """Verify a QA pair. Returns (is_valid, llm_verdict, exact_substring_match)."""
    # Deterministic gate for condition (2).
    exact_match = is_exact_substring(source_passage, original_chunk)

    response = await client.messages.parse(
        model=MODEL,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    question=question,
                    answer=ground_truth_answer,
                    passage=source_passage,
                    chunk=original_chunk,
                ),
            }
        ],
        output_format=LLMVerdict,
    )
    verdict = response.parsed_output

    # Valid only if the answer is supported AND the passage is verifiably a
    # verbatim substring (Python is authoritative; the LLM's substring opinion
    # acts as an extra guard).
    is_valid = (
        verdict.answer_supported_by_passage
        and verdict.passage_is_exact_substring
        and exact_match
    )
    return is_valid, verdict, exact_match


async def verify_qa_pair(
    question: str,
    ground_truth_answer: str,
    source_passage: str,
    original_chunk: str,
) -> bool:
    """Return True if the QA pair passes both checks, else False."""
    is_valid, _, _ = await verify_qa_pair_detailed(
        question, ground_truth_answer, source_passage, original_chunk
    )
    return is_valid


# --- Demo --------------------------------------------------------------------

async def _demo() -> None:
    chunk = (
        "Total revenue increased 16% to $245.1 billion in fiscal year 2024, "
        "driven by growth in our Intelligent Cloud segment. Operating income "
        "was $109.4 billion."
    )

    # A faithful pair (passage is a verbatim substring; answer is supported).
    good = (
        "What was total revenue in fiscal year 2024?",
        "$245.1 billion",
        "Total revenue increased 16% to $245.1 billion in fiscal year 2024",
    )
    # A bad pair: the passage is NOT a verbatim substring (paraphrased number).
    bad = (
        "What was total revenue in fiscal year 2024?",
        "$250 billion",
        "Total revenue increased 16% to $250.1 billion in fiscal year 2024",
    )

    # The deterministic substring gate works without an API key:
    print("Deterministic substring check:")
    print("  good passage exact substring:", is_exact_substring(good[2], chunk))
    print("  bad  passage exact substring:", is_exact_substring(bad[2], chunk))

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\nSet ANTHROPIC_API_KEY (in .env) to run the full LLM verification.")
        return

    for label, (q, a, p) in (("GOOD", good), ("BAD", bad)):
        valid, verdict, exact = await verify_qa_pair_detailed(q, a, p, chunk)
        print(f"\n[{label}] is_valid={valid} (exact_substring={exact})")
        print(f"  answer_supported={verdict.answer_supported_by_passage}")
        print(f"  llm_substring={verdict.passage_is_exact_substring}")
        print(f"  reasoning: {verdict.reasoning}")


if __name__ == "__main__":
    asyncio.run(_demo())
