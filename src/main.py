"""End-to-end pipeline: parse 10-Ks -> generate QA pairs -> verify -> dataset.

Flow:
  1. Ensure each configured ticker's chunks exist (run parser.py if needed).
  2. Stream chunks through the generator (async) and verifier concurrently.
  3. Discard any QA pair the verifier rejects (hallucination / mismatch).
  4. Collect exactly TARGET verified pairs, then stop.
  5. Save the dataset to data/output/caliper_dataset.csv with pandas.

A single 10-K yields too few pairs to reach 100, so TICKERS lists several
companies. The scheduler stops requesting new chunks as soon as TARGET is met,
so it only processes as many filings as it actually needs.

Run:  python src/main.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Make sibling modules importable whether run as `python src/main.py` or otherwise.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parser  # noqa: E402  (local module: src/parser.py)
from generator import generate_qa_pairs  # noqa: E402
from verifier import verify_qa_pair  # noqa: E402

load_dotenv()

# --- Configuration -----------------------------------------------------------

TARGET = 100          # exactly this many verified QA pairs
CONCURRENCY = 5       # chunks processed in parallel (each makes a few API calls)
TICKERS = ["MSFT", "AAPL", "AMZN", "GOOGL", "NVDA"]  # enough to clear TARGET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_CSV = OUTPUT_DIR / "caliper_dataset.csv"

CSV_COLUMNS = [
    "ticker",
    "chunk_id",
    "Question",
    "Ground_Truth_Answer",
    "Source_Passage",
    "Question_Type",
    "Difficulty",
]


# --- Chunk loading -----------------------------------------------------------

def ensure_chunks(ticker: str) -> list[dict]:
    """Return the parsed chunks for a ticker, running the parser if needed."""
    chunks_path = INPUT_DIR / f"{ticker}_10K_chunks.json"
    if not chunks_path.exists():
        print(f"[{ticker}] chunks not found — running parser ...")
        parser.main(ticker)
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    return data["chunks"]


def iter_all_chunks():
    """Yield (ticker, chunk) lazily so parsing happens only as chunks are pulled."""
    for ticker in TICKERS:
        for chunk in ensure_chunks(ticker):
            yield ticker, chunk


# --- Per-chunk processing (generate + verify) --------------------------------

async def process_chunk(ticker: str, chunk: dict) -> list[dict]:
    """Generate QA pairs for one chunk and return only the verified ones."""
    text = chunk["text"]
    chunk_id = chunk["chunk_id"]
    try:
        result = await generate_qa_pairs(text)
    except Exception as exc:  # noqa: BLE001 - keep the pipeline going
        print(f"[{ticker} #{chunk_id}] generation failed: {exc}")
        return []

    async def verified_row(pair) -> dict | None:
        try:
            ok = await verify_qa_pair(
                pair.Question, pair.Ground_Truth_Answer, pair.Source_Passage, text
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{ticker} #{chunk_id}] verification failed: {exc}")
            return None
        if not ok:
            return None
        return {
            "ticker": ticker,
            "chunk_id": chunk_id,
            "Question": pair.Question,
            "Ground_Truth_Answer": pair.Ground_Truth_Answer,
            "Source_Passage": pair.Source_Passage,
            "Question_Type": pair.Question_Type,
            "Difficulty": pair.Difficulty,
        }

    rows = await asyncio.gather(*(verified_row(p) for p in result.qa_pairs))
    return [r for r in rows if r is not None]


# --- Streaming scheduler -----------------------------------------------------

async def collect_verified_pairs(target: int = TARGET) -> list[dict]:
    """Stream chunks through process_chunk, keeping CONCURRENCY in flight, until
    `target` verified pairs are collected (then stop scheduling new work)."""
    verified: list[dict] = []
    chunks = iter_all_chunks()
    pending: set[asyncio.Task] = set()

    def schedule_next() -> bool:
        nxt = next(chunks, None)
        if nxt is None:
            return False
        pending.add(asyncio.create_task(process_chunk(*nxt)))
        return True

    # Prime the pool.
    for _ in range(CONCURRENCY):
        if not schedule_next():
            break

    while pending and len(verified) < target:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            verified.extend(task.result())
            print(f"  verified so far: {len(verified)}/{target}")
            if len(verified) < target:
                schedule_next()  # backfill the pool

    # Cancel any still-running tasks once we've hit the target.
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    return verified[:target]


# --- Output ------------------------------------------------------------------

def save_dataset(rows: list[dict], path: Path = OUTPUT_CSV) -> None:
    """Write the verified QA pairs to CSV with pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(path, index=False)
    try:
        display = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display = path
    print(f"Saved {len(df)} verified QA pairs to {display}")


# --- Entrypoint --------------------------------------------------------------

async def _run() -> None:
    rows = await collect_verified_pairs()
    if len(rows) < TARGET:
        print(
            f"WARNING: only {len(rows)} verified pairs collected "
            f"(wanted {TARGET}). Add more tickers to TICKERS to reach the target."
        )
    save_dataset(rows)


if __name__ == "__main__":
    import os

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY (in .env) to run the pipeline.")
    asyncio.run(_run())
