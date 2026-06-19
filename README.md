# Caliper Lab 10-K QA Pipeline

A pipeline that turns SEC 10-K annual reports into a verified question-answer
evaluation dataset. It downloads the most recent 10-K for a set of tickers,
cleans the filing HTML into narrative text chunks, uses Claude to generate
highly specific QA pairs grounded in each chunk, runs a **separate** verification
pass to discard hallucinations, and emits a CSV of exactly 100 verified pairs.

## Objective

Produce a high-quality, **grounded** QA dataset from financial filings — every
answer must be supported by an exact verbatim quote from the source document.
The dataset is intended for evaluating financial-domain LLM systems, where
unfaithful or fabricated ground-truth answers would silently corrupt eval
results. The pipeline therefore treats verification as a first-class stage, not
an afterthought.

## Pipeline stages

| Stage | File | Responsibility |
|-------|------|----------------|
| Parse | `src/parser.py` | Download the latest 10-K (`sec-edgar-downloader`), clean HTML (`BeautifulSoup`), drop boilerplate tables / TOC, chunk into ~1000-word segments, save JSON. |
| Generate | `src/generator.py` | Async Claude call per chunk → 2 QA pairs as a validated Pydantic object (structured outputs). |
| Verify | `src/verifier.py` | Separate Claude call + deterministic substring check; returns `True`/`False`. |
| Orchestrate | `src/main.py` | Stream chunks through generate→verify concurrently, discard rejects, collect exactly 100, write `data/output/caliper_dataset.csv` with pandas. |

## Project layout

```
caliper-10k-pipeline/
├── src/
│   ├── parser.py        # 10-K download + HTML cleaning + chunking
│   ├── generator.py     # async QA-pair generation (Pydantic structured output)
│   ├── verifier.py      # hallucination / exact-quote verification
│   └── main.py          # end-to-end orchestration
├── data/
│   ├── input/           # parsed chunks (JSON) + raw downloaded filings
│   └── output/          # caliper_dataset.csv
├── requirements.txt
├── .env                 # ANTHROPIC_API_KEY, SEC_EDGAR_USER_AGENT (gitignored)
└── README.md
```

## Install

Requires Python 3.10+ (developed on 3.13).

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
#    (also installs lxml, the BeautifulSoup parser backend)

# 3. Configure credentials in .env
#    ANTHROPIC_API_KEY=sk-ant-...
#    SEC_EDGAR_USER_AGENT="Your Name your@email.com"   # SEC requires a descriptive UA
```

> **SEC User-Agent:** the SEC asks every EDGAR client to send a descriptive
> `User-Agent` (a name + contact email). Set `SEC_EDGAR_USER_AGENT` in `.env`;
> the parser falls back to a placeholder if unset, but you should provide your
> own to be a good citizen and avoid throttling.

## Run

```bash
source venv/bin/activate

# Run a single stage (each module has a runnable demo)
python src/parser.py        # downloads MSFT 10-K, writes data/input/MSFT_10K_chunks.json
python src/generator.py     # demo: 2 QA pairs from the first cached chunk
python src/verifier.py      # demo: GOOD/BAD verification example

# Run the full pipeline → data/output/caliper_dataset.csv (100 verified pairs)
python src/main.py
```

The full run makes many API calls (each chunk = 1 generation + 2 verification
calls), so it takes a few minutes and incurs API cost. Tune `TARGET`,
`CONCURRENCY`, and `TICKERS` at the top of `src/main.py`. For a cheap dry run,
set `TARGET=4` and `TICKERS=["MSFT"]`.

### Output schema

`data/output/caliper_dataset.csv` columns:

| Column | Description |
|--------|-------------|
| `ticker` | Source company ticker |
| `chunk_id` | Index of the source chunk within that filing |
| `Question` | Generated question |
| `Ground_Truth_Answer` | Answer derivable from the source passage |
| `Source_Passage` | **Verbatim** quote from the chunk supporting the answer |
| `Question_Type` | `Literal` / `Fact extraction` / `Numeric calculation` / `Comparison` / `Multi-step reasoning` |
| `Difficulty` | `Easy` / `Medium` / `Hard` |

## Architecture & design choices

- **Async generation and verification (`asyncio` + `AsyncAnthropic`).** Each
  chunk is independent and the work is I/O-bound (network latency dominates), so
  the orchestrator keeps `CONCURRENCY` chunks in flight at once. A **streaming
  scheduler** (`collect_verified_pairs`) primes a bounded task pool, backfills as
  tasks complete, and **stops scheduling new work the moment the target is met**
  — so we never process more filings than necessary and we trim to *exactly* 100.

- **Pydantic + structured outputs for generation.** `generator.py` defines the
  output as a Pydantic model (`QAPair` with `Literal` enums for `Question_Type`
  and `Difficulty`) and uses the Anthropic structured-outputs API
  (`messages.parse(..., output_format=...)`). This guarantees schema-valid,
  parseable JSON — no brittle regex/`json.loads` on free-form text, and invalid
  enum values are impossible by construction.

- **A separate verification LLM call.** Generation and verification are
  deliberately decoupled into independent prompts/calls. Asking the same call
  that *wrote* an answer to also grade it invites self-confirmation bias; a fresh
  call with a strict fact-checker persona, given only the question/answer/passage
  and the original chunk, is far more likely to catch a hallucination. Verified
  pairs must pass **both** semantic support *and* exactness.

- **Deterministic substring check as the authoritative exactness gate.** Whether
  `Source_Passage` is a verbatim substring of the chunk is a pure string
  operation, and LLMs are unreliable at exact-substring matching. So `verifier.py`
  does `passage in chunk` in Python as the source of truth for that condition,
  and uses the LLM's opinion only as a secondary guard. This is cheaper and
  strictly more reliable than trusting the model.

- **Boilerplate filtering at parse time.** `parser.py` drops numeric/financial
  tables (digit-ratio heuristic) and table-of-contents blocks before chunking, so
  the generator sees narrative prose rather than grids of numbers that produce
  low-quality questions.

- **Resilient orchestration.** Per-chunk generation/verification failures are
  caught and logged, not fatal — one bad chunk or transient API error doesn't
  sink the whole run. The Anthropic SDK already retries 429/5xx with exponential
  backoff.

## Known limitations

- **SEC EDGAR HTML variability.** Filings are inline-XBRL/HTML with no consistent
  structure across companies or years (inline styles, nested tables, vendor
  quirks). The cleaning heuristics in `parser.py` are tuned to common patterns
  and will not be optimal for every filer; some narrative may be dropped or some
  boilerplate retained.

- **Table parsing is hard.** The digit-ratio heuristic that discards financial
  tables is coarse. It can drop a narrative-heavy table that happens to contain
  many numbers, or keep a borderline one. Genuinely table-derived facts (e.g.
  values that only appear in a financial statement grid) are largely excluded,
  so the dataset skews toward prose-grounded questions.

- **Exact-substring strictness.** The verifier rejects a passage that differs
  from the source by even whitespace or a normalized character. This favors
  precision over recall — some legitimate pairs are discarded because the model
  slightly reformatted the quote. A whitespace-normalized comparison would raise
  yield at some risk to strictness.

- **Cost and latency.** Every QA pair costs ~3 LLM calls (1 generate + 2 verify).
  Producing 100 verified pairs is on the order of 150+ calls; scaling to
  thousands multiplies cost and runtime linearly under the current design.

- **No deduplication.** Nothing prevents two chunks (or two filings) from
  producing near-duplicate questions. See Scaling Strategy.

- **Single-filing-per-ticker, most-recent only.** The pipeline pulls only the
  latest 10-K per ticker and no other form types.

## Scaling Strategy

The current design is a single local async script — appropriate for 100 pairs
from a handful of filings. Scaling to **multiple documents and 1,000+ QA pairs**
requires moving from "one process orchestrating everything" to a distributed,
queue-driven, cloud-managed pipeline:

### 1. Distributed task queue (Celery / Ray)

Decompose the pipeline into independent, idempotent tasks — `download_filing`,
`chunk_document`, `generate_qa`, `verify_qa` — and dispatch them onto a
distributed task queue such as **Celery** (with Redis/RabbitMQ as broker) or
**Ray** for Python-native parallelism. Each chunk becomes a unit of work that any
worker can pick up, so throughput scales horizontally by adding workers rather
than being bound to one machine's event loop. Failed tasks retry independently
with backoff, and partial progress survives crashes (work is checkpointed in the
broker/result backend, not held in process memory).

### 2. Vector database for deduplication

At 1,000+ questions across many filings, near-duplicate questions become a real
problem (e.g. every tech company's risk-factors section yields similar
questions). Before accepting a verified pair, embed the question and query a
**vector database** (Pinecone, Weaviate, pgvector, or Milvus) for nearest
neighbors above a cosine-similarity threshold; if a semantically equivalent
question already exists, discard the new one. This guarantees diversity and
prevents the dataset from collapsing onto a few common question templates. The
vector store also doubles as the dataset index for downstream retrieval/eval.

### 3. Batch API processing to avoid rate limits

Generation and verification are not latency-sensitive when building a dataset
offline. Switch from synchronous per-chunk calls to the **Anthropic Message
Batches API**, which processes up to 100,000 requests asynchronously at **50% of
standard cost** and sidesteps per-minute rate limits entirely. The flow becomes:
submit all generation requests as one batch → poll for completion → submit all
verification requests as a second batch → poll → assemble results. This trades a
small amount of wall-clock latency (batches typically finish within an hour) for
dramatically higher throughput, lower cost, and no 429 handling.

### 4. From local script to a managed cloud pipeline (AWS Step Functions / Airflow)

Replace `main.py`'s in-process orchestration with a managed workflow engine —
**AWS Step Functions** or **Apache Airflow** — that models the pipeline as a DAG:
`fetch tickers → download filings → chunk → submit generation batch → wait →
submit verification batch → wait → dedupe via vector DB → write dataset`. The
orchestrator handles scheduling, retries, fan-out/fan-in over thousands of
documents, observability, and recovery from partial failure. Filings and chunks
live in object storage (S3), the dataset lands in a warehouse or data lake, and
the whole pipeline can run on a schedule (e.g. re-process new filings as they're
published) without anyone babysitting a terminal. Workers (the Celery/Ray tasks
above) run on autoscaling compute (ECS/EKS/Batch), so capacity tracks load.

Together these changes move the system from a synchronous, single-host,
rate-limit-bound script to an asynchronous, horizontally-scalable, fault-tolerant
data pipeline capable of producing tens of thousands of verified, de-duplicated
QA pairs across a large corpus of filings.
