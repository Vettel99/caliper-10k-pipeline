"""Download the most recent 10-K for a ticker, clean the HTML, and chunk the
narrative text into ~1000-word segments saved as JSON in data/input/.

Pipeline:
    1. Download the latest 10-K filing with sec-edgar-downloader.
    2. Locate the primary HTML document of the filing.
    3. Clean the HTML with BeautifulSoup, dropping boilerplate tables and the
       table of contents.
    4. Chunk the remaining narrative text into ~1000-word segments.
    5. Save the chunks to data/input/<TICKER>_10K_chunks.json.

Run directly:  python src/parser.py
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

load_dotenv()

# Modern 10-Ks are inline-XBRL (XML-ish); parsing them as HTML works fine, so
# silence BeautifulSoup's advisory warning about it.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# --- Configuration -----------------------------------------------------------

TICKER = "MSFT"
TARGET_WORDS_PER_CHUNK = 1000

# Project paths (parser.py lives in src/, so the project root is its parent).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
RAW_DIR = INPUT_DIR / "raw"  # where sec-edgar-downloader writes filings

# SEC requires a descriptive User-Agent: a company/name and an email address.
# Configure via .env (SEC_EDGAR_USER_AGENT="Your Name your@email.com") or fall
# back to placeholders that still satisfy the SEC's format requirement.
_user_agent = os.getenv("SEC_EDGAR_USER_AGENT", "Caliper Pipeline contact@example.com")
_parts = _user_agent.split()
COMPANY_NAME = " ".join(_parts[:-1]) if len(_parts) > 1 else "Caliper Pipeline"
EMAIL_ADDRESS = _parts[-1] if _parts and "@" in _parts[-1] else "contact@example.com"


# --- Step 1: download --------------------------------------------------------

def download_latest_10k(ticker: str, dest: Path = RAW_DIR) -> Path:
    """Download the most recent 10-K and return the path to its HTML document."""
    dest.mkdir(parents=True, exist_ok=True)
    dl = Downloader(COMPANY_NAME, EMAIL_ADDRESS, str(dest))
    # download_details=True fetches the human-readable primary HTML document
    # alongside the raw full-submission text file.
    num = dl.get("10-K", ticker, limit=1, download_details=True)
    if num == 0:
        raise RuntimeError(f"No 10-K filing found for ticker {ticker!r}.")

    filing_root = dest / "sec-edgar-filings" / ticker / "10-K"
    return _find_primary_document(filing_root)


def _find_primary_document(filing_root: Path) -> Path:
    """Find the readable HTML document for the (single) downloaded filing."""
    # Prefer the rendered details file; fall back to any .htm/.html, then the
    # raw full submission text.
    candidates = (
        sorted(filing_root.glob("*/primary-document.html"))
        or sorted(filing_root.glob("*/*.htm"))
        or sorted(filing_root.glob("*/*.html"))
        or sorted(filing_root.glob("*/full-submission.txt"))
    )
    if not candidates:
        raise FileNotFoundError(f"No filing document found under {filing_root}.")
    return candidates[0]


# --- Step 2/3: clean HTML and extract text -----------------------------------

def _table_is_boilerplate(table) -> bool:
    """Return True for tables that are financial/numeric grids or a TOC.

    Narrative-bearing tables (mostly words) are kept; tables dominated by
    digits, currency/percent symbols, or page-number leaders are dropped.
    """
    text = table.get_text(" ", strip=True)
    if not text:
        return True

    # Table-of-contents tables typically reference "Item N" and page numbers.
    if re.search(r"\bitem\s+\d+", text, re.IGNORECASE) and re.search(r"\bpage\b", text, re.IGNORECASE):
        return True

    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    total = letters + digits
    if total == 0:
        return True
    # If at least 35% of alphanumerics are digits, treat it as a numeric grid.
    return digits / total >= 0.35


def extract_text(html_path: Path) -> str:
    """Clean the filing HTML and return narrative text, one paragraph per line."""
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")

    # Drop non-content elements outright.
    for tag in soup(["script", "style", "head", "title", "meta", "link"]):
        tag.decompose()

    # Drop boilerplate / numeric tables; keep narrative ones.
    for table in soup.find_all("table"):
        if _table_is_boilerplate(table):
            table.decompose()

    # Separator preserves block boundaries so we can treat lines as paragraphs.
    text = soup.get_text(separator="\n")

    # Normalize non-breaking spaces and collapse intra-line whitespace.
    lines = []
    for line in text.replace("\xa0", " ").split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


# --- Step 4: chunk -----------------------------------------------------------

_TOC_PATTERNS = [
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^part\s+[ivx]+$", re.IGNORECASE),
    re.compile(r"\.{4,}\s*\d+$"),            # dotted leader to a page number
    re.compile(r"^item\s+\d+[a-z]?\.?\s*$", re.IGNORECASE),  # bare TOC item entry
]


def _is_boilerplate_paragraph(para: str) -> bool:
    """Filter TOC entries, page numbers, and numeric/symbol-heavy fragments."""
    if any(p.search(para) for p in _TOC_PATTERNS):
        return True

    # Bare page numbers or short numeric fragments.
    if re.fullmatch(r"[\d\s.,$%()\-]+", para):
        return True

    words = para.split()
    if len(words) < 4:  # stray fragments, headers, page markers
        return True

    # Symbol/number-dominated fragments (leftover table cells).
    letters = sum(c.isalpha() for c in para)
    if letters / max(len(para), 1) < 0.5:
        return True

    return False


def chunk_text(text: str, target_words: int = TARGET_WORDS_PER_CHUNK) -> list[dict]:
    """Chunk cleaned text into ~target_words segments at paragraph boundaries."""
    paragraphs = [
        p for p in text.split("\n") if not _is_boilerplate_paragraph(p)
    ]

    chunks: list[dict] = []
    buffer: list[str] = []
    word_count = 0

    def flush() -> None:
        nonlocal buffer, word_count
        if buffer:
            body = "\n\n".join(buffer)
            chunks.append({
                "chunk_id": len(chunks),
                "word_count": word_count,
                "text": body,
            })
            buffer = []
            word_count = 0

    for para in paragraphs:
        buffer.append(para)
        word_count += len(para.split())
        if word_count >= target_words:
            flush()
    flush()  # remaining tail

    return chunks


# --- Step 5: orchestrate -----------------------------------------------------

def main(ticker: str = TICKER) -> Path:
    print(f"Downloading most recent 10-K for {ticker} ...")
    html_path = download_latest_10k(ticker)
    print(f"  -> filing document: {html_path.relative_to(PROJECT_ROOT)}")

    print("Cleaning HTML and extracting text ...")
    text = extract_text(html_path)
    print(f"  -> {len(text.split()):,} words after cleaning")

    print(f"Chunking into ~{TARGET_WORDS_PER_CHUNK}-word segments ...")
    chunks = chunk_text(text)
    print(f"  -> {len(chunks)} chunks")

    out_path = INPUT_DIR / f"{ticker}_10K_chunks.json"
    payload = {
        "ticker": ticker,
        "source": str(html_path.relative_to(PROJECT_ROOT)),
        "num_chunks": len(chunks),
        "target_words_per_chunk": TARGET_WORDS_PER_CHUNK,
        "chunks": chunks,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks to {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


if __name__ == "__main__":
    main()
