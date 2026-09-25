"""Extract PDF text page by page and split it into fixed-size, page-tagged chunks."""

import json
import re
from pathlib import Path

import pymupdf

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/chunks.jsonl")

CHUNK_CHARS = 2000  # ~500 tokens
OVERLAP_CHARS = 300  # ~75 tokens

# Running header printed at the top of every page: "Prompt Engineering\nFebruary 2025\n12\n"
HEADER_RE = re.compile(r"^\s*Prompt Engineering\s*\n\s*February 2025\s*\n\s*\d+\s*\n")
# Marker at the bottom of tables/code that spill onto the next page
CONTINUES_RE = re.compile(r"^\s*Continues next page\.\.\.\s*$", re.MULTILINE)
# Stray control characters (e.g. \x08 in the table of contents)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

SUPERSCRIPT_FLAG = 1  # footnote markers like the "11" in "Self-consistency11"
PULL_QUOTE_SIZE = 28  # large callouts that repeat sentences from the body text


def _page_text(page: pymupdf.Page) -> str:
    """Rebuild page text from spans, dropping footnote superscripts and pull-quotes."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [
                s for s in line["spans"]
                if not s["flags"] & SUPERSCRIPT_FLAG and round(s["size"]) != PULL_QUOTE_SIZE
            ]
            if spans:
                lines.append("".join(s["text"] for s in spans).rstrip())
    return "\n".join(lines)


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text) for each non-empty page, 1-indexed."""
    pages = []
    with pymupdf.open(path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = HEADER_RE.sub("", _page_text(page))
            text = CONTROL_RE.sub("", CONTINUES_RE.sub("", text)).strip()
            if text:
                pages.append((page_num, text))
    return pages


def chunk_page(page_num: int, text: str) -> list[dict]:
    """Split one page into overlapping fixed-size chunks; chunks never cross pages."""
    step = CHUNK_CHARS - OVERLAP_CHARS
    chunks = []
    for i, start in enumerate(range(0, len(text), step)):
        chunks.append({
            "chunk_id": f"p{page_num:03d}-c{i:02d}",
            "page": page_num,
            "text": text[start:start + CHUNK_CHARS],
        })
        if start + CHUNK_CHARS >= len(text):
            break
    return chunks


def main() -> None:
    pages = [p for pdf in sorted(RAW_DIR.glob("*.pdf")) for p in extract_pages(pdf)]
    chunks = [c for page_num, text in pages for c in chunk_page(page_num, text)]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"{len(pages)} pages -> {len(chunks)} chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()
