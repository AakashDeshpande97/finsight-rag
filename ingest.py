import json
import re
from pathlib import Path

DATA = Path("data")
PAGE_MARKER = re.compile(r"^\{(\d+)\}-{10,}\s*$", re.MULTILINE)
IMAGE_LINK = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MAX_CHARS = 1200      # target maximum chunk size
OVERLAP_CHARS = 200   # short trailing blocks (e.g. headings) carry into the next chunk


def find(filename):
    """Find a file anywhere inside data/ (handles subfolders)."""
    return next(DATA.rglob(filename))


def clean(text):
    text = IMAGE_LINK.sub("", text)          # remove image links
    text = text.replace("\\$", "$")          # \$40.4 -> $40.4
    text = re.sub(r" {2,}", " ", text)       # squeeze table padding
    text = re.sub(r"-{4,}", "---", text)     # shorten table separator lines
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse extra blank lines
    return text.strip()


def load_pages(md_path):
    raw = md_path.read_text(encoding="utf-8", errors="ignore")
    parts = PAGE_MARKER.split(raw)   # ['', '0', page0_text, '1', page1_text, ...]
    pages = []
    for i in range(1, len(parts), 2):
        text = clean(parts[i + 1])
        if text:
            pages.append({"page": int(parts[i]), "text": text})
    return pages


def split_big_block(block):
    """Split a block that is larger than MAX_CHARS."""
    if block.lstrip().startswith("|"):                 # TABLE: split by rows, repeat header
        lines = block.splitlines()
        header, rows = lines[:2], lines[2:]
        pieces, current = [], []
        for row in rows:
            if current and len("\n".join(header + current + [row])) > MAX_CHARS:
                pieces.append("\n".join(header + current))
                current = []
            current.append(row)
        if current:
            pieces.append("\n".join(header + current))
        return pieces

    sentences = re.split(r"(?<=[.!?])\s+", block)      # TEXT: split at sentence ends
    pieces, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > MAX_CHARS:
            pieces.append(current)
            current = ""
        current = f"{current} {s}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_page(text):
    """Pack a page's paragraphs and tables into chunks of up to MAX_CHARS."""
    blocks = []
    for b in text.split("\n\n"):
        b = b.strip()
        if b:
            blocks.extend(split_big_block(b) if len(b) > MAX_CHARS else [b])

    chunks, current = [], []
    for b in blocks:
        if current and len("\n\n".join(current + [b])) > MAX_CHARS:
            chunks.append("\n\n".join(current))
            last = current[-1]
            current = [last] if len(last) <= OVERLAP_CHARS else []   # overlap
        current.append(b)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def build_chunks(company):
    """Turn one company's report into a list of chunks with metadata."""
    pages = load_pages(find(f"{company['sha1']}.md"))
    chunks = []
    for p in pages:
        for i, text in enumerate(chunk_page(p["text"])):
            page_no = p["page"] + 1   # human-friendly page numbers start at 1
            header = f"Company: {company['company_name']} | Page: {page_no}"
            chunks.append({
                "id": f"{company['sha1']}_p{p['page']}_c{i}",
                "text": f"{header}\n\n{text}",
                "metadata": {
                    "company": company["company_name"],
                    "industry": company["major_industry"],
                    "sha1": company["sha1"],
                    "page": page_no,
                },
            })
    return chunks


if __name__ == "__main__":
    companies = json.loads(find("subset.json").read_text(encoding="utf-8"))
    all_chunks = []
    for c in companies[:5]:
        chunks = build_chunks(c)
        all_chunks += chunks
        sizes = [len(ch["text"]) for ch in chunks]
        print(f"{c['company_name'][:35]:35} | {len(chunks):5} chunks | "
              f"avg {sum(sizes) // len(sizes)} chars | max {max(sizes)}")
    print(f"\nTotal: {len(all_chunks)} chunks")

    table_chunks = [ch for ch in all_chunks if "\n|" in ch["text"]]
    text_chunks = [ch for ch in all_chunks if "\n|" not in ch["text"]]
    for label, ch in [("TEXT CHUNK", text_chunks[len(text_chunks) // 2]),
                      ("TABLE CHUNK", table_chunks[len(table_chunks) // 2])]:
        print(f"\n===== {label} | id={ch['id']} | {len(ch['text'])} chars =====")
        print(ch["text"])