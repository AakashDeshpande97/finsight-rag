import json
import re
from pathlib import Path

DATA = Path("data")
PAGE_MARKER = re.compile(r"^\{(\d+)\}-{10,}\s*$", re.MULTILINE)
IMAGE_LINK = re.compile(r"!\[[^\]]*\]\([^)]*\)")

def find(filename):
    """Find a file anywhere inside data/ (handles subfolders)."""
    return next(DATA.rglob(filename))

def clean(text):
    text = IMAGE_LINK.sub("", text)          # remove image links
    text = text.replace("\\$", "$")          # \$40.4 -> $40.4
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse extra blank lines
    return text.strip()

def load_pages(md_path):
    raw = md_path.read_text(encoding="utf-8", errors="ignore")
    parts = PAGE_MARKER.split(raw)   # ['', '0', page0_text, '1', page1_text, ...]
    pages = []
    for i in range(1, len(parts), 2):
        text = clean(parts[i + 1])
        if text:                         # skip empty pages (e.g. image-only)
            pages.append({"page": int(parts[i]), "text": text})
    return pages

companies = json.loads(find("subset.json").read_text(encoding="utf-8"))
print(f"{len(companies)} companies in subset.json\n")

for c in companies[:5]:
    pages = load_pages(find(f"{c['sha1']}.md"))
    chars = sum(len(p["text"]) for p in pages)
    print(f"{c['company_name'][:40]:40} | {c['major_industry'][:22]:22} | "
          f"{len(pages):4} pages | {chars / 1000:6.0f}k chars")

first = companies[0]
sample = load_pages(find(f"{first['sha1']}.md"))
page = sample[min(5, len(sample) - 1)]
print(f"\n--- {first['company_name']}, page {page['page']} (cleaned) ---")
print(page["text"][:1500])