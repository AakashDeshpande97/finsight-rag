import json
from pathlib import Path
from collections import Counter

DATA = Path("data")
files = [p for p in DATA.rglob("*") if p.is_file()]
print(f"Total files: {len(files)}")
print("File types:", Counter(p.suffix for p in files))

md_files = sorted(p for p in files if p.suffix == ".md")
if md_files:
    sizes = [p.stat().st_size for p in md_files]
    print(f"\nMarkdown files: {len(md_files)} | total {sum(sizes)/1e6:.1f} MB | "
          f"smallest {min(sizes)/1e3:.0f} KB | largest {max(sizes)/1e3:.0f} KB")
    sample = min(md_files, key=lambda p: p.stat().st_size)
    print(f"\n--- First 1500 characters of {sample.name} ---")
    print(sample.read_text(encoding="utf-8", errors="ignore")[:1500])

for jf in (p for p in files if p.suffix == ".json"):
    data = json.loads(jf.read_text(encoding="utf-8"))
    print(f"\n--- {jf.name}: {type(data).__name__} with {len(data)} items ---")
    items = data[:2] if isinstance(data, list) else list(data.items())[:2]
    print(json.dumps(items, indent=2, ensure_ascii=False)[:800])