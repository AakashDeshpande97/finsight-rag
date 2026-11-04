"""v3 step 1: extract key financial figures from every annual report into a SQL database.

For each company:
  1. retrieve the most relevant chunks for each figure (hybrid search, filtered by company)
  2. one Gemini call with structured output (Pydantic) fills in the figures + source pages
  3. values are converted to full currency units in code and stored in SQLite

The script is resumable: companies already in the database are skipped, so if it stops
(rate limit, crash, Ctrl+C), just run it again and it continues where it left off.

Usage:
    python extract_financials.py 5      # first 5 companies (test)
    python extract_financials.py        # all 100
"""
import json
import sqlite3
import sys
import time
from datetime import datetime
from typing import Literal

from google.genai import types
from pydantic import BaseModel, Field

from tracing import langfuse, observe
from hybrid_search import hybrid_search
from ingest import find
from llm import generate

DB_PATH = "financials.db"
SLEEP_SECONDS = 13        # stay under the free tier's requests-per-minute limit
N_COMPANIES = int(sys.argv[1]) if len(sys.argv) > 1 else 100

SEARCHES = {              # one targeted search per figure
    "revenue": "total revenue sales for the year",
    "net_income": "net income profit after tax for the year attributable to members consolidated",
    "total_assets": "total assets consolidated balance sheet",
    "employees": "number of employees headcount staff",
}
MULTIPLIER = {"units": 1, "thousands": 1_000, "millions": 1_000_000, "billions": 1_000_000_000}


# ---------- The "form" Gemini must fill in ----------
class Figure(BaseModel):
    value: float | None = Field(description="The number exactly as written in the report, "
                                            "negative for losses, null if not found")
    unit: Literal["units", "thousands", "millions", "billions"] | None = Field(
        description="Scale stated in the report, e.g. 'in thousands' -> thousands")
    page: int | None = Field(description="PDF page of the source this figure comes from")


class KeyFigures(BaseModel):
    fiscal_year: int | None = Field(description="The main fiscal year the report covers")
    currency: str | None = Field(description="ISO currency code, e.g. USD, AUD, EUR")
    revenue: Figure
    net_income: Figure
    total_assets: Figure
    employees: int | None = Field(description="Total number of employees, null if not stated")
    employees_page: int | None


PROMPT = """Below are excerpts from the annual report of {company}.
Extract the key figures for the MAIN fiscal year of the report (the most recent year, not
prior-year comparisons).
Rules:
- Use ONLY the excerpts. If a figure is not in them, set it to null. Never guess.
- Copy numbers as written and give the unit stated near the table (e.g. "in thousands").
- - Net income is the company's total net income (loss) for the year, not the amount allocable
  to common shares. In Australian and UK reports it is often called "profit after tax",
  "net profit" or "profit for the year attributable to members/owners".
- For each figure, give the PDF page from the source header, e.g. "(PDF page 95)".

EXCERPTS:
{excerpts}"""

CONFIG = types.GenerateContentConfig(
    temperature=0,
    response_mime_type="application/json",
    response_schema=KeyFigures,
)


# ---------- Database ----------
def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS financials (
            company           TEXT PRIMARY KEY,
            industry          TEXT,
            fiscal_year       INTEGER,
            currency          TEXT,
            revenue           REAL,
            revenue_page      INTEGER,
            net_income        REAL,
            net_income_page   INTEGER,
            total_assets      REAL,
            total_assets_page INTEGER,
            employees         INTEGER,
            employees_page    INTEGER,
            extracted_with    TEXT,
            extracted_at      TEXT
        )""")


def to_full_units(fig):
    """1,234 'in thousands' -> 1,234,000. Done in code, not by the LLM."""
    if fig.value is None:
        return None
    return fig.value * MULTIPLIER.get(fig.unit or "units", 1)


# ---------- Extraction ----------
@observe(name="extract_financials")
def extract(company):
    seen, excerpts = set(), []
    for query in SEARCHES.values():
        for doc, meta, _ in hybrid_search(query, company, k=4):
            if doc not in seen:
                seen.add(doc)
                excerpts.append(f"(PDF page {meta['page']})\n{doc}")
    prompt = PROMPT.format(company=company, excerpts="\n\n---\n\n".join(excerpts))
    response, model = generate(prompt, CONFIG)
    langfuse.update_current_span(metadata={"company": company, "n_excerpts": len(excerpts)})
    return response.parsed, model


def main():
    companies = json.loads(find("subset.json").read_text(encoding="utf-8"))[:N_COMPANIES]
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    done = {row[0] for row in conn.execute("SELECT company FROM financials")}
    todo = [c for c in companies if c["company_name"] not in done]
    print(f"{len(done)} companies already in {DB_PATH}, {len(todo)} to extract\n")

    for i, c in enumerate(todo, 1):
        name = c["company_name"]
        try:
            figs, model = extract(name)
        except Exception as e:                     # keep going; this company is retried next run
            print(f"{i:3}. {name[:40]:40} FAILED: {e}")
            continue

        conn.execute(
            "INSERT OR REPLACE INTO financials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (name, c["major_industry"], figs.fiscal_year, figs.currency,
             to_full_units(figs.revenue), figs.revenue.page,
             to_full_units(figs.net_income), figs.net_income.page,
             to_full_units(figs.total_assets), figs.total_assets.page,
             figs.employees, figs.employees_page,
             model, datetime.now().isoformat(timespec="seconds")))
        conn.commit()                              # save after every company (checkpointing)

        ni = to_full_units(figs.net_income)
        print(f"{i:3}. {name[:40]:40} FY{figs.fiscal_year} {figs.currency or '?':3} "
              f"net income: {ni:,.0f} (p.{figs.net_income.page})" if ni is not None
              else f"{i:3}. {name[:40]:40} FY{figs.fiscal_year} net income: not found")
        time.sleep(SLEEP_SECONDS)

    total = conn.execute("SELECT COUNT(*) FROM financials").fetchone()[0]
    missing = conn.execute("SELECT COUNT(*) FROM financials WHERE net_income IS NULL").fetchone()[0]
    print(f"\n{total} companies in {DB_PATH} ({missing} without net income)")
    conn.close()
    langfuse.flush()


if __name__ == "__main__":
    main()
