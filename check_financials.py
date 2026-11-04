"""Quick data-quality report for financials.db."""
import sqlite3

conn = sqlite3.connect("financials.db")
total = conn.execute("SELECT COUNT(*) FROM financials").fetchone()[0]
print(f"{total} companies\n\nFound per field:")
for col in ["fiscal_year", "currency", "revenue", "net_income", "total_assets", "employees"]:
    found = conn.execute(f"SELECT COUNT({col}) FROM financials").fetchone()[0]
    print(f"  {col:13} {found:3}/{total}  ({100 * found / total:.0f}%)")

print("\nCurrencies:", conn.execute(
    "SELECT currency, COUNT(*) FROM financials GROUP BY currency ORDER BY 2 DESC").fetchall())

print("\nCompanies without net income:")
for (name,) in conn.execute("SELECT company FROM financials WHERE net_income IS NULL ORDER BY company"):
    print("  -", name)
