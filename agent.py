"""v3: a LangGraph agent that chooses between two tools for each question:
  - query_financials: SQL over the key figures of all 100 companies (structured data)
  - search_reports:   the RAG pipeline over the report text (unstructured data)

    python agent.py "Which 3 companies reporting in USD had the highest net income?"
"""
import json
import os
import sqlite3
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tracing import langfuse          # loads .env first
from ingest import find
from rag import answer_question
from text_utils import suggest_companies

DB_PATH = "financials.db"
COMPANIES = sorted(c["company_name"] for c in json.loads(find("subset.json").read_text(encoding="utf-8")))
BY_LOWER = {c.lower(): c for c in COMPANIES}
MAX_ROWS = 50


# ---------- Tool 1: SQL over structured figures ----------
@tool
def query_financials(sql: str) -> str:
    """Run a read-only SQLite SELECT query on the table `financials` (one row per company).
    Columns: company TEXT, industry TEXT, fiscal_year INTEGER, currency TEXT (USD, AUD, EUR...),
    revenue REAL, net_income REAL, total_assets REAL (all in FULL currency units, negative
    net_income = loss), employees INTEGER, and *_page INTEGER columns with the PDF page of
    each figure. Values can be NULL when a figure was not found.
    Use this for numbers, rankings, comparisons and counts across companies.
    Find exact company names with e.g. WHERE company LIKE '%acres%'."""
    if not sql.strip().lower().startswith("select"):
        return "Error: only SELECT queries are allowed."
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)   # read-only connection
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(MAX_ROWS)
    finally:
        conn.close()
    if not rows:
        return "No rows."
    return "\n".join([" | ".join(cols)] + [" | ".join(str(v) for v in r) for r in rows])


# ---------- Tool 2: RAG over the report text ----------
@tool
def search_reports(question: str, company: str) -> str:
    """Answer a question from the text of ONE company's annual report, with PDF page citations.
    Use this for explanations, risks, strategy, management, policies, or any figure that is
    not in the financials table. `company` must be the exact company name."""
    name = BY_LOWER.get(company.strip().lower())
    if name is None:
        return f"Unknown company '{company}'. Did you mean: {suggest_companies(company, COMPANIES)}?"
    out = answer_question(question, name)
    pages = sorted({meta["page"] for _, meta, _ in out["results"]})
    return f"{out['answer']}\n(Retrieved PDF pages: {pages})"


TOOLS = [query_financials, search_reports]

SYSTEM_PROMPT = """You are a financial analyst assistant for 100 corporate annual reports.
- For numbers, rankings or comparisons across companies, use query_financials (SQL).
- For explanations, risks, strategy or details, use search_reports.
- You may use several tools and combine their results.
- Amounts in the database are in full currency units. Mention the currency, and never compare
  or rank amounts in different currencies as if they were the same.
- Base every statement on tool results and mention the PDF page when you have one.
- If the tools do not provide the answer, say so. Do not guess."""


def make_llm(model):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        max_retries=3,                      # retry temporary errors (503, 429) a few times
        google_api_key=os.environ["GEMINI_API_KEY"],
    ).bind_tools(TOOLS)


# Same strategy as llm.py: lite model first, the other as fallback if it keeps failing
llm = make_llm("gemini-flash-lite-latest").with_fallbacks([make_llm("gemini-flash-latest")])


# ---------- The graph: agent <-> tools loop ----------
def call_model(state: MessagesState):
    return {"messages": [llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)   # tool call -> "tools", else -> END
builder.add_edge("tools", "agent")                         # tool results go back to the model
graph = builder.compile()


def _text(content):
    if isinstance(content, list):          # some Gemini versions return content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content


def ask_agent(question):
    """Run the agent and RETURN the answer plus every tool call (used by the API and evaluation)."""
    result = graph.invoke(
        {"messages": [HumanMessage(question)]},
        config={"recursion_limit": 12, "callbacks": [CallbackHandler()]},  # cap the loop + trace it
    )
    tool_calls, tool_results = [], []
    for msg in result["messages"][1:]:
        for call in getattr(msg, "tool_calls", None) or []:
            tool_calls.append({"tool": call["name"], "args": call["args"]})
        if msg.type == "tool":
            tool_results.append(_text(msg.content))
    return {"answer": _text(result["messages"][-1].content),
            "tool_calls": tool_calls, "tool_results": tool_results}


def run(question):
    """Run the agent and PRINT the steps (for manual testing)."""
    out = ask_agent(question)
    for call, res in zip(out["tool_calls"], out["tool_results"]):
        print(f"  🔧 {call['tool']}({call['args']})")
        print(f"  📦 {res[:300]}")
    print(f"\nAnswer:\n{out['answer']}")
    return out


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Which 3 companies reporting in USD had the highest net income?"
    print(f"Q: {question}\n")
    run(question)
    langfuse.flush()
