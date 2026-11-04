"""Evaluate the agent: right tools, right answer. Scores go to Langfuse like evaluate.py.

    python evaluate_agent.py
"""
import json
import re
import time
from datetime import datetime
from pathlib import Path

from google.genai import types
from pydantic import BaseModel

from tracing import langfuse
from agent import ask_agent
from llm import generate

EVAL_FILE = "agent_eval.json"
SLEEP_SECONDS = 20          # agent runs use several LLM calls, so pause a bit longer
RUN_NAME = f"agent_eval {datetime.now():%Y-%m-%d %H:%M}"


class Verdict(BaseModel):
    correct: bool
    reason: str


JUDGE_PROMPT = """You are grading an AI assistant that answers questions about annual reports.

QUESTION: {question}
EXPECTED ANSWER: {expected}
ASSISTANT ANSWER: {actual}

Is the assistant answer correct? It is correct if it contains the same key information as the
expected answer; different wording and extra correct details are fine. If the expected answer
says the information is not available, it is correct only if the assistant also says so
without inventing an answer."""

JUDGE_CONFIG = types.GenerateContentConfig(
    temperature=0, response_mime_type="application/json", response_schema=Verdict)


def normalise(text):
    return re.sub(r"[\s,$]", "", text.lower())


eval_set = json.loads(Path(EVAL_FILE).read_text(encoding="utf-8"))
print(f"Evaluating {len(eval_set)} agent questions  (Langfuse run: {RUN_NAME})\n")
results = []

for i, item in enumerate(eval_set, 1):
    with langfuse.start_as_current_observation(as_type="span", name="eval_agent_item") as span:
        span.update(input=item["question"], metadata={"eval_run": RUN_NAME, "type": item["type"]})
        out = ask_agent(item["question"])
        used = [c["tool"] for c in out["tool_calls"]]

        # Tools: every expected tool must have been used (extra calls are fine)
        tools_ok = all(t in used for t in item["expected_tools"])

        # Answer: key fact first (free), otherwise LLM judge
        key_fact = item.get("key_fact", "")
        if key_fact and normalise(key_fact) in normalise(out["answer"]):
            answer_ok, method, reason = True, "key fact", ""
        else:
            prompt = JUDGE_PROMPT.format(question=item["question"],
                                         expected=item["expected_answer"], actual=out["answer"])
            verdict = generate(prompt, JUDGE_CONFIG)[0].parsed
            answer_ok, method, reason = verdict.correct, "judge", verdict.reason

        span.update(output=out["answer"])
        span.score_trace(name="answer_correct", value=int(answer_ok), data_type="BOOLEAN",
                         comment=f"{method}: {reason}" if reason else method)
        span.score_trace(name="tools_correct", value=int(tools_ok), data_type="BOOLEAN",
                         comment=f"used: {used}, expected: {item['expected_tools']}")

    print(f"{i}. tools {'OK  ' if tools_ok else 'MISS'} | answer {'OK  ' if answer_ok else 'MISS'} "
          f"({method:8}) | {item['type']:12} | {item['question'][:55]}")
    print(f"     tools used: {used}")
    if method == "judge":
        print(f"     judge: {reason}")
    if not answer_ok:
        print(f"     answer: {out['answer'][:300]}")

    results.append({**item, "agent_answer": out["answer"], "tools_used": used,
                    "tools_ok": tools_ok, "answer_ok": answer_ok, "judge_reason": reason})
    time.sleep(SLEEP_SECONDS)

n = len(results)
t = sum(r["tools_ok"] for r in results)
a = sum(r["answer_ok"] for r in results)
print(f"\nTool choice correct: {t}/{n} = {100 * t / n:.0f}%")
print(f"Answer correct:      {a}/{n} = {100 * a / n:.0f}%")

Path("agent_eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
print("Details saved to agent_eval_results.json")
langfuse.flush()
