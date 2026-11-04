"""Chat with the FinSight agent in the terminal. It remembers the conversation.

    python chat.py
Type 'quit' to exit, 'new' to start a new conversation.
"""
import uuid

from tracing import langfuse
from agent import ask_agent

thread_id = str(uuid.uuid4())
print("FinSight agent. Ask about 100 annual reports. Type 'new' to reset, 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if not question:
        continue
    if question.lower() in ("quit", "exit"):
        break
    if question.lower() == "new":
        thread_id = str(uuid.uuid4())
        print("(new conversation)\n")
        continue
    try:
        out = ask_agent(question, thread_id=thread_id)
    except Exception as e:
        print(f"Agent error: {e}\n")
        continue
    for call in out["tool_calls"]:
        print(f"  🔧 {call['tool']}({call['args']})")
    print(f"\nAgent: {out['answer']}\n")

langfuse.flush()
