"""Langfuse tracing setup in one place.

Every module imports `observe` and `langfuse` from here, so .env is always loaded
before Langfuse reads its keys. Without keys, Langfuse logs a warning and tracing
is simply skipped: the pipeline still works.
"""
from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client, observe  # noqa: E402  (must come after load_dotenv)

langfuse = get_client()

__all__ = ["langfuse", "observe"]
