"""Guardrails: rules checked in code before a tool runs.

Some rules are too important to leave to the prompt. With "pesos" alone, a
small local model (qwen3:8b) assumed Mexican pesos even when the prompt told
it to ask. A guardrail blocks that tool call and tells the model what to do
instead, and it works with any model.

A guardrail gets the tool name, its arguments and everything the user has
said so far. It returns None to allow the call, or a message for the LLM.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

Guardrail = Callable[[str, dict, str], Optional[str]]

PESO_CURRENCIES = {"MXN", "COP", "ARS", "CLP", "UYU", "DOP", "PHP", "CUP"}
PESO_COUNTRY_HINTS = re.compile(
    r"mex|colomb|argentin|chile|chilen|urugu|dominican|dominicano|philippin|filipin|cuba"
    r"|\b(mxn|cop|ars|clp|uyu|dop|php|cup)\b",
    re.IGNORECASE,
)


def ambiguous_pesos(tool_name: str, args: dict, user_text: str) -> Optional[str]:
    """Block conversions to/from a peso currency the user never actually specified."""
    if tool_name != "convert_currency":
        return None
    codes = {str(args.get("from_currency", "")).upper(), str(args.get("to_currency", "")).upper()}
    if not codes & PESO_CURRENCIES:
        return None
    if re.search(r"\bpesos?\b", user_text, re.IGNORECASE) and not PESO_COUNTRY_HINTS.search(user_text):
        return (
            "The user said 'pesos' without saying which country. Don't convert yet: "
            "share the other results and ask which pesos they mean "
            "(Mexican MXN, Colombian COP, Argentine ARS, Chilean CLP...)."
        )
    return None


DEFAULT_GUARDRAILS: List[Guardrail] = [ambiguous_pesos]
