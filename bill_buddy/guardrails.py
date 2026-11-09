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
from decimal import Decimal, InvalidOperation
from typing import Callable, List, Optional, Set

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


NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def numbers_in(text: str) -> Set[Decimal]:
    """Every number the user wrote, read both ways: "1,200" -> 1200 and "12,50" -> 12.5."""
    found: Set[Decimal] = set()
    for raw in NUMBER.findall(text):
        for candidate in (raw.replace(",", ""), raw.replace(".", "").replace(",", "."), raw):
            try:
                found.add(Decimal(candidate))
            except InvalidOperation:
                pass
    return found


def numbers_from_user(tool_names=("propose_expense",), fields=("amount", "tip_percent")) -> Guardrail:
    """Block tool calls whose amounts the user never said.

    In testing, "Paid lunch" made the model invent "25 EUR + 10% tip", and
    "Dinner was 80 dollars" got an unrequested 15% tip. Money must come from
    the person, not from the model's imagination.
    """
    def check(tool_name: str, args: dict, user_text: str) -> Optional[str]:
        if tool_name not in tool_names:
            return None
        said = numbers_in(user_text)
        for field in fields:
            value = args.get(field)
            if value in (None, "", 0):
                continue
            try:
                number = Decimal(str(value))
            except InvalidOperation:
                continue  # the tool's own validation will explain the problem
            if number not in said:
                return (
                    f"The user never said the {field} {value}. Don't guess amounts: "
                    f"ask the user for the {field.replace('_', ' ')}"
                    + (" or leave it at 0." if field != "amount" else ".")
                )
        return None

    return check


DEFAULT_GUARDRAILS: List[Guardrail] = [ambiguous_pesos]
