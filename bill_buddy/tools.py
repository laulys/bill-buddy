"""LangChain tools the agent can call: thin `@tool` wrappers around money.py and currency.py.

Problems the user can fix (negative amount, unknown currency...) come back as
{"error": ...} so the LLM can read them, explain them, or retry.
"""

from __future__ import annotations

from typing import Callable, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from . import currency, money
from .money import MoneyError


def _safely(func: Callable, *args) -> dict:
    try:
        return func(*args)
    except MoneyError as exc:
        return {"error": str(exc)}


class Item(BaseModel):
    """Something one person ordered."""

    person: str = Field(description="Name of the person who ordered it")
    amount: float = Field(description="Price of the item")


@tool
def calculate_tip(bill_amount: float, tip_percent: float) -> dict:
    """Calculate the tip for a bill and the total including the tip.

    Example: $60 with a 20% tip -> bill_amount=60, tip_percent=20
    """
    return _safely(money.calculate_tip, bill_amount, tip_percent)


@tool
def split_evenly(total: float, people: int) -> dict:
    """Split a total equally between a number of people.

    Shares always add up exactly to the total (some may pay 1 cent more).
    Example: $100 between 3 friends -> total=100, people=3
    """
    return _safely(money.split_evenly, total, people)


@tool
def split_itemized(
    items: List[Item],
    shared_amount: float = 0,
    tip_percent: float = 0,
    tax_percent: float = 0,
) -> dict:
    """Split a bill where each person pays what they ordered.

    - items: what each person ordered (one entry per item; a person can appear several times)
    - shared_amount: total of things everyone shared (split equally)
    - tip_percent / tax_percent: added in proportion to what each person consumed
    Example: Ana had pasta 18, Luis steak 32, shared wine 24, 15% tip ->
    items=[{"person": "Ana", "amount": 18}, {"person": "Luis", "amount": 32}],
    shared_amount=24, tip_percent=15
    """
    pairs = [(item.person, item.amount) for item in items]
    return _safely(money.split_itemized, pairs, shared_amount, tip_percent, tax_percent)


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert money between currencies with today's exchange rate.

    Currencies are 3-letter ISO codes: USD, EUR, GBP, MXN, COP, ARS, CLP, PEN, BRL...
    Example: 50 euros to Mexican pesos -> amount=50, from_currency="EUR", to_currency="MXN"
    """
    return _safely(currency.convert, amount, from_currency, to_currency)


ALL_TOOLS = [calculate_tip, split_evenly, split_itemized, convert_currency]
