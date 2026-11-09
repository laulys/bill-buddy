"""Money math for the agent's tools (no LangChain here).

- Uses Decimal, never float, so 0.1 + 0.2 is exactly 0.3 and cents don't drift.
- Every split adds up to the exact total: leftover cents are handed out one
  by one (largest remainder method), so 100 / 3 -> 33.34 + 33.33 + 33.33.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Dict, Iterable, List, Sequence, Tuple

CENT = Decimal("0.01")
MAX_PEOPLE = 100


class MoneyError(ValueError):
    """A problem the user (or the LLM) can fix: negative amount, unknown currency..."""


def to_decimal(value, name: str = "amount") -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise MoneyError(f"{name} must be a number, got {value!r}.") from exc
    if not number.is_finite() or number < 0:
        raise MoneyError(f"{name} must be a positive number, got {value!r}.")
    return number


def round_cents(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def to_cents(value: Decimal) -> int:
    return int(round_cents(value) * 100)


def from_cents(cents: int) -> float:
    return float(Decimal(cents) / 100)


def check_percent(value, name: str) -> Decimal:
    percent = to_decimal(value, name)
    if percent > 100:
        raise MoneyError(f"{name} looks too high ({value}%). Use a number between 0 and 100.")
    return percent


def allocate(total_cents: int, weights: Sequence[Decimal]) -> List[int]:
    """Split total_cents proportionally to weights so the parts add up exactly."""
    weight_sum = sum(weights)
    if weight_sum == 0:
        weights, weight_sum = [Decimal(1)] * len(weights), Decimal(len(weights))
    exact = [Decimal(total_cents) * w / weight_sum for w in weights]
    parts = [int(x) for x in exact]  # round down first...
    leftover = total_cents - sum(parts)
    # ...then give the remaining cents to the largest fractional parts (ties: first listed).
    order = sorted(range(len(parts)), key=lambda i: exact[i] - parts[i], reverse=True)
    for i in order[:leftover]:
        parts[i] += 1
    return parts


def calculate_tip(bill_amount, tip_percent) -> Dict:
    bill = to_decimal(bill_amount, "bill_amount")
    percent = check_percent(tip_percent, "tip_percent")
    tip = round_cents(bill * percent / 100)
    return {
        "bill": float(round_cents(bill)),
        "tip_percent": float(percent),
        "tip": float(tip),
        "total_with_tip": float(round_cents(bill) + tip),
    }


def split_evenly(total, people) -> Dict:
    amount = to_decimal(total, "total")
    if int(people) != people or not 1 <= int(people) <= MAX_PEOPLE:
        raise MoneyError(f"people must be a whole number between 1 and {MAX_PEOPLE}, got {people!r}.")
    people = int(people)
    shares = allocate(to_cents(amount), [Decimal(1)] * people)
    result = {"total": float(round_cents(amount)), "people": people, "shares": [from_cents(s) for s in shares]}
    extra = shares.count(shares[0]) if len(set(shares)) > 1 else 0
    if extra:
        who = "1 person pays" if extra == 1 else f"{extra} people pay"
        result["note"] = f"{who} 1 cent more so the shares add up exactly to the total."
    return result


def split_itemized(
    items: Iterable[Tuple[str, float]],
    shared_amount=0,
    tip_percent=0,
    tax_percent=0,
) -> Dict:
    """Each person pays their own items, an equal part of the shared items, and
    tax and tip in proportion to what they consumed. Tip is on the pre-tax subtotal."""
    own: Dict[str, Decimal] = {}
    names: Dict[str, str] = {}  # "ana" -> "Ana": merge names case-insensitively
    for person, amount in items:
        key = str(person).strip().lower()
        if not key:
            raise MoneyError("Every item needs a person's name.")
        names.setdefault(key, str(person).strip())
        own[key] = own.get(key, Decimal(0)) + to_decimal(amount, f"amount for {person}")
    if not own:
        raise MoneyError("Give at least one item with a person and an amount.")

    people = list(own)
    shared_cents = allocate(to_cents(to_decimal(shared_amount, "shared_amount")), [Decimal(1)] * len(people))
    consumed = [to_cents(own[p]) + shared_cents[i] for i, p in enumerate(people)]
    subtotal = sum(consumed)

    tax_total = to_cents(Decimal(subtotal) / 100 * check_percent(tax_percent, "tax_percent") / 100)
    tip_total = to_cents(Decimal(subtotal) / 100 * check_percent(tip_percent, "tip_percent") / 100)
    weights = [Decimal(c) for c in consumed]
    taxes, tips = allocate(tax_total, weights), allocate(tip_total, weights)

    breakdown = {}
    for i, p in enumerate(people):
        breakdown[names[p]] = {
            "own_items": from_cents(to_cents(own[p])),
            "shared_part": from_cents(shared_cents[i]),
            "tax": from_cents(taxes[i]),
            "tip": from_cents(tips[i]),
            "pays": from_cents(consumed[i] + taxes[i] + tips[i]),
        }
    return {
        "per_person": breakdown,
        "subtotal": from_cents(subtotal),
        "tax": from_cents(tax_total),
        "tip": from_cents(tip_total),
        "grand_total": from_cents(subtotal + tax_total + tip_total),
    }
