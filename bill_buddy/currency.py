"""Currency conversion with live exchange rates.

Rates come from ExchangeRate-API's free open endpoint (no API key, 160+
currencies, updated daily): https://www.exchangerate-api.com
They are cached in memory for an hour to avoid repeated requests.
"""

from __future__ import annotations

import re
import time
from decimal import Decimal
from typing import Dict, Tuple

import requests

from .money import MoneyError, round_cents, to_decimal

API_URL = "https://open.er-api.com/v6/latest/{base}"
SOURCE = "Rates by ExchangeRate-API (https://www.exchangerate-api.com)"
CACHE_SECONDS = 3600
_cache: Dict[str, Tuple[float, dict]] = {}


def _currency_code(value: str, name: str) -> str:
    code = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", code):
        raise MoneyError(
            f"{name} must be a 3-letter currency code like USD, EUR or MXN, got {value!r}."
        )
    return code


def get_rates(base: str) -> dict:
    """Latest rates for 1 unit of `base`, from the cache when fresh."""
    cached = _cache.get(base)
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        return cached[1]
    try:
        response = requests.get(API_URL.format(base=base), timeout=10)
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise MoneyError(f"Couldn't get exchange rates right now ({exc}). Try again later.") from exc
    if data.get("result") != "success":
        if data.get("error-type") == "unsupported-code":
            raise MoneyError(f"Unknown currency code '{base}'.")
        raise MoneyError(f"The exchange-rate service returned an error: {data.get('error-type', 'unknown')}.")
    _cache[base] = (time.time(), data)
    return data


def convert(amount, from_currency: str, to_currency: str) -> dict:
    value = to_decimal(amount)
    source = _currency_code(from_currency, "from_currency")
    target = _currency_code(to_currency, "to_currency")
    if source == target:
        return {"amount": float(round_cents(value)), "from": source, "to": target, "rate": 1.0,
                "converted": float(round_cents(value))}

    data = get_rates(source)
    if target not in data["rates"]:
        raise MoneyError(f"Unknown currency code '{target}'.")
    rate = Decimal(str(data["rates"][target]))
    return {
        "amount": float(round_cents(value)),
        "from": source,
        "to": target,
        "rate": float(rate),
        "converted": float(round_cents(value * rate)),
        "rates_updated": data.get("time_last_update_utc", "unknown"),
        "source": SOURCE,
    }
