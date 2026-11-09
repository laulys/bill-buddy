"""Currency conversion (with a fake HTTP API) and the 'pesos' guardrail."""

import pytest
import requests

from bill_buddy import currency
from bill_buddy.guardrails import ambiguous_pesos
from bill_buddy.money import MoneyError

FAKE_USD = {
    "result": "success",
    "time_last_update_utc": "Mon, 09 Nov 2026 00:02:31 +0000",
    "rates": {"USD": 1, "EUR": 0.9, "MXN": 18.5, "COP": 3250.123},
}


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


@pytest.fixture
def fake_api(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        base = url.rsplit("/", 1)[1]
        if base == "USD":
            return FakeResponse(FAKE_USD)
        return FakeResponse({"result": "error", "error-type": "unsupported-code"})

    currency._cache.clear()
    monkeypatch.setattr(currency.requests, "get", fake_get)
    yield calls
    currency._cache.clear()


def test_convert_uses_the_rate_and_rounds_to_cents(fake_api):
    result = currency.convert(46, "usd", " cop ")
    assert (result["from"], result["to"], result["rate"]) == ("USD", "COP", 3250.123)
    assert result["converted"] == 149505.66  # 46 * 3250.123 = 149505.658
    assert "ExchangeRate-API" in result["source"]


def test_rates_are_cached(fake_api):
    currency.convert(1, "USD", "EUR")
    currency.convert(2, "USD", "MXN")
    assert len(fake_api) == 1


def test_same_currency_needs_no_request(fake_api):
    assert currency.convert(10, "EUR", "EUR")["converted"] == 10.0
    assert fake_api == []


@pytest.mark.parametrize(
    "args, message",
    [
        ((1, "USD", "XYZ"), "Unknown currency code 'XYZ'"),
        ((1, "XYZ", "USD"), "Unknown currency code 'XYZ'"),
        ((1, "pesos", "USD"), "3-letter currency code"),
        ((-1, "USD", "EUR"), "positive"),
    ],
)
def test_conversion_errors(fake_api, args, message):
    with pytest.raises(MoneyError, match=message):
        currency.convert(*args)


def test_network_problems_become_a_friendly_error(monkeypatch):
    def broken(url, timeout):
        raise requests.ConnectionError("no internet")

    currency._cache.clear()
    monkeypatch.setattr(currency.requests, "get", broken)
    with pytest.raises(MoneyError, match="Couldn't get exchange rates"):
        currency.convert(1, "USD", "EUR")


@pytest.mark.parametrize(
    "user_text, to, blocked",
    [
        ("¿Cuánto es cada parte en pesos?", "MXN", True),
        ("Convert 100 dollars to pesos", "COP", True),
        ("Convert to Mexican pesos", "MXN", False),
        ("en pesos\nPesos colombianos", "COP", False),  # clarified in a follow-up
        ("to pesos chilenos", "CLP", False),
        ("Convert 100 USD to MXN pesos", "MXN", False),
        ("Convert 50 euros to dollars", "USD", False),   # not a peso currency
        ("Convert 50 EUR to MXN", "MXN", False),         # code given, no word "pesos"
    ],
)
def test_ambiguous_pesos_guardrail(user_text, to, blocked):
    args = {"amount": 46, "from_currency": "USD", "to_currency": to}
    assert (ambiguous_pesos("convert_currency", args, user_text) is not None) == blocked


def test_guardrail_ignores_other_tools():
    assert ambiguous_pesos("split_evenly", {"total": 10, "people": 2}, "en pesos") is None
