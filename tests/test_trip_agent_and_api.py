"""The trip agent and the HTTP API, with a scripted fake LLM (no API key or Ollama)."""

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from bill_buddy import currency
from bill_buddy.api import create_app
from bill_buddy.guardrails import numbers_from_user, numbers_in


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def calls(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


@pytest.fixture
def script():
    """Replies the fake LLM will give, in order. Tests append to it."""
    return []


@pytest.fixture
def client(tmp_path, monkeypatch, script):
    monkeypatch.setattr(currency, "convert", lambda amount, f, t: {"rate": 0.9})
    app = create_app(str(tmp_path / "api.db"), llm_factory=lambda: ScriptedModel(messages=iter(script)))
    return TestClient(app)


@pytest.fixture
def trip(client):
    response = client.post("/trips", json={"name": "Lisboa 2026", "currency": "EUR",
                                           "members": ["Laura", "Ana", "Luis", "Marta"]})
    assert response.status_code == 201
    return response.json()


# ---------------------------------------------------------------- plain API

def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_manual_expenses_and_balances(client, trip):
    r = client.post(f"/trips/{trip['id']}/expenses",
                    json={"description": "Hotel", "amount": 300, "currency": "USD", "paid_by": "Luis"})
    assert r.status_code == 201 and r.json()["status"] == "confirmed"
    assert r.json()["trip_amount"] == 270.0

    full = client.get(f"/trips/{trip['id']}").json()
    assert len(full["expenses"]) == 1 and full["drafts"] == []
    assert full["balances"]["net"]["Luis"] == 202.5


def test_errors_have_clear_status_codes(client, trip):
    assert client.get("/trips/999").status_code == 404
    assert client.post("/expenses/999/confirm").status_code == 404
    bad = client.post(f"/trips/{trip['id']}/expenses",
                      json={"description": "x", "amount": 5, "currency": "EUR", "paid_by": "Pedro"})
    assert bad.status_code == 400 and "isn't a member" in bad.json()["detail"]
    assert client.post("/trips", json={"name": "x", "currency": "MXN", "members": ["A", "B"]}).status_code == 422


# ---------------------------------------------------------------- chat -> draft -> confirm

def test_chat_proposes_a_draft_that_only_counts_after_confirmation(client, trip, script):
    script += [
        calls("propose_expense", {"description": "Taxi", "amount": 45, "currency": "USD",
                                  "paid_by": "I", "split_between": ["Ana", "me"]}),
        AIMessage(content="Here's the taxi card, please confirm it."),
    ]
    r = client.post(f"/trips/{trip['id']}/chat",
                    json={"user": "laura", "message": "I paid the taxi, 45 dollars, for Ana and me"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "Here's the taxi card, please confirm it."
    [draft] = body["drafts"]
    assert (draft["paid_by"], draft["amount"], draft["trip_amount"], draft["status"]) == ("Laura", 45.0, 40.5, "draft")
    assert draft["shares"] == {"Laura": 20.25, "Ana": 20.25}
    assert body["steps"][0]["tool"] == "propose_expense"

    assert client.get(f"/trips/{trip['id']}/balances").json()["transfers"] == ["Everyone is even."]
    assert client.post(f"/expenses/{draft['id']}/confirm").json()["status"] == "confirmed"
    assert client.get(f"/trips/{trip['id']}/balances").json()["transfers"] == ["Ana pays Laura 20.25 EUR"]


def test_guardrail_stops_invented_amounts(client, trip, script):
    script += [
        calls("propose_expense", {"description": "Lunch", "amount": 25, "currency": "EUR", "paid_by": "Ana"}),
        AIMessage(content="How much was lunch?"),
    ]
    body = client.post(f"/trips/{trip['id']}/chat", json={"user": "Ana", "message": "Paid lunch"}).json()
    assert body["reply"] == "How much was lunch?"
    assert body["drafts"] == []
    assert "never said the amount 25" in body["steps"][0]["result"]["error"]


def test_guardrail_stops_invented_tips(client, trip, script):
    script += [
        calls("propose_expense", {"description": "Dinner", "amount": 80, "currency": "USD",
                                  "paid_by": "Ana", "tip_percent": 15}),
        AIMessage(content="Any tip?"),
    ]
    body = client.post(f"/trips/{trip['id']}/chat",
                       json={"user": "Ana", "message": "Dinner was 80 dollars"}).json()
    assert "never said the tip_percent 15" in body["steps"][0]["result"]["error"]


def test_chat_tip_is_added_to_the_amount(client, trip, script):
    script += [
        calls("propose_expense", {"description": "Cena", "amount": 120, "currency": "EUR",
                                  "paid_by": "yo", "tip_percent": 15}),
        AIMessage(content="Listo, confírmalo."),
    ]
    body = client.post(f"/trips/{trip['id']}/chat",
                       json={"user": "Laura", "message": "Pagué la cena, 120 euros con 15% de propina"}).json()
    assert body["drafts"][0]["amount"] == 138.0


def test_chat_balances_tool(client, trip, script):
    client.post(f"/trips/{trip['id']}/expenses",
                json={"description": "Cena", "amount": 120, "currency": "EUR", "paid_by": "Laura"})
    script += [calls("trip_balances", {}), AIMessage(content="Each owes Laura 30 EUR.")]
    body = client.post(f"/trips/{trip['id']}/chat", json={"user": "Ana", "message": "who owes whom?"}).json()
    assert body["steps"][0]["result"]["transfers"][0] == "Marta pays Laura 30.00 EUR"


def test_chat_rejects_unknown_users(client, trip):
    r = client.post(f"/trips/{trip['id']}/chat", json={"user": "Pedro", "message": "hi"})
    assert r.status_code == 400


def test_chat_returns_503_when_the_model_is_down(tmp_path):
    class Down(ScriptedModel):
        def _generate(self, *args, **kwargs):
            raise ConnectionError("Ollama is not running")

    client = TestClient(create_app(str(tmp_path / "x.db"), llm_factory=lambda: Down(messages=iter([]))))
    trip_id = client.post("/trips", json={"name": "x", "members": ["A", "B"]}).json()["id"]
    r = client.post(f"/trips/{trip_id}/chat", json={"user": "A", "message": "hi"})
    assert r.status_code == 503


# ---------------------------------------------------------------- amounts guardrail details

@pytest.mark.parametrize(
    "text, expected",
    [("45 dollars", {45}), ("12,50 €", {12.5}), ("1,200 USD", {1200}), ("15% tip on 80.5", {15, 80.5})],
)
def test_numbers_in_reads_common_formats(text, expected):
    found = {float(n) for n in numbers_in(text)}
    assert set(map(float, expected)) <= found


def test_numbers_from_user_only_checks_its_tools():
    check = numbers_from_user()
    assert check("trip_balances", {}, "") is None
    assert check("propose_expense", {"amount": 45.0, "tip_percent": 0}, "paid 45") is None
    assert check("propose_expense", {"amount": 46}, "paid 45") is not None
