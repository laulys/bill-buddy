"""Trip ledger, balances and the settle-up algorithm. No LLM needed."""

import random

import pytest

from bill_buddy import currency, ledger
from bill_buddy.settle import net_balances, settle_up

MEMBERS = ["Laura", "Ana", "Luis", "Marta"]


@pytest.fixture
def db(tmp_path, monkeypatch):
    # Fixed exchange rate so tests don't need the internet.
    monkeypatch.setattr(currency, "convert", lambda amount, f, t: {"rate": 0.9 if f == "USD" else 1 / 0.9})
    path = str(tmp_path / "trips.db")
    with ledger.session(path) as conn:
        yield conn


@pytest.fixture
def trip(db):
    return ledger.create_trip(db, "Lisboa 2026", "EUR", MEMBERS)


# ---------------------------------------------------------------- settle-up

def test_settle_up_simple():
    assert settle_up({"Laura": 9000, "Ana": -3000, "Luis": -3000, "Marta": -3000}) == [
        ("Marta", "Laura", 3000), ("Luis", "Laura", 3000), ("Ana", "Laura", 3000)
    ]


def test_settle_up_everyone_even():
    assert settle_up({"Laura": 0, "Ana": 0}) == []


def test_settle_up_rejects_balances_that_dont_add_up():
    with pytest.raises(ValueError):
        settle_up({"Laura": 100, "Ana": -99})


@pytest.mark.parametrize("seed", range(25))
def test_settle_up_always_clears_everything_in_at_most_n_minus_1_transfers(seed):
    rng = random.Random(seed)
    people = [f"P{i}" for i in range(rng.randint(2, 9))]
    net = {p: rng.randint(-50_000, 50_000) for p in people[:-1]}
    net[people[-1]] = -sum(net.values())

    transfers = settle_up(net)

    remaining = dict(net)
    for frm, to, cents in transfers:
        assert cents > 0 and frm != to
        remaining[frm] += cents
        remaining[to] -= cents
    assert all(v == 0 for v in remaining.values())
    assert len(transfers) <= len(people) - 1


def test_net_balances():
    expenses = [("Laura", 1200, {"Laura": 600, "Ana": 600}), ("Ana", 300, {"Laura": 300})]
    assert net_balances(["Laura", "Ana"], expenses) == {"Laura": 300, "Ana": -300}


# ---------------------------------------------------------------- trips & expenses

def test_create_trip(trip):
    assert trip["members"] == MEMBERS and trip["currency"] == "EUR"


@pytest.mark.parametrize(
    "name, code, members, message",
    [
        ("", "EUR", MEMBERS, "needs a name"),
        ("x", "MXN", MEMBERS, "Use USD or EUR"),
        ("x", "EUR", ["Ana"], "between 2 and 20"),
        ("x", "EUR", ["Ana", "ana"], "must be different"),
        ("x", "EUR", ["Ana", " "], "can't be empty"),
    ],
)
def test_create_trip_validation(db, name, code, members, message):
    with pytest.raises(ledger.LedgerError, match=message):
        ledger.create_trip(db, name, code, members)


def test_expense_split_between_everyone_by_default(db, trip):
    e = ledger.add_expense(db, trip["id"], "Dinner", 100, "EUR", "Laura")
    assert e["status"] == "draft"
    assert e["shares"] == {"Laura": 25.0, "Ana": 25.0, "Luis": 25.0, "Marta": 25.0}


def test_expense_in_the_other_currency_keeps_original_and_converted(db, trip):
    e = ledger.add_expense(db, trip["id"], "Taxi", 45, "dollars", "ana", ["Ana", "LUIS"])
    assert (e["amount"], e["currency"], e["trip_amount"], e["trip_currency"]) == (45.0, "USD", 40.5, "EUR")
    assert e["paid_by"] == "Ana"
    assert e["shares"] == {"Ana": 20.25, "Luis": 20.25}


def test_self_words_mean_the_current_user(db, trip):
    e = ledger.add_expense(db, trip["id"], "Tickets", 32, "€", "yo", ["Marta", "me"], current_user="Laura")
    assert e["paid_by"] == "Laura"
    assert e["shares"] == {"Laura": 16.0, "Marta": 16.0}


def test_leftover_cents_are_not_lost(db, trip):
    e = ledger.add_expense(db, trip["id"], "Snacks", 10, "EUR", "Luis", ["Ana", "Luis", "Marta"])
    assert sorted(e["shares"].values()) == [3.33, 3.33, 3.34]


@pytest.mark.parametrize(
    "args, message",
    [
        (("x", 10, "MXN", "Ana"), "Use USD or EUR"),
        (("x", 10, "EUR", "Pedro"), "isn't a member"),
        (("x", 0, "EUR", "Ana"), "greater than zero"),
        (("x", -5, "EUR", "Ana"), "positive"),
    ],
)
def test_expense_validation(db, trip, args, message):
    with pytest.raises(ledger.LedgerError, match=message):
        ledger.add_expense(db, trip["id"], *args)


def test_only_confirmed_expenses_count_in_balances(db, trip):
    draft = ledger.add_expense(db, trip["id"], "Dinner", 120, "EUR", "Laura")
    assert ledger.balances(db, trip["id"])["transfers"] == ["Everyone is even."]

    ledger.confirm_expense(db, draft["id"])
    result = ledger.balances(db, trip["id"])
    assert result["net"] == {"Laura": 90.0, "Ana": -30.0, "Luis": -30.0, "Marta": -30.0}
    assert result["transfers"] == [
        "Marta pays Laura 30.00 EUR", "Luis pays Laura 30.00 EUR", "Ana pays Laura 30.00 EUR"
    ]


def test_delete_expense(db, trip):
    e = ledger.add_expense(db, trip["id"], "Oops", 5, "EUR", "Ana")
    ledger.delete_expense(db, e["id"])
    with pytest.raises(ledger.NotFound):
        ledger.get_expense(db, e["id"])


def test_balances_always_add_up_to_zero(db, trip):
    rng = random.Random(1)
    for i in range(30):
        e = ledger.add_expense(db, trip["id"], f"E{i}", round(rng.uniform(1, 300), 2),
                               rng.choice(["USD", "EUR"]), rng.choice(MEMBERS),
                               rng.sample(MEMBERS, rng.randint(1, 4)))
        ledger.confirm_expense(db, e["id"])
    result = ledger.balances(db, trip["id"])
    assert round(sum(result["net"].values()), 2) == 0
    assert len(result["settle_up"]) <= len(MEMBERS) - 1
