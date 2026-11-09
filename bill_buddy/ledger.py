"""Trip ledger: trips, members and expenses stored in SQLite.

Key design rule: the AI can only create DRAFT expenses. A person confirms
them in the app. Balances only count confirmed expenses, so the AI can
never change who owes what on its own.

Money is stored in whole cents. Each expense keeps its original amount and
currency, plus the exchange rate and the amount in the trip's currency at
the moment it was recorded ("$45 (≈ 41.20 €)").
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Iterator, List, Optional, Sequence

from . import currency
from .money import MoneyError, allocate, from_cents, round_cents, to_cents, to_decimal
from .settle import net_balances, settle_up

SUPPORTED_CURRENCIES = ("USD", "EUR")
CURRENCY_ALIASES = {"$": "USD", "DOLLAR": "USD", "DOLLARS": "USD", "DÓLAR": "USD", "DÓLARES": "USD",
                    "DOLARES": "USD", "€": "EUR", "EURO": "EUR", "EUROS": "EUR"}
MIN_MEMBERS, MAX_MEMBERS = 2, 20
SELF_WORDS = {"i", "me", "myself", "yo", "mí", "mi"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS trips (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    currency TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS members (
    id INTEGER PRIMARY KEY,
    trip_id INTEGER NOT NULL REFERENCES trips(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    UNIQUE (trip_id, name COLLATE NOCASE)
);
CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY,
    trip_id INTEGER NOT NULL REFERENCES trips(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    currency TEXT NOT NULL,
    rate TEXT NOT NULL,
    trip_amount_cents INTEGER NOT NULL,
    paid_by INTEGER NOT NULL REFERENCES members(id),
    status TEXT NOT NULL CHECK (status IN ('draft', 'confirmed')),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shares (
    expense_id INTEGER NOT NULL REFERENCES expenses(id) ON DELETE CASCADE,
    member_id INTEGER NOT NULL REFERENCES members(id),
    cents INTEGER NOT NULL,
    PRIMARY KEY (expense_id, member_id)
);
"""


class LedgerError(ValueError):
    """Invalid input the user (or the LLM) can fix."""


class NotFound(LedgerError):
    """A trip or expense that doesn't exist."""


def default_db_path() -> str:
    return os.getenv("BILL_BUDDY_DB", "bill_buddy.db")


@contextmanager
def session(db_path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    """Open the database, commit on success, roll back on error, always close."""
    conn = sqlite3.connect(db_path or default_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_currency(value: str) -> str:
    code = str(value).strip().upper()
    code = CURRENCY_ALIASES.get(code, code)
    if code not in SUPPORTED_CURRENCIES:
        raise LedgerError(f"Currency '{value}' isn't supported. Use USD or EUR.")
    return code


# ---------------------------------------------------------------- trips

def create_trip(conn: sqlite3.Connection, name: str, currency_code: str, members: Sequence[str]) -> Dict:
    name = str(name).strip()
    if not name:
        raise LedgerError("The trip needs a name.")
    code = normalize_currency(currency_code)
    clean = [str(m).strip() for m in members]
    if any(not m for m in clean):
        raise LedgerError("Member names can't be empty.")
    if len({m.lower() for m in clean}) != len(clean):
        raise LedgerError("Member names must be different.")
    if not MIN_MEMBERS <= len(clean) <= MAX_MEMBERS:
        raise LedgerError(f"A trip needs between {MIN_MEMBERS} and {MAX_MEMBERS} members.")

    trip_id = conn.execute(
        "INSERT INTO trips (name, currency, created_at) VALUES (?, ?, ?)", (name, code, _now())
    ).lastrowid
    conn.executemany("INSERT INTO members (trip_id, name) VALUES (?, ?)", [(trip_id, m) for m in clean])
    return get_trip(conn, trip_id)


def get_trip(conn: sqlite3.Connection, trip_id: int) -> Dict:
    row = conn.execute("SELECT * FROM trips WHERE id = ?", (trip_id,)).fetchone()
    if row is None:
        raise NotFound(f"Trip {trip_id} not found.")
    members = conn.execute("SELECT id, name FROM members WHERE trip_id = ? ORDER BY id", (trip_id,)).fetchall()
    return {
        "id": row["id"],
        "name": row["name"],
        "currency": row["currency"],
        "created_at": row["created_at"],
        "members": [m["name"] for m in members],
    }


def list_trips(conn: sqlite3.Connection) -> List[Dict]:
    ids = [r["id"] for r in conn.execute("SELECT id FROM trips ORDER BY id DESC")]
    return [get_trip(conn, i) for i in ids]


def _member_ids(conn: sqlite3.Connection, trip_id: int) -> Dict[str, int]:
    rows = conn.execute("SELECT id, name FROM members WHERE trip_id = ? ORDER BY id", (trip_id,))
    return {r["name"]: r["id"] for r in rows}


def resolve_member(members: Sequence[str], name: str, current_user: Optional[str] = None) -> str:
    """Match a name from the LLM to a trip member: case-insensitive, "I/me/yo" = current user."""
    wanted = str(name).strip().lower()
    if wanted in SELF_WORDS and current_user:
        wanted = current_user.lower()
    for member in members:
        if member.lower() == wanted:
            return member
    raise LedgerError(f"'{name}' isn't a member of this trip. Members: {', '.join(members)}.")


# ---------------------------------------------------------------- expenses

def add_expense(
    conn: sqlite3.Connection,
    trip_id: int,
    description: str,
    amount,
    currency_code: str,
    paid_by: str,
    split_between: Optional[Sequence[str]] = None,
    status: str = "draft",
    current_user: Optional[str] = None,
) -> Dict:
    """Record an expense split equally between some (default: all) members."""
    trip = get_trip(conn, trip_id)
    description = str(description).strip() or "Expense"
    try:
        value = to_decimal(amount)
    except MoneyError as exc:
        raise LedgerError(str(exc)) from exc
    if value == 0:
        raise LedgerError("The amount must be greater than zero.")
    code = normalize_currency(currency_code)
    payer = resolve_member(trip["members"], paid_by, current_user)
    names = trip["members"] if not split_between else [
        resolve_member(trip["members"], n, current_user) for n in split_between
    ]
    names = list(dict.fromkeys(names))  # drop duplicates, keep order

    if code == trip["currency"]:
        rate = Decimal(1)
    else:
        try:
            rate = Decimal(str(currency.convert(1, code, trip["currency"])["rate"]))
        except MoneyError as exc:
            raise LedgerError(str(exc)) from exc
    trip_cents = to_cents(round_cents(value * rate))

    ids = _member_ids(conn, trip_id)
    ordered = [n for n in trip["members"] if n in names]  # stable order for leftover cents
    shares = allocate(trip_cents, [Decimal(1)] * len(ordered))

    expense_id = conn.execute(
        """INSERT INTO expenses (trip_id, description, amount_cents, currency, rate,
                                 trip_amount_cents, paid_by, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (trip_id, description, to_cents(value), code, str(rate), trip_cents, ids[payer], status, _now()),
    ).lastrowid
    conn.executemany(
        "INSERT INTO shares (expense_id, member_id, cents) VALUES (?, ?, ?)",
        [(expense_id, ids[n], c) for n, c in zip(ordered, shares)],
    )
    return get_expense(conn, expense_id)


def get_expense(conn: sqlite3.Connection, expense_id: int) -> Dict:
    row = conn.execute(
        """SELECT e.*, m.name AS payer, t.currency AS trip_currency
           FROM expenses e JOIN members m ON m.id = e.paid_by JOIN trips t ON t.id = e.trip_id
           WHERE e.id = ?""",
        (expense_id,),
    ).fetchone()
    if row is None:
        raise NotFound(f"Expense {expense_id} not found.")
    shares = conn.execute(
        """SELECT m.name, s.cents FROM shares s JOIN members m ON m.id = s.member_id
           WHERE s.expense_id = ? ORDER BY m.id""",
        (expense_id,),
    ).fetchall()
    return {
        "id": row["id"],
        "trip_id": row["trip_id"],
        "description": row["description"],
        "amount": from_cents(row["amount_cents"]),
        "currency": row["currency"],
        "rate": float(row["rate"]),
        "trip_amount": from_cents(row["trip_amount_cents"]),
        "trip_currency": row["trip_currency"],
        "paid_by": row["payer"],
        "shares": {s["name"]: from_cents(s["cents"]) for s in shares},
        "status": row["status"],
        "created_at": row["created_at"],
    }


def list_expenses(conn: sqlite3.Connection, trip_id: int, status: Optional[str] = None) -> List[Dict]:
    get_trip(conn, trip_id)  # 404 if the trip doesn't exist
    query, params = "SELECT id FROM expenses WHERE trip_id = ?", [trip_id]
    if status:
        query += " AND status = ?"
        params.append(status)
    ids = [r["id"] for r in conn.execute(query + " ORDER BY id", params)]
    return [get_expense(conn, i) for i in ids]


def confirm_expense(conn: sqlite3.Connection, expense_id: int) -> Dict:
    get_expense(conn, expense_id)
    conn.execute("UPDATE expenses SET status = 'confirmed' WHERE id = ?", (expense_id,))
    return get_expense(conn, expense_id)


def delete_expense(conn: sqlite3.Connection, expense_id: int) -> None:
    get_expense(conn, expense_id)
    conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))


# ---------------------------------------------------------------- balances

def balances(conn: sqlite3.Connection, trip_id: int) -> Dict:
    """Net balance per member and the transfers to settle up (confirmed expenses only)."""
    trip = get_trip(conn, trip_id)
    rows = conn.execute(
        """SELECT e.id, m.name AS payer, e.trip_amount_cents FROM expenses e
           JOIN members m ON m.id = e.paid_by WHERE e.trip_id = ? AND e.status = 'confirmed'""",
        (trip_id,),
    ).fetchall()
    expenses = []
    for row in rows:
        shares = conn.execute(
            """SELECT m.name, s.cents FROM shares s JOIN members m ON m.id = s.member_id
               WHERE s.expense_id = ?""",
            (row["id"],),
        ).fetchall()
        expenses.append((row["payer"], row["trip_amount_cents"], {s["name"]: s["cents"] for s in shares}))

    net = net_balances(trip["members"], expenses)
    transfers = settle_up(net)
    return {
        "currency": trip["currency"],
        "total_spent": from_cents(sum(total for _, total, _ in expenses)),
        "net": {name: from_cents(cents) for name, cents in net.items()},
        "settle_up": [
            {"from": frm, "to": to, "amount": from_cents(cents)} for frm, to, cents in transfers
        ],
        # Ready-made sentences, so the LLM copies them instead of re-summarizing numbers.
        "transfers": [f"{frm} pays {to} {from_cents(cents):.2f} {trip['currency']}" for frm, to, cents in transfers]
        or ["Everyone is even."],
    }
