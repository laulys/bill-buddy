"""The agent inside a trip: turns messy messages into expense DRAFTS.

It reuses the hand-written loop from agent.py with a different set of tools,
each one bound to a single trip and to the person who is chatting.

    "I paid the taxi, 45 dollars, for Ana and me"
        -> propose_expense(description="Taxi", amount=45, currency="USD",
                           paid_by="I", split_between=["Ana", "I"])
        -> a draft card the user confirms (or discards) in the app

The agent has no tool to confirm, delete or edit expenses: only people can.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import tool

from . import ledger, money
from .agent import BillBuddyAgent, StepCallback
from .guardrails import numbers_from_user

TRIP_PROMPT = """You are Bill Buddy, the assistant inside a group-trip expense app.
Trip: "{name}". Main currency: {currency}. Members: {members}.
The person chatting with you is {user}.

Your job: turn what the user tells you into expense proposals, and answer
questions about the trip's spending.

Rules:
1. To record an expense, call propose_expense. It only creates a DRAFT card that
   the user confirms in the app. Never say an expense is saved or confirmed.
2. "I", "me", "yo" means {user}. If nobody else is mentioned, split between all
   members. If it's unclear who paid, ask before proposing.
3. Only USD and EUR are supported ("dollars" = USD, "euros" = EUR).
4. Use tools for every number: tips, balances, who owes whom. Only report
   amounts that a tool returned. Never invent an amount or a tip: if the user
   didn't say it, ask. When reporting who owes whom, list every line of
   "transfers" exactly as given.
5. People add and confirm expenses in the app between messages, so earlier
   results go stale: call trip_balances or list_trip_expenses again for EVERY
   question about balances or expenses. Never answer them from memory.
6. Reply in the user's language, in one or two short sentences."""


def build_trip_agent(
    llm,
    trip_id: int,
    user: str,
    db_path: Optional[str] = None,
    on_step: Optional[StepCallback] = None,
) -> BillBuddyAgent:
    """Create an agent for one trip and one member. Its `drafts` list collects proposals."""
    with ledger.session(db_path) as conn:
        trip = ledger.get_trip(conn, trip_id)
    user = ledger.resolve_member(trip["members"], user)
    drafts: List[dict] = []

    def run(func, *args, **kwargs):
        try:
            with ledger.session(db_path) as conn:
                return func(conn, *args, **kwargs)
        except (ledger.LedgerError, money.MoneyError) as exc:
            return {"error": str(exc)}

    @tool
    def propose_expense(
        description: str,
        amount: float,
        currency: str,
        paid_by: str,
        split_between: Optional[List[str]] = None,
        tip_percent: float = 0,
    ) -> dict:
        """Propose a new expense as a DRAFT for the user to confirm in the app.

        - description: short label, e.g. "Taxi", "Dinner at Bairro Alto"
        - amount: the price before tip; currency: "USD" or "EUR"
        - paid_by: member who paid ("I" for the person chatting)
        - split_between: members who share it; leave empty for everyone
        - tip_percent: optional tip added to the amount, e.g. 15
        """
        total = amount
        if tip_percent:
            try:
                total = money.calculate_tip(amount, tip_percent)["total_with_tip"]
            except money.MoneyError as exc:
                return {"error": str(exc)}
        result = run(ledger.add_expense, trip_id, description, total, currency, paid_by,
                     split_between, status="draft", current_user=user)
        if "error" in result:
            return result
        drafts.append(result)
        return {"draft": result, "note": "Draft created. The user must confirm it in the app."}

    @tool
    def trip_balances() -> dict:
        """Current balances (confirmed expenses only) and the transfers that settle the trip:
        who should pay whom, and how much."""
        return run(ledger.balances, trip_id)

    @tool
    def list_trip_expenses() -> dict:
        """The trip's confirmed expenses and the drafts still waiting for confirmation."""
        def both(conn):
            return {
                "confirmed": ledger.list_expenses(conn, trip_id, "confirmed")[-20:],
                "drafts": ledger.list_expenses(conn, trip_id, "draft"),
            }
        return run(both)

    prompt = TRIP_PROMPT.format(
        name=trip["name"], currency=trip["currency"], members=", ".join(trip["members"]), user=user
    )
    agent = BillBuddyAgent(
        llm,
        tools=[propose_expense, trip_balances, list_trip_expenses],
        system_prompt=prompt,
        on_step=on_step,
        # Amounts must come from the user. (Only USD/EUR here, so no "pesos" guardrail.)
        guardrails=[numbers_from_user()],
    )
    agent.drafts = drafts
    return agent
