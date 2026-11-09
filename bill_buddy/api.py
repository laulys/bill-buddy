"""HTTP API for the trip app (FastAPI).

    uvicorn bill_buddy.api:app --reload
    open http://localhost:8000/docs     # interactive docs: try every endpoint

The mobile app talks to these endpoints. The flow for the AI part:

    POST /trips/{id}/chat  {"user": "Laura", "message": "I paid the taxi, 45 dollars"}
      -> {"reply": "...", "drafts": [ {expense card} ]}
    POST /expenses/{id}/confirm     (the person taps ✓)
    DELETE /expenses/{id}           (or discards it)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__, ledger
from .money import MoneyError
from .trip_agent import build_trip_agent


class TripIn(BaseModel):
    name: str = Field(examples=["Lisboa 2026"])
    currency: Literal["USD", "EUR"] = "EUR"
    members: List[str] = Field(examples=[["Laura", "Ana", "Luis", "Marta"]])


class ExpenseIn(BaseModel):
    description: str = Field(examples=["Dinner"])
    amount: float = Field(gt=0, examples=[120])
    currency: Literal["USD", "EUR"] = "EUR"
    paid_by: str = Field(examples=["Laura"])
    split_between: Optional[List[str]] = Field(default=None, description="Empty = everyone")


class ChatIn(BaseModel):
    user: str = Field(description="The member who is chatting", examples=["Laura"])
    message: str = Field(examples=["I paid the taxi, 45 dollars, for Ana and me"])


class ChatOut(BaseModel):
    reply: str
    drafts: List[dict]
    steps: List[dict]


def create_app(db_path: Optional[str] = None, llm_factory: Optional[Callable] = None) -> FastAPI:
    """Build the API. Tests pass a temporary database and a fake LLM."""
    load_dotenv()
    app = FastAPI(
        title="Bill Buddy API",
        version=__version__,
        description="Group-trip expenses with an AI assistant that proposes expenses for you to confirm.",
    )
    agents: Dict[Tuple[int, str], object] = {}  # one conversation per (trip, member)

    def make_llm():
        if llm_factory:
            return llm_factory()
        from .llm import get_llm
        return get_llm()

    @app.exception_handler(ledger.LedgerError)
    def ledger_error(request: Request, exc: ledger.LedgerError):
        status = 404 if isinstance(exc, ledger.NotFound) else 400
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    @app.exception_handler(MoneyError)
    def money_error(request: Request, exc: MoneyError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.post("/trips", status_code=201)
    def create_trip(body: TripIn):
        with ledger.session(db_path) as conn:
            return ledger.create_trip(conn, body.name, body.currency, body.members)

    @app.get("/trips")
    def list_trips():
        with ledger.session(db_path) as conn:
            return ledger.list_trips(conn)

    @app.get("/trips/{trip_id}")
    def get_trip(trip_id: int):
        """Everything a trip screen needs: members, expenses, drafts and who owes whom."""
        with ledger.session(db_path) as conn:
            return {
                **ledger.get_trip(conn, trip_id),
                "expenses": ledger.list_expenses(conn, trip_id, "confirmed"),
                "drafts": ledger.list_expenses(conn, trip_id, "draft"),
                "balances": ledger.balances(conn, trip_id),
            }

    @app.get("/trips/{trip_id}/balances")
    def get_balances(trip_id: int):
        with ledger.session(db_path) as conn:
            return ledger.balances(conn, trip_id)

    @app.post("/trips/{trip_id}/expenses", status_code=201)
    def add_expense(trip_id: int, body: ExpenseIn):
        """Manual entry (the classic form). Saved as confirmed right away."""
        with ledger.session(db_path) as conn:
            return ledger.add_expense(conn, trip_id, body.description, body.amount, body.currency,
                                      body.paid_by, body.split_between, status="confirmed")

    @app.post("/expenses/{expense_id}/confirm")
    def confirm_expense(expense_id: int):
        with ledger.session(db_path) as conn:
            return ledger.confirm_expense(conn, expense_id)

    @app.delete("/expenses/{expense_id}", status_code=204)
    def delete_expense(expense_id: int):
        with ledger.session(db_path) as conn:
            ledger.delete_expense(conn, expense_id)

    @app.post("/trips/{trip_id}/chat", response_model=ChatOut)
    def chat(trip_id: int, body: ChatIn):
        """Talk to the assistant. Expenses it proposes come back as drafts to confirm."""
        with ledger.session(db_path) as conn:
            trip = ledger.get_trip(conn, trip_id)
        user = ledger.resolve_member(trip["members"], body.user)
        key = (trip_id, user)
        if key not in agents:
            agents[key] = build_trip_agent(make_llm(), trip_id, user, db_path)
        agent = agents[key]

        steps: List[dict] = []
        agent.on_step = lambda name, args, result: steps.append({"tool": name, "args": args, "result": result})
        first_new_draft = len(agent.drafts)
        try:
            reply = agent.ask(body.message)
        except Exception as exc:  # the model is unreachable or misconfigured
            raise HTTPException(status_code=503, detail=f"The AI assistant is unavailable: {exc}") from exc
        return {"reply": reply, "drafts": agent.drafts[first_new_draft:], "steps": steps}

    return app


app = create_app()
