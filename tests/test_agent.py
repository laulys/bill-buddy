"""The hand-written agent loop, driven by a scripted fake LLM (no API key or Ollama)."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from bill_buddy.agent import BillBuddyAgent
from bill_buddy.tools import ALL_TOOLS


class ScriptedModel(GenericFakeChatModel):
    """Replies from a script and records every conversation it was sent."""

    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def model(*replies):
    return ScriptedModel(messages=iter(replies), seen=[])


def calls(*specs):
    """AIMessage requesting one or more tool calls: calls(("name", {args}), ...)."""
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"call_{i}"} for i, (name, args) in enumerate(specs)
    ])


def tool_results(agent):
    return [json.loads(m.content) for m in agent.messages if isinstance(m, ToolMessage)]


def test_answers_directly_when_no_tool_is_needed():
    # The course lab's TipAgent crashed here (tool_calls[0] on an empty list).
    agent = BillBuddyAgent(model(AIMessage(content="Hi! Send me a bill.")))
    assert agent.ask("hello") == "Hi! Send me a bill."


def test_multi_step_question_feeds_each_result_into_the_next_call():
    llm = model(
        calls(("calculate_tip", {"bill_amount": 230, "tip_percent": 10})),
        calls(("split_evenly", {"total": 253, "people": 4})),
        AIMessage(content="Each pays 63.25 EUR."),
    )
    agent = BillBuddyAgent(llm)

    assert agent.ask("4 friends, 230 EUR, 10% tip") == "Each pays 63.25 EUR."
    assert tool_results(agent)[0]["total_with_tip"] == 253.0
    assert tool_results(agent)[1]["shares"] == [63.25] * 4
    assert len(llm.seen) == 3  # the loop called the model three times
    assert isinstance(llm.seen[1][-1], ToolMessage)  # ...and showed it the tool result


def test_runs_parallel_tool_calls_and_links_them_by_id():
    # The course lab's agent only ran tool_calls[0].
    llm = model(
        calls(("calculate_tip", {"bill_amount": 100, "tip_percent": 10}),
              ("split_evenly", {"total": 90, "people": 3})),
        AIMessage(content="done"),
    )
    agent = BillBuddyAgent(llm)
    agent.ask("two things")

    tool_messages = [m for m in agent.messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["call_0", "call_1"]
    assert [m.name for m in tool_messages] == ["calculate_tip", "split_evenly"]


def test_reports_each_step_to_the_callback():
    steps = []
    llm = model(calls(("split_evenly", {"total": 10, "people": 2})), AIMessage(content="5 each"))
    BillBuddyAgent(llm, on_step=lambda *step: steps.append(step)).ask("split 10 in 2")
    assert steps == [("split_evenly", {"total": 10, "people": 2}, {"total": 10.0, "people": 2, "shares": [5.0, 5.0]})]


@pytest.mark.parametrize(
    "bad_call, expected",
    [
        (("pay_bill", {"amount": 5}), "Unknown tool 'pay_bill'"),
        (("split_evenly", {"total": "lots", "people": 2}), "Invalid arguments for split_evenly: total"),
        (("split_evenly", {"total": 10}), "Invalid arguments for split_evenly: people"),
        (("split_evenly", {"total": 10, "people": 0}), "whole number between 1 and 100"),
    ],
)
def test_bad_tool_calls_become_errors_the_model_can_read(bad_call, expected):
    agent = BillBuddyAgent(model(calls(bad_call), AIMessage(content="Sorry!")))
    assert agent.ask("?") == "Sorry!"
    assert expected in tool_results(agent)[0]["error"]


def test_invalid_json_tool_calls_get_an_error_reply():
    broken = AIMessage(content="", invalid_tool_calls=[
        {"name": "split_evenly", "args": "{total: 10", "id": "bad_1", "error": "bad json"}
    ])
    agent = BillBuddyAgent(model(broken, AIMessage(content="Let me retry.")))
    assert agent.ask("?") == "Let me retry."
    assert "not valid JSON" in tool_results(agent)[0]["error"]


def test_stops_after_max_steps():
    looping = [calls(("split_evenly", {"total": 10, "people": 2})) for _ in range(10)]
    agent = BillBuddyAgent(model(*looping), max_steps=3)
    assert agent.ask("loop forever").startswith("Sorry, I couldn't finish")
    assert len(tool_results(agent)) == 3


def test_guardrail_blocks_ambiguous_pesos():
    llm = model(
        calls(("convert_currency", {"amount": 46, "from_currency": "USD", "to_currency": "MXN"})),
        AIMessage(content="Which pesos?"),
    )
    agent = BillBuddyAgent(llm)
    assert agent.ask("¿Cuánto es en pesos?") == "Which pesos?"
    assert "without saying which country" in tool_results(agent)[0]["error"]


def test_conversation_is_kept_for_follow_ups():
    llm = model(AIMessage(content="Which pesos?"), AIMessage(content="OK, Colombian."))
    agent = BillBuddyAgent(llm)
    agent.ask("in pesos")
    agent.ask("Colombian")
    second_request = llm.seen[1]
    assert isinstance(second_request[0], SystemMessage)
    assert [m.content for m in second_request if isinstance(m, HumanMessage)] == ["in pesos", "Colombian"]


def test_a_failed_turn_is_rolled_back():
    class Broken(ScriptedModel):
        def _generate(self, *args, **kwargs):
            raise ConnectionError("Ollama is down")

    agent = BillBuddyAgent(Broken(messages=iter([]), seen=[]))
    with pytest.raises(ConnectionError):
        agent.ask("hello")
    assert len(agent.messages) == 1  # only the system prompt


def test_tools_have_descriptions_and_schemas():
    assert [t.name for t in ALL_TOOLS] == ["calculate_tip", "split_evenly", "split_itemized", "convert_currency"]
    for t in ALL_TOOLS:
        assert t.description
    item_schema = ALL_TOOLS[2].args["items"]
    assert item_schema["type"] == "array"
