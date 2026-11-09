"""A tool-calling agent written by hand, without `create_agent` or `AgentExecutor`.

This is the loop those helpers run for you:

    1. Send the conversation (and the tool schemas) to the LLM.
    2. If the reply has no tool calls, it is the final answer: stop.
    3. Otherwise run every requested tool and add each result to the
       conversation as a ToolMessage linked by its tool_call_id.
    4. Go back to step 1 (up to max_steps times).

Compared with the course lab's ToolCallingAgent, this version:
- loops, so multi-step questions work ("tip, then split, then convert");
- runs every tool call in a reply, not just the first (parallel tool calls);
- answers directly when no tool is needed (the lab's TipAgent crashed);
- turns unknown tools, invalid arguments and tool crashes into error
  messages the LLM can read, instead of exceptions;
- stops after max_steps, so a confused model can't loop forever;
- keeps the conversation between questions, for follow-ups;
- checks guardrails in code before running a tool (see guardrails.py).
"""

from __future__ import annotations

import json
from typing import Any, Callable, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from .guardrails import DEFAULT_GUARDRAILS, Guardrail
from .tools import ALL_TOOLS

SYSTEM_PROMPT = """You are Bill Buddy, a friendly assistant that helps friends split bills,
calculate tips and convert currencies.

Rules:
1. Never do money math in your head. Use a tool for EVERY calculation, even simple ones.
2. For multi-step questions, call one tool, read its result, then call the next one
   with that result. Independent calculations can be requested at the same time.
3. Currencies must be 3-letter codes (USD, EUR, MXN, COP...). "Pesos" alone is
   ambiguous (MXN, COP, ARS, CLP...): if the country isn't clear, do NOT guess.
   Do the other calculations, then ask which pesos before converting.
4. Only report amounts that a tool returned. If a tool returns an error, fix the call
   or explain the problem.
5. Answer in the user's language, briefly, showing each person's amount clearly."""

# Called for every tool call and its result: on_step(name, args, result)
StepCallback = Callable[[str, dict, Any], None]


def message_text(message) -> str:
    """Plain text of a message, whether its content is a string or a list of blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


class BillBuddyAgent:
    def __init__(
        self,
        llm,
        tools: Optional[List] = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = 6,
        on_step: Optional[StepCallback] = None,
        guardrails: Optional[List[Guardrail]] = None,
    ):
        tools = tools if tools is not None else ALL_TOOLS
        self.model = llm.bind_tools(tools)          # sends the tool schemas with every request
        self.tools = {t.name: t for t in tools}     # name -> tool, to run what the LLM asks for
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.on_step = on_step
        self.guardrails = guardrails if guardrails is not None else DEFAULT_GUARDRAILS
        self.reset()

    def reset(self) -> None:
        """Forget the conversation."""
        self.messages: list = [SystemMessage(self.system_prompt)]

    def ask(self, question: str) -> str:
        """Answer one question, calling tools as needed. The conversation is kept for follow-ups."""
        start = len(self.messages)
        self.messages.append(HumanMessage(question))
        try:
            for _ in range(self.max_steps):
                reply = self.model.invoke(self.messages)
                self.messages.append(reply)

                calls = list(reply.tool_calls) + [self._broken(c) for c in reply.invalid_tool_calls]
                if not calls:
                    return message_text(reply)  # no tools requested: this is the answer

                for call in calls:  # the model may request several tools in one reply
                    result = call.get("error_result") or self.run_tool(call["name"], call["args"])
                    if self.on_step:
                        self.on_step(call["name"], call["args"], result)
                    self.messages.append(ToolMessage(
                        content=json.dumps(result, ensure_ascii=False, default=str),
                        tool_call_id=call["id"],
                        name=call["name"],
                    ))
        except Exception:
            del self.messages[start:]  # don't keep a half-finished turn
            raise

        give_up = "Sorry, I couldn't finish that calculation. Could you rephrase it in smaller steps?"
        self.messages.append(AIMessage(give_up))
        return give_up

    def run_tool(self, name: str, args: dict) -> Any:
        """Run one tool call. Every failure becomes a message for the LLM, never an exception."""
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(self.tools)}."}
        user_text = "\n".join(m.content for m in self.messages if isinstance(m, HumanMessage))
        for guardrail in self.guardrails:
            problem = guardrail(name, args, user_text)
            if problem:
                return {"error": problem}
        try:
            return tool.invoke(args)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return {"error": f"Invalid arguments for {name}: {problems}"}
        except Exception as exc:
            return {"error": f"{name} failed: {type(exc).__name__}: {exc}"}

    @staticmethod
    def _broken(call: dict) -> dict:
        """A tool call whose arguments weren't valid JSON: answer it with an error."""
        return {
            "name": call.get("name") or "unknown",
            "args": {},
            "id": call.get("id") or "invalid",
            "error_result": {"error": f"Your tool call arguments were not valid JSON: {call.get('args')!r}"},
        }
