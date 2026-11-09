# 💸 Bill Buddy

A chat assistant that splits restaurant bills, calculates tips and converts currencies, built on a **tool-calling agent loop written from scratch** (no `create_agent`, no `AgentExecutor`).

Ask in plain language, in English or Spanish. Bill Buddy picks the right tools, chains them (tip → split → convert) and does every calculation in code with exact cents.

> 🇪🇸 [Resumen en español abajo](#-resumen-en-español)

![Demo: Bill Buddy splits a bill with tip, asks which pesos, then converts to Colombian pesos](docs/demo.gif)

## What it can do

Real output, running locally with Ollama (`qwen3:8b`):

```text
You: Ana had pasta for 18, Luis had steak for 32, and we shared a 24 dollar bottle of wine. Add 15% tip. How much does each pay?
  🔧 split_itemized(tip_percent=15, items=[{"amount": 18, "person": "Ana"}, {"amount": 32, "person": "Luis"}], shared_amount=24)
     → {"per_person": {"Ana": {"own_items": 18.0, "shared_part": 12.0, "tax": 0.0, "tip": 4.5, "pays": 34.5}, ...
Ana pays $34.50 and Luis pays $50.60.

You: Convert 50 euros to Mexican pesos and 50 euros to Colombian pesos
  🔧 convert_currency(amount=50, from_currency="EUR", to_currency="MXN")
  🔧 convert_currency(amount=50, from_currency="EUR", to_currency="COP")
50 euros is equivalent to:
- 1,038.71 Mexican Pesos (MXN)
- 180,937.58 Colombian Pesos (COP)
```

| Tool | What it does |
|---|---|
| `calculate_tip` | Tip and total for a bill |
| `split_evenly` | Equal shares that **always add up to the exact total** (100 / 3 → 33.34 + 33.33 + 33.33) |
| `split_itemized` | Everyone pays what they ordered, plus an equal part of shared items, plus tax and tip **in proportion to what they consumed** |
| `convert_currency` | Today's exchange rate for 160+ currencies (USD, EUR, MXN, COP, ARS, CLP...) |

## The agent loop, from scratch

Frameworks hide the most important part of an agent. [`bill_buddy/agent.py`](bill_buddy/agent.py) writes it out in about 60 lines:

```mermaid
flowchart TD
    Q([User question]) --> M[Send conversation + tool schemas to the LLM]
    M --> D{Tool calls in the reply?}
    D -- no --> A([Final answer])
    D -- yes --> G{Guardrails OK?}
    G -- no --> E[Error message for the LLM]
    G -- yes --> R[Run each tool<br/>errors become messages too]
    R --> T[Add ToolMessages,<br/>linked by tool_call_id]
    E --> T
    T --> S{max_steps reached?}
    S -- no --> M
    S -- yes --> X([Polite give-up message])
```

```python
for _ in range(self.max_steps):
    reply = self.model.invoke(self.messages)          # 1. ask the LLM
    self.messages.append(reply)
    if not reply.tool_calls:
        return message_text(reply)                    # 2. no tools -> final answer
    for call in reply.tool_calls:                     # 3. run EVERY requested tool
        result = self.run_tool(call["name"], call["args"])
        self.messages.append(ToolMessage(json.dumps(result), tool_call_id=call["id"]))
                                                      # 4. loop: the LLM reads the results
```

### Guardrails: rules in code, not just in the prompt

The prompt says *"if the user only says 'pesos', ask which country"*. In testing, `qwen3:8b` ignored that and assumed Mexican pesos every time. So the rule is now checked **in code** ([`guardrails.py`](bill_buddy/guardrails.py)) before `convert_currency` runs: an ambiguous peso conversion is blocked, and the model is told to ask. That's what the demo GIF shows. Guardrails work the same with any model, big or small.

## Getting started

**Requirements:** Python 3.10+ and [Ollama](https://ollama.com) (free, no API key).

```bash
ollama pull qwen3:8b                      # a local model with tool calling (~5 GB)

git clone https://github.com/laulys/bill-buddy.git
cd bill-buddy
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                      # works as-is with Ollama
```

```bash
python -m bill_buddy                                           # chat
python -m bill_buddy "How much should I tip on \$60 at 20%?"   # one question
python -m bill_buddy --quiet "Split 100 between 3"             # hide the tool calls
```

**Other providers:** set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` in `.env` to use OpenAI. For IBM watsonx, run `pip install langchain-ibm` and fill in the `WATSONX_*` values.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

49 tests, with no API key, no Ollama and no internet needed:
- **Money:** rounding, splits that never lose a cent, proportional tax and tip.
- **Currency:** a fake HTTP API, caching, network errors.
- **Guardrail:** 8 phrasings of "pesos", with and without a country.
- **Agent loop:** a scripted fake LLM covers multi-step chains, parallel calls, unknown tools, invalid arguments, broken JSON, `max_steps`, follow-up memory and rollback after a failure.

GitHub Actions runs them on every push.

## Improvements over the course lab

This project started as the guided lab *"Build Interactive LLM Agents with Tools"* from IBM Skills Network, which builds a calculator and tip agent by hand. I kept the from-scratch approach and fixed its limits:

| In the lab | In Bill Buddy |
|---|---|
| Runs only `tool_calls[0]` | Runs every tool call in a reply (parallel calls) |
| One round only: can't chain "tip, then split" | Loops until the answer is ready, up to `max_steps` |
| `TipAgent` crashes when the model answers without a tool | Returns the answer directly |
| Unknown tool or bad arguments raise exceptions | Become error messages the LLM can read and fix |
| The class reads global `tools` / `tool_map` | Tools are passed in, so the agent is reusable and testable |
| `int` amounts (no $60.50, no 12.5%) and float math | `Decimal` money, exact cents |
| No memory between questions | Keeps the conversation for follow-ups ("Colombian pesos") |
| Rules only in the prompt | Guardrails enforced in code |

## What I learned

- **What `create_agent` actually does:** a loop of *LLM → tool calls → ToolMessages → LLM*, with `tool_call_id` tying each result to its request.
- **Every failure should become a message.** An agent that crashes on a bad tool call can't recover; one that returns `{"error": ...}` usually fixes itself on the next step.
- **Prompts are suggestions; code is a rule.** Guardrails make critical behavior reliable even with a small local model.
- **Money needs `Decimal`** and a deliberate way to hand out leftover cents.

## Credits

Based on the IBM Skills Network guided project *Build Interactive LLM Agents with Tools* (author: Joshua Zhou; contributors: Kunal Makwana and Karan Goswami). The code in this repository is my own. Exchange rates by [ExchangeRate-API](https://www.exchangerate-api.com).

---

## 🇪🇸 Resumen en español

**Bill Buddy** es un asistente para dividir cuentas entre amigos: calcula propinas, reparte el total en partes iguales o según lo que pidió cada uno, y convierte monedas con el tipo de cambio del día. Le escribes en español o inglés y él usa herramientas de Python para hacer cada cálculo con exactitud de centavos.

**Qué demuestra este proyecto:**
- Un **ciclo de agente programado desde cero**, sin `create_agent`, para entender qué hacen los frameworks por dentro.
- Llamadas a varias herramientas en cadena y en paralelo, con manejo de errores y límite de pasos.
- **Guardrails**: reglas importantes verificadas en código. Por ejemplo, si dices solo "pesos", te pregunta de qué país.
- Dinero con `Decimal`: las partes siempre suman el total exacto.
- 49 tests automatizados con un LLM simulado, e integración continua con GitHub Actions.

**Cómo usarlo:** sigue los pasos de [Getting started](#getting-started). Funciona gratis en tu computadora con Ollama.

Basado en el laboratorio guiado de IBM Skills Network, reimplementado y ampliado como proyecto independiente.
