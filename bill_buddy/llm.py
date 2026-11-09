"""Chooses the chat model from environment variables (see `.env.example`)."""

from __future__ import annotations

import os

PROVIDERS = ("ollama", "openai", "watsonx")


class ConfigError(RuntimeError):
    """Raised when the environment is missing something the chosen provider needs."""


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Missing {name}. Copy .env.example to .env and fill it in.")
    return value


def get_llm():
    """Return a LangChain chat model for LLM_PROVIDER: 'ollama' (default), 'openai' or 'watsonx'."""
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "ollama":
        # Free and local: the model runs on your own computer, no API key needed.
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            reasoning=False,  # skip the "thinking" phase of reasoning models: faster answers
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        _require("OPENAI_API_KEY")
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.4-nano"))

    if provider == "watsonx":
        try:
            from langchain_ibm import ChatWatsonx
        except ImportError as exc:
            raise ConfigError("watsonx needs an extra package: pip install langchain-ibm") from exc

        _require("WATSONX_APIKEY")  # read automatically by ChatWatsonx
        return ChatWatsonx(
            model_id=os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small"),
            url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            project_id=_require("WATSONX_PROJECT_ID"),
        )

    raise ConfigError(f"Unknown LLM_PROVIDER {provider!r}. Use one of: {', '.join(PROVIDERS)}.")
