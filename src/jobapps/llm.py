"""LLM provider abstraction: OpenAI, Anthropic (cached system), Cursor fallback."""

from __future__ import annotations

import os
import re

from typing import TypeVar

from pydantic import BaseModel

from jobapps.config import ROOT, require_env

_ANTHROPIC_ALIASES = {
    "claude-4.5-sonnet": "claude-sonnet-4-5",
    "claude-sonnet-4.5": "claude-sonnet-4-5",
    "claude-4-5-sonnet": "claude-sonnet-4-5",
    "claude-opus-5": "claude-opus-4-1",
    "claude-opus-4.5": "claude-opus-4-1",
    "claude-4-opus": "claude-opus-4-1",
    "claude-opus-4": "claude-opus-4-1",
}

_OPENAI_PREFIX = re.compile(r"^(gpt|o\d)", re.IGNORECASE)


def extract_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        if lines and lines[0].strip().lower() in {"json", "jsonc"}:
            lines = lines[1:]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("Model did not return JSON.")
    return cleaned[start : end + 1]


def anthropic_model_id(model: str) -> str:
    key = model.strip().lower()
    return _ANTHROPIC_ALIASES.get(key, model.strip())


def resolve_provider(model: str) -> str:
    override = os.getenv("LLM_PROVIDER", "").strip().lower()
    if override in {"openai", "anthropic", "cursor"}:
        return override
    key = model.strip().lower()
    if key.startswith("claude"):
        if os.getenv("ANTHROPIC_API_KEY", "").strip():
            return "anthropic"
        return "cursor"
    if _OPENAI_PREFIX.match(key):
        if os.getenv("OPENAI_API_KEY", "").strip():
            return "openai"
        return "cursor"
    return "cursor"


T = TypeVar("T", bound=BaseModel)


def complete(*, system: str, user: str, model: str, cache_system: bool = True) -> str:
    provider = resolve_provider(model)
    if provider == "anthropic":
        return _complete_anthropic(system, user, model, cache_system)
    if provider == "openai":
        return _complete_openai(system, user, model)
    return _complete_cursor(system, user, model)


def generate_text(*, system: str, user: str, model: str, cache_system: bool = True) -> str:
    return complete(system=system, user=user, model=model, cache_system=cache_system)


def generate_structured(
    *,
    system: str,
    user: str,
    model: str,
    schema: type[T],
    cache_system: bool = True,
) -> T:
    text = complete(system=system, user=user, model=model, cache_system=cache_system)
    try:
        return schema.model_validate_json(extract_json(text))
    except Exception as error:
        raise RuntimeError(f"Model returned invalid {schema.__name__} JSON: {error}") from error


def review(
    *,
    system: str,
    user: str,
    model: str,
    schema: type[T] | None = None,
    cache_system: bool = True,
) -> T:
    from jobapps.models import ReviewResult

    target: type[BaseModel] = schema or ReviewResult
    return generate_structured(
        system=system,
        user=user,
        model=model,
        schema=target,  # type: ignore[arg-type]
        cache_system=cache_system,
    )


def _complete_anthropic(system: str, user: str, model: str, cache_system: bool) -> str:
    try:
        import anthropic
    except ImportError as error:
        raise RuntimeError("The anthropic package is required for Claude models.") from error
    client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
    if system and cache_system:
        system_payload: object = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
    else:
        system_payload = system or ""
    try:
        response = client.messages.create(
            model=anthropic_model_id(model),
            max_tokens=8192,
            system=system_payload,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as error:
        raise RuntimeError(f"Anthropic request failed ({model}): {error}") from error
    parts = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    text = "".join(parts).strip()
    if not text:
        raise RuntimeError(f"Anthropic returned no text ({model}).")
    return text


def _complete_openai(system: str, user: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("The openai package is required for GPT models.") from error
    client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    try:
        response = client.chat.completions.create(model=model, messages=messages)
    except Exception as error:
        raise RuntimeError(f"OpenAI request failed ({model}): {error}") from error
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(f"OpenAI returned no text ({model}).")
    return text


def _complete_cursor(system: str, user: str, model: str) -> str:
    try:
        from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions
    except ImportError as error:
        raise RuntimeError("cursor-sdk is required for the Cursor LLM provider.") from error
    prompt = f"{system.rstrip()}\n\n{user}" if system.strip() else user
    try:
        result = Agent.prompt(
            prompt,
            AgentOptions(
                api_key=require_env("CURSOR_API_KEY"),
                model=model,
                local=LocalAgentOptions(cwd=str(ROOT)),
                tools=[],
            ),
        )
    except CursorAgentError as error:
        raise RuntimeError(f"Cursor agent failed to start ({model}): {error}") from error
    if getattr(result, "status", None) == "error":
        detail = getattr(result, "result", None) or result.status
        raise RuntimeError(f"Cursor run failed ({model}): {detail}")
    text = getattr(result, "result", None)
    if not text or not str(text).strip():
        raise RuntimeError(f"Cursor returned no text ({model}).")
    return str(text)
