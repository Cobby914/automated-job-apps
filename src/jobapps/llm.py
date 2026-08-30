"""LLM provider abstraction. Callers must not branch on OpenAI / Anthropic / Cursor."""

from __future__ import annotations

import json
import os
import random
import re
import time
from contextvars import ContextVar
from datetime import datetime, timedelta
from typing import TypeVar

from pydantic import BaseModel

from jobapps.config import (
    USAGE_LOG_PATH,
    llm_daily_budget_usd,
    llm_max_retries,
    llm_retry_base_seconds,
    openai_reasoning_effort,
    require_env,
)
from jobapps.errors import PROVIDER_FAILURE, PipelineError
from jobapps.models import CostSummary, UsageRecord

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
_USAGE: ContextVar[list[UsageRecord] | None] = ContextVar("llm_usage", default=None)

# input / cached-input / output USD per 1M tokens. Unknown models use gpt-4.1 rates.
_PRICING: dict[str, tuple[float, float, float]] = {
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-5.6-sol": (2.00, 0.50, 8.00),
    "claude-sonnet-4-5": (3.00, 0.30, 15.00),
    "claude-4.5-sonnet": (3.00, 0.30, 15.00),
    "claude-opus-4-1": (15.00, 1.50, 75.00),
    "claude-opus-5": (15.00, 1.50, 75.00),
}

T = TypeVar("T", bound=BaseModel)


class ProviderError(PipelineError):
    def __init__(self, message: str, *, retryable: bool = False, status_code: int | None = None) -> None:
        super().__init__(PROVIDER_FAILURE, message)
        self.retryable = retryable
        self.status_code = status_code


def extract_json(text: str) -> str:
    """Used only for Anthropic/Cursor text responses. OpenAI uses native structured outputs."""
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
        raise ProviderError("Model did not return JSON.", retryable=False)
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


def begin_usage_collection() -> list[UsageRecord]:
    records: list[UsageRecord] = []
    _USAGE.set(records)
    return records


def usage_records() -> list[UsageRecord]:
    return list(_USAGE.get() or [])


def reset_usage_collection() -> None:
    _USAGE.set(None)


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> float:
    rates = _PRICING.get(model.strip().lower()) or _PRICING["gpt-4.1"]
    input_rate, cached_rate, output_rate = rates
    uncached = max(0, input_tokens - cached_input_tokens)
    return (
        uncached * input_rate
        + cached_input_tokens * cached_rate
        + output_tokens * output_rate
    ) / 1_000_000


def _record_usage(record: UsageRecord) -> None:
    bucket = _USAGE.get()
    if bucket is not None:
        bucket.append(record)


def _check_daily_budget() -> None:
    budget = llm_daily_budget_usd()
    if budget is None:
        return
    summary = aggregate_costs()
    if summary.daily_usd >= budget:
        raise ProviderError(
            f"Daily LLM budget of ${budget:.2f} exceeded (${summary.daily_usd:.2f}).",
            retryable=False,
        )


def complete(
    *,
    system: str,
    user: str,
    model: str,
    cache_system: bool = True,
    purpose: str = "",
) -> str:
    provider = resolve_provider(model)
    return _with_retry(
        provider,
        model,
        purpose,
        lambda: _complete_once(provider, system, user, model, cache_system, purpose),
    )


def generate_text(*, system: str, user: str, model: str, cache_system: bool = True, purpose: str = "") -> str:
    return complete(
        system=system, user=user, model=model, cache_system=cache_system, purpose=purpose
    )


def generate_structured(
    *,
    system: str,
    user: str,
    model: str,
    schema: type[T],
    cache_system: bool = True,
    purpose: str = "",
) -> T:
    provider = resolve_provider(model)
    if provider == "openai":
        return _with_retry(
            provider,
            model,
            purpose,
            lambda: _openai_structured(system, user, model, schema, purpose),
        )
    instruction = (
        f"{system.rstrip()}\n\nReturn a JSON object matching this schema. "
        f"No markdown, no code fences.\n{json.dumps(schema.model_json_schema())}"
    )
    text = complete(
        system=instruction,
        user=user,
        model=model,
        cache_system=cache_system,
        purpose=purpose,
    )
    try:
        return schema.model_validate_json(extract_json(text))
    except ProviderError:
        raise
    except Exception as error:
        raise ProviderError(
            f"Model returned invalid {schema.__name__} JSON: {error}",
            retryable=False,
        ) from error


def review(
    *,
    system: str,
    user: str,
    model: str,
    schema: type[T] | None = None,
    cache_system: bool = True,
    purpose: str = "review",
) -> T:
    from jobapps.models import ReviewResult

    target: type[BaseModel] = schema or ReviewResult
    return generate_structured(
        system=system,
        user=user,
        model=model,
        schema=target,  # type: ignore[arg-type]
        cache_system=cache_system,
        purpose=purpose,
    )


def _with_retry(provider: str, model: str, purpose: str, call):
    _check_daily_budget()
    attempts = llm_max_retries() + 1
    base = llm_retry_base_seconds()
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except ProviderError as error:
            last_error = error
            if not error.retryable or attempt >= attempts - 1:
                raise
            delay = base * (2**attempt) + random.uniform(0, 0.25)
            time.sleep(delay)
        except Exception as error:
            last_error = error
            wrapped = _wrap_provider_exception(provider, model, error)
            if not wrapped.retryable or attempt >= attempts - 1:
                raise wrapped from error
            delay = base * (2**attempt) + random.uniform(0, 0.25)
            time.sleep(delay)
    raise last_error or ProviderError(f"{provider} request failed ({model}, {purpose}).")


def _wrap_provider_exception(provider: str, model: str, error: Exception) -> ProviderError:
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None
    text = str(error).casefold()
    retryable = status_code in {408, 409, 429, 500, 502, 503, 504, 529} or any(
        token in text for token in ("rate limit", "overloaded", "temporarily", "timeout", "timed out", "connection")
    )
    if status_code in {400, 401, 403, 404, 422}:
        retryable = False
    return ProviderError(
        f"{provider.title()} request failed ({model}): {error}",
        retryable=retryable,
        status_code=status_code,
    )


def _complete_once(
    provider: str,
    system: str,
    user: str,
    model: str,
    cache_system: bool,
    purpose: str,
) -> str:
    if provider == "anthropic":
        return _complete_anthropic(system, user, model, cache_system, purpose)
    if provider == "openai":
        return _complete_openai(system, user, model, purpose)
    return _complete_cursor(system, user, model, purpose)


def _usage_from_openai(response: object, model: str, latency_ms: float, purpose: str) -> UsageRecord:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(
        getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
    )
    details = getattr(usage, "prompt_tokens_details", None) or getattr(
        usage, "input_tokens_details", None
    )
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    completion_details = getattr(usage, "completion_tokens_details", None) or getattr(
        usage, "output_tokens_details", None
    )
    reasoning = (
        int(getattr(completion_details, "reasoning_tokens", 0) or 0)
        if completion_details is not None
        else 0
    )
    record = UsageRecord(
        provider="openai",
        model=model,
        purpose=purpose,
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning,
        latency_ms=latency_ms,
        estimated_cost_usd=estimate_cost_usd(model, input_tokens, cached, output_tokens),
    )
    _record_usage(record)
    return record


def _openai_client():
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ProviderError("The openai package is required for GPT models.") from error
    return OpenAI(api_key=require_env("OPENAI_API_KEY"))


def _model_supports_reasoning(model: str) -> bool:
    key = model.strip().lower()
    return key.startswith(("gpt-5", "o1", "o3", "o4"))


def _openai_reasoning_kwargs(model: str, purpose: str) -> dict[str, object]:
    if not _model_supports_reasoning(model):
        return {}
    effort = openai_reasoning_effort(purpose)
    if not effort:
        return {}
    return {"reasoning": {"effort": effort}}


def _complete_openai(system: str, user: str, model: str, purpose: str) -> str:
    client = _openai_client()
    extra = _openai_reasoning_kwargs(model, purpose)
    started = time.perf_counter()
    responses = getattr(client, "responses", None)
    try:
        if responses is not None:
            kwargs: dict[str, object] = {"model": model, "input": user}
            if system:
                kwargs["instructions"] = system
            kwargs.update(extra)
            response = client.responses.create(**kwargs)
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            response = client.chat.completions.create(model=model, messages=messages)
    except Exception as error:
        raise _wrap_provider_exception("openai", model, error) from error
    latency = (time.perf_counter() - started) * 1000
    _usage_from_openai(response, model, latency, purpose)
    text = (getattr(response, "output_text", None) or "").strip()
    if not text:
        choices = getattr(response, "choices", None) or []
        if choices:
            text = (choices[0].message.content or "").strip()
    if not text:
        raise ProviderError(f"OpenAI returned no text ({model}).")
    return text


def _openai_structured(system: str, user: str, model: str, schema: type[T], purpose: str) -> T:
    client = _openai_client()
    extra = _openai_reasoning_kwargs(model, purpose)
    started = time.perf_counter()
    responses = getattr(client, "responses", None)
    try:
        if responses is not None and hasattr(responses, "parse"):
            kwargs: dict[str, object] = {
                "model": model,
                "input": user,
                "text_format": schema,
            }
            if system:
                kwargs["instructions"] = system
            kwargs.update(extra)
            response = client.responses.parse(**kwargs)
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            parse = getattr(getattr(client, "beta", None), "chat", None)
            if parse is not None:
                response = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=schema,
                )
            else:
                response = client.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=schema,
                )
    except Exception as error:
        raise _wrap_provider_exception("openai", model, error) from error
    latency = (time.perf_counter() - started) * 1000
    _usage_from_openai(response, model, latency, purpose)
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        choices = getattr(response, "choices", None) or [None]
        message = getattr(choices[0], "message", None)
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            refusal = getattr(message, "refusal", None) if message is not None else None
            raise ProviderError(
                f"OpenAI structured output was empty ({model}): {refusal or 'no parsed object'}."
            )
    if isinstance(parsed, schema):
        return parsed
    return schema.model_validate(parsed)


def _usage_from_anthropic(response: object, model: str, latency_ms: float, purpose: str) -> UsageRecord:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    created = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    total_input = input_tokens + cached + created
    record = UsageRecord(
        provider="anthropic",
        model=model,
        purpose=purpose,
        input_tokens=total_input,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        estimated_cost_usd=estimate_cost_usd(model, total_input, cached, output_tokens),
    )
    _record_usage(record)
    return record


def _complete_anthropic(system: str, user: str, model: str, cache_system: bool, purpose: str) -> str:
    try:
        import anthropic
    except ImportError as error:
        raise ProviderError("The anthropic package is required for Claude models.") from error
    client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
    if system and cache_system:
        system_payload: object = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
    else:
        system_payload = system or ""
    started = time.perf_counter()
    try:
        response = client.messages.create(
            model=anthropic_model_id(model),
            max_tokens=8192,
            system=system_payload,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as error:
        raise _wrap_provider_exception("anthropic", model, error) from error
    latency = (time.perf_counter() - started) * 1000
    _usage_from_anthropic(response, model, latency, purpose)
    parts = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    text = "".join(parts).strip()
    if not text:
        raise ProviderError(f"Anthropic returned no text ({model}).")
    return text


def _complete_cursor(system: str, user: str, model: str, purpose: str) -> str:
    try:
        from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions
    except ImportError as error:
        raise ProviderError("cursor-sdk is required for the Cursor LLM provider.") from error
    from jobapps.config import ROOT

    prompt = f"{system.rstrip()}\n\n{user}" if system.strip() else user
    started = time.perf_counter()
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
        raise ProviderError(f"Cursor agent failed to start ({model}): {error}", retryable=True) from error
    latency = (time.perf_counter() - started) * 1000
    if getattr(result, "status", None) == "error":
        detail = getattr(result, "result", None) or result.status
        raise ProviderError(f"Cursor run failed ({model}): {detail}")
    text = getattr(result, "result", None)
    if not text or not str(text).strip():
        raise ProviderError(f"Cursor returned no text ({model}).")
    record = UsageRecord(
        provider="cursor",
        model=model,
        purpose=purpose,
        latency_ms=latency,
    )
    _record_usage(record)
    return str(text)


def summarize_usage(records: list[UsageRecord] | None = None) -> CostSummary:
    items = records if records is not None else usage_records()
    input_tokens = sum(item.input_tokens for item in items)
    cached = sum(item.cached_input_tokens for item in items)
    output_tokens = sum(item.output_tokens for item in items)
    reasoning = sum(item.reasoning_tokens for item in items)
    cost = sum(item.estimated_cost_usd for item in items)
    cached_pct = (cached / input_tokens) if input_tokens else 0.0
    return CostSummary(
        application_usd=cost,
        call_count=len(items),
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning,
        cached_input_pct=cached_pct,
    )


def append_usage_log(records: list[UsageRecord], extra: dict[str, object] | None = None) -> None:
    if not records:
        return
    USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    with USAGE_LOG_PATH.open("a", encoding="utf-8") as handle:
        for item in records:
            payload = item.model_dump()
            payload["recorded_at"] = stamp
            if extra:
                payload.update(extra)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def aggregate_costs(now: datetime | None = None) -> CostSummary:
    """Daily / weekly / average cost from the shared usage ledger."""
    moment = now or datetime.now()
    day_start = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = day_start - timedelta(days=day_start.weekday())
    daily = 0.0
    weekly = 0.0
    total = 0.0
    apps: set[str] = set()
    if not USAGE_LOG_PATH.is_file():
        return CostSummary()
    for line in USAGE_LOG_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        cost = float(row.get("estimated_cost_usd") or 0.0)
        total += cost
        app_key = str(row.get("application") or row.get("recorded_at") or "")
        if app_key:
            apps.add(app_key)
        stamp_raw = str(row.get("recorded_at") or "")
        try:
            stamp = datetime.fromisoformat(stamp_raw)
        except ValueError:
            continue
        if stamp >= week_start:
            weekly += cost
        if stamp >= day_start:
            daily += cost
    average = (total / len(apps)) if apps else 0.0
    summary = summarize_usage()
    summary.daily_usd = daily
    summary.weekly_usd = weekly
    summary.average_usd_per_application = average
    return summary
