"""Project root, folder paths, environment loading, and explicit model config."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def find_root() -> Path:
    env = os.getenv("JOBAPPS_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "resume_templates").is_dir():
        return package_root

    cwd = Path.cwd()
    if (cwd / "resume_templates").is_dir():
        return cwd

    raise RuntimeError(
        "Cannot find the AutomatedJobApps project root. "
        "Run from the repo directory or set JOBAPPS_ROOT."
    )


ROOT = find_root()
JOBS_DIR = ROOT / "jobs"
PROCESSED_DIR = JOBS_DIR / "processed"
SAMPLES_DIR = JOBS_DIR / "samples"
OUTPUT_DIR = ROOT / "output"
RESUME_TEMPLATES_DIR = ROOT / "resume_templates"
COVER_LETTER_TEMPLATES_DIR = ROOT / "cover_letter_templates"
COVER_LETTER_EXAMPLES_DIR = ROOT / "cover_letter_examples"
COVER_LETTER_EXAMPLE_PATH = COVER_LETTER_EXAMPLES_DIR / "Spacex_Cover_Letter.md"
WRITING_SAMPLES_DIR = ROOT / "writing_samples"
RESUME_ADDITIONS_DIR = ROOT / "resume_additions"
CAREER_DIR = ROOT / "career"
SKILLS_BANK_PATH = CAREER_DIR / "skills.yaml"
CONNECTIONS_PATH = ROOT / "connections" / "connections.yaml"
USAGE_LOG_PATH = OUTPUT_DIR / "usage.jsonl"
RANKING_LOG_PATH = OUTPUT_DIR / "ranking_log.jsonl"


def load_env() -> None:
    load_dotenv(ROOT / ".env")


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to .env (see .env.example).")
    return value


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def configured_provider() -> str:
    override = _env("LLM_PROVIDER").lower()
    if override in {"openai", "anthropic", "cursor"}:
        return override
    if _env("OPENAI_API_KEY"):
        return "openai"
    if _env("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "cursor"


def reviewer_provider() -> str:
    override = _env("LLM_REVIEWER_PROVIDER").lower()
    if override in {"openai", "anthropic", "cursor"}:
        return override
    return configured_provider()


def cursor_writer_model() -> str:
    return _env("CURSOR_WRITER_MODEL") or _env("CURSOR_MODEL") or "gpt-5.6-sol"


def cursor_checker_model() -> str:
    return _env("CURSOR_CHECKER_MODEL") or "claude-4.5-sonnet"


def cursor_escalation_model() -> str:
    return _env("CURSOR_ESCALATION_MODEL") or "claude-opus-5"


def writer_model() -> str:
    explicit = _env("OPENAI_WRITER_MODEL")
    if explicit:
        return explicit
    if _env("OPENAI_API_KEY") and configured_provider() == "openai":
        raise RuntimeError(
            "OPENAI_WRITER_MODEL is not set. Add it to .env (see .env.example)."
        )
    return cursor_writer_model()


def checker_model() -> str:
    """Cheap semantic reviewer. Alias: reviewer_model()."""
    provider = reviewer_provider()
    if provider == "openai":
        explicit = _env("OPENAI_REVIEWER_MODEL")
        if explicit:
            return explicit
        if _env("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_REVIEWER_MODEL is not set. Add it to .env (see .env.example)."
            )
    if provider == "anthropic":
        explicit = _env("ANTHROPIC_CHECKER_MODEL")
        if explicit:
            return explicit
        if _env("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_CHECKER_MODEL is not set. Add it to .env (see .env.example)."
            )
    return cursor_checker_model()


def reviewer_model() -> str:
    return checker_model()


def repair_model() -> str:
    explicit = _env("OPENAI_REPAIR_MODEL")
    if explicit:
        return explicit
    if _env("OPENAI_API_KEY") and configured_provider() == "openai":
        raise RuntimeError(
            "OPENAI_REPAIR_MODEL is not set. Add it to .env (see .env.example)."
        )
    return writer_model()


def escalation_model() -> str:
    provider = reviewer_provider()
    if provider == "openai":
        explicit = _env("OPENAI_ESCALATION_MODEL") or _env("OPENAI_WRITER_MODEL")
        if explicit:
            return explicit
        if _env("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_ESCALATION_MODEL is not set. Add it to .env (see .env.example)."
            )
    if provider == "anthropic":
        explicit = _env("ANTHROPIC_ESCALATION_MODEL")
        if explicit:
            return explicit
        if _env("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_ESCALATION_MODEL is not set. Add it to .env (see .env.example)."
            )
    return cursor_escalation_model()


def llm_max_retries() -> int:
    raw = _env("LLM_MAX_RETRIES") or "3"
    return max(0, int(raw))


def llm_retry_base_seconds() -> float:
    raw = _env("LLM_RETRY_BASE_SECONDS") or "1"
    return max(0.1, float(raw))


def llm_daily_budget_usd() -> float | None:
    raw = _env("LLM_DAILY_BUDGET_USD")
    if not raw:
        return None
    return float(raw)


_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)


def _purpose_reasoning_role(purpose: str) -> str:
    key = purpose.strip().lower()
    if "escalat" in key:
        return "escalation"
    if "review" in key:
        return "reviewer"
    if any(token in key for token in ("repair", "shorten", "fit")):
        return "repair"
    return "writer"


def openai_reasoning_effort(purpose: str = "") -> str | None:
    """Responses API reasoning.effort for GPT-5 / o-series. None = model default."""
    role = _purpose_reasoning_role(purpose)
    overrides = {
        "writer": "OPENAI_WRITER_REASONING_EFFORT",
        "reviewer": "OPENAI_REVIEWER_REASONING_EFFORT",
        "repair": "OPENAI_REPAIR_REASONING_EFFORT",
        "escalation": "OPENAI_ESCALATION_REASONING_EFFORT",
    }
    raw = _env(overrides.get(role, "")) or _env("OPENAI_REASONING_EFFORT")
    if not raw:
        return None
    key = raw.casefold()
    if key not in _REASONING_EFFORTS:
        raise RuntimeError(
            f"Invalid reasoning effort {raw!r}. Use one of: "
            + ", ".join(sorted(_REASONING_EFFORTS))
        )
    return key


def notion_configured() -> bool:
    return bool(_env("NOTION_TOKEN") and _env("NOTION_DATABASE_ID"))
