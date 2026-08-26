"""Project root, folder paths, and environment loading."""

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
WRITING_SAMPLES_DIR = ROOT / "writing_samples"
RESUME_ADDITIONS_DIR = ROOT / "resume_additions"
SKILLS_BANK_PATH = RESUME_ADDITIONS_DIR / "skills.md"
CONNECTIONS_PATH = ROOT / "connections" / "connections.yaml"


def load_env() -> None:
    load_dotenv(ROOT / ".env")


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to .env (see .env.example).")
    return value


def cursor_writer_model() -> str:
    return (
        os.getenv("CURSOR_WRITER_MODEL", os.getenv("CURSOR_MODEL", "gpt-5.6-sol")).strip()
        or "gpt-5.6-sol"
    )


def cursor_checker_model() -> str:
    return os.getenv("CURSOR_CHECKER_MODEL", "claude-opus-5").strip() or "claude-opus-5"


def notion_configured() -> bool:
    return bool(os.getenv("NOTION_TOKEN", "").strip() and os.getenv("NOTION_DATABASE_ID", "").strip())
