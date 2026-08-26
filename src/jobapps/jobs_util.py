"""Shared helpers for discovering job YAML files."""

from __future__ import annotations

from pathlib import Path

from jobapps.config import JOBS_DIR


def is_job_file(path: Path) -> bool:
    if path.name.startswith(".") or path.name.startswith("_"):
        return False
    if path.name.endswith(".error.txt"):
        return False
    if path.parent.resolve() != JOBS_DIR.resolve():
        return False
    return path.suffix.lower() in {".yaml", ".yml"}


def list_job_files() -> list[Path]:
    if not JOBS_DIR.is_dir():
        return []
    return sorted(
        path for path in JOBS_DIR.iterdir() if path.is_file() and is_job_file(path)
    )
