"""Detect already-completed applications so workers do not spend another LLM run."""

from __future__ import annotations

import hashlib
from pathlib import Path

from jobapps.config import OUTPUT_DIR, PROCESSED_DIR
from jobapps.models import Job, load_job


def normalize_portal_url(url: str) -> str:
    return url.strip().casefold().rstrip("/")


def job_fingerprint(job: Job) -> str:
    url = normalize_portal_url(job.portal_url)
    if url:
        return f"url:{url}"
    company = " ".join(job.company.casefold().split())
    title = " ".join(job.title.casefold().split())
    digest = hashlib.sha256(job.description.encode("utf-8")).hexdigest()[:16]
    return f"post:{company}|{title}|{digest}"


def _job_from_yaml(path: Path) -> Job | None:
    try:
        return load_job(path)
    except Exception:
        return None


def find_duplicate(job: Job, *, exclude_output: Path | None = None) -> Path | None:
    """Return a completed output or processed YAML path if this posting already shipped."""
    target = job_fingerprint(job)
    exclude = exclude_output.resolve() if exclude_output is not None else None

    if PROCESSED_DIR.is_dir():
        for path in sorted(PROCESSED_DIR.iterdir()):
            if path.suffix.lower() not in {".yaml", ".yml"}:
                continue
            previous = _job_from_yaml(path)
            if previous is not None and job_fingerprint(previous) == target:
                return path

    if OUTPUT_DIR.is_dir():
        for folder in sorted(OUTPUT_DIR.iterdir()):
            if not folder.is_dir():
                continue
            if exclude is not None and folder.resolve() == exclude:
                continue
            meta = folder / "meta" / "meta.yaml"
            job_yaml = folder / "inputs" / "job.yaml"
            if not meta.is_file() or not job_yaml.is_file():
                continue
            previous = _job_from_yaml(job_yaml)
            if previous is not None and job_fingerprint(previous) == target:
                return folder
    return None
