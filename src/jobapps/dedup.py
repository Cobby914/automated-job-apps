"""Detect already-completed applications and similar postings worth reusing."""

from __future__ import annotations

import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from jobapps.config import OUTPUT_DIR, PROCESSED_DIR
from jobapps.models import ApplicationPlan, Job, load_job

_TRACKING_KEYS = frozenset(
    {
        "fbclid",
        "gclid",
        "gclsrc",
        "yclid",
        "msclkid",
        "igshid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
        "source",
    }
)
_NEAR_DUPLICATE_RATIO = 0.92
PLAN_REUSE_MIN_SIMILARITY = 0.65


def normalize_text(text: str) -> str:
    return " ".join((text or "").casefold().split())


def strip_tracking_params(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    kept = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        low = key.casefold()
        if low in _TRACKING_KEYS or low.startswith("utm_"):
            continue
        kept.append((key, value))
    cleaned = urlunsplit(
        (parts.scheme, parts.netloc, parts.path.rstrip("/"), urlencode(kept), "")
    )
    return cleaned.casefold()


def normalize_portal_url(url: str) -> str:
    cleaned = strip_tracking_params(url)
    return cleaned or url.strip().casefold().rstrip("/")


def job_fingerprint(job: Job) -> str:
    url = normalize_portal_url(job.portal_url)
    if url:
        return f"url:{url}"
    company = normalize_text(job.company)
    title = normalize_text(job.title)
    digest = hashlib.sha256(normalize_text(job.description).encode("utf-8")).hexdigest()[:16]
    return f"post:{company}|{title}|{digest}"


def description_similarity_ratio(left: str, right: str) -> float:
    a = normalize_text(left)
    b = normalize_text(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def descriptions_similar(left: str, right: str, threshold: float = _NEAR_DUPLICATE_RATIO) -> bool:
    return description_similarity_ratio(left, right) >= threshold


def _job_from_yaml(path: Path) -> Job | None:
    try:
        return load_job(path)
    except Exception:
        return None


def find_duplicate(job: Job, *, exclude_output: Path | None = None) -> Path | None:
    """Return a completed output or processed YAML path if this posting already shipped."""
    target = job_fingerprint(job)
    company = normalize_text(job.company)
    title = normalize_text(job.title)
    exclude = exclude_output.resolve() if exclude_output is not None else None

    if PROCESSED_DIR.is_dir():
        for path in sorted(PROCESSED_DIR.iterdir()):
            if path.suffix.lower() not in {".yaml", ".yml"}:
                continue
            previous = _job_from_yaml(path)
            if previous is None:
                continue
            if job_fingerprint(previous) == target:
                return path
            if (
                normalize_text(previous.company) == company
                and normalize_text(previous.title) == title
                and descriptions_similar(previous.description, job.description)
            ):
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
            if previous is None:
                continue
            if job_fingerprint(previous) == target:
                return folder
            if (
                normalize_text(previous.company) == company
                and normalize_text(previous.title) == title
                and descriptions_similar(previous.description, job.description)
            ):
                return folder
    return None


def find_reusable_plan(job: Job, *, exclude_output: Path | None = None) -> ApplicationPlan | None:
    """Reuse rankings from a same-company posting similar enough to share a plan."""
    company = normalize_text(job.company)
    title = normalize_text(job.title)
    exclude = exclude_output.resolve() if exclude_output is not None else None
    if not OUTPUT_DIR.is_dir():
        return None
    for folder in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        if not folder.is_dir():
            continue
        if exclude is not None and folder.resolve() == exclude:
            continue
        job_yaml = folder / "inputs" / "job.yaml"
        plan_path = folder / "meta" / "application_plan.json"
        if not job_yaml.is_file() or not plan_path.is_file():
            continue
        previous = _job_from_yaml(job_yaml)
        if previous is None:
            continue
        if normalize_text(previous.company) != company:
            continue
        if normalize_text(previous.title) != title:
            continue
        ratio = description_similarity_ratio(previous.description, job.description)
        if ratio < PLAN_REUSE_MIN_SIMILARITY:
            continue
        if ratio >= _NEAR_DUPLICATE_RATIO:
            continue
        try:
            return ApplicationPlan.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return None
