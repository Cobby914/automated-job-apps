"""Pick June 2027 vs Dec. 2027 from when the role starts."""

from __future__ import annotations

import re
from typing import TypeVar

from jobapps.models import CoverLetter, Education, GenerationResult, Job, Resume, TailoredResume

GRADUATION_JUNE = "June 2027"
GRADUATION_DEC = "Dec. 2027"
ALLOWED_GRADUATION_DATES = frozenset({GRADUATION_JUNE, GRADUATION_DEC})

# Roles that begin on or before this month in 2027 use June; later starts use Dec.
_CUTOFF_YEAR = 2027
_CUTOFF_MONTH = 6

_MONTH_NAMES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_SEASON_START_MONTH = {
    "spring": 3,
    "summer": 6,
    "fall": 9,
    "autumn": 9,
    "winter": 1,
}

_SEASON_RE = re.compile(
    r"\b(spring|summer|fall|autumn|winter)\s+(20\d{2})\b",
    re.IGNORECASE,
)
_MONTH_YEAR_RE = re.compile(
    r"\b("
    + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True))
    + r")\.?\s+(20\d{2})\b",
    re.IGNORECASE,
)
_ISO_RE = re.compile(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b")
_GRAD_MENTION_RE = re.compile(
    r"\b(?:June|Dec(?:ember|\.)?)\s+2027\b",
    re.IGNORECASE,
)

T = TypeVar("T", Resume, TailoredResume)


def normalize_graduation_date(value: str) -> str | None:
    """Return a canonical graduation label, or None if unrecognized."""
    key = value.strip().lower().replace(".", "")
    key = re.sub(r"\s+", " ", key)
    if key in {"june 2027", "jun 2027"}:
        return GRADUATION_JUNE
    if key in {"dec 2027", "december 2027"}:
        return GRADUATION_DEC
    return None


def _graduation_for_start(year: int, month: int) -> str:
    if (year, month) <= (_CUTOFF_YEAR, _CUTOFF_MONTH):
        return GRADUATION_JUNE
    return GRADUATION_DEC


def _parse_start_token(text: str) -> tuple[int, int] | None:
    """Extract the first role-start (year, month) hint from free text."""
    if not text or not text.strip():
        return None

    season = _SEASON_RE.search(text)
    if season:
        season_name = season.group(1).lower()
        year = int(season.group(2))
        month = _SEASON_START_MONTH[season_name]
        # "Winter 2027" is treated as early 2027 (Jan); winter after a year
        # label like "Winter 2028" falls after the June cutoff via the year.
        return year, month

    month_year = _MONTH_YEAR_RE.search(text)
    if month_year:
        month = _MONTH_NAMES[month_year.group(1).lower().rstrip(".")]
        year = int(month_year.group(2))
        return year, month

    iso = _ISO_RE.search(text)
    if iso:
        return int(iso.group(1)), int(iso.group(2))

    return None


def infer_role_start(job: Job) -> tuple[int, int] | None:
    """Best-effort start month from explicit starts, then notes/title/description."""
    for blob in (job.starts, job.notes, job.title, job.description):
        parsed = _parse_start_token(blob)
        if parsed:
            return parsed
    return None


def resolve_graduation_date(job: Job) -> str:
    """Choose June 2027 or Dec. 2027 for this application.

    Priority:
    1. Explicit job.graduation override
    2. Role start from job.starts, else notes/title/description
    3. Default June 2027 when the start date is unknown
    """
    if job.graduation.strip():
        normalized = normalize_graduation_date(job.graduation)
        if normalized is None:
            allowed = ", ".join(sorted(ALLOWED_GRADUATION_DATES))
            raise ValueError(
                f"Unknown graduation {job.graduation!r}; use one of: {allowed}."
            )
        return normalized

    start = infer_role_start(job)
    if start is None:
        return GRADUATION_JUNE
    return _graduation_for_start(*start)


def set_education_year(education: list[Education], year: str) -> list[Education]:
    return [item.model_copy(update={"year": year}) for item in education]


def apply_graduation_to_resume(resume: T, year: str) -> T:
    return resume.model_copy(update={"education": set_education_year(resume.education, year)})


def rewrite_graduation_mentions(text: str, year: str) -> str:
    """Normalize June/Dec/December 2027 mentions to the chosen graduation label."""
    if not text:
        return text
    return _GRAD_MENTION_RE.sub(year, text)


def apply_graduation_to_cover_letter(cover: CoverLetter, year: str) -> CoverLetter:
    return cover.model_copy(
        update={
            "paragraphs": [rewrite_graduation_mentions(p, year) for p in cover.paragraphs],
        }
    )


def apply_graduation(materials: GenerationResult, year: str) -> GenerationResult:
    update: dict[str, object] = {
        "resume": apply_graduation_to_resume(materials.resume, year),
    }
    if materials.cover_letter is not None:
        update["cover_letter"] = apply_graduation_to_cover_letter(materials.cover_letter, year)
    return materials.model_copy(update=update)
