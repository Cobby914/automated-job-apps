"""Every application uses June 2028.

The owner is taking a 5th year, so internships, co-ops, new-grad roles,
full-time roles, and postings with no start date all use the same label.
"""

from __future__ import annotations

import re
from typing import TypeVar

from jobapps.models import CoverLetter, Education, GenerationResult, Job, Resume, TailoredResume

GRADUATION_JUNE_2028 = "June 2028"
ALLOWED_GRADUATION_DATES = frozenset({GRADUATION_JUNE_2028})

_GRAD_MENTION_RE = re.compile(
    r"\b(?:June|Dec(?:ember|\.)?)\s+202[78]\b",
    re.IGNORECASE,
)

T = TypeVar("T", Resume, TailoredResume)


def normalize_graduation_date(value: str) -> str | None:
    """Return June 2028, or None when the value is any other date."""
    key = value.strip().lower().replace(".", "")
    key = re.sub(r"\s+", " ", key)
    if key in {"june 2028", "jun 2028"}:
        return GRADUATION_JUNE_2028
    return None


def resolve_graduation_date(job: Job) -> str:
    """Return June 2028 for every role.

    An empty ``job.graduation`` is June 2028. Any other explicit value,
    including June 2027 and Dec. 2027, is rejected.
    """
    if job.graduation.strip():
        normalized = normalize_graduation_date(job.graduation)
        if normalized is None:
            allowed = ", ".join(sorted(ALLOWED_GRADUATION_DATES))
            raise ValueError(
                f"Unknown graduation {job.graduation!r}; use one of: {allowed}."
            )
        return normalized
    return GRADUATION_JUNE_2028


def set_education_year(education: list[Education], year: str) -> list[Education]:
    return [item.model_copy(update={"year": year}) for item in education]


def apply_graduation_to_resume(resume: T, year: str) -> T:
    return resume.model_copy(update={"education": set_education_year(resume.education, year)})


def rewrite_graduation_mentions(text: str, year: str) -> str:
    """Normalize June/Dec 2027 and 2028 mentions to the chosen graduation label."""
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
