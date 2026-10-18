"""Targeted LLM repairs for a single bullet, paragraph, letter, or answer."""

from __future__ import annotations

import json

from jobapps.career import ExperienceRecord, ProjectRecord
from jobapps.config import repair_model
from jobapps.llm import generate_structured
from jobapps.models import (
    ApplicationAnswer,
    CoverLetter,
    MAX_BULLET_CHARS,
    SourcedBullet,
    TextPayload,
)


def _dump(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _source_context(record: ExperienceRecord | ProjectRecord | None) -> str:
    if record is None:
        return "(no extra source record)"
    return _dump(record.prompt_payload())


def _text_repair(system: str, user: str, purpose: str) -> str:
    payload = generate_structured(
        system=system,
        user=user,
        model=repair_model(),
        schema=TextPayload,
        purpose=purpose,
    )
    return payload.text.strip()


def shorten_bullet(
    bullet: str,
    max_chars: int = MAX_BULLET_CHARS,
    record: ExperienceRecord | ProjectRecord | None = None,
) -> str:
    system = f"""\
Shorten this resume bullet to at most {max_chars} characters. Keep facts grounded \
in the source record. Do not invent. Prefer the same meaning with tighter wording. \
Do not change relative metrics into absolute ones or the reverse.
"""
    user = f"""\
Bullet ({len(bullet)} chars):
{bullet}

Source record:
{_source_context(record)}
"""
    return _text_repair(system, user, "bullet_shorten")


def rewrite_bullet(
    bullet: str,
    feedback: str,
    record: ExperienceRecord | ProjectRecord | None = None,
    max_chars: int = MAX_BULLET_CHARS,
) -> str:
    system = f"""\
Rewrite this one resume bullet to address the feedback. Keep it at most \
{max_chars} characters. Do not invent facts. Stay grounded in the source record. \
Preserve source-backed metrics as relative or absolute according to the record.
"""
    user = f"""\
Bullet:
{bullet}

Feedback:
{feedback}

Source record:
{_source_context(record)}
"""
    return _text_repair(system, user, "bullet_repair")


def rewrite_cover_letter_paragraph(
    paragraph: str,
    index: int,
    cover: CoverLetter,
    feedback: str,
    supporting: list[dict[str, object]] | None = None,
) -> str:
    system = f"""\
Rewrite paragraph {index} of this cover letter to address the feedback. Keep the \
same role in the letter. Do not invent facts. Do not rewrite other paragraphs.
"""
    user = f"""\
Current paragraph:
{paragraph}

Full letter:
{_dump(cover.model_dump())}

Feedback:
{feedback}

Supporting records:
{_dump(supporting or [])}
"""
    return _text_repair(system, user, "cover_paragraph_repair")


COVER_LETTER_TARGET_WORDS = (280, 300)


def cover_letter_word_count(cover: CoverLetter) -> int:
    return sum(len(paragraph.split()) for paragraph in cover.paragraphs)


def shorten_cover_letter(
    cover: CoverLetter,
    *,
    pages: int | None = None,
    overfull: list[str] | None = None,
    attempt: int = 1,
) -> CoverLetter:
    """Cut the letter so the closing and signature stay on page 1.

    Later attempts tighten the word target when the previous cut still overflowed.
    """
    low, high = COVER_LETTER_TARGET_WORDS
    current = cover_letter_word_count(cover)
    if attempt > 1:
        low = max(200, low - 30 * (attempt - 1))
        high = max(230, high - 30 * (attempt - 1))
    problems: list[str] = []
    if pages is not None and pages > 1:
        problems.append(
            f"The rendered PDF is {pages} pages; it must be exactly 1 page, and the "
            "closing line and signature must stay on page 1."
        )
    if overfull:
        problems.append(
            "Some lines overflow the right margin; avoid long unbreakable tokens "
            "(URLs, long hyphenated or slash-joined terms)."
        )
    problem_text = " ".join(problems) or "The letter does not fit on one page."
    system = f"""\
Shorten this cover letter so it fits on one page. {problem_text}
The body (all paragraphs combined) is currently {current} words. Cut it to \
{low}-{high} words total. Keep exactly 4 substantive paragraphs (never fewer); drop the \
weakest material rather than trimming every sentence. Tighten wording; do not \
invent facts; keep every remaining claim grounded in the current letter; keep \
the greeting and closing unchanged.
"""
    user = f"""\
Current letter:
{_dump(cover.model_dump())}
"""
    return generate_structured(
        system=system,
        user=user,
        model=repair_model(),
        schema=CoverLetter,
        purpose="cover_letter_shorten",
    )


def rewrite_answer(
    item: ApplicationAnswer,
    feedback: str,
    sources: list[dict[str, object]] | None = None,
) -> str:
    limit = (
        f"at most {item.max_length} characters"
        if item.max_length is not None
        else "no character limit"
    )
    system = f"""\
Rewrite this application-question answer. Stay grounded in the source records. \
Do not invent. The answer must be {limit}.
"""
    user = f"""\
Question:
{item.prompt}

Current answer ({len(item.answer)} chars):
{item.answer}

Feedback:
{feedback}

Source records:
{_dump(sources or [])}
"""
    return _text_repair(system, user, "answer_repair")


def replace_experience_bullet(
    draft_bullets: list[SourcedBullet],
    index: int,
    new_text: str,
) -> list[SourcedBullet]:
    updated = list(draft_bullets)
    current = updated[index]
    updated[index] = current.model_copy(update={"text": new_text})
    return updated
