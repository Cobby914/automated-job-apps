"""Targeted LLM repairs for a single bullet, paragraph, letter, or answer."""

from __future__ import annotations

import json

from jobapps.career import ExperienceRecord, ProjectRecord
from jobapps.config import writer_model
from jobapps.llm import extract_json, generate_structured, generate_text
from jobapps.models import (
    ApplicationAnswer,
    CoverLetter,
    MAX_BULLET_CHARS,
    SourcedBullet,
)


def _dump(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _parse_text_payload(raw: str) -> str:
    data = json.loads(extract_json(raw))
    if isinstance(data, dict) and "text" in data:
        return str(data["text"]).strip()
    if isinstance(data, dict) and "answer" in data:
        return str(data["answer"]).strip()
    raise RuntimeError("Repair did not return a text payload.")


def _source_context(record: ExperienceRecord | ProjectRecord | None) -> str:
    if record is None:
        return "(no extra source record)"
    return _dump(record.prompt_payload())


def _repair_complete(system: str, user: str) -> str:
    return generate_text(system=system, user=user, model=writer_model())


def shorten_bullet(
    bullet: str,
    max_chars: int = MAX_BULLET_CHARS,
    record: ExperienceRecord | ProjectRecord | None = None,
) -> str:
    system = f"""\
Shorten this resume bullet to at most {max_chars} characters. Keep facts grounded \
in the source record. Do not invent. Prefer the same meaning with tighter wording.

JSON schema: {{"text": "string"}}
Return JSON only. No markdown, no code fences.
"""
    user = f"""\
Bullet ({len(bullet)} chars):
{bullet}

Source record:
{_source_context(record)}
"""
    return _parse_text_payload(_repair_complete(system, user))


def rewrite_bullet(
    bullet: str,
    feedback: str,
    record: ExperienceRecord | ProjectRecord | None = None,
    max_chars: int = MAX_BULLET_CHARS,
) -> str:
    system = f"""\
Rewrite this one resume bullet to address the feedback. Keep it at most \
{max_chars} characters. Do not invent facts. Stay grounded in the source record.

JSON schema: {{"text": "string"}}
Return JSON only. No markdown, no code fences.
"""
    user = f"""\
Bullet:
{bullet}

Feedback:
{feedback}

Source record:
{_source_context(record)}
"""
    return _parse_text_payload(_repair_complete(system, user))


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

JSON schema: {{"text": "string"}}
Return JSON only. No markdown, no code fences.
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
    return _parse_text_payload(_repair_complete(system, user))


def shorten_cover_letter(cover: CoverLetter) -> CoverLetter:
    system = f"""\
Shorten this cover letter so it fits on one page. Keep 4-6 substantive paragraphs. \
Tighten wording; do not invent facts; do not drop below 4 paragraphs if the current \
letter has 4 or more.

JSON schema:
{_dump(CoverLetter.model_json_schema())}
Return JSON only. No markdown, no code fences.
"""
    user = f"""\
Current letter:
{_dump(cover.model_dump())}
"""
    try:
        return generate_structured(
            system=system, user=user, model=writer_model(), schema=CoverLetter
        )
    except Exception as error:
        raise RuntimeError(f"Cover letter shorten returned invalid JSON: {error}") from error


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

JSON schema: {{"text": "string"}}
Return JSON only. No markdown, no code fences.
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
    return _parse_text_payload(_repair_complete(system, user))


def replace_experience_bullet(
    draft_bullets: list[SourcedBullet],
    index: int,
    new_text: str,
) -> list[SourcedBullet]:
    updated = list(draft_bullets)
    current = updated[index]
    updated[index] = current.model_copy(update={"text": new_text})
    return updated
