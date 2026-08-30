"""Targeted LLM repairs for a single bullet, paragraph, letter, or answer."""

from __future__ import annotations

import json

from jobapps.career import ExperienceRecord, ProjectRecord
from jobapps.config import cursor_writer_model
from jobapps.generate import extract_json, run_prompt
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


def shorten_bullet(
    bullet: str,
    max_chars: int = MAX_BULLET_CHARS,
    record: ExperienceRecord | ProjectRecord | None = None,
) -> str:
    prompt = f"""\
Shorten this resume bullet to at most {max_chars} characters. Keep facts grounded \
in the source record. Do not invent. Prefer the same meaning with tighter wording.

JSON schema: {{"text": "string"}}
Return JSON only. No markdown, no code fences.

Bullet ({len(bullet)} chars):
{bullet}

Source record:
{_source_context(record)}
"""
    text = _parse_text_payload(run_prompt(prompt, cursor_writer_model()))
    return text


def rewrite_bullet(
    bullet: str,
    feedback: str,
    record: ExperienceRecord | ProjectRecord | None = None,
    max_chars: int = MAX_BULLET_CHARS,
) -> str:
    prompt = f"""\
Rewrite this one resume bullet to address the feedback. Keep it at most \
{max_chars} characters. Do not invent facts. Stay grounded in the source record.

JSON schema: {{"text": "string"}}
Return JSON only. No markdown, no code fences.

Bullet:
{bullet}

Feedback:
{feedback}

Source record:
{_source_context(record)}
"""
    return _parse_text_payload(run_prompt(prompt, cursor_writer_model()))


def rewrite_cover_letter_paragraph(
    paragraph: str,
    index: int,
    cover: CoverLetter,
    feedback: str,
    supporting: list[dict[str, object]] | None = None,
) -> str:
    prompt = f"""\
Rewrite paragraph {index} of this cover letter to address the feedback. Keep the \
same role in the letter. Do not invent facts. Do not rewrite other paragraphs.

JSON schema: {{"text": "string"}}
Return JSON only. No markdown, no code fences.

Current paragraph:
{paragraph}

Full letter:
{_dump(cover.model_dump())}

Feedback:
{feedback}

Supporting records:
{_dump(supporting or [])}
"""
    return _parse_text_payload(run_prompt(prompt, cursor_writer_model()))


def shorten_cover_letter(cover: CoverLetter) -> CoverLetter:
    prompt = f"""\
Shorten this cover letter so it fits on one page. Keep 4-6 substantive paragraphs. \
Tighten wording; do not invent facts; do not drop below 4 paragraphs if the current \
letter has 4 or more.

JSON schema:
{_dump(CoverLetter.model_json_schema())}
Return JSON only. No markdown, no code fences.

Current letter:
{_dump(cover.model_dump())}
"""
    try:
        return CoverLetter.model_validate_json(extract_json(run_prompt(prompt, cursor_writer_model())))
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
    prompt = f"""\
Rewrite this application-question answer. Stay grounded in the source records. \
Do not invent. The answer must be {limit}.

JSON schema: {{"text": "string"}}
Return JSON only. No markdown, no code fences.

Question:
{item.prompt}

Current answer ({len(item.answer)} chars):
{item.answer}

Feedback:
{feedback}

Source records:
{_dump(sources or [])}
"""
    return _parse_text_payload(run_prompt(prompt, cursor_writer_model()))


def replace_experience_bullet(
    draft_bullets: list[SourcedBullet],
    index: int,
    new_text: str,
) -> list[SourcedBullet]:
    updated = list(draft_bullets)
    current = updated[index]
    updated[index] = current.model_copy(update={"text": new_text})
    return updated
