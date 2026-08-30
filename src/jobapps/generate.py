"""Cursor-powered resume, cover letter, and screening-answer generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions

from jobapps.career import CareerBank, ExperienceRecord, ProjectRecord
from jobapps.config import (
    COVER_LETTER_EXAMPLE_PATH,
    ROOT,
    cursor_checker_model,
    cursor_escalation_model,
    cursor_writer_model,
    require_env,
)
from jobapps.models import (
    ApplicationAnswer,
    ApplicationAnswersResult,
    ApplicationPlan,
    CoverLetter,
    DraftResume,
    Job,
    ReviewResult,
    TailoredResume,
    overlong_answer_issues,
)
from jobapps.plan import selected_experiences, selected_projects

RESUME_WRITER_PROMPT = """\
You tailor a resume for one job from an ApplicationPlan and selected career records.

Rules:
- Never invent employers, job titles, dates, degrees, schools, projects, or skills.
- Use only the selected experiences and projects provided. Do not add others.
- Copy the provided skill_groups as-is. Do not add, invent, or rename skill items.
- Leave resume.summary empty.
- Keep education exactly as provided (including the graduation year).
- Every experience must have {min_exp_bullets}-{max_exp_bullets} content bullets.
- Every project must have exactly {proj_bullets} content bullets plus a trailing \
"Stack: ..." line using the provided stack.
- Each content bullet must be one printed line of {min_chars}-{max_chars} characters. \
Prefer filling the line when facts allow; never exceed {max_chars}.
- Start from the canonical bullets and expand with grounded tech and impact from \
facts/metrics. Attach source ids on every bullet.
- Write Jake Gutierrez-style lines: what + how (tech) + impact.
- Do not mention that you are an AI.
- Return JSON only matching the schema. No markdown, no code fences, no commentary.
- Do not edit files, run commands, or use tools.
"""

COVER_LETTER_WRITER_PROMPT = """\
You write a cover letter for one job using a finalized resume and a few supporting records.

Rules:
- Never invent employers, titles, dates, degrees, schools, projects, skills, or outcomes.
- Ground every claim in the finalized resume or the supporting records.
- First person, specific, no cliches, not a rewrite of the resume.
- Write 4-6 substantive paragraphs that fill most of one page.
- Structure: connection to the role/company; strongest relevant experience; second \
technical experience or project; optional research/project depth; company-specific close.
- Greeting should be "Dear Hiring Manager," unless the job text names a person.
- Closing should be "Sincerely,"
- Match the example's length and tone, not its company-specific content.
- Use the exact education year from the resume. Do not change June 2027 vs Dec. 2027.
- Do not mention that you are an AI.
- Return JSON only matching the schema. No markdown, no code fences, no commentary.
- Do not edit files, run commands, or use tools.
"""

CHECKER_PROMPT = """\
You review a tailored resume and optional cover letter for grounding, relevance, and voice.

Python already checked bullet length, skill whitelist, bullet counts, required sections, \
Stack lines, and education fields. Do not reject for those mechanical issues.

Approve only if all of these are true:
- No invented employers, titles, dates, degrees, schools, projects, skills, or outcomes.
- Every claim is grounded in the selected career records or the finalized resume.
- Experiences and projects are relevant to the job.
- Resume bullets read like a strong human resume (specific tech + impact).
- If a cover letter is present: first person, specific, not a resume restatement, \
not generic, 4-6 paragraphs of real substance.

Return JSON only matching the schema. No markdown, no code fences, no commentary.
Do not edit files, run commands, or use tools.
"""

ANSWERS_WRITER_PROMPT = """\
You write answers to optional job-application / screening questions for one role.

Rules:
- Never invent employers, titles, dates, degrees, schools, projects, skills, or outcomes.
- Ground every claim in the selected career records (summaries, facts, metrics, bullets).
- Choose the best-matching experiences and projects for each question and the job.
- If a question asks for N experiences or projects, use exactly that many when enough \
grounded material exists.
- Match a clear, specific, technical first-person voice.
- Respect each question's max_length (character count of the answer string). If \
max_length is null/omitted, there is no limit.
- Use the character budget: short limits stay tight. Larger limits should add more \
grounded experiences and projects, not fluff.
- Answer every question in the same order. Copy each prompt and max_length into the output.
- Do not mention that you are an AI.
- Return JSON only matching the schema. No markdown, no code fences, no commentary.
- Do not edit files, run commands, or use tools.
"""

ANSWERS_CHECKER_PROMPT = """\
You are reviewing application-question answers before they are submitted.

Python already checked character limits. Do not reject only for length.

Approve only if all of these are true:
- No invented employers, titles, dates, degrees, schools, projects, skills, or outcomes.
- Every fact is grounded in the selected career records.
- Experiences/projects chosen are relevant to the job and the question.
- When a question asks for a specific number of experiences or projects, the answer \
uses that count when enough grounded material exists.
- Tone is first-person, specific, and appropriate for a real application.

Return JSON only matching the schema. No markdown, no code fences, no commentary.
Do not edit files, run commands, or use tools.
"""


def _dump(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def extract_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        if lines and lines[0].strip().lower() in {"json", "jsonc"}:
            lines = lines[1:]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("Cursor did not return JSON.")
    return cleaned[start : end + 1]


def run_prompt(prompt: str, model: str) -> str:
    try:
        result = Agent.prompt(
            prompt,
            AgentOptions(
                api_key=require_env("CURSOR_API_KEY"),
                model=model,
                local=LocalAgentOptions(cwd=str(ROOT)),
                tools=[],
            ),
        )
    except CursorAgentError as error:
        raise RuntimeError(f"Cursor agent failed to start ({model}): {error}") from error

    if getattr(result, "status", None) == "error":
        detail = getattr(result, "result", None) or result.status
        raise RuntimeError(f"Cursor run failed ({model}): {detail}")

    text = getattr(result, "result", None)
    if not text or not str(text).strip():
        raise RuntimeError(f"Cursor returned no text ({model}).")
    return str(text)


def _parse_draft_resume(text: str) -> DraftResume:
    try:
        return DraftResume.model_validate_json(extract_json(text))
    except Exception as error:
        raise RuntimeError(f"Writer returned invalid resume JSON: {error}") from error


def _parse_cover_letter(text: str) -> CoverLetter:
    try:
        return CoverLetter.model_validate_json(extract_json(text))
    except Exception as error:
        raise RuntimeError(f"Writer returned invalid cover letter JSON: {error}") from error


def _parse_review(text: str) -> ReviewResult:
    try:
        return ReviewResult.model_validate_json(extract_json(text))
    except Exception as error:
        raise RuntimeError(f"Checker returned invalid review JSON: {error}") from error


def _parse_answers(text: str) -> ApplicationAnswersResult:
    try:
        return ApplicationAnswersResult.model_validate_json(extract_json(text))
    except Exception as error:
        raise RuntimeError(f"Writer returned invalid application answers JSON: {error}") from error


def load_cover_letter_example(path: Path | None = None) -> str:
    target = path or COVER_LETTER_EXAMPLE_PATH
    if not target.is_file():
        return ""
    return target.read_text(encoding="utf-8").strip()


def _job_header(job: Job) -> str:
    return f"""\
## Job
Company: {job.company}
Title: {job.title}
Portal: {job.portal_url or "(none)"}
Notes: {job.notes or "(none)"}
Starts: {job.starts or "(infer from description)"}

### Description
{job.description}
"""


def _records_payload(
    experiences: list[ExperienceRecord],
    projects: list[ProjectRecord],
) -> dict[str, object]:
    return {
        "experiences": [item.prompt_payload() for item in experiences],
        "projects": [item.prompt_payload() for item in projects],
    }


def generate_resume_draft(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
) -> tuple[DraftResume, int]:
    """Write a tailored resume from the plan. Returns (draft, context_chars)."""
    experiences = selected_experiences(plan, bank)
    projects = selected_projects(plan, bank)
    layout = plan.layout
    prompt_body = f"""\
{RESUME_WRITER_PROMPT.format(
    min_exp_bullets=layout.min_experience_bullets,
    max_exp_bullets=layout.max_experience_bullets,
    proj_bullets=layout.project_bullets,
    min_chars=layout.min_bullet_chars,
    max_chars=layout.max_bullet_chars,
)}

JSON schema:
{_dump(DraftResume.model_json_schema())}

{_job_header(job)}

## ApplicationPlan
{_dump(plan.model_dump())}

## Selected career records (only these may appear)
{_dump(_records_payload(experiences, projects))}

## Education (copy exactly, including year)
{_dump([item.model_dump() for item in bank.profile.education])}

## Approved skills (copy as-is; do not invent names)
{_dump([group.model_dump() for group in plan.skill_groups])}
"""
    draft = _parse_draft_resume(run_prompt(prompt_body, cursor_writer_model()))
    if not draft.skills:
        draft = draft.model_copy(update={"skills": list(plan.skill_groups)})
    if not draft.education:
        draft = draft.model_copy(update={"education": list(bank.profile.education)})
    return draft, len(prompt_body)


def generate_cover_letter_draft(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
    resume: TailoredResume,
) -> CoverLetter:
    source_ids = plan.cover_letter_source_ids
    experiences = [
        item for item in selected_experiences(plan, bank) if item.id in source_ids
    ]
    projects = [item for item in selected_projects(plan, bank) if item.id in source_ids]
    # Preserve plan order for supporting records.
    ordered: list[ExperienceRecord | ProjectRecord] = []
    lookup = {item.id: item for item in [*experiences, *projects]}
    for record_id in source_ids:
        if record_id in lookup:
            ordered.append(lookup[record_id])
    supporting = [item.prompt_payload() for item in ordered]
    example = load_cover_letter_example()
    prompt_body = f"""\
{COVER_LETTER_WRITER_PROMPT}

JSON schema:
{_dump(CoverLetter.model_json_schema())}

{_job_header(job)}

## Finalized resume
{_dump(resume.model_dump())}

## Supporting experiences/projects
{_dump(supporting)}

## Cover letter example (structure, length, and tone only)
{example or "(none — use a professional first-person cover letter format)"}
"""
    return _parse_cover_letter(run_prompt(prompt_body, cursor_writer_model()))


def _review_prompt(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
    resume: TailoredResume,
    cover: CoverLetter | None,
) -> str:
    experiences = selected_experiences(plan, bank)
    projects = selected_projects(plan, bank)
    cover_block = _dump(cover.model_dump()) if cover is not None else "(cover letter skipped)"
    return f"""\
{CHECKER_PROMPT}

JSON schema:
{_dump(ReviewResult.model_json_schema())}

{_job_header(job)}

## ApplicationPlan
{_dump(plan.model_dump())}

## Selected career records
{_dump(_records_payload(experiences, projects))}

## Draft resume
{_dump(resume.model_dump())}

## Draft cover letter
{cover_block}
"""


def review_materials(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
    resume: TailoredResume,
    cover: CoverLetter | None,
) -> tuple[ReviewResult, bool, str]:
    """Sonnet review; escalate to Opus only when Sonnet flags an issue.

    Returns (review, escalated, model_used).
    """
    prompt = _review_prompt(job, plan, bank, resume, cover)
    checker = cursor_checker_model()
    review = _parse_review(run_prompt(prompt, checker))
    if review.approved and not review.issues:
        return review, False, checker
    escalation = cursor_escalation_model()
    review = _parse_review(run_prompt(prompt, escalation))
    return review, True, escalation


@dataclass
class GeneratedAnswers:
    answers: ApplicationAnswersResult
    review: ReviewResult
    writer_model: str
    checker_model: str
    revised: bool
    escalated: bool = False


def _align_answers(job: Job, result: ApplicationAnswersResult) -> ApplicationAnswersResult:
    aligned: list[ApplicationAnswer] = []
    for index, question in enumerate(job.questions):
        if index < len(result.answers):
            item = result.answers[index]
            aligned.append(
                item.model_copy(
                    update={"prompt": question.prompt, "max_length": question.max_length}
                )
            )
        else:
            aligned.append(
                ApplicationAnswer(
                    prompt=question.prompt,
                    answer="",
                    max_length=question.max_length,
                )
            )
    return ApplicationAnswersResult(answers=aligned)


def _answers_context(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
) -> str:
    experiences = selected_experiences(plan, bank)
    projects = selected_projects(plan, bank)
    questions_payload = [
        {"prompt": q.prompt, "max_length": q.max_length} for q in job.questions
    ]
    return f"""\
{_job_header(job)}

## Selected career records (summaries, facts, metrics, bullets)
{_dump(_records_payload(experiences, projects))}

## Questions to answer (preserve order, prompt, and max_length)
{_dump(questions_payload)}
"""


def _write_answers_draft(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
    extra: str = "",
) -> ApplicationAnswersResult:
    prompt = f"""\
{ANSWERS_WRITER_PROMPT}

JSON schema:
{_dump(ApplicationAnswersResult.model_json_schema())}

{_answers_context(job, plan, bank)}
{extra}
Return answers as one JSON object matching the schema.
"""
    draft = _parse_answers(run_prompt(prompt, cursor_writer_model()))
    return _align_answers(job, draft)


def _review_answers(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
    draft: ApplicationAnswersResult,
) -> tuple[ReviewResult, bool, str]:
    prompt = f"""\
{ANSWERS_CHECKER_PROMPT}

JSON schema:
{_dump(ReviewResult.model_json_schema())}

{_answers_context(job, plan, bank)}

## Draft answers to review
{_dump(draft.model_dump())}
"""
    checker = cursor_checker_model()
    review = _parse_review(run_prompt(prompt, checker))
    if review.approved and not review.issues:
        return review, False, checker
    escalation = cursor_escalation_model()
    review = _parse_review(run_prompt(prompt, escalation))
    return review, True, escalation


def generate_answers(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
) -> GeneratedAnswers:
    if not job.questions:
        raise ValueError("generate_answers called with no questions on the job.")

    draft = _write_answers_draft(job, plan, bank)
    review, escalated, checker_model = _review_answers(job, plan, bank, draft)
    revised = False
    length_issues = overlong_answer_issues(draft)
    if not review.approved or length_issues:
        issues = list(review.issues) if review.issues else []
        if not review.approved and not issues:
            issues.append(review.summary or "The answers did not look strong enough.")
        issues.extend(length_issues)
        extra = f"""
## Checker / length feedback — revise the answers to address every issue
{_dump({"summary": review.summary, "issues": issues})}

Respect each question's max_length. Keep every fact grounded in the selected records.
Return a full replacement JSON object.
"""
        draft = _write_answers_draft(job, plan, bank, extra=extra)
        review, escalated, checker_model = _review_answers(job, plan, bank, draft)
        revised = True

    return GeneratedAnswers(
        answers=draft,
        review=review,
        writer_model=cursor_writer_model(),
        checker_model=checker_model,
        revised=revised,
        escalated=escalated,
    )
