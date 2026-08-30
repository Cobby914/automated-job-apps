"""Resume, cover letter, and screening-answer generation. Provider-agnostic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from jobapps.career import CareerBank, ExperienceRecord, ProjectRecord
from jobapps.config import (
    COVER_LETTER_EXAMPLE_PATH,
    checker_model,
    escalation_model,
    writer_model,
)
from jobapps.llm import generate_structured, review as llm_review
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
You tailor a resume for one job from an ApplicationPlan and the career bank.

Rules:
- Never invent employers, job titles, dates, degrees, schools, projects, or skills.
- Use only experiences and projects whose ids appear in ApplicationPlan. \
Do not add others.
- Copy the provided skill_groups as-is. Do not add, invent, or rename skill items.
- Leave resume.summary empty.
- Keep education exactly as provided (including the graduation year).
- Every experience must have {min_exp_bullets}-{max_exp_bullets} content bullets.
- Every project must have exactly {proj_bullets} content bullets plus a trailing \
"Stack: ..." line using the provided stack.
- Each content bullet must be one printed line of {min_chars}-{max_chars} characters. \
Prefer filling the line when facts allow; never exceed {max_chars}.
- Start from the canonical bullets and expand with grounded tech and impact from \
facts/metrics. Attach at least one valid fact/metric source id on every content bullet.
- Cite sources that belong to that experience or project only. Unknown ids are invalid.
- Numeric claims must match a cited metric. Relative vs absolute improvements are \
labeled on each metric (kind: relative, absolute, or count). Do not convert one into \
the other.
- Write Jake Gutierrez-style lines: what + how (tech) + impact.
- Do not mention that you are an AI.
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
- Do not edit files, run commands, or use tools.
"""

CHECKER_PROMPT = """\
You review a tailored resume and optional cover letter for grounding, relevance, and voice.

Python already checked bullet length, skill whitelist, bullet counts, required sections, \
Stack lines, education fields, and source ids. Do not reject for those mechanical issues.

Approve only if all of these are true:
- No invented employers, titles, dates, degrees, schools, projects, skills, or outcomes.
- Every claim is grounded in the selected career records or the finalized resume.
- Experiences and projects are relevant to the job.
- Resume bullets read like a strong human resume (specific tech + impact).
- Relative vs absolute metrics are not restated as the other kind.
- If a cover letter is present: first person, specific, not a resume restatement, \
not generic, 4-6 paragraphs of real substance.

Every issue must be an object with:
- type: ungrounded | invention | generic | irrelevant | voice | unsupported_claim | other
- section: experience | project | cover_letter | resume | answers
- item_id: career record id such as uci-scalesense (empty for cover_letter/resume)
- bullet_index: 0-based content bullet index when section is experience or project
- paragraph_index: 0-based when section is cover_letter
- location: experience[i].bullets[j], projects[i].bullets[j], cover_letter.paragraphs[k], \
or resume (fallback if item_id is unknown)
- code: same as type
- message: one specific sentence
- severity: error or warning
Identify experiences and projects by career record id, not only array index.
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

Every issue must be an object with:
- type: ungrounded | invention | generic | irrelevant | voice | unsupported_claim | other
- section: answers
- item_id: empty
- paragraph_index: omitted
- location: answers[i]
- code: same as type
- message: one specific sentence
- severity: error or warning
Do not edit files, run commands, or use tools.
"""


def _dump(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


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


def career_prompt_block(bank: CareerBank) -> str:
    """Stable career payload for prompt-prefix caching."""
    return _dump(
        {
            "experiences": [item.prompt_payload() for item in bank.experiences],
            "projects": [item.prompt_payload() for item in bank.projects],
            "education": [item.model_dump() for item in bank.profile.education],
        }
    )


def generate_resume_draft(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
) -> tuple[DraftResume, int]:
    """Write a tailored resume from the plan. Returns (draft, context_chars)."""
    layout = plan.layout
    system = f"""\
{RESUME_WRITER_PROMPT.format(
    min_exp_bullets=layout.min_experience_bullets,
    max_exp_bullets=layout.max_experience_bullets,
    proj_bullets=layout.project_bullets,
    min_chars=layout.min_bullet_chars,
    max_chars=layout.max_bullet_chars,
)}

## Career bank (source of truth; use only ids listed in the ApplicationPlan)
{career_prompt_block(bank)}
"""
    user = f"""\
{_job_header(job)}

## ApplicationPlan
{_dump(plan.model_dump())}

## Approved skills (copy as-is; do not invent names)
{_dump([group.model_dump() for group in plan.skill_groups])}
"""
    draft = generate_structured(
        system=system,
        user=user,
        model=writer_model(),
        schema=DraftResume,
        purpose="resume_write",
    )
    if not draft.skills:
        draft = draft.model_copy(update={"skills": list(plan.skill_groups)})
    if not draft.education:
        draft = draft.model_copy(update={"education": list(bank.profile.education)})
    return draft, len(system) + len(user)


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
    ordered: list[ExperienceRecord | ProjectRecord] = []
    lookup = {item.id: item for item in [*experiences, *projects]}
    for record_id in source_ids:
        if record_id in lookup:
            ordered.append(lookup[record_id])
    supporting = [item.prompt_payload() for item in ordered]
    example = load_cover_letter_example()
    system = f"""\
{COVER_LETTER_WRITER_PROMPT}

## Cover letter example (structure, length, and tone only)
{example or "(none — use a professional first-person cover letter format)"}

## Career bank excerpts available as supporting records
{_dump(supporting)}
"""
    user = f"""\
{_job_header(job)}

## Finalized resume
{_dump(resume.model_dump())}
"""
    return generate_structured(
        system=system,
        user=user,
        model=writer_model(),
        schema=CoverLetter,
        purpose="cover_letter_write",
    )


def _review_messages(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
    resume: TailoredResume,
    cover: CoverLetter | None,
) -> tuple[str, str]:
    cover_block = _dump(cover.model_dump()) if cover is not None else "(cover letter skipped)"
    system = f"""\
{CHECKER_PROMPT}

## Career bank (source of truth)
{career_prompt_block(bank)}
"""
    user = f"""\
{_job_header(job)}

## ApplicationPlan
{_dump(plan.model_dump())}

## Draft resume
{_dump(resume.model_dump())}

## Draft cover letter
{cover_block}
"""
    return system, user


def review_materials(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
    resume: TailoredResume,
    cover: CoverLetter | None,
) -> tuple[ReviewResult, bool, str]:
    """Cheap reviewer first; escalate only when something is flagged.

    Returns (review, escalated, model_used).
    """
    system, user = _review_messages(job, plan, bank, resume, cover)
    checker = checker_model()
    review = llm_review(
        system=system, user=user, model=checker, schema=ReviewResult, purpose="review"
    )
    if review.approved and not review.issues:
        return review, False, checker
    escalation = escalation_model()
    review = llm_review(
        system=system,
        user=user,
        model=escalation,
        schema=ReviewResult,
        purpose="review_escalation",
    )
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


def _answers_user(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
    extra: str = "",
) -> str:
    questions_payload = [
        {"prompt": q.prompt, "max_length": q.max_length} for q in job.questions
    ]
    return f"""\
{_job_header(job)}

## ApplicationPlan
{_dump(plan.model_dump())}

## Questions to answer (preserve order, prompt, and max_length)
{_dump(questions_payload)}
{extra}
"""


def _write_answers_draft(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
    extra: str = "",
) -> ApplicationAnswersResult:
    system = f"""\
{ANSWERS_WRITER_PROMPT}

## Career bank (source of truth; use only ids listed in the ApplicationPlan)
{career_prompt_block(bank)}
"""
    user = _answers_user(job, plan, bank, extra)
    draft = generate_structured(
        system=system,
        user=user,
        model=writer_model(),
        schema=ApplicationAnswersResult,
        purpose="answers_write",
    )
    return _align_answers(job, draft)


def _review_answers(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
    draft: ApplicationAnswersResult,
) -> tuple[ReviewResult, bool, str]:
    system = f"""\
{ANSWERS_CHECKER_PROMPT}

## Career bank (source of truth)
{career_prompt_block(bank)}
"""
    user = f"""\
{_job_header(job)}

## ApplicationPlan
{_dump(plan.model_dump())}

## Draft answers to review
{_dump(draft.model_dump())}
"""
    checker = checker_model()
    review = llm_review(
        system=system, user=user, model=checker, schema=ReviewResult, purpose="answers_review"
    )
    if review.approved and not review.issues:
        return review, False, checker
    escalation = escalation_model()
    review = llm_review(
        system=system,
        user=user,
        model=escalation,
        schema=ReviewResult,
        purpose="answers_review_escalation",
    )
    return review, True, escalation


def generate_answers(
    job: Job,
    plan: ApplicationPlan,
    bank: CareerBank,
) -> GeneratedAnswers:
    if not job.questions:
        raise ValueError("generate_answers called with no questions on the job.")

    draft = _write_answers_draft(job, plan, bank)
    review, escalated, used_checker = _review_answers(job, plan, bank, draft)
    revised = False
    length_issues = overlong_answer_issues(draft)
    if not review.approved or length_issues:
        issue_payload = [item.model_dump() for item in review.issues]
        if not review.approved and not issue_payload:
            issue_payload.append(
                {
                    "location": "answers[0]",
                    "type": "other",
                    "section": "answers",
                    "code": "other",
                    "message": review.summary or "The answers did not look strong enough.",
                    "severity": "error",
                }
            )
        for length_issue in length_issues:
            issue_payload.append(
                {
                    "location": "answers",
                    "type": "other",
                    "section": "answers",
                    "code": "other",
                    "message": length_issue,
                    "severity": "error",
                }
            )
        extra = f"""
## Checker / length feedback — revise the answers to address every issue
{_dump({"summary": review.summary, "issues": issue_payload})}

Respect each question's max_length. Keep every fact grounded in the selected records.
"""
        draft = _write_answers_draft(job, plan, bank, extra=extra)
        review, escalated, used_checker = _review_answers(job, plan, bank, draft)
        revised = True

    return GeneratedAnswers(
        answers=draft,
        review=review,
        writer_model=writer_model(),
        checker_model=used_checker,
        revised=revised,
        escalated=escalated,
    )
