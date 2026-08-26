"""Cursor-powered resume and cover letter tailoring."""

from __future__ import annotations

import json
from dataclasses import dataclass

from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions

from jobapps.config import ROOT, cursor_checker_model, cursor_writer_model, require_env
from jobapps.models import (
    MAX_BULLET_CHARS,
    MIN_BULLET_CHARS,
    ApplicationAnswer,
    ApplicationAnswersResult,
    GenerationResult,
    Job,
    Resume,
    ReviewResult,
    TemplateChoice,
    allowed_skill_names,
    overlong_answer_issues,
    overlong_bullet_issues,
    unknown_skill_issues,
)

TEMPLATE_SELECTOR_PROMPT = """\
You choose the best base resume template for one job application.

Templates:
- swe: general software engineering — backend, frontend, full-stack, APIs, databases, \
cloud, web apps, DevOps, security engineering, and product engineering. Prefer when \
the role emphasizes building/shipping software systems rather than ML research.
- ai: AI/ML, computer vision, robotics, autonomy, perception, deep learning, research, \
data science, embedded sensing, and simulation. Prefer when ML, PyTorch, CARLA, radar/camera, \
or research-style work is central to the role.
- default: mixed or ambiguous roles that do not clearly favor swe or ai, or broad \
software roles where either mix could work equally well.

Pick exactly one template that best matches the job title and description.
Return JSON only matching the schema. No markdown, no code fences, no commentary.
Do not edit files, run commands, or use tools.
"""

def _system_prompt() -> str:
    return f"""\
You tailor an existing resume and write a cover letter for one job.

Rules:
- Never invent employers, job titles, dates, degrees, schools, projects, or skills.
- Every Technical Skills item must appear in the skills bank. Do not invent skills.
- Output 3-5 compact skill lines (one SkillGroup each); typical is 4. Start from the \
bank's recommended general-purpose mix and swap or reweight categories for the job \
(frontend → Frontend; ML/autonomy → AI/ML; embedded → Systems & Embedded).
- Category labels may be compact (e.g. Backend/Data) even if they combine bank headings. \
Item names must match the bank. Do not dump the whole bank. Do not print the \
"Additional Technologies" heading; those names may still appear in other lines.
- Only use facts from the provided resume and resume additions.
- Start from "Resume bullets:" in additions (or matching base-resume bullets) and expand \
each line with grounded tech, what you built, and impact. Use prose summaries and \
"Best statistics" to add detail — do not leave bullets thin when richer facts exist.
- Write full Jake Gutierrez-style lines: what + how (tech) + impact. Prefer \
{MIN_BULLET_CHARS}-{MAX_BULLET_CHARS} characters per bullet when facts allow. Do not \
compress to short stubs when the source supports a fuller line.
- Shorten bullets only when they exceed {MAX_BULLET_CHARS} characters or when trimming \
for a one-page overflow. Drop weaker bullets and reorder for the job when needed.
- Do not add an employer, role, or project that is not named in the base resume or additions.
- Keep education accurate; do not invent GPA, honors, or coursework.
- Use the exact education year (graduation date) from the base resume in both \
the resume and the cover letter. Do not change June 2027 vs Dec. 2027.
- Leave resume.summary empty. Do not write a Summary section.
- Fill most of one page (Jake Gutierrez density). Default target when material exists: \
4 experiences and 3-4 projects. Drop a role/project only for weak relevance or after \
a real one-page overflow — do not leave large empty space.
- Bullet counts for each kept experience or project (content bullets only; \
"Stack:" lines do not count): default to at least 3; use 4 when highly relevant; \
allow fewer than 3 only when weakly relevant but still worth keeping.
- Every kept project must end with a trailing "Stack: tech, ..." bullet so the \
heading can show Name | tech. Use stacks from the base resume or additions.
- Preserve education details that become Awards when present in the base resume.
- The resume and cover letter must each fit on ONE page. Prefer dense, \
non-wrapping content that fills the page; shorten only when needed to stay on one page.
- Every resume bullet must be a single printed line: at most {MAX_BULLET_CHARS} characters. \
No wrapping.
- Cover letter structure, length, and tone must match the cover letter examples: \
first person, specific, no cliches, not a rewrite of the resume. Write 4-6 substantive \
paragraphs that fill most of one page. Each paragraph should develop one thread \
(role fit, a key experience, research/project depth, team/leadership, why this company). \
Base the cover letter on the examples; let additional writing samples inform \
sentence-level prose style only (clarity, technical specificity, depth) — do not copy \
their academic format or section structure.
- Greeting should be "Dear Hiring Manager," unless the job text names a person.
- Closing should be "Sincerely,"
- Do not mention that you are an AI or that the letter was generated.
- Return JSON only. No markdown, no code fences, no commentary.
- Do not edit files, run commands, or use tools.
"""


def _checker_prompt() -> str:
    return f"""\
You are reviewing a tailored resume and cover letter before they are sent.

Approve only if all of these are true:
- resume.summary is empty (no Summary section).
- No invented employers, titles, dates, degrees, schools, projects, or skills.
- Every Technical Skills item appears in the skills bank; the skills section is 3-5 \
compact lines, not a dump of the bank.
- Every fact is grounded in the base resume or the experience/project notes.
- Bullets are expanded from "Resume bullets:" (or base-resume equivalents) with grounded \
tech + impact — not thin stubs when richer source bullets or statistics exist.
- Density matches a strong one-page Jake Gutierrez resume: typically ~4 experiences and \
3-4 projects when enough grounded material exists. Reject sparse drafts that leave large \
empty space (e.g. only 1 project when several relevant ones are available).
- Highly relevant roles/projects have at least 3 content bullets \
("Stack:" lines do not count). Weakly relevant kept items may have fewer.
- Every kept project includes a trailing "Stack: ..." bullet.
- Resume bullets are specific, relevant to the job, and read like a strong human resume.
- Bullets use most of the allowed line length when facts allow (typically \
{MIN_BULLET_CHARS}-{MAX_BULLET_CHARS} chars). Reject unnecessarily short bullets \
under {MIN_BULLET_CHARS} characters when the source material supports a fuller line.
- Every resume bullet is one printed line (at most {MAX_BULLET_CHARS} characters); \
reject any longer bullet.
- The resume and cover letter each fit on one page without large unused whitespace \
and without overflowing to a second page.
- Cover letter matches the example structure and tone: first person, specific, not a \
resume restatement, not generic. Write 4-6 substantive paragraphs that fill most of \
one page when grounded material exists. Reject thin or generic letters with fewer than \
4 paragraphs when the resume and additions support more depth. Reject overflow beyond \
one page.
- Length and tone are appropriate for a real application.

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


def _run_prompt(prompt: str, model: str) -> str:
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


def _parse_generation(text: str) -> GenerationResult:
    try:
        return GenerationResult.model_validate_json(extract_json(text))
    except Exception as error:
        raise RuntimeError(f"Writer returned invalid resume/cover letter JSON: {error}") from error


def _parse_review(text: str) -> ReviewResult:
    try:
        return ReviewResult.model_validate_json(extract_json(text))
    except Exception as error:
        raise RuntimeError(f"Checker returned invalid review JSON: {error}") from error


def _parse_template_choice(text: str) -> TemplateChoice:
    try:
        return TemplateChoice.model_validate_json(extract_json(text))
    except Exception as error:
        raise RuntimeError(f"Template selector returned invalid JSON: {error}") from error


def select_resume_template(job: Job) -> TemplateChoice:
    """Choose swe, ai, or default from the job title and description."""
    schema = _dump(TemplateChoice.model_json_schema())
    prompt = f"""\
{TEMPLATE_SELECTOR_PROMPT}

JSON schema:
{schema}

Company: {job.company}
Title: {job.title}
Notes: {job.notes or "(none)"}

### Job description
{job.description}
"""
    return _parse_template_choice(_run_prompt(prompt, cursor_writer_model()))


def _context_block(
    job: Job,
    resume: Resume,
    additions: str,
    examples: str,
    skills_bank: str = "",
    writing_samples: str = "",
) -> str:
    return f"""\
## Job
Company: {job.company}
Title: {job.title}
Portal: {job.portal_url or "(none)"}
Notes: {job.notes or "(none)"}
Starts: {job.starts or "(infer from description)"}
Graduation date (required): {resume.education[0].year if resume.education else "(none)"}
Template track: {job.template}

### Description
{job.description}

## Base resume (source of truth)
{_dump(resume.model_dump())}

## Skills bank (allowed inventory — select a short subset; do not invent)
{skills_bank or "(none)"}

## Extra experience/project notes — expand "Resume bullets:" with grounded tech + impact
{additions or "(none)"}

## Cover letter examples (match structure, length, and tone)
{examples or "(none — use a professional first-person cover letter format)"}

## Additional writing samples (prose style only — do not copy format or content)
These are research/project writeups, not cover letters. Use them only to match \
the author's writing style: clarity, technical specificity, and depth. Do not \
mirror their section structure or academic tone in the cover letter.
{writing_samples or "(none)"}
"""


def _write_draft(
    job: Job,
    resume: Resume,
    additions: str,
    examples: str,
    skills_bank: str = "",
    writing_samples: str = "",
    extra: str = "",
) -> GenerationResult:
    schema = _dump(GenerationResult.model_json_schema())
    prompt = f"""\
{_system_prompt()}

JSON schema:
{schema}

{_context_block(job, resume, additions, examples, skills_bank, writing_samples)}
{extra}
Return a tailored resume and cover letter as one JSON object matching the schema. \
Contact information is applied separately; you do not need to include it.
"""
    return _parse_generation(_run_prompt(prompt, cursor_writer_model()))


def _review_draft(
    job: Job,
    resume: Resume,
    additions: str,
    examples: str,
    draft: GenerationResult,
    skills_bank: str = "",
    writing_samples: str = "",
) -> ReviewResult:
    schema = _dump(ReviewResult.model_json_schema())
    prompt = f"""\
{_checker_prompt()}

JSON schema:
{schema}

{_context_block(job, resume, additions, examples, skills_bank, writing_samples)}

## Draft to review
{_dump(draft.model_dump())}
"""
    return _parse_review(_run_prompt(prompt, cursor_checker_model()))


@dataclass
class GeneratedMaterials:
    materials: GenerationResult
    review: ReviewResult
    writer_model: str
    checker_model: str
    revised: bool


def _bullet_feedback(issues: list[str]) -> str:
    return f"""
## Bullet length violations — shorten every listed bullet to at most {MAX_BULLET_CHARS} characters
Each resume bullet must fit on one printed line. Shorten wording; do not invent facts. \
Prefer keeping wording close to "Resume bullets:" from the additions. \
Return a full replacement JSON object.
{_dump({"issues": issues})}
"""


def _skills_feedback(issues: list[str]) -> str:
    return f"""
## Invented skills — every item must appear in the skills bank
Drop or replace each listed item with a name from the skills bank. Do not invent skills. \
Keep 3-5 compact skill lines. Return a full replacement JSON object.
{_dump({"issues": issues})}
"""


def rewrite_draft(
    job: Job,
    resume: Resume,
    additions: str,
    examples: str,
    draft: GenerationResult,
    feedback: str,
    skills_bank: str = "",
    writing_samples: str = "",
) -> GenerationResult:
    """Rewrite materials using concrete feedback (page overflow, bullets, etc.)."""
    extra = f"""
## Revision feedback — address every issue
Keep every fact grounded in the base resume or additions. Expand "Resume bullets:" \
with grounded tech + impact. Every skill item must appear in the skills bank. \
Return a full replacement JSON object.

Current draft:
{_dump(draft.model_dump())}

{feedback}
"""
    return _write_draft(
        job, resume, additions, examples, skills_bank, writing_samples, extra=extra
    )


def _ensure_bullet_length(
    job: Job,
    resume: Resume,
    additions: str,
    examples: str,
    draft: GenerationResult,
    skills_bank: str = "",
    writing_samples: str = "",
) -> tuple[GenerationResult, bool]:
    issues = overlong_bullet_issues(draft.resume)
    if not issues:
        return draft, False
    draft = rewrite_draft(
        job,
        resume,
        additions,
        examples,
        draft,
        _bullet_feedback(issues),
        skills_bank,
        writing_samples,
    )
    issues = overlong_bullet_issues(draft.resume)
    if issues:
        joined = "; ".join(issues[:5])
        raise RuntimeError(
            f"Resume bullets still exceed {MAX_BULLET_CHARS} characters after rewrite: {joined}"
        )
    return draft, True


def _ensure_known_skills(
    job: Job,
    resume: Resume,
    additions: str,
    examples: str,
    draft: GenerationResult,
    skills_bank: str,
    writing_samples: str = "",
) -> tuple[GenerationResult, bool]:
    if not skills_bank.strip():
        return draft, False
    allowed = allowed_skill_names(skills_bank)
    issues = unknown_skill_issues(draft.resume, allowed)
    if not issues:
        return draft, False
    draft = rewrite_draft(
        job,
        resume,
        additions,
        examples,
        draft,
        _skills_feedback(issues),
        skills_bank,
        writing_samples,
    )
    issues = unknown_skill_issues(draft.resume, allowed)
    if issues:
        joined = "; ".join(issues[:5])
        raise RuntimeError(f"Resume still has skills not in the skills bank: {joined}")
    return draft, True


def generate(
    job: Job,
    resume: Resume,
    additions: str,
    examples: str,
    skills_bank: str = "",
    writing_samples: str = "",
) -> GeneratedMaterials:
    draft = _write_draft(job, resume, additions, examples, skills_bank, writing_samples)
    review = _review_draft(
        job, resume, additions, examples, draft, skills_bank, writing_samples
    )
    revised = False
    bullet_issues = overlong_bullet_issues(draft.resume)
    skill_issues = (
        unknown_skill_issues(draft.resume, allowed_skill_names(skills_bank))
        if skills_bank.strip()
        else []
    )
    if not review.approved or bullet_issues or skill_issues:
        issues = list(review.issues) if review.issues else []
        if not review.approved and not issues:
            issues.append(review.summary or "The draft did not look strong enough.")
        issues.extend(bullet_issues)
        issues.extend(skill_issues)
        extra = f"""
## Checker / length / skills feedback — revise the draft to address every issue
{_dump({"summary": review.summary, "issues": issues})}

Every resume bullet must be at most {MAX_BULLET_CHARS} characters (one printed line). \
Prefer {MIN_BULLET_CHARS}-{MAX_BULLET_CHARS} chars when facts allow; expand thin bullets \
from the source material.
Every skill item must appear in the skills bank. Keep 3-5 compact skill lines.
Keep every fact grounded in the base resume or additions. Expand "Resume bullets:" \
with grounded tech + impact rather than compressing to stubs.
Return a full replacement JSON object.
"""
        draft = _write_draft(
            job, resume, additions, examples, skills_bank, writing_samples, extra=extra
        )
        review = _review_draft(
            job, resume, additions, examples, draft, skills_bank, writing_samples
        )
        revised = True

    draft, bullet_revised = _ensure_bullet_length(
        job, resume, additions, examples, draft, skills_bank, writing_samples
    )
    draft, skills_revised = _ensure_known_skills(
        job, resume, additions, examples, draft, skills_bank, writing_samples
    )
    return GeneratedMaterials(
        materials=draft,
        review=review,
        writer_model=cursor_writer_model(),
        checker_model=cursor_checker_model(),
        revised=revised or bullet_revised or skills_revised,
    )


ANSWERS_WRITER_PROMPT = """\
You write answers to optional job-application / screening questions for one role.

Rules:
- Never invent employers, titles, dates, degrees, schools, projects, skills, or outcomes.
- Ground every claim in the base resume, resume additions, or writing samples.
- Lean on the opening prose summaries in resume additions (experiences/projects) as \
primary story context — problem, approach, and impact. Use "Best statistics" and \
"Resume bullets" for concrete numbers and tech. Use writing samples when a question \
needs deeper project/research narrative.
- Choose the best-matching experiences and projects for each question and for the job \
description. If a question asks for N experiences or projects, use exactly that many \
when enough grounded material exists.
- Match the author's voice: clear, specific, technical, first person when natural.
- Respect each question's max_length (character count of the answer string). If \
max_length is null/omitted, there is no limit.
- Use the character budget: short limits (e.g. ~200) stay tight and pick the single \
best story. Larger limits (e.g. ~2000) or unlimited should fill most of the allowed \
space with additional grounded experiences and projects that strengthen the answer \
— more relevant roles, technical depth, and concrete outcomes — not fluff, \
repetition, or padding. Prefer adding another well-matched experience/project over \
stopping early when room remains and source material supports it.
- Answer every question in the same order as given. Copy each prompt and max_length \
into the output answers list.
- Do not mention that you are an AI or that the answers were generated.
- Return JSON only matching the schema. No markdown, no code fences, no commentary.
- Do not edit files, run commands, or use tools.
"""

ANSWERS_CHECKER_PROMPT = """\
You are reviewing application-question answers before they are submitted.

Approve only if all of these are true:
- No invented employers, titles, dates, degrees, schools, projects, skills, or outcomes.
- Every fact is grounded in the base resume, resume additions, or writing samples.
- Answers lean on addition summary-level detail when available, not vague restatements \
of thin bullets.
- Experiences/projects chosen are relevant to the job description and to the question.
- When a question asks for a specific number of experiences or projects, the answer \
uses that count when enough grounded material exists.
- Each answer with a max_length is at most that many characters (len of the answer string).
- Budget use: short max_length answers may be brief. For large max_length (roughly \
1000+) or unlimited, reject thin answers that leave most of the budget unused when \
more grounded experiences/projects could strengthen the response. Unused room should \
be filled with additional relevant substance, not padding.
- Tone is first-person, specific, and appropriate for a real application.

Return JSON only matching the schema. No markdown, no code fences, no commentary.
Do not edit files, run commands, or use tools.
"""


@dataclass
class GeneratedAnswers:
    answers: ApplicationAnswersResult
    review: ReviewResult
    writer_model: str
    checker_model: str
    revised: bool


def _answers_context_block(
    job: Job,
    resume: Resume,
    additions: str,
    writing_samples: str = "",
) -> str:
    questions_payload = [
        {"prompt": q.prompt, "max_length": q.max_length} for q in job.questions
    ]
    return f"""\
## Job
Company: {job.company}
Title: {job.title}
Portal: {job.portal_url or "(none)"}
Notes: {job.notes or "(none)"}

### Description
{job.description}

## Base resume (source of truth)
{_dump(resume.model_dump())}

## Experience/project notes — lean on opening summaries for story context
Each note typically starts with a prose summary, then optional "Best statistics" and \
"Resume bullets". Prefer the summary paragraphs when answering screening questions; \
use stats/bullets for concrete numbers and tech names.
{additions or "(none)"}

## Writing samples (deeper project/research prose — content + voice)
Use these when a question needs more depth than the addition summaries provide. \
Match clarity and technical specificity; do not invent beyond what is written.
{writing_samples or "(none)"}

## Questions to answer (preserve order, prompt, and max_length)
{_dump(questions_payload)}
"""


def _parse_answers(text: str) -> ApplicationAnswersResult:
    try:
        return ApplicationAnswersResult.model_validate_json(extract_json(text))
    except Exception as error:
        raise RuntimeError(f"Writer returned invalid application answers JSON: {error}") from error


def _align_answers(
    job: Job, result: ApplicationAnswersResult
) -> ApplicationAnswersResult:
    """Ensure prompts/max_length match the job questions; pad or trim if needed."""
    aligned: list = []
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


def _write_answers_draft(
    job: Job,
    resume: Resume,
    additions: str,
    writing_samples: str = "",
    extra: str = "",
) -> ApplicationAnswersResult:
    schema = _dump(ApplicationAnswersResult.model_json_schema())
    prompt = f"""\
{ANSWERS_WRITER_PROMPT}

JSON schema:
{schema}

{_answers_context_block(job, resume, additions, writing_samples)}
{extra}
Return answers as one JSON object matching the schema.
"""
    draft = _parse_answers(_run_prompt(prompt, cursor_writer_model()))
    return _align_answers(job, draft)


def _review_answers_draft(
    job: Job,
    resume: Resume,
    additions: str,
    draft: ApplicationAnswersResult,
    writing_samples: str = "",
) -> ReviewResult:
    schema = _dump(ReviewResult.model_json_schema())
    prompt = f"""\
{ANSWERS_CHECKER_PROMPT}

JSON schema:
{schema}

{_answers_context_block(job, resume, additions, writing_samples)}

## Draft answers to review
{_dump(draft.model_dump())}
"""
    return _parse_review(_run_prompt(prompt, cursor_checker_model()))


def _ensure_answer_length(
    job: Job,
    resume: Resume,
    additions: str,
    draft: ApplicationAnswersResult,
    writing_samples: str = "",
) -> tuple[ApplicationAnswersResult, bool]:
    issues = overlong_answer_issues(draft)
    if not issues:
        return draft, False
    extra = f"""
## Length violations — shorten every listed answer to its max_length
Do not invent facts. Preserve grounding in resume additions summaries and writing \
samples. Return a full replacement JSON object.
{_dump({"issues": issues})}
"""
    draft = _write_answers_draft(job, resume, additions, writing_samples, extra=extra)
    issues = overlong_answer_issues(draft)
    if issues:
        joined = "; ".join(issues[:5])
        raise RuntimeError(f"Application answers still exceed max_length after rewrite: {joined}")
    return draft, True


def generate_answers(
    job: Job,
    resume: Resume,
    additions: str,
    writing_samples: str = "",
) -> GeneratedAnswers:
    """Generate screening-question answers with writer then checker. Caller skips if empty."""
    if not job.questions:
        raise ValueError("generate_answers called with no questions on the job.")

    draft = _write_answers_draft(job, resume, additions, writing_samples)
    review = _review_answers_draft(job, resume, additions, draft, writing_samples)
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

Respect each question's max_length. Keep every fact grounded in the base resume, \
resume addition summaries, or writing samples. Return a full replacement JSON object.
"""
        draft = _write_answers_draft(job, resume, additions, writing_samples, extra=extra)
        review = _review_answers_draft(job, resume, additions, draft, writing_samples)
        revised = True

    draft, length_revised = _ensure_answer_length(
        job, resume, additions, draft, writing_samples
    )
    return GeneratedAnswers(
        answers=draft,
        review=review,
        writer_model=cursor_writer_model(),
        checker_model=cursor_checker_model(),
        revised=revised or length_revised,
    )
