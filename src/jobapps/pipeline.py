"""End-to-end processing for one job file."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jobapps.config import (
    CONNECTIONS_PATH,
    COVER_LETTER_EXAMPLES_DIR,
    JOBS_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RESUME_ADDITIONS_DIR,
    RESUME_TEMPLATES_DIR,
    SKILLS_BANK_PATH,
    WRITING_SAMPLES_DIR,
)
from jobapps.generate import generate, generate_answers, rewrite_draft, select_resume_template
from jobapps.graduation import apply_graduation, apply_graduation_to_resume, resolve_graduation_date
from jobapps.latex import pdf_page_count, render_documents
from jobapps.match import match_connection
from jobapps.models import (
    MAX_BULLET_CHARS,
    MIN_BULLET_CHARS,
    Connection,
    GenerationResult,
    Job,
    allowed_skill_names,
    dump_yaml,
    format_answers_markdown,
    load_additions,
    load_connections,
    load_cover_letter_examples,
    load_job,
    load_resume,
    load_skills_bank,
    load_writing_samples,
    overlong_bullet_issues,
    resolve_template_request,
    unknown_skill_issues,
)
from jobapps.notion import create_application_page, referral_text
from jobapps.notify import notify, reveal

MAX_PAGE_REVISIONS = 2


@dataclass
class ProcessResult:
    output_dir: Path
    job: Job
    referral: Connection | None
    notion_url: str | None
    checker_approved: bool = True
    checker_summary: str = ""
    answers_checker_approved: bool | None = None
    answers_checker_summary: str = ""


def slug(company: str, title: str) -> str:
    raw = f"{company}-{title}"
    cleaned = re.sub(r"[^\w]+", "-", raw, flags=re.UNICODE).strip("-")
    return cleaned or "application"


def output_dir_for(job: Job, now: datetime | None = None) -> Path:
    stamp_day = (now or datetime.now()).strftime("%Y-%m-%d")
    base_name = f"{stamp_day}_{slug(job.company, job.title)}"
    base = OUTPUT_DIR / base_name
    if not base.exists():
        return base
    time_stamp = (now or datetime.now()).strftime("%H%M%S")
    return OUTPUT_DIR / f"{base_name}-{time_stamp}"


def ensure_output_tree(destination: Path) -> tuple[Path, Path, Path]:
    inputs = destination / "inputs"
    materials = destination / "materials"
    meta = destination / "meta"
    for path in (inputs, materials, meta):
        path.mkdir(parents=True, exist_ok=True)
    return inputs, materials, meta


def error_sidecar(job_path: Path) -> Path:
    return job_path.with_name(f"{job_path.name}.error.txt")


def _page_overflow_issues(materials_dir: Path) -> list[str]:
    issues: list[str] = []
    resume_pdf = materials_dir / "resume.pdf"
    cover_pdf = materials_dir / "cover_letter.pdf"
    resume_pages = pdf_page_count(resume_pdf)
    cover_pages = pdf_page_count(cover_pdf)
    if resume_pages > 1:
        issues.append(
            f"Resume is {resume_pages} pages; must be exactly 1. "
            "Trim the weakest bullets first; only then drop the least relevant project or role."
        )
    if cover_pages > 1:
        issues.append(
            f"Cover letter is {cover_pages} pages; must be exactly 1. "
            "Shorten paragraphs or drop the weakest paragraph; keep 4+ substantive "
            "paragraphs when possible."
        )
    return issues


def _render_and_fit(
    job: Job,
    contact,
    draft: GenerationResult,
    materials_dir: Path,
    resume_base,
    additions: str,
    examples: str,
    skills_bank: str,
    writing_samples: str = "",
    graduation_year: str = "",
) -> tuple[GenerationResult, int]:
    """Render PDFs; if over one page, rewrite up to MAX_PAGE_REVISIONS times."""
    page_revisions = 0
    materials = draft
    allowed = allowed_skill_names(skills_bank) if skills_bank.strip() else set()
    while True:
        if graduation_year:
            materials = apply_graduation(materials, graduation_year)
        render_documents(
            job,
            contact,
            materials.resume,
            materials.cover_letter,
            materials_dir,
            template_name=job.template,
        )
        overflow = _page_overflow_issues(materials_dir)
        if not overflow:
            return materials, page_revisions
        if page_revisions >= MAX_PAGE_REVISIONS:
            raise RuntimeError(
                "Resume/cover letter still exceed 1 page after "
                f"{MAX_PAGE_REVISIONS} revisions: " + "; ".join(overflow)
            )
        feedback = f"""
## One-page limit violated
{_dump_issues(overflow)}

Shorten the resume and cover letter so each PDF is exactly one page. First trim or \
shorten the weakest bullets; only if still over one page, drop the least relevant \
project or role. Keep remaining bullets full ({MIN_BULLET_CHARS}-{MAX_BULLET_CHARS} chars \
when facts allow). Every resume bullet must stay at most {MAX_BULLET_CHARS} characters. \
Every skill item must appear in the skills bank. Every kept project must still end with \
a "Stack: ..." bullet.
"""
        materials = rewrite_draft(
            job,
            resume_base,
            additions,
            examples,
            materials,
            feedback,
            skills_bank,
            writing_samples,
        )
        bullet_issues = overlong_bullet_issues(materials.resume)
        if bullet_issues:
            materials = rewrite_draft(
                job,
                resume_base,
                additions,
                examples,
                materials,
                f"""
## Also fix overlong bullets (max {MAX_BULLET_CHARS} chars)
{_dump_issues(bullet_issues)}
""",
                skills_bank,
                writing_samples,
            )
            still = overlong_bullet_issues(materials.resume)
            if still:
                raise RuntimeError(
                    "Resume bullets still exceed "
                    f"{MAX_BULLET_CHARS} characters during page revise: "
                    + "; ".join(still[:5])
                )
        if allowed:
            skill_issues = unknown_skill_issues(materials.resume, allowed)
            if skill_issues:
                materials = rewrite_draft(
                    job,
                    resume_base,
                    additions,
                    examples,
                    materials,
                    f"""
## Also drop invented skills — every item must appear in the skills bank
{_dump_issues(skill_issues)}
""",
                    skills_bank,
                    writing_samples,
                )
                still_skills = unknown_skill_issues(materials.resume, allowed)
                if still_skills:
                    raise RuntimeError(
                        "Resume still has skills not in the skills bank during page revise: "
                        + "; ".join(still_skills[:5])
                    )
        page_revisions += 1


def _dump_issues(issues: list[str]) -> str:
    import json

    return json.dumps({"issues": issues}, indent=2, ensure_ascii=False)


def process_job_file(job_path: Path) -> ProcessResult:
    job = load_job(job_path)
    template_request = job.template
    explicit = resolve_template_request(template_request)
    if explicit:
        job = job.model_copy(update={"template": explicit})
        template_reason = "Set explicitly in job YAML."
        template_auto_selected = False
    else:
        choice = select_resume_template(job)
        job = job.model_copy(update={"template": choice.template})
        template_reason = choice.reason
        template_auto_selected = True

    resume_path = RESUME_TEMPLATES_DIR / f"{job.template}.yaml"
    if not resume_path.is_file():
        raise FileNotFoundError(f"Resume content not found: {resume_path}")

    resume = load_resume(resume_path)
    graduation_year = resolve_graduation_date(job)
    resume = apply_graduation_to_resume(resume, graduation_year)
    additions = load_additions(RESUME_ADDITIONS_DIR)
    skills_bank = load_skills_bank(SKILLS_BANK_PATH)
    examples = load_cover_letter_examples(COVER_LETTER_EXAMPLES_DIR)
    writing_samples = load_writing_samples(WRITING_SAMPLES_DIR)
    connections = load_connections(CONNECTIONS_PATH)
    referral = match_connection(job.company, connections)
    generated = generate(job, resume, additions, examples, skills_bank, writing_samples)
    tailored = apply_graduation(generated.materials, graduation_year)

    destination = output_dir_for(job)
    inputs_dir, materials_dir, meta_dir = ensure_output_tree(destination)
    dump_yaml(job.model_dump(), inputs_dir / "job.yaml")
    tailored, page_revisions = _render_and_fit(
        job,
        resume.contact,
        tailored,
        materials_dir,
        resume,
        additions,
        examples,
        skills_bank,
        writing_samples,
        graduation_year=graduation_year,
    )

    answers_payload: dict | None = None
    answers_checker_approved: bool | None = None
    answers_checker_summary = ""
    if job.questions:
        answered = generate_answers(job, resume, additions, writing_samples)
        (materials_dir / "answers.md").write_text(
            format_answers_markdown(answered.answers), encoding="utf-8"
        )
        answers_payload = {
            "answers": answered.answers.model_dump(),
            "writer_model": answered.writer_model,
            "checker_model": answered.checker_model,
            "checker_approved": answered.review.approved,
            "checker_summary": answered.review.summary,
            "checker_issues": answered.review.issues,
            "revised_after_review": answered.revised,
        }
        answers_checker_approved = answered.review.approved
        answers_checker_summary = answered.review.summary

    notion_url = create_application_page(job, destination, referral)
    dump_yaml(
        {
            "company": job.company,
            "title": job.title,
            "portal_url": job.portal_url,
            "referral": referral_text(referral),
            "notion_url": notion_url,
            "writer_model": generated.writer_model,
            "checker_model": generated.checker_model,
            "checker_approved": generated.review.approved,
            "checker_summary": generated.review.summary,
            "checker_issues": generated.review.issues,
            "revised_after_review": generated.revised,
            "page_revisions": page_revisions,
            "template_request": template_request,
            "selected_template": job.template,
            "template_reason": template_reason,
            "template_auto_selected": template_auto_selected,
            "graduation": graduation_year,
            "answers": answers_payload,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        meta_dir / "meta.yaml",
    )
    return ProcessResult(
        output_dir=destination,
        job=job,
        referral=referral,
        notion_url=notion_url,
        checker_approved=generated.review.approved,
        checker_summary=generated.review.summary,
        answers_checker_approved=answers_checker_approved,
        answers_checker_summary=answers_checker_summary,
    )


def finish_success(job_path: Path, result: ProcessResult) -> None:
    sidecar = error_sidecar(job_path)
    if sidecar.exists():
        sidecar.unlink()
    if job_path.parent.resolve() == JOBS_DIR.resolve():
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        dest = PROCESSED_DIR / job_path.name
        if dest.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            dest = PROCESSED_DIR / f"{job_path.stem}-{stamp}{job_path.suffix}"
        shutil.move(str(job_path), dest)

    referral = referral_text(result.referral)
    portal = result.job.portal_url or "No portal URL"
    body = f"{result.job.company} — {result.job.title}\n{portal}\n{referral}"
    if result.notion_url is None:
        body += "\nNotion skipped (not configured)"
    if not result.checker_approved:
        note = result.checker_summary or "Checker still has concerns — review the PDFs"
        body += f"\nChecker: {note[:120]}"
    if result.answers_checker_approved is False:
        note = result.answers_checker_summary or "Answers checker has concerns — review answers.md"
        body += f"\nAnswers: {note[:120]}"
    notify("Resume + cover letter created", body)
    reveal(result.output_dir / "materials" / "resume.pdf")


def finish_failure(job_path: Path, error: Exception) -> None:
    message = str(error)
    error_sidecar(job_path).write_text(message + "\n", encoding="utf-8")
    notify("Job application failed", f"{job_path.name}: {message[:180]}")


def run_job_file(job_path: Path) -> ProcessResult:
    try:
        result = process_job_file(job_path)
        finish_success(job_path, result)
        return result
    except Exception as error:
        finish_failure(job_path, error)
        raise
