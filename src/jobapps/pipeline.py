"""End-to-end processing for one job file."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from jobapps.career import CareerBank, get_career_bank
from jobapps.config import CONNECTIONS_PATH, JOBS_DIR, OUTPUT_DIR, PROCESSED_DIR, cursor_writer_model
from jobapps.fit import (
    OverflowReport,
    apply_python_trim,
    longest_content_bullet,
    parse_overfull_hbox,
)
from jobapps.generate import (
    generate_answers,
    generate_cover_letter_draft,
    generate_resume_draft,
    review_materials,
)
from jobapps.graduation import (
    apply_graduation_to_cover_letter,
    apply_graduation_to_resume,
    resolve_graduation_date,
    set_education_year,
)
from jobapps.latex import compile_tex_with_log, pdf_page_count, render_documents
from jobapps.match import match_connection
from jobapps.models import (
    ApplicationPlan,
    Connection,
    CoverLetter,
    DraftResume,
    Job,
    PipelineMetrics,
    ReviewResult,
    TailoredResume,
    dump_yaml,
    format_answers_markdown,
    is_stack_bullet,
    load_connections,
    load_job,
    overlong_bullet_issues,
)
from jobapps.notion import create_application_page, referral_text
from jobapps.notify import notify, reveal
from jobapps.plan import build_application_plan, selected_experiences
from jobapps.repair import rewrite_bullet, rewrite_cover_letter_paragraph, shorten_bullet, shorten_cover_letter
from jobapps.validate import (
    MAX_BULLET_REPAIRS,
    MAX_COVER_LETTER_REPAIRS,
    MAX_PAGE_FIT_REPAIRS,
    MAX_SEMANTIC_REVISIONS,
    apply_mechanical_fixes,
    validate_cover_letter,
    validate_draft_resume,
)


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
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)


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


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _project_stacks(bank: CareerBank) -> dict[str, str]:
    return {
        item.name.strip().casefold(): item.stack
        for item in bank.projects
        if item.stack.strip()
    }


def _experience_record(bank: CareerBank, company: str, role: str):
    for item in bank.experiences:
        if item.company == company and item.role == role:
            return item
    for item in bank.experiences:
        if item.company == company:
            return item
    return None


def _project_record(bank: CareerBank, name: str):
    for item in bank.projects:
        if item.name == name:
            return item
    return None


def _repair_overlong_bullets(
    draft: DraftResume,
    plan: ApplicationPlan,
    bank: CareerBank,
    limit: int,
) -> tuple[DraftResume, int]:
    rewrites = 0
    max_chars = plan.layout.max_bullet_chars
    while rewrites < limit:
        issues = overlong_bullet_issues(draft.to_tailored(), max_chars)
        if not issues:
            return draft, rewrites
        updated = False
        experiences = list(draft.experience)
        for exp_index, item in enumerate(experiences):
            bullets = list(item.bullets)
            for bullet_index, bullet in enumerate(bullets):
                if is_stack_bullet(bullet.text):
                    continue
                if len(bullet.text) <= max_chars:
                    continue
                record = _experience_record(bank, item.company, item.role)
                shortened = shorten_bullet(bullet.text, max_chars=max_chars, record=record)
                bullets[bullet_index] = bullet.model_copy(update={"text": shortened})
                experiences[exp_index] = item.model_copy(update={"bullets": bullets})
                draft = draft.model_copy(update={"experience": experiences})
                rewrites += 1
                updated = True
                break
            if updated:
                break
        if updated:
            continue
        projects = list(draft.projects)
        for proj_index, item in enumerate(projects):
            bullets = list(item.bullets)
            for bullet_index, bullet in enumerate(bullets):
                if is_stack_bullet(bullet.text):
                    continue
                if len(bullet.text) <= max_chars:
                    continue
                record = _project_record(bank, item.name)
                shortened = shorten_bullet(bullet.text, max_chars=max_chars, record=record)
                bullets[bullet_index] = bullet.model_copy(update={"text": shortened})
                projects[proj_index] = item.model_copy(update={"bullets": bullets})
                draft = draft.model_copy(update={"projects": projects})
                rewrites += 1
                updated = True
                break
            if updated:
                break
        if not updated:
            break
    return draft, rewrites


def _repair_from_review(
    draft: DraftResume,
    cover: CoverLetter | None,
    review: ReviewResult,
    plan: ApplicationPlan,
    bank: CareerBank,
) -> tuple[DraftResume, CoverLetter | None, int, int]:
    issues = " ".join(review.issues or [review.summary]).casefold()
    bullet_rewrites = 0
    cover_repairs = 0
    if cover is not None and "cover letter" in issues:
        index = 0
        if cover.paragraphs:
            rewritten = rewrite_cover_letter_paragraph(
                cover.paragraphs[index],
                index + 1,
                cover,
                review.summary or "; ".join(review.issues),
                [item.prompt_payload() for item in selected_experiences(plan, bank)[:2]],
            )
            paragraphs = list(cover.paragraphs)
            paragraphs[index] = rewritten
            cover = cover.model_copy(update={"paragraphs": paragraphs})
            cover_repairs = 1
        return draft, cover, bullet_rewrites, cover_repairs

    tailored = draft.to_tailored()
    loc = longest_content_bullet(tailored)
    if loc is None:
        return draft, cover, bullet_rewrites, cover_repairs
    kind, item_index, bullet_index = loc
    feedback = review.summary or "; ".join(review.issues)
    if kind == "experience":
        item = draft.experience[item_index]
        record = _experience_record(bank, item.company, item.role)
        new_text = rewrite_bullet(item.bullets[bullet_index].text, feedback, record=record)
        bullets = list(item.bullets)
        bullets[bullet_index] = bullets[bullet_index].model_copy(update={"text": new_text})
        experiences = list(draft.experience)
        experiences[item_index] = item.model_copy(update={"bullets": bullets})
        draft = draft.model_copy(update={"experience": experiences})
    else:
        item = draft.projects[item_index]
        record = _project_record(bank, item.name)
        new_text = rewrite_bullet(item.bullets[bullet_index].text, feedback, record=record)
        bullets = list(item.bullets)
        bullets[bullet_index] = bullets[bullet_index].model_copy(update={"text": new_text})
        projects = list(draft.projects)
        projects[item_index] = item.model_copy(update={"bullets": bullets})
        draft = draft.model_copy(update={"projects": projects})
    return draft, cover, 1, cover_repairs


def _overflow_issues(report: OverflowReport, with_cover: bool) -> list[str]:
    issues: list[str] = []
    if report.resume_pages > 1:
        issues.append(f"Resume is {report.resume_pages} pages; must be exactly 1.")
    if report.resume_horizontal:
        issues.append(
            "Resume has Overfull \\hbox (horizontal overflow): "
            + ", ".join(report.resume_overfull[:5])
        )
    if with_cover and report.cover_pages is not None and report.cover_pages > 1:
        issues.append(f"Cover letter is {report.cover_pages} pages; must be exactly 1.")
    if with_cover and report.cover_horizontal:
        issues.append(
            "Cover letter has Overfull \\hbox (horizontal overflow): "
            + ", ".join(report.cover_overfull[:5])
        )
    return issues


def _set_resume_bullet(
    resume: TailoredResume, kind: str, item_index: int, bullet_index: int, text: str
) -> TailoredResume:
    if kind == "experience":
        item = resume.experience[item_index]
        bullets = list(item.bullets)
        bullets[bullet_index] = text
        experiences = list(resume.experience)
        experiences[item_index] = item.model_copy(update={"bullets": bullets})
        return resume.model_copy(update={"experience": experiences})
    item = resume.projects[item_index]
    bullets = list(item.bullets)
    bullets[bullet_index] = text
    projects = list(resume.projects)
    projects[item_index] = item.model_copy(update={"bullets": bullets})
    return resume.model_copy(update={"projects": projects})


def _render_and_fit(
    job: Job,
    contact,
    resume: TailoredResume,
    cover: CoverLetter | None,
    materials_dir: Path,
    plan: ApplicationPlan,
    bank: CareerBank,
    metrics: PipelineMetrics,
) -> tuple[TailoredResume, CoverLetter | None]:
    page_fit = 0
    captured_initial = False
    while True:
        render_documents(
            job,
            contact,
            resume,
            cover,
            materials_dir,
            template_name=job.template,
            compile_pdf=False,
        )
        _pdf, resume_log = compile_tex_with_log(materials_dir / "resume.tex")
        cover_log = ""
        cover_pages: int | None = None
        if cover is not None:
            _pdf, cover_log = compile_tex_with_log(materials_dir / "cover_letter.tex")
            cover_pages = pdf_page_count(materials_dir / "cover_letter.pdf")
        resume_pages = pdf_page_count(materials_dir / "resume.pdf")
        report = OverflowReport(
            resume_pages=resume_pages,
            cover_pages=cover_pages,
            resume_overfull=parse_overfull_hbox(resume_log),
            cover_overfull=parse_overfull_hbox(cover_log),
        )
        if not captured_initial:
            metrics.initial_resume_pages = resume_pages
            metrics.initial_cover_pages = cover_pages
            captured_initial = True
        if not report.resume_overflow and not report.cover_overflow:
            metrics.final_resume_pages = resume_pages
            metrics.final_cover_pages = cover_pages
            metrics.page_fit_attempts = page_fit
            return resume, cover

        if report.resume_vertical:
            trimmed = apply_python_trim(resume, plan)
            if trimmed is not None:
                resume = trimmed
                continue

        if page_fit >= MAX_PAGE_FIT_REPAIRS:
            raise RuntimeError(
                "Resume/cover letter still exceed 1 page after "
                f"{MAX_PAGE_FIT_REPAIRS} repairs: "
                + "; ".join(_overflow_issues(report, cover is not None))
            )

        if report.resume_overflow:
            loc = longest_content_bullet(resume)
            if loc is not None:
                kind, item_index, bullet_index = loc
                if kind == "experience":
                    item = resume.experience[item_index]
                    record = _experience_record(bank, item.company, item.role)
                    current = item.bullets[bullet_index]
                else:
                    item = resume.projects[item_index]
                    record = _project_record(bank, item.name)
                    current = item.bullets[bullet_index]
                shortened = shorten_bullet(
                    current, max_chars=plan.layout.max_bullet_chars, record=record
                )
                resume = _set_resume_bullet(resume, kind, item_index, bullet_index, shortened)
                page_fit += 1
                metrics.bullet_rewrites += 1
                continue
            trimmed = apply_python_trim(resume, plan)
            if trimmed is not None:
                resume = trimmed
                page_fit += 1
                continue

        if cover is not None and report.cover_overflow:
            cover = shorten_cover_letter(cover)
            page_fit += 1
            metrics.cover_letter_repairs += 1
            continue

        raise RuntimeError(
            "Could not repair PDF overflow: "
            + "; ".join(_overflow_issues(report, cover is not None))
        )


def process_job_file(job_path: Path) -> ProcessResult:
    job = load_job(job_path)
    template_request = job.template
    bank = get_career_bank()
    graduation_year = resolve_graduation_date(job)
    plan = build_application_plan(job, bank)
    job = job.model_copy(update={"template": plan.template})
    allowed = bank.skills.allowed_names()
    connections = load_connections(CONNECTIONS_PATH)
    referral = match_connection(job.company, connections)
    metrics = PipelineMetrics(
        experience_count=len(plan.experience_ids),
        project_count=len(plan.project_ids),
    )

    draft, context_chars = generate_resume_draft(job, plan, bank)
    metrics.context_chars = context_chars
    draft = apply_mechanical_fixes(
        draft, plan, bank.profile.education, _project_stacks(bank)
    )
    draft = draft.model_copy(
        update={"education": set_education_year(draft.education, graduation_year)}
    )
    draft, bullet_rewrites = _repair_overlong_bullets(
        draft, plan, bank, MAX_BULLET_REPAIRS
    )
    metrics.bullet_rewrites += bullet_rewrites
    draft = apply_mechanical_fixes(
        draft, plan, bank.profile.education, _project_stacks(bank)
    )
    issues = validate_draft_resume(draft, plan, allowed, bank.profile)
    metrics.validation_failures += len(issues)
    if issues:
        still_long = overlong_bullet_issues(draft.to_tailored(), plan.layout.max_bullet_chars)
        if still_long:
            raise RuntimeError(
                "Resume bullets still exceed the line limit after targeted repairs: "
                + "; ".join(still_long[:5])
            )
        # Remaining issues are layout/section problems the writer should have hit.
        # Keep going only if they are empty-summary / skills-already-fixed class;
        # otherwise fail so a bad draft cannot ship.
        blocking = [
            item
            for item in issues
            if "summary must be empty" not in item.casefold()
        ]
        if blocking:
            raise RuntimeError("Resume failed Python validation: " + "; ".join(blocking[:8]))

    destination = output_dir_for(job)
    inputs_dir, materials_dir, meta_dir = ensure_output_tree(destination)
    dump_yaml(job.model_dump(), inputs_dir / "job.yaml")
    _write_json(meta_dir / "application_plan.json", plan.model_dump())
    _write_json(meta_dir / "resume_draft.json", draft.model_dump())

    cover: CoverLetter | None = None
    if plan.cover_letter:
        cover = generate_cover_letter_draft(job, plan, bank, draft.to_tailored())
        cover = apply_graduation_to_cover_letter(cover, graduation_year)
        _write_json(meta_dir / "cover_letter_draft.json", cover.model_dump())
        cl_issues = validate_cover_letter(cover)
        metrics.validation_failures += len(cl_issues)
        if cl_issues:
            cover = shorten_cover_letter(cover)
            metrics.cover_letter_repairs += 1
            cl_issues = validate_cover_letter(cover)
            if cl_issues and metrics.cover_letter_repairs >= MAX_COVER_LETTER_REPAIRS:
                raise RuntimeError(
                    "Cover letter failed Python validation: " + "; ".join(cl_issues[:8])
                )

    resume = apply_graduation_to_resume(draft.to_tailored(), graduation_year)
    review, escalated, checker_model = review_materials(job, plan, bank, resume, cover)
    metrics.checker_escalated = escalated
    if (not review.approved or review.issues) and metrics.semantic_revisions < MAX_SEMANTIC_REVISIONS:
        draft, cover, extra_bullets, extra_cover = _repair_from_review(
            draft, cover, review, plan, bank
        )
        metrics.bullet_rewrites += extra_bullets
        metrics.cover_letter_repairs += extra_cover
        metrics.semantic_revisions += 1
        resume = apply_graduation_to_resume(draft.to_tailored(), graduation_year)
        if cover is not None:
            cover = apply_graduation_to_cover_letter(cover, graduation_year)

    _write_json(meta_dir / "resume_final.json", draft.model_dump())
    if cover is not None:
        _write_json(meta_dir / "cover_letter_final.json", cover.model_dump())

    resume, cover = _render_and_fit(
        job,
        bank.profile.contact,
        resume,
        cover,
        materials_dir,
        plan,
        bank,
        metrics,
    )

    answers_payload: dict | None = None
    answers_checker_approved: bool | None = None
    answers_checker_summary = ""
    if job.questions:
        answered = generate_answers(job, plan, bank)
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
            "checker_escalated": answered.escalated,
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
            "writer_model": cursor_writer_model(),
            "checker_model": checker_model,
            "checker_approved": review.approved,
            "checker_summary": review.summary,
            "checker_issues": review.issues,
            "checker_escalated": escalated,
            "revised_after_review": metrics.semantic_revisions > 0,
            "page_revisions": metrics.page_fit_attempts,
            "template_request": template_request,
            "selected_template": job.template,
            "template_reason": plan.template_reason,
            "template_auto_selected": template_request.strip().lower() in {"auto", ""},
            "graduation": graduation_year,
            "cover_letter": plan.cover_letter,
            "answers": answers_payload,
            "metrics": metrics.model_dump(),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        meta_dir / "meta.yaml",
    )
    return ProcessResult(
        output_dir=destination,
        job=job,
        referral=referral,
        notion_url=notion_url,
        checker_approved=review.approved,
        checker_summary=review.summary,
        answers_checker_approved=answers_checker_approved,
        answers_checker_summary=answers_checker_summary,
        metrics=metrics,
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
