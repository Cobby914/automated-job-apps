"""End-to-end processing for one job file."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from jobapps.career import CareerBank, get_career_bank
from jobapps.config import CONNECTIONS_PATH, JOBS_DIR, OUTPUT_DIR, PROCESSED_DIR, writer_model
from jobapps.dedup import find_duplicate
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
    DraftExperience,
    DraftProject,
    DraftResume,
    Job,
    ParsedLocation,
    PipelineMetrics,
    ReviewIssue,
    ReviewResult,
    SourcedBullet,
    TailoredResume,
    dump_yaml,
    format_answers_markdown,
    is_stack_bullet,
    issue_feedback,
    load_connections,
    load_job,
    overlong_bullet_issues,
    parse_issue_location,
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
    skipped_duplicate: bool = False
    duplicate_of: str = ""


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


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


_STAGE_ORDER = [
    "planned",
    "resume_drafted",
    "cover_drafted",
    "reviewed",
    "fitted",
    "complete",
]


def progress_sidecar(job_path: Path) -> Path:
    return job_path.with_name(job_path.name + ".progress.json")


def _stage_at_least(current: str | None, needed: str) -> bool:
    if not current:
        return False
    try:
        return _STAGE_ORDER.index(current) >= _STAGE_ORDER.index(needed)
    except ValueError:
        return False


def _load_progress(job_path: Path) -> dict | None:
    path = progress_sidecar(job_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data.get("output_dir"):
        return None
    return data


def _save_progress(job_path: Path, payload: dict) -> None:
    progress_sidecar(job_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _merge_sourced_bullets(
    original: list[SourcedBullet], new_texts: list[str]
) -> list[SourcedBullet]:
    """Keep source ids when PDF fitting shortens a bullet instead of dropping it."""
    unused = list(original)
    assigned: list[SourcedBullet | None] = [None] * len(new_texts)
    for index, text in enumerate(new_texts):
        for unused_index, bullet in enumerate(unused):
            if bullet.text == text:
                assigned[index] = unused.pop(unused_index)
                break
    leftover = 0
    merged: list[SourcedBullet] = []
    for index, text in enumerate(new_texts):
        current = assigned[index]
        if current is not None:
            merged.append(current)
        elif leftover < len(unused):
            merged.append(unused[leftover].model_copy(update={"text": text}))
            leftover += 1
        else:
            merged.append(SourcedBullet(text=text, sources=[]))
    return merged


def merge_fitted_into_draft(draft: DraftResume, resume: TailoredResume) -> DraftResume:
    """Keep source ids on bullets whose text survived PDF fitting."""
    exp_lookup = {(item.company, item.role): item for item in draft.experience}
    experiences: list[DraftExperience] = []
    for item in resume.experience:
        orig = exp_lookup.get((item.company, item.role))
        bullets = _merge_sourced_bullets(orig.bullets if orig else [], item.bullets)
        if orig is not None:
            experiences.append(orig.model_copy(update={"bullets": bullets}))
        else:
            experiences.append(
                DraftExperience(
                    company=item.company,
                    role=item.role,
                    location=item.location,
                    start=item.start,
                    end=item.end,
                    bullets=bullets,
                )
            )
    proj_lookup = {item.name: item for item in draft.projects}
    projects: list[DraftProject] = []
    for item in resume.projects:
        orig = proj_lookup.get(item.name)
        bullets = _merge_sourced_bullets(orig.bullets if orig else [], item.bullets)
        if orig is not None:
            projects.append(orig.model_copy(update={"bullets": bullets, "url": item.url}))
        else:
            projects.append(DraftProject(name=item.name, url=item.url, bullets=bullets))
    return draft.model_copy(
        update={
            "summary": resume.summary,
            "experience": experiences,
            "projects": projects,
            "education": resume.education,
            "skills": resume.skills,
        }
    )


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


def _first_repairable_issue(review: ReviewResult) -> ReviewIssue | None:
    for item in review.issues:
        if item.severity.casefold() != "warning":
            return item
    return review.issues[0] if review.issues else None


def _experience_index_for_id(draft: DraftResume, bank: CareerBank, item_id: str) -> int | None:
    record = bank.experience_by_id().get(item_id)
    if record is None:
        return None
    for index, item in enumerate(draft.experience):
        if item.company == record.company and item.role == record.role:
            return index
    for index, item in enumerate(draft.experience):
        if item.company == record.company:
            return index
    return None


def _project_index_for_id(draft: DraftResume, bank: CareerBank, item_id: str) -> int | None:
    record = bank.project_by_id().get(item_id)
    if record is None:
        return None
    for index, item in enumerate(draft.projects):
        if item.name == record.name:
            return index
    return None


def _resolve_review_issue(
    issue: ReviewIssue,
    draft: DraftResume,
    bank: CareerBank,
) -> ParsedLocation | None:
    item_id = issue.item_id.strip()
    section = issue.section.strip().casefold()
    if item_id:
        exp_index = _experience_index_for_id(draft, bank, item_id)
        if exp_index is not None:
            return ParsedLocation(
                kind="experience",
                item_index=exp_index,
                part_index=issue.bullet_index or 0,
                item_id=item_id,
            )
        proj_index = _project_index_for_id(draft, bank, item_id)
        if proj_index is not None:
            return ParsedLocation(
                kind="project",
                item_index=proj_index,
                part_index=issue.bullet_index or 0,
                item_id=item_id,
            )
    if section == "cover_letter":
        return ParsedLocation(
            kind="cover_letter",
            part_index=issue.paragraph_index or 0,
        )
    if section in {"experience", "project"} and issue.bullet_index is not None:
        loc = parse_issue_location(issue.location)
        if loc is not None and loc.kind in {"experience", "project"}:
            return loc
    return parse_issue_location(issue.location)


def _repair_experience_bullet(
    draft: DraftResume,
    item_index: int,
    bullet_index: int,
    feedback: str,
    bank: CareerBank,
) -> tuple[DraftResume, int]:
    if item_index < 0 or item_index >= len(draft.experience):
        return draft, 0
    item = draft.experience[item_index]
    if bullet_index < 0 or bullet_index >= len(item.bullets):
        return draft, 0
    if is_stack_bullet(item.bullets[bullet_index].text):
        return draft, 0
    record = _experience_record(bank, item.company, item.role)
    new_text = rewrite_bullet(item.bullets[bullet_index].text, feedback, record=record)
    bullets = list(item.bullets)
    bullets[bullet_index] = bullets[bullet_index].model_copy(update={"text": new_text})
    experiences = list(draft.experience)
    experiences[item_index] = item.model_copy(update={"bullets": bullets})
    return draft.model_copy(update={"experience": experiences}), 1


def _repair_project_bullet(
    draft: DraftResume,
    item_index: int,
    bullet_index: int,
    feedback: str,
    bank: CareerBank,
) -> tuple[DraftResume, int]:
    if item_index < 0 or item_index >= len(draft.projects):
        return draft, 0
    item = draft.projects[item_index]
    if bullet_index < 0 or bullet_index >= len(item.bullets):
        return draft, 0
    if is_stack_bullet(item.bullets[bullet_index].text):
        return draft, 0
    record = _project_record(bank, item.name)
    new_text = rewrite_bullet(item.bullets[bullet_index].text, feedback, record=record)
    bullets = list(item.bullets)
    bullets[bullet_index] = bullets[bullet_index].model_copy(update={"text": new_text})
    projects = list(draft.projects)
    projects[item_index] = item.model_copy(update={"bullets": bullets})
    return draft.model_copy(update={"projects": projects}), 1


def _repair_from_review(
    draft: DraftResume,
    cover: CoverLetter | None,
    review: ReviewResult,
    plan: ApplicationPlan,
    bank: CareerBank,
) -> tuple[DraftResume, CoverLetter | None, int, int]:
    issue = _first_repairable_issue(review)
    if issue is None:
        return draft, cover, 0, 0
    feedback = issue_feedback(issue)
    loc = _resolve_review_issue(issue, draft, bank)
    if loc is not None and loc.kind == "cover_letter" and cover is not None and cover.paragraphs:
        index = loc.part_index if 0 <= loc.part_index < len(cover.paragraphs) else 0
        rewritten = rewrite_cover_letter_paragraph(
            cover.paragraphs[index],
            index + 1,
            cover,
            feedback,
            [item.prompt_payload() for item in selected_experiences(plan, bank)[:2]],
        )
        paragraphs = list(cover.paragraphs)
        paragraphs[index] = rewritten
        cover = cover.model_copy(update={"paragraphs": paragraphs})
        return draft, cover, 0, 1

    if loc is not None and loc.kind == "experience":
        draft, rewritten = _repair_experience_bullet(
            draft, loc.item_index, loc.part_index, feedback, bank
        )
        if rewritten:
            return draft, cover, rewritten, 0

    if loc is not None and loc.kind == "project":
        draft, rewritten = _repair_project_bullet(
            draft, loc.item_index, loc.part_index, feedback, bank
        )
        if rewritten:
            return draft, cover, rewritten, 0

    tailored = draft.to_tailored()
    longest = longest_content_bullet(tailored)
    if loc is None or loc.kind in {"resume", "answers"} or longest is not None:
        if longest is None:
            return draft, cover, 0, 0
        kind, item_index, bullet_index = longest
        if kind == "experience":
            draft, rewritten = _repair_experience_bullet(
                draft, item_index, bullet_index, feedback, bank
            )
            return draft, cover, rewritten, 0
        draft, rewritten = _repair_project_bullet(
            draft, item_index, bullet_index, feedback, bank
        )
        return draft, cover, rewritten, 0
    return draft, cover, 0, 0


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
                metrics.generation_attempts += 1
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
            metrics.generation_attempts += 1
            continue

        raise RuntimeError(
            "Could not repair PDF overflow: "
            + "; ".join(_overflow_issues(report, cover is not None))
        )


def _blocking_validation_issues(issues: list[str]) -> list[str]:
    return [item for item in issues if "summary must be empty" not in item.casefold()]


def process_job_file(job_path: Path) -> ProcessResult:
    job = load_job(job_path)
    template_request = job.template
    bank = get_career_bank()
    connections = load_connections(CONNECTIONS_PATH)
    referral = match_connection(job.company, connections)
    allowed = bank.skills.allowed_names()

    duplicate = find_duplicate(job)
    if duplicate is not None:
        return ProcessResult(
            output_dir=duplicate if duplicate.is_dir() else OUTPUT_DIR,
            job=job,
            referral=referral,
            notion_url=None,
            skipped_duplicate=True,
            duplicate_of=str(duplicate),
        )

    progress = _load_progress(job_path)
    if progress:
        destination = Path(str(progress["output_dir"]))
        stage = str(progress.get("stage") or "")
        template_request = str(progress.get("template_request") or template_request)
        graduation_year = str(progress.get("graduation_year") or "") or resolve_graduation_date(job)
        metrics = PipelineMetrics.model_validate(progress.get("metrics") or {})
        checker_used = str(progress.get("checker_model") or "")
        escalated = bool(progress.get("escalated"))
    else:
        destination = output_dir_for(job)
        stage = ""
        graduation_year = resolve_graduation_date(job)
        metrics = PipelineMetrics()
        checker_used = ""
        escalated = False

    inputs_dir, materials_dir, meta_dir = ensure_output_tree(destination)

    def persist(next_stage: str) -> None:
        _save_progress(
            job_path,
            {
                "output_dir": str(destination),
                "stage": next_stage,
                "template_request": template_request,
                "graduation_year": graduation_year,
                "checker_model": checker_used,
                "escalated": escalated,
                "metrics": metrics.model_dump(),
            },
        )

    if _stage_at_least(stage, "planned"):
        plan = ApplicationPlan.model_validate(_read_json(meta_dir / "application_plan.json"))
        job = job.model_copy(update={"template": plan.template})
    else:
        plan = build_application_plan(job, bank)
        job = job.model_copy(update={"template": plan.template})
        dump_yaml(job.model_dump(), inputs_dir / "job.yaml")
        _write_json(meta_dir / "application_plan.json", plan.model_dump())
        metrics.experience_count = len(plan.experience_ids)
        metrics.project_count = len(plan.project_ids)
        metrics.selected_experiences = len(plan.experience_ids)
        metrics.selected_projects = len(plan.project_ids)
        metrics.candidate_experiences = len(bank.experiences)
        metrics.candidate_projects = len(bank.projects)
        persist("planned")
        stage = "planned"

    stacks = _project_stacks(bank)
    if _stage_at_least(stage, "resume_drafted"):
        draft = DraftResume.model_validate(_read_json(meta_dir / "resume_draft.json"))
    else:
        draft, context_chars = generate_resume_draft(job, plan, bank)
        metrics.generation_attempts += 1
        metrics.context_chars = context_chars
        draft = apply_mechanical_fixes(draft, plan, bank.profile.education, stacks)
        draft = draft.model_copy(
            update={"education": set_education_year(draft.education, graduation_year)}
        )
        draft, bullet_rewrites = _repair_overlong_bullets(
            draft, plan, bank, MAX_BULLET_REPAIRS
        )
        metrics.bullet_rewrites += bullet_rewrites
        metrics.generation_attempts += bullet_rewrites
        draft = apply_mechanical_fixes(draft, plan, bank.profile.education, stacks)
        issues = validate_draft_resume(draft, plan, allowed, bank.profile, bank)
        metrics.validation_failures += len(issues)
        if issues:
            still_long = overlong_bullet_issues(
                draft.to_tailored(), plan.layout.max_bullet_chars
            )
            if still_long:
                raise RuntimeError(
                    "Resume bullets still exceed the line limit after targeted repairs: "
                    + "; ".join(still_long[:5])
                )
            blocking = _blocking_validation_issues(issues)
            if blocking:
                raise RuntimeError(
                    "Resume failed Python validation: " + "; ".join(blocking[:8])
                )
        _write_json(meta_dir / "resume_draft.json", draft.model_dump())
        persist("resume_drafted")
        stage = "resume_drafted"

    cover: CoverLetter | None = None
    if _stage_at_least(stage, "cover_drafted"):
        cover_path = meta_dir / "cover_letter_draft.json"
        if cover_path.is_file():
            cover = CoverLetter.model_validate(_read_json(cover_path))
    elif plan.cover_letter:
        cover = generate_cover_letter_draft(job, plan, bank, draft.to_tailored())
        metrics.generation_attempts += 1
        cover = apply_graduation_to_cover_letter(cover, graduation_year)
        _write_json(meta_dir / "cover_letter_draft.json", cover.model_dump())
        cl_issues = validate_cover_letter(cover)
        metrics.validation_failures += len(cl_issues)
        if cl_issues:
            cover = shorten_cover_letter(cover)
            metrics.cover_letter_repairs += 1
            metrics.generation_attempts += 1
            cl_issues = validate_cover_letter(cover)
            if cl_issues and metrics.cover_letter_repairs >= MAX_COVER_LETTER_REPAIRS:
                raise RuntimeError(
                    "Cover letter failed Python validation: " + "; ".join(cl_issues[:8])
                )
            _write_json(meta_dir / "cover_letter_draft.json", cover.model_dump())
        persist("cover_drafted")
        stage = "cover_drafted"
    else:
        persist("cover_drafted")
        stage = "cover_drafted"

    if _stage_at_least(stage, "reviewed"):
        review = ReviewResult.model_validate(_read_json(meta_dir / "review.json"))
        draft = DraftResume.model_validate(_read_json(meta_dir / "resume_draft.json"))
        cover_path = meta_dir / "cover_letter_draft.json"
        if cover is None and cover_path.is_file():
            cover = CoverLetter.model_validate(_read_json(cover_path))
        resume = apply_graduation_to_resume(draft.to_tailored(), graduation_year)
        if cover is not None:
            cover = apply_graduation_to_cover_letter(cover, graduation_year)
    else:
        resume = apply_graduation_to_resume(draft.to_tailored(), graduation_year)
        review, escalated, checker_used = review_materials(job, plan, bank, resume, cover)
        metrics.checker_escalated = escalated
        if (
            (not review.approved or review.issues)
            and metrics.semantic_revisions < MAX_SEMANTIC_REVISIONS
        ):
            if not review.approved or any(
                item.severity.casefold() != "warning" for item in review.issues
            ):
                metrics.semantic_review_failures += 1
            draft, cover, extra_bullets, extra_cover = _repair_from_review(
                draft, cover, review, plan, bank
            )
            metrics.bullet_rewrites += extra_bullets
            metrics.cover_letter_repairs += extra_cover
            metrics.semantic_revisions += 1
            metrics.generation_attempts += extra_bullets + extra_cover
            draft = apply_mechanical_fixes(draft, plan, bank.profile.education, stacks)
            draft, extra_long = _repair_overlong_bullets(
                draft, plan, bank, MAX_BULLET_REPAIRS
            )
            metrics.bullet_rewrites += extra_long
            metrics.generation_attempts += extra_long
            issues = validate_draft_resume(
                draft, plan, allowed, bank.profile, bank
            )
            metrics.validation_failures += len(issues)
            blocking = _blocking_validation_issues(issues)
            if blocking:
                raise RuntimeError(
                    "Resume failed Python validation after repair: "
                    + "; ".join(blocking[:8])
                )
            if cover is not None:
                cl_issues = validate_cover_letter(cover)
                metrics.validation_failures += len(cl_issues)
                if cl_issues:
                    cover = shorten_cover_letter(cover)
                    metrics.cover_letter_repairs += 1
                    metrics.generation_attempts += 1
                    cl_issues = validate_cover_letter(cover)
                    if cl_issues:
                        raise RuntimeError(
                            "Cover letter failed Python validation after repair: "
                            + "; ".join(cl_issues[:8])
                        )
            resume = apply_graduation_to_resume(draft.to_tailored(), graduation_year)
            if cover is not None:
                cover = apply_graduation_to_cover_letter(cover, graduation_year)
            review, escalated, checker_used = review_materials(
                job, plan, bank, resume, cover
            )
            metrics.checker_escalated = metrics.checker_escalated or escalated
        _write_json(meta_dir / "review.json", review.model_dump())
        _write_json(meta_dir / "resume_draft.json", draft.model_dump())
        if cover is not None:
            _write_json(meta_dir / "cover_letter_draft.json", cover.model_dump())
        persist("reviewed")
        stage = "reviewed"

    if _stage_at_least(stage, "fitted"):
        draft = DraftResume.model_validate(_read_json(meta_dir / "resume_final.json"))
        resume = apply_graduation_to_resume(draft.to_tailored(), graduation_year)
        cover_final = meta_dir / "cover_letter_final.json"
        if cover_final.is_file():
            cover = CoverLetter.model_validate(_read_json(cover_final))
    else:
        resume = apply_graduation_to_resume(draft.to_tailored(), graduation_year)
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
        draft = merge_fitted_into_draft(draft, resume)
        issues = validate_draft_resume(
            draft,
            plan,
            allowed,
            bank.profile,
            bank,
            enforce_layout_mins=False,
        )
        metrics.validation_failures += len(issues)
        blocking = _blocking_validation_issues(issues)
        if blocking:
            raise RuntimeError(
                "Resume failed Python validation after PDF fitting: "
                + "; ".join(blocking[:8])
            )
        if cover is not None:
            cl_issues = validate_cover_letter(cover)
            metrics.validation_failures += len(cl_issues)
            if cl_issues:
                raise RuntimeError(
                    "Cover letter failed Python validation after PDF fitting: "
                    + "; ".join(cl_issues[:8])
                )
        _write_json(meta_dir / "resume_final.json", draft.model_dump())
        if cover is not None:
            _write_json(meta_dir / "cover_letter_final.json", cover.model_dump())
        persist("fitted")
        stage = "fitted"

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
            "checker_issues": [item.model_dump() for item in answered.review.issues],
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
            "writer_model": writer_model(),
            "checker_model": checker_used,
            "checker_approved": review.approved,
            "checker_summary": review.summary,
            "checker_issues": [item.model_dump() for item in review.issues],
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
    persist("complete")
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


def _move_job_yaml(job_path: Path) -> None:
    if job_path.parent.resolve() != JOBS_DIR.resolve():
        return
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROCESSED_DIR / job_path.name
    if dest.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = PROCESSED_DIR / f"{job_path.stem}-{stamp}{job_path.suffix}"
    shutil.move(str(job_path), dest)


def finish_success(job_path: Path, result: ProcessResult) -> None:
    sidecar = error_sidecar(job_path)
    if sidecar.exists():
        sidecar.unlink()
    progress = progress_sidecar(job_path)
    if progress.exists():
        progress.unlink()
    _move_job_yaml(job_path)

    if result.skipped_duplicate:
        notify(
            "Skipped duplicate job",
            f"{result.job.company} — {result.job.title}\nAlready processed: {result.duplicate_of}",
        )
        return

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
