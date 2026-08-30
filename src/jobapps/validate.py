"""Deterministic resume, cover-letter, and answer checks. No LLM."""

from __future__ import annotations

from jobapps.career import CareerBank, CareerProfile, ExperienceRecord, ProjectRecord
from jobapps.models import (
    ApplicationAnswersResult,
    ApplicationPlan,
    CoverLetter,
    DraftExperience,
    DraftProject,
    DraftResume,
    SourcedBullet,
    TailoredResume,
    is_stack_bullet,
    numeric_claim_tokens,
    overlong_answer_issues,
    overlong_bullet_issues,
    unknown_skill_issues,
)
from jobapps.plan import selected_experiences, selected_projects

MAX_SEMANTIC_REVISIONS = 1
MAX_BULLET_REPAIRS = 2
MAX_PAGE_FIT_REPAIRS = 2
MAX_PDF_FIT_LLM_REPAIRS = 2
MAX_COVER_LETTER_REPAIRS = 1
MAX_ANSWER_REPAIRS = 1


def content_bullets(bullets: list[str]) -> list[str]:
    return [bullet for bullet in bullets if not is_stack_bullet(bullet)]


def validate_resume(
    resume: TailoredResume,
    plan: ApplicationPlan,
    allowed_skills: set[str],
    profile: CareerProfile | None = None,
    *,
    enforce_layout_mins: bool = True,
) -> list[str]:
    issues: list[str] = []
    layout = plan.layout
    issues.extend(overlong_bullet_issues(resume, layout.max_bullet_chars))
    issues.extend(unknown_skill_issues(resume, allowed_skills))

    if resume.summary.strip():
        issues.append("resume.summary must be empty (no Summary section).")
    if not resume.experience:
        issues.append("Resume is missing the Experience section.")
    if not resume.projects:
        issues.append("Resume is missing the Projects section.")
    if not resume.education:
        issues.append("Resume is missing the Education section.")
    if not resume.skills:
        issues.append("Resume is missing the Skills section.")

    n_exp = len(resume.experience)
    if enforce_layout_mins and n_exp < layout.min_experiences:
        issues.append(
            f"Resume has {n_exp} experiences; need at least {layout.min_experiences}."
        )
    if n_exp > layout.max_experiences:
        issues.append(
            f"Resume has {n_exp} experiences; keep at most {layout.max_experiences}."
        )

    n_proj = len(resume.projects)
    if enforce_layout_mins and n_proj < layout.min_projects:
        issues.append(
            f"Resume has {n_proj} projects; need at least {layout.min_projects}."
        )
    if n_proj > layout.max_projects:
        issues.append(
            f"Resume has {n_proj} projects; keep at most {layout.max_projects}."
        )

    for item in resume.experience:
        if not item.company.strip() or not item.role.strip():
            issues.append("Experience is missing company or role.")
        count = len(content_bullets(item.bullets))
        if enforce_layout_mins and count < layout.min_experience_bullets:
            issues.append(
                f"Experience {item.company} / {item.role} has {count} bullets; "
                f"need at least {layout.min_experience_bullets}."
            )
        if count > layout.max_experience_bullets:
            issues.append(
                f"Experience {item.company} / {item.role} has {count} bullets; "
                f"keep at most {layout.max_experience_bullets}."
            )

    for item in resume.projects:
        if not item.name.strip():
            issues.append("A project is missing a name.")
        if not any(is_stack_bullet(bullet) for bullet in item.bullets):
            issues.append(f"Project {item.name} is missing a trailing Stack: line.")
        count = len(content_bullets(item.bullets))
        if enforce_layout_mins and count < layout.project_bullets:
            issues.append(
                f"Project {item.name} has {count} content bullets; "
                f"need {layout.project_bullets}."
            )
        if count > layout.max_experience_bullets:
            issues.append(
                f"Project {item.name} has {count} content bullets; "
                f"keep at most {layout.max_experience_bullets}."
            )

    n_skills = len(resume.skills)
    if enforce_layout_mins and n_skills < layout.min_skill_groups:
        issues.append(
            f"Skills section has {n_skills} groups; need at least {layout.min_skill_groups}."
        )
    if n_skills > layout.max_skill_groups:
        issues.append(
            f"Skills section has {n_skills} groups; keep at most {layout.max_skill_groups}."
        )

    if profile and profile.education:
        expected = profile.education[0]
        if not resume.education:
            issues.append("Resume education does not match the career profile.")
        else:
            got = resume.education[0]
            if got.school.strip() != expected.school.strip():
                issues.append("Education school does not match the career profile.")
            if got.degree.strip() != expected.degree.strip():
                issues.append("Education degree does not match the career profile.")
            if not got.year.strip():
                issues.append("Education is missing a graduation year.")

    return issues


def _experience_record(
    item: DraftExperience, plan: ApplicationPlan, bank: CareerBank
) -> ExperienceRecord | None:
    for record in selected_experiences(plan, bank):
        if record.company == item.company and record.role == item.role:
            return record
    for record in selected_experiences(plan, bank):
        if record.company == item.company:
            return record
    return None


def _project_record(
    item: DraftProject, plan: ApplicationPlan, bank: CareerBank
) -> ProjectRecord | None:
    for record in selected_projects(plan, bank):
        if record.name == item.name:
            return record
    return None


def _metric_provenance_issues(
    bullet: SourcedBullet,
    record: ExperienceRecord | ProjectRecord,
    label: str,
    index: int,
) -> list[str]:
    claims = numeric_claim_tokens(bullet.text)
    if not claims:
        return []
    cited_ids = set(bullet.sources)
    cited_metrics = [item for item in record.metrics if item.id in cited_ids]
    if not cited_metrics:
        return [
            f"{label} bullet {index + 1} has numeric claims but cites no metric sources."
        ]
    metric_nums = set()
    for item in cited_metrics:
        metric_nums.update(numeric_claim_tokens(item.text))
    missing = [claim for claim in claims if claim not in metric_nums]
    if not missing:
        return []
    return [
        f"{label} bullet {index + 1} numeric claims "
        + ", ".join(missing)
        + " are not supported by cited metrics."
    ]


def validate_sources(
    draft: DraftResume,
    plan: ApplicationPlan,
    bank: CareerBank,
) -> list[str]:
    issues: list[str] = []
    for item in draft.experience:
        record = _experience_record(item, plan, bank)
        label = f"Experience {item.company} / {item.role}"
        if record is None:
            issues.append(f"{label} is not in the selected career records.")
            continue
        allowed = record.fact_metric_ids()
        for index, bullet in enumerate(item.bullets):
            if is_stack_bullet(bullet.text):
                continue
            if not bullet.sources:
                issues.append(f"{label} bullet {index + 1} is missing sources.")
                continue
            unknown = [sid for sid in bullet.sources if sid not in allowed]
            if unknown:
                issues.append(
                    f"{label} bullet {index + 1} cites unknown sources: "
                    + ", ".join(unknown)
                )
                continue
            issues.extend(_metric_provenance_issues(bullet, record, label, index))
    for item in draft.projects:
        record = _project_record(item, plan, bank)
        if record is None:
            issues.append(f"Project {item.name} is not in the selected career records.")
            continue
        allowed = record.fact_metric_ids()
        label = f"Project {item.name}"
        for index, bullet in enumerate(item.bullets):
            if is_stack_bullet(bullet.text):
                continue
            if not bullet.sources:
                issues.append(f"{label} bullet {index + 1} is missing sources.")
                continue
            unknown = [sid for sid in bullet.sources if sid not in allowed]
            if unknown:
                issues.append(
                    f"{label} bullet {index + 1} cites unknown sources: "
                    + ", ".join(unknown)
                )
                continue
            issues.extend(_metric_provenance_issues(bullet, record, label, index))
    return issues


def source_issues(
    draft: DraftResume,
    plan: ApplicationPlan,
    bank: CareerBank,
) -> list[str]:
    return validate_sources(draft, plan, bank)


def validate_draft_resume(
    draft: DraftResume,
    plan: ApplicationPlan,
    allowed_skills: set[str],
    profile: CareerProfile | None = None,
    bank: CareerBank | None = None,
    *,
    enforce_layout_mins: bool = True,
) -> list[str]:
    issues = validate_resume(
        draft.to_tailored(),
        plan,
        allowed_skills,
        profile,
        enforce_layout_mins=enforce_layout_mins,
    )
    if bank is not None:
        issues.extend(validate_sources(draft, plan, bank))
    return issues


def validate_cover_letter(cover: CoverLetter) -> list[str]:
    issues: list[str] = []
    if not cover.greeting.strip():
        issues.append("Cover letter is missing a greeting.")
    paragraphs = [part.strip() for part in cover.paragraphs if part.strip()]
    if len(paragraphs) < 4:
        issues.append(
            f"Cover letter has {len(paragraphs)} paragraphs; need at least 4."
        )
    if len(paragraphs) > 6:
        issues.append(
            f"Cover letter has {len(paragraphs)} paragraphs; keep at most 6."
        )
    for index, paragraph in enumerate(paragraphs, start=1):
        if not paragraph:
            issues.append(f"Cover letter paragraph {index} is empty.")
    if not cover.closing.strip():
        issues.append("Cover letter is missing a closing.")
    return issues


def validate_answers(result: ApplicationAnswersResult) -> list[str]:
    return overlong_answer_issues(result)


def apply_mechanical_fixes(
    draft: DraftResume,
    plan: ApplicationPlan,
    profile_education: list,
    project_stacks: dict[str, str],
) -> DraftResume:
    """Fix issues Python can correct without an LLM call."""
    layout = plan.layout
    experience = []
    for item in draft.experience:
        content = [b for b in item.bullets if not is_stack_bullet(b.text)]
        if len(content) > layout.max_experience_bullets:
            content = content[: layout.max_experience_bullets]
        experience.append(item.model_copy(update={"bullets": content}))

    projects = []
    for item in draft.projects:
        content = [b for b in item.bullets if not is_stack_bullet(b.text)]
        stack = next((b for b in item.bullets if is_stack_bullet(b.text)), None)
        if len(content) > layout.max_experience_bullets:
            content = content[: layout.project_bullets]
        elif len(content) > layout.project_bullets:
            content = content[: layout.project_bullets]
        if stack is None:
            known = project_stacks.get(item.name.strip().casefold(), "")
            if known:
                stack = SourcedBullet(text=f"Stack: {known}")
        bullets = list(content)
        if stack is not None:
            bullets.append(stack)
        projects.append(item.model_copy(update={"bullets": bullets}))

    education = list(draft.education) or list(profile_education)
    if len(experience) > layout.max_experiences:
        experience = experience[: layout.max_experiences]
    if len(projects) > layout.max_projects:
        projects = projects[: layout.max_projects]
    skills = list(plan.skill_groups) if plan.skill_groups else list(draft.skills)
    return draft.model_copy(
        update={
            "summary": "",
            "experience": experience,
            "projects": projects,
            "education": education,
            "skills": skills,
        }
    )
