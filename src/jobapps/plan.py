"""Build an ApplicationPlan from the career bank and a job posting."""

from __future__ import annotations

from jobapps.career import CareerBank, ExperienceRecord, ProjectRecord
from jobapps.models import ApplicationPlan, Job, LayoutBudget, RankedSelection
from jobapps.ranking import (
    rank_experiences,
    rank_projects,
    select_ranked,
    select_skills,
    select_template,
)


DEFAULT_LAYOUT = LayoutBudget()


def build_application_plan(
    job: Job,
    bank: CareerBank,
    layout: LayoutBudget | None = None,
) -> ApplicationPlan:
    budget = layout or DEFAULT_LAYOUT
    template, reason, _auto = select_template(job)

    exp_ranked = rank_experiences(job, bank, template)
    proj_ranked = rank_projects(job, bank, template)

    exp_selected = select_ranked(
        exp_ranked,
        min_count=budget.min_experiences,
        max_count=budget.max_experiences,
    )
    proj_selected = select_ranked(
        proj_ranked,
        min_count=budget.min_projects,
        max_count=budget.max_projects,
    )
    experience_ids = [item.record_id for item in exp_selected]
    project_ids = [item.record_id for item in proj_selected]

    # Highest relevance first; trim drops from the end.
    resume_priorities = [*experience_ids, *project_ids]

    combined = [
        *(item.record_id for item in exp_ranked if item.record_id in experience_ids),
        *(item.record_id for item in proj_ranked if item.record_id in project_ids),
    ]
    cover_sources = combined[:3]

    skill_groups = select_skills(job, bank.skills, template)
    if len(skill_groups) > budget.max_skill_groups:
        skill_groups = skill_groups[: budget.max_skill_groups]

    return ApplicationPlan(
        template=template,
        template_reason=reason,
        experience_ids=experience_ids,
        project_ids=project_ids,
        skill_groups=skill_groups,
        layout=budget,
        cover_letter=job.cover_letter,
        cover_letter_source_ids=cover_sources,
        resume_priorities=resume_priorities,
        experience_scores=[
            RankedSelection(record_id=item.record_id, score=item.score)
            for item in exp_selected
        ],
        project_scores=[
            RankedSelection(record_id=item.record_id, score=item.score)
            for item in proj_selected
        ],
    )


def selected_experiences(plan: ApplicationPlan, bank: CareerBank) -> list[ExperienceRecord]:
    lookup = bank.experience_by_id()
    return [lookup[record_id] for record_id in plan.experience_ids if record_id in lookup]


def selected_projects(plan: ApplicationPlan, bank: CareerBank) -> list[ProjectRecord]:
    lookup = bank.project_by_id()
    return [lookup[record_id] for record_id in plan.project_ids if record_id in lookup]
