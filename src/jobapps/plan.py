"""Build an ApplicationPlan from the career bank and a job posting."""

from __future__ import annotations

from jobapps.career import CareerBank, ExperienceRecord, ProjectRecord
from jobapps.models import ApplicationPlan, Job, LayoutBudget, RankedSelection
from jobapps.ranking import (
    append_ranking_log,
    rank_experiences,
    rank_projects,
    select_courses,
    select_ranked,
    select_skills,
    select_template,
)


DEFAULT_LAYOUT = LayoutBudget()


def layout_for_template(template: str, layout: LayoutBudget | None = None) -> LayoutBudget:
    """AI resumes get a fuller Research section; other tracks stay tighter."""
    budget = (layout or DEFAULT_LAYOUT).model_copy()
    if template.strip().casefold() == "ai":
        budget = budget.model_copy(update={"max_research_bullets": 4})
    else:
        budget = budget.model_copy(update={"max_research_bullets": 2})
    return budget


def _selections(ranked_items, selected_ids: list[str]) -> list[RankedSelection]:
    lookup = {item.record_id: item for item in ranked_items}
    selections: list[RankedSelection] = []
    for index, record_id in enumerate(selected_ids, start=1):
        item = lookup.get(record_id)
        if item is None:
            continue
        selections.append(
            RankedSelection(
                record_id=item.record_id,
                score=item.score,
                priority=index,
                explanation=item.explanation,
                matched_terms=list(item.matched_terms),
            )
        )
    return selections


def build_application_plan(
    job: Job,
    bank: CareerBank,
    layout: LayoutBudget | None = None,
    reuse: ApplicationPlan | None = None,
) -> ApplicationPlan:
    template, reason, _auto = select_template(job)
    budget = layout_for_template(template, layout)
    research_ids = [item.id for item in bank.research_experiences()]

    if reuse is not None and reuse.template == template:
        blocked = set(research_ids)
        experience_ids = [record_id for record_id in reuse.experience_ids if record_id not in blocked]
        priorities = [record_id for record_id in reuse.resume_priorities if record_id not in blocked]
        for record_id in research_ids:
            if record_id not in priorities:
                priorities.append(record_id)
        coursework = list(reuse.coursework) or select_courses(
            bank.courses, template, max_count=budget.max_courses
        )
        plan = reuse.model_copy(
            update={
                "cover_letter": job.cover_letter,
                "template_reason": f"{reason} Reused ranking from a similar posting.",
                "experience_ids": experience_ids,
                "research_ids": research_ids,
                "coursework": coursework,
                "layout": budget,
                "resume_priorities": priorities,
                "experience_scores": [
                    item for item in reuse.experience_scores if item.record_id not in blocked
                ],
            }
        )
        append_ranking_log(
            job,
            rank_experiences(job, bank, template),
            rank_projects(job, bank, template),
            plan.experience_ids,
            plan.project_ids,
            reused=True,
        )
        return plan

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
    resume_priorities = [*experience_ids, *project_ids, *research_ids]

    combined = [
        *(item.record_id for item in exp_ranked if item.record_id in experience_ids),
        *(item.record_id for item in proj_ranked if item.record_id in project_ids),
    ]
    cover_sources = combined[:3]

    skill_groups = select_skills(job, bank.skills, template)
    if len(skill_groups) > budget.max_skill_groups:
        skill_groups = skill_groups[: budget.max_skill_groups]
    coursework = select_courses(bank.courses, template, max_count=budget.max_courses)

    append_ranking_log(job, exp_ranked, proj_ranked, experience_ids, project_ids)

    return ApplicationPlan(
        template=template,
        template_reason=reason,
        experience_ids=experience_ids,
        research_ids=research_ids,
        project_ids=project_ids,
        coursework=coursework,
        skill_groups=skill_groups,
        layout=budget,
        cover_letter=job.cover_letter,
        cover_letter_source_ids=cover_sources,
        resume_priorities=resume_priorities,
        experience_scores=_selections(exp_ranked, experience_ids),
        project_scores=_selections(proj_ranked, project_ids),
    )


def selected_experiences(plan: ApplicationPlan, bank: CareerBank) -> list[ExperienceRecord]:
    lookup = bank.experience_by_id()
    return [lookup[record_id] for record_id in plan.experience_ids if record_id in lookup]


def selected_research(plan: ApplicationPlan, bank: CareerBank) -> list[ExperienceRecord]:
    lookup = bank.experience_by_id()
    return [lookup[record_id] for record_id in plan.research_ids if record_id in lookup]


def selected_projects(plan: ApplicationPlan, bank: CareerBank) -> list[ProjectRecord]:
    lookup = bank.project_by_id()
    return [lookup[record_id] for record_id in plan.project_ids if record_id in lookup]
