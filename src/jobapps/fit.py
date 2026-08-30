"""PDF overflow classification and deterministic resume trimming."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from jobapps.models import (
    ApplicationPlan,
    CoverLetter,
    TailoredResume,
    is_stack_bullet,
)

OVERFULL_HBOX_RE = re.compile(r"Overfull \\hbox \(([\d.]+)pt")


@dataclass
class OverflowReport:
    resume_pages: int = 1
    cover_pages: int | None = None
    resume_overfull: list[str] = field(default_factory=list)
    cover_overfull: list[str] = field(default_factory=list)

    @property
    def resume_vertical(self) -> bool:
        return self.resume_pages > 1

    @property
    def cover_vertical(self) -> bool:
        return self.cover_pages is not None and self.cover_pages > 1

    @property
    def resume_horizontal(self) -> bool:
        return bool(self.resume_overfull)

    @property
    def cover_horizontal(self) -> bool:
        return bool(self.cover_overfull)

    @property
    def resume_overflow(self) -> bool:
        return self.resume_vertical or self.resume_horizontal

    @property
    def cover_overflow(self) -> bool:
        return self.cover_vertical or self.cover_horizontal


def parse_overfull_hbox(log_text: str) -> list[str]:
    return [f"{match.group(1)}pt" for match in OVERFULL_HBOX_RE.finditer(log_text or "")]


def _content_indices(bullets: list[str]) -> list[int]:
    return [index for index, bullet in enumerate(bullets) if not is_stack_bullet(bullet)]


def longest_content_bullet(resume: TailoredResume) -> tuple[str, int, int] | None:
    """Return (kind, item_index, bullet_index) for the longest content bullet."""
    best: tuple[int, str, int, int] | None = None
    for item_index, item in enumerate(resume.experience):
        for bullet_index, bullet in enumerate(item.bullets):
            if is_stack_bullet(bullet):
                continue
            candidate = (len(bullet), "experience", item_index, bullet_index)
            if best is None or candidate[0] > best[0]:
                best = candidate
    for item_index, item in enumerate(resume.projects):
        for bullet_index, bullet in enumerate(item.bullets):
            if is_stack_bullet(bullet):
                continue
            candidate = (len(bullet), "project", item_index, bullet_index)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        return None
    return best[1], best[2], best[3]


def _priority_index(plan: ApplicationPlan, record_key: str) -> int:
    try:
        return plan.resume_priorities.index(record_key)
    except ValueError:
        return len(plan.resume_priorities)


def _experience_key(resume: TailoredResume, index: int, plan: ApplicationPlan) -> str:
    if index < len(plan.experience_ids):
        return plan.experience_ids[index]
    return f"experience:{index}:{resume.experience[index].company}"


def _project_key(resume: TailoredResume, index: int, plan: ApplicationPlan) -> str:
    if index < len(plan.project_ids):
        return plan.project_ids[index]
    return f"project:{index}:{resume.projects[index].name}"


def drop_last_content_bullet(bullets: list[str]) -> list[str] | None:
    indices = _content_indices(bullets)
    if len(indices) <= 1:
        return None
    drop_at = indices[-1]
    return [bullet for index, bullet in enumerate(bullets) if index != drop_at]


def apply_python_trim(resume: TailoredResume, plan: ApplicationPlan) -> TailoredResume | None:
    """Apply one deterministic trim step. Lowest-priority content first.

    Order:
    1. Reduce project content bullets toward the layout target
    2. Reduce experience content bullets toward the layout minimum
    3. Remove the lowest-priority extra bullet
    4. Remove the lowest-priority project
    5. Remove the lowest-priority experience
    """
    layout = plan.layout

    over_projects = [
        index
        for index, item in enumerate(resume.projects)
        if len(_content_indices(item.bullets)) > layout.project_bullets
    ]
    if over_projects:
        index = max(over_projects, key=lambda i: _priority_index(plan, _project_key(resume, i, plan)))
        trimmed = drop_last_content_bullet(resume.projects[index].bullets)
        if trimmed is not None:
            projects = list(resume.projects)
            projects[index] = projects[index].model_copy(update={"bullets": trimmed})
            return resume.model_copy(update={"projects": projects})

    over_experiences = [
        index
        for index, item in enumerate(resume.experience)
        if len(_content_indices(item.bullets)) > layout.min_experience_bullets
    ]
    if over_experiences:
        index = max(
            over_experiences,
            key=lambda i: _priority_index(plan, _experience_key(resume, i, plan)),
        )
        trimmed = drop_last_content_bullet(resume.experience[index].bullets)
        if trimmed is not None:
            experiences = list(resume.experience)
            experiences[index] = experiences[index].model_copy(update={"bullets": trimmed})
            return resume.model_copy(update={"experience": experiences})

    # Drop the lowest-priority remaining extra content bullet across the resume.
    candidates: list[tuple[int, str, int]] = []
    for index, item in enumerate(resume.experience):
        if len(_content_indices(item.bullets)) > 1:
            candidates.append(
                (_priority_index(plan, _experience_key(resume, index, plan)), "experience", index)
            )
    for index, item in enumerate(resume.projects):
        if len(_content_indices(item.bullets)) > 1:
            candidates.append(
                (_priority_index(plan, _project_key(resume, index, plan)), "project", index)
            )
    if candidates:
        _prio, kind, index = max(candidates, key=lambda row: row[0])
        if kind == "experience":
            trimmed = drop_last_content_bullet(resume.experience[index].bullets)
            if trimmed is not None:
                experiences = list(resume.experience)
                experiences[index] = experiences[index].model_copy(update={"bullets": trimmed})
                return resume.model_copy(update={"experience": experiences})
        else:
            trimmed = drop_last_content_bullet(resume.projects[index].bullets)
            if trimmed is not None:
                projects = list(resume.projects)
                projects[index] = projects[index].model_copy(update={"bullets": trimmed})
                return resume.model_copy(update={"projects": projects})

    if len(resume.projects) > layout.min_projects:
        index = max(
            range(len(resume.projects)),
            key=lambda i: _priority_index(plan, _project_key(resume, i, plan)),
        )
        projects = [item for i, item in enumerate(resume.projects) if i != index]
        return resume.model_copy(update={"projects": projects})

    if len(resume.experience) > layout.min_experiences:
        index = max(
            range(len(resume.experience)),
            key=lambda i: _priority_index(plan, _experience_key(resume, i, plan)),
        )
        experiences = [item for i, item in enumerate(resume.experience) if i != index]
        return resume.model_copy(update={"experience": experiences})

    return None


def longest_cover_paragraph_index(cover: CoverLetter) -> int | None:
    if not cover.paragraphs:
        return None
    return max(range(len(cover.paragraphs)), key=lambda i: len(cover.paragraphs[i]))
