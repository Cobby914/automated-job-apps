"""Load and cache the structured career bank."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from jobapps.config import CAREER_DIR
from jobapps.models import Contact, Education, SkillGroup, allowed_skill_names, parse_skills_bank


class SourcedText(BaseModel):
    id: str
    text: str


class CanonicalBullet(BaseModel):
    id: str
    text: str
    sources: list[str] = Field(default_factory=list)


class ExperienceRecord(BaseModel):
    id: str
    company: str
    role: str
    location: str = ""
    start: str = ""
    end: str = ""
    summary: str = ""
    technologies: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    facts: list[SourcedText] = Field(default_factory=list)
    metrics: list[SourcedText] = Field(default_factory=list)
    bullets: list[CanonicalBullet] = Field(default_factory=list)
    tracks: list[str] = Field(default_factory=list)

    def source_ids(self) -> set[str]:
        ids = {self.id}
        ids.update(item.id for item in self.facts)
        ids.update(item.id for item in self.metrics)
        ids.update(item.id for item in self.bullets)
        return ids

    def prompt_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "company": self.company,
            "role": self.role,
            "location": self.location,
            "start": self.start,
            "end": self.end,
            "summary": self.summary,
            "technologies": self.technologies,
            "facts": [item.model_dump() for item in self.facts],
            "metrics": [item.model_dump() for item in self.metrics],
            "bullets": [item.model_dump() for item in self.bullets],
        }


class ProjectRecord(BaseModel):
    id: str
    name: str
    url: str = ""
    summary: str = ""
    technologies: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    facts: list[SourcedText] = Field(default_factory=list)
    metrics: list[SourcedText] = Field(default_factory=list)
    bullets: list[CanonicalBullet] = Field(default_factory=list)
    stack: str = ""
    tracks: list[str] = Field(default_factory=list)

    def source_ids(self) -> set[str]:
        ids = {self.id}
        ids.update(item.id for item in self.facts)
        ids.update(item.id for item in self.metrics)
        ids.update(item.id for item in self.bullets)
        return ids

    def content_bullets(self) -> list[CanonicalBullet]:
        return [item for item in self.bullets if not item.text.strip().lower().startswith("stack:")]

    def prompt_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "summary": self.summary,
            "technologies": self.technologies,
            "stack": self.stack,
            "facts": [item.model_dump() for item in self.facts],
            "metrics": [item.model_dump() for item in self.metrics],
            "bullets": [item.model_dump() for item in self.bullets],
        }


class CareerProfile(BaseModel):
    contact: Contact
    education: list[Education] = Field(default_factory=list)


class SkillsInventory(BaseModel):
    categories: dict[str, list[str]] = Field(default_factory=dict)
    recommended: list[SkillGroup] = Field(default_factory=list)

    def allowed_names(self) -> set[str]:
        lines = []
        for heading, items in self.categories.items():
            lines.append(f"## {heading}")
            lines.extend(f"- {item}" for item in items)
        if self.recommended:
            lines.append("## Recommended General-Purpose Skills Section")
            for group in self.recommended:
                lines.append(f"**{group.category}:** {group.items}")
        return allowed_skill_names("\n".join(lines))

    def all_skill_items(self) -> list[tuple[str, str]]:
        """Return (category, item) pairs from inventory categories."""
        pairs: list[tuple[str, str]] = []
        for category, items in self.categories.items():
            for item in items:
                pairs.append((category, item))
        return pairs


class CareerBank(BaseModel):
    profile: CareerProfile
    experiences: list[ExperienceRecord] = Field(default_factory=list)
    projects: list[ProjectRecord] = Field(default_factory=list)
    skills: SkillsInventory = Field(default_factory=SkillsInventory)

    def experience_by_id(self) -> dict[str, ExperienceRecord]:
        return {item.id: item for item in self.experiences}

    def project_by_id(self) -> dict[str, ProjectRecord]:
        return {item.id: item for item in self.projects}

    def record_by_id(self, record_id: str) -> ExperienceRecord | ProjectRecord:
        exp = self.experience_by_id().get(record_id)
        if exp is not None:
            return exp
        project = self.project_by_id().get(record_id)
        if project is not None:
            return project
        raise KeyError(f"Unknown career record id: {record_id}")


def _read_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_skills_inventory(path: Path) -> SkillsInventory:
    data = _read_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"Skills file must be a YAML mapping: {path}")
    categories = data.get("categories") or {}
    if not isinstance(categories, dict):
        raise ValueError(f"Skills categories must be a mapping: {path}")
    recommended_raw = data.get("recommended") or []
    recommended = [SkillGroup.model_validate(item) for item in recommended_raw]
    parsed: dict[str, list[str]] = {}
    for heading, items in categories.items():
        parsed[str(heading)] = [str(item) for item in items or []]
    return SkillsInventory(categories=parsed, recommended=recommended)


def _validate_unique_ids(bank: CareerBank) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in [*bank.experiences, *bank.projects]:
        for source_id in sorted(record.source_ids()):
            if source_id in seen:
                duplicates.append(source_id)
            seen.add(source_id)
    if duplicates:
        raise ValueError("Duplicate career source ids: " + ", ".join(duplicates[:10]))


def load_career_bank(directory: Path | None = None) -> CareerBank:
    root = directory or CAREER_DIR
    profile_data = _read_yaml(root / "profile.yaml")
    if not isinstance(profile_data, dict):
        raise ValueError(f"Career profile must be a YAML mapping: {root / 'profile.yaml'}")
    skills_data = profile_data.get("skills")
    if isinstance(skills_data, dict):
        profile_data = dict(profile_data)
        profile_data["skills"] = [
            {"category": key, "items": value} for key, value in skills_data.items()
        ]
    profile = CareerProfile.model_validate(profile_data)

    experiences_raw = _read_yaml(root / "experiences.yaml")
    if not isinstance(experiences_raw, list):
        raise ValueError(f"Experiences file must be a YAML list: {root / 'experiences.yaml'}")
    projects_raw = _read_yaml(root / "projects.yaml")
    if not isinstance(projects_raw, list):
        raise ValueError(f"Projects file must be a YAML list: {root / 'projects.yaml'}")

    bank = CareerBank(
        profile=profile,
        experiences=[ExperienceRecord.model_validate(item) for item in experiences_raw],
        projects=[ProjectRecord.model_validate(item) for item in projects_raw],
        skills=load_skills_inventory(root / "skills.yaml"),
    )
    _validate_unique_ids(bank)
    return bank


@lru_cache(maxsize=1)
def get_career_bank() -> CareerBank:
    """Process-level cache: parse career YAML once per worker."""
    return load_career_bank()


def skills_bank_markdown(skills: SkillsInventory) -> str:
    """Render inventory as markdown so existing skill helpers can reuse it."""
    lines: list[str] = []
    for heading, items in skills.categories.items():
        lines.append(f"## {heading}")
        lines.append("")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
    if skills.recommended:
        lines.append("## Recommended General-Purpose Skills Section")
        lines.append("")
        for group in skills.recommended:
            lines.append(f"**{group.category}:** {group.items}")
    return "\n".join(lines).strip() + "\n"


def skills_as_parseable_text(skills: SkillsInventory) -> str:
    return skills_bank_markdown(skills)


def parse_inventory_as_bank(skills: SkillsInventory) -> tuple[dict[str, list[str]], list[SkillGroup]]:
    return parse_skills_bank(skills_bank_markdown(skills))
