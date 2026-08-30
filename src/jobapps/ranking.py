"""Deterministic template scoring, experience/project ranking, and skill matching."""

from __future__ import annotations

import re
from dataclasses import dataclass

from jobapps.career import CareerBank, ExperienceRecord, ProjectRecord, SkillsInventory
from jobapps.models import (
    AUTO_TEMPLATE_ALIASES,
    RESUME_TEMPLATE_NAMES,
    Job,
    SkillGroup,
    split_skill_items,
)

_TOKEN_RE = re.compile(r"[a-z0-9+#]+")

SWE_KEYWORDS = frozenset(
    {
        "backend",
        "frontend",
        "fullstack",
        "full-stack",
        "api",
        "apis",
        "database",
        "databases",
        "sql",
        "cloud",
        "web",
        "devops",
        "sre",
        "security",
        "platform",
        "infrastructure",
        "microservices",
        "distributed",
        "react",
        "node",
        "typescript",
        "javascript",
        "postgresql",
        "rest",
        "graphql",
        "shipping",
        "production",
        "software",
        "engineer",
        "services",
        "product",
    }
)

AI_KEYWORDS = frozenset(
    {
        "ml",
        "ai",
        "pytorch",
        "tensorflow",
        "robotics",
        "autonomy",
        "autonomous",
        "perception",
        "research",
        "neural",
        "nlp",
        "llm",
        "vision",
        "radar",
        "camera",
        "multimodal",
        "carla",
        "simulation",
        "sensor",
        "pointnet",
        "gnn",
        "reinforcement",
        "ppo",
        "dataset",
        "inference",
        "training",
    }
)

AI_PHRASES = (
    "machine learning",
    "deep learning",
    "computer vision",
    "data science",
    "sensor fusion",
    "autonomous driving",
    "reinforcement learning",
)

SWE_PHRASES = (
    "software engineer",
    "full stack",
    "full-stack",
    "rest api",
    "web application",
    "backend engineer",
    "frontend engineer",
)

_SCORE_MARGIN = 2

_CATEGORY_ALIASES = {
    "backend & apis": "Backend/Data",
    "databases & data": "Backend/Data",
    "frontend": "Frontend",
    "ai / machine learning": "AI/ML",
    "autonomy & simulation": "AI/ML",
    "systems & embedded": "Systems/Tools",
    "cloud & devops": "Systems/Tools",
    "developer tools": "Systems/Tools",
    "languages": "Languages",
}


def tokenize(text: str) -> set[str]:
    lowered = (text or "").casefold().replace("c++", "cplusplus").replace("c#", "csharp")
    return set(_TOKEN_RE.findall(lowered))


def job_text(job: Job) -> str:
    return "\n".join(part for part in (job.title, job.description, job.notes) if part)


def score_template(job: Job) -> tuple[str, str, dict[str, int]]:
    """Return (template, reason, raw scores) from keyword overlap."""
    blob = job_text(job).casefold()
    tokens = tokenize(blob)
    swe = sum(1 for word in SWE_KEYWORDS if word in tokens or word in blob)
    ai = sum(1 for word in AI_KEYWORDS if word in tokens or word in blob)
    swe += sum(2 for phrase in SWE_PHRASES if phrase in blob)
    ai += sum(2 for phrase in AI_PHRASES if phrase in blob)
    scores = {"swe": swe, "ai": ai}
    if ai >= swe + _SCORE_MARGIN:
        return "ai", f"AI keywords outscored SWE ({ai} vs {swe}).", scores
    if swe >= ai + _SCORE_MARGIN:
        return "swe", f"SWE keywords outscored AI ({swe} vs {ai}).", scores
    return "default", f"SWE and AI scores were close ({swe} vs {ai}).", scores


def select_template(job: Job) -> tuple[str, str, bool]:
    """Honor an explicit YAML override; otherwise score the job description."""
    key = job.template.strip().lower()
    if key not in AUTO_TEMPLATE_ALIASES:
        if key not in RESUME_TEMPLATE_NAMES:
            allowed = ", ".join(sorted(RESUME_TEMPLATE_NAMES | AUTO_TEMPLATE_ALIASES))
            raise ValueError(f"Unknown template {job.template!r}; use auto or one of: {allowed}.")
        return key, "Set explicitly in job YAML.", False
    template, reason, _scores = score_template(job)
    return template, reason, True


@dataclass(frozen=True)
class RankedItem:
    record_id: str
    score: float
    kind: str  # "experience" or "project"


def _record_corpus(record: ExperienceRecord | ProjectRecord) -> str:
    parts = [
        " ".join(record.technologies),
        " ".join(record.tags),
        " ".join(record.domains),
        record.summary,
    ]
    if isinstance(record, ExperienceRecord):
        parts.extend([record.company, record.role])
    else:
        parts.append(record.name)
        parts.extend(record.stack.split(","))
    return " ".join(parts)


def score_record(job: Job, record: ExperienceRecord | ProjectRecord, template: str) -> float:
    hay = _record_corpus(record).casefold()
    blob = job_text(job).casefold()
    job_tokens = tokenize(blob)
    record_tokens = tokenize(hay)
    overlap = len(job_tokens & record_tokens)
    tech_hits = 0
    for tech in record.technologies:
        name = tech.casefold()
        if name and name in blob:
            tech_hits += 2
    tag_hits = sum(1 for tag in record.tags if tag.casefold().replace("-", " ") in blob)
    domain_hits = sum(1 for domain in record.domains if domain.casefold().replace("-", " ") in blob)
    track_bonus = 3.0 if template in {track.casefold() for track in record.tracks} else 0.0
    # Untagged extras (unity-rl, unix-shell, emg) can still win on overlap.
    return float(overlap + tech_hits + tag_hits + domain_hits + track_bonus)


def rank_experiences(job: Job, bank: CareerBank, template: str) -> list[RankedItem]:
    ranked = [
        RankedItem(record.id, score_record(job, record, template), "experience")
        for record in bank.experiences
    ]
    ranked.sort(key=lambda item: (-item.score, item.record_id))
    return ranked


def rank_projects(job: Job, bank: CareerBank, template: str) -> list[RankedItem]:
    ranked = [
        RankedItem(record.id, score_record(job, record, template), "project")
        for record in bank.projects
    ]
    ranked.sort(key=lambda item: (-item.score, item.record_id))
    return ranked


def _normalize_skill(name: str) -> str:
    return " ".join(name.strip().split()).casefold()


def _compact_category(heading: str) -> str:
    key = heading.strip().casefold()
    return _CATEGORY_ALIASES.get(key, heading.strip())


def select_skills(job: Job, skills: SkillsInventory, template: str) -> list[SkillGroup]:
    """Whitelist skill names from the job description; never invent."""
    blob = job_text(job)
    blob_cf = blob.casefold()
    inventory = skills.all_skill_items()
    # Longest names first so "GitHub Actions" wins over "Git".
    inventory.sort(key=lambda pair: len(pair[1]), reverse=True)
    matched: dict[str, list[str]] = {}
    used: set[str] = set()
    for category, item in inventory:
        key = _normalize_skill(item)
        if key in used:
            continue
        pattern = re.compile(r"(?<![a-z0-9])" + re.escape(item.casefold()) + r"(?![a-z0-9])")
        if pattern.search(blob_cf):
            compact = _compact_category(category)
            matched.setdefault(compact, [])
            if item not in matched[compact]:
                matched[compact].append(item)
            used.add(key)

    groups: list[SkillGroup] = []
    recommended = list(skills.recommended)
    if template == "ai":
        recommended.sort(key=lambda g: 0 if "ai" in g.category.casefold() else 1)
    elif template == "swe":
        recommended.sort(key=lambda g: 0 if "backend" in g.category.casefold() else 1)

    seed_categories = [group.category for group in recommended]
    for extra in matched:
        if extra not in seed_categories:
            seed_categories.append(extra)

    for category in seed_categories:
        default = next((g for g in recommended if g.category == category), None)
        names: list[str] = []
        if category in matched:
            names.extend(matched[category])
        if default:
            for item in split_skill_items(default.items):
                if _normalize_skill(item) not in {_normalize_skill(n) for n in names}:
                    names.append(item)
        if not names:
            continue
        groups.append(SkillGroup(category=category, items=", ".join(names)))
        if len(groups) >= 5:
            break

    if len(groups) < 3:
        for group in recommended:
            if all(g.category != group.category for g in groups):
                groups.append(group)
            if len(groups) >= 3:
                break
    return groups[:5]
