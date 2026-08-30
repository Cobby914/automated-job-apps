"""Pydantic models and YAML loaders."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

MAX_BULLET_CHARS = 113
MIN_BULLET_CHARS = 90
RECOMMENDED_SKILLS_HEADING = "Recommended General-Purpose Skills Section"
RESUME_TEMPLATE_NAMES = frozenset({"swe", "ai", "default"})
AUTO_TEMPLATE_ALIASES = frozenset({"auto", ""})

_SKILLS_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
_SKILLS_BULLET_RE = re.compile(r"^-\s+(.+?)\s*$")
_SKILLS_BOLD_LINE_RE = re.compile(r"^\*\*(.+?):\*\*\s*(.+?)\s*$")


class ApplicationQuestion(BaseModel):
    prompt: str
    # Character limit for the answer; None means unlimited.
    max_length: int | None = None

    @field_validator("prompt", mode="before")
    @classmethod
    def require_prompt(cls, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("Question prompt must be non-empty.")
        return text

    @field_validator("max_length", mode="before")
    @classmethod
    def normalize_max_length(cls, value: object) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, str) and value.strip().lower() in {"unlimited", "none", "null"}:
            return None
        length = int(value)  # type: ignore[arg-type]
        if length < 1:
            raise ValueError("max_length must be a positive integer or omitted for unlimited.")
        return length


class Job(BaseModel):
    company: str
    title: str
    portal_url: str = ""
    description: str
    notes: str = ""
    template: str = "auto"
    # Optional role start hint, e.g. "Summer 2027", "Fall 2027", "2027-09".
    starts: str = ""
    # Optional force: "June 2027" or "Dec. 2027". When empty, inferred from starts.
    graduation: str = ""
    # Skip cover-letter generation, review, and PDF when false.
    cover_letter: bool = True
    # Optional screening / portal questions. Empty → skip answer generation.
    questions: list[ApplicationQuestion] = Field(default_factory=list)


def resolve_template_request(value: str) -> str | None:
    """Return an explicit template name, or None when the model should choose."""
    key = value.strip().lower()
    if key in AUTO_TEMPLATE_ALIASES:
        return None
    if key in RESUME_TEMPLATE_NAMES:
        return key
    allowed = ", ".join(sorted(RESUME_TEMPLATE_NAMES | AUTO_TEMPLATE_ALIASES))
    raise ValueError(f"Unknown template {value!r}; use auto or one of: {allowed}.")


class TemplateChoice(BaseModel):
    template: str
    reason: str = ""

    @field_validator("template", mode="before")
    @classmethod
    def normalize_template(cls, value: object) -> str:
        key = str(value).strip().lower()
        if key not in RESUME_TEMPLATE_NAMES:
            raise ValueError(f"Template must be one of: {', '.join(sorted(RESUME_TEMPLATE_NAMES))}")
        return key


class Contact(BaseModel):
    name: str
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    website: str = ""


def is_stack_bullet(bullet: str) -> bool:
    return bullet.strip().lower().startswith("stack:")


class Experience(BaseModel):
    company: str
    role: str
    location: str = ""
    start: str = ""
    end: str = ""
    bullets: list[str] = Field(default_factory=list)


class Education(BaseModel):
    school: str
    degree: str
    location: str = ""
    year: str = ""
    details: str = ""

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)


class Project(BaseModel):
    name: str
    url: str = ""
    bullets: list[str] = Field(default_factory=list)


class SkillGroup(BaseModel):
    category: str
    items: str


class Resume(BaseModel):
    contact: Contact
    summary: str = ""
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: list[SkillGroup] = Field(default_factory=list)


class Connection(BaseModel):
    name: str
    company: str
    relationship: str = ""
    notes: str = ""
    aliases: list[str] = Field(default_factory=list)


class TailoredResume(BaseModel):
    summary: str = ""
    experience: list[Experience]
    projects: list[Project] = Field(default_factory=list)
    education: list[Education]
    skills: list[SkillGroup] = Field(default_factory=list)


class CoverLetter(BaseModel):
    greeting: str
    paragraphs: list[str]
    closing: str


class SourcedBullet(BaseModel):
    text: str
    sources: list[str] = Field(default_factory=list)


def coerce_sourced_bullets(value: object) -> object:
    if not isinstance(value, list):
        return value
    coerced: list[object] = []
    for item in value:
        if isinstance(item, str):
            coerced.append({"text": item, "sources": []})
        else:
            coerced.append(item)
    return coerced


class DraftExperience(BaseModel):
    company: str
    role: str
    location: str = ""
    start: str = ""
    end: str = ""
    bullets: list[SourcedBullet] = Field(default_factory=list)

    @field_validator("bullets", mode="before")
    @classmethod
    def coerce_bullets(cls, value: object) -> object:
        return coerce_sourced_bullets(value)

    def to_experience(self) -> Experience:
        return Experience(
            company=self.company,
            role=self.role,
            location=self.location,
            start=self.start,
            end=self.end,
            bullets=[item.text for item in self.bullets],
        )


class DraftProject(BaseModel):
    name: str
    url: str = ""
    bullets: list[SourcedBullet] = Field(default_factory=list)

    @field_validator("bullets", mode="before")
    @classmethod
    def coerce_bullets(cls, value: object) -> object:
        return coerce_sourced_bullets(value)

    def to_project(self) -> Project:
        return Project(
            name=self.name,
            url=self.url,
            bullets=[item.text for item in self.bullets],
        )


class DraftResume(BaseModel):
    summary: str = ""
    experience: list[DraftExperience] = Field(default_factory=list)
    projects: list[DraftProject] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[SkillGroup] = Field(default_factory=list)

    def to_tailored(self) -> TailoredResume:
        return TailoredResume(
            summary=self.summary,
            experience=[item.to_experience() for item in self.experience],
            projects=[item.to_project() for item in self.projects],
            education=self.education,
            skills=self.skills,
        )


def draft_from_tailored(resume: TailoredResume) -> DraftResume:
    """Wrap a rendered resume as a draft with empty source lists."""
    return DraftResume(
        summary=resume.summary,
        experience=[
            DraftExperience(
                company=item.company,
                role=item.role,
                location=item.location,
                start=item.start,
                end=item.end,
                bullets=[SourcedBullet(text=bullet) for bullet in item.bullets],
            )
            for item in resume.experience
        ],
        projects=[
            DraftProject(
                name=item.name,
                url=item.url,
                bullets=[SourcedBullet(text=bullet) for bullet in item.bullets],
            )
            for item in resume.projects
        ],
        education=resume.education,
        skills=resume.skills,
    )


class GenerationResult(BaseModel):
    resume: TailoredResume
    cover_letter: CoverLetter | None = None


class ApplicationAnswer(BaseModel):
    prompt: str
    answer: str
    max_length: int | None = None


class ApplicationAnswersResult(BaseModel):
    answers: list[ApplicationAnswer]


class ReviewIssue(BaseModel):
    location: str = "resume"
    code: str = "other"
    type: str = ""
    section: str = ""
    item_id: str = ""
    bullet_index: int | None = None
    paragraph_index: int | None = None
    message: str
    severity: str = "error"

    @model_validator(mode="before")
    @classmethod
    def sync_type_code_and_section(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        code = str(payload.get("code") or "").strip()
        kind = str(payload.get("type") or "").strip()
        if kind and not code:
            payload["code"] = kind
            code = kind
        elif code and not kind:
            payload["type"] = code
            kind = code
        elif not code and not kind:
            payload["code"] = "other"
            payload["type"] = "other"
        section = str(payload.get("section") or "").strip()
        if not section:
            inferred = _section_from_location(str(payload.get("location") or ""))
            if inferred:
                payload["section"] = inferred
        return payload


class ParsedLocation(BaseModel):
    kind: str
    item_index: int = 0
    part_index: int = 0
    item_id: str = ""


def _section_from_location(location: str) -> str:
    parsed = parse_issue_location(location)
    if parsed is None:
        return ""
    return parsed.kind


def coerce_review_issues(value: object) -> object:
    if not isinstance(value, list):
        return value
    coerced: list[object] = []
    for item in value:
        if isinstance(item, str):
            coerced.append(
                {
                    "location": "resume",
                    "code": "other",
                    "type": "other",
                    "section": "resume",
                    "message": item,
                    "severity": "error",
                }
            )
        else:
            coerced.append(item)
    return coerced


def parse_issue_location(location: str) -> ParsedLocation | None:
    """Parse checker locations like experience[0].bullets[1] or cover_letter.paragraphs[0]."""
    text = (location or "").strip().casefold().replace(" ", "").replace("_", "")
    if not text or text in {"resume", "skills", "education", "summary"}:
        return ParsedLocation(kind="resume")
    match = re.fullmatch(r"experience\[(\d+)\](?:\.bullets\[(\d+)\])?", text)
    if match:
        return ParsedLocation(
            kind="experience",
            item_index=int(match.group(1)),
            part_index=int(match.group(2) or 0),
        )
    match = re.fullmatch(r"projects?\[(\d+)\](?:\.bullets\[(\d+)\])?", text)
    if match:
        return ParsedLocation(
            kind="project",
            item_index=int(match.group(1)),
            part_index=int(match.group(2) or 0),
        )
    match = re.fullmatch(r"coverletter(?:\.paragraphs\[(\d+)\])?", text)
    if match:
        return ParsedLocation(
            kind="cover_letter",
            part_index=int(match.group(1) or 0),
        )
    match = re.fullmatch(r"answers\[(\d+)\]", text)
    if match:
        return ParsedLocation(kind="answers", part_index=int(match.group(1)))
    return None


def review_issue_text(review: ReviewResult) -> str:
    if review.issues:
        parts = []
        for item in review.issues:
            loc = item.location.strip()
            parts.append(f"{loc}: {item.message}" if loc else item.message)
        return "; ".join(parts)
    return review.summary


def issue_feedback(issue: ReviewIssue) -> str:
    """Feedback for a single review issue, used by targeted repair."""
    if issue.item_id.strip():
        loc = issue.item_id.strip()
        if issue.bullet_index is not None:
            loc = f"{loc} bullet {issue.bullet_index}"
        elif issue.paragraph_index is not None:
            loc = f"{loc} paragraph {issue.paragraph_index}"
    else:
        loc = issue.location.strip()
    return f"{loc}: {issue.message}" if loc else issue.message


class ReviewResult(BaseModel):
    approved: bool
    summary: str = ""
    issues: list[ReviewIssue] = Field(default_factory=list)

    @field_validator("issues", mode="before")
    @classmethod
    def coerce_issues(cls, value: object) -> object:
        return coerce_review_issues(value)


class RankedSelection(BaseModel):
    record_id: str
    score: float


class LayoutBudget(BaseModel):
    min_experiences: int = 3
    max_experiences: int = 4
    min_projects: int = 2
    max_projects: int = 3
    min_experience_bullets: int = 2
    max_experience_bullets: int = 3
    project_bullets: int = 2
    min_bullet_chars: int = MIN_BULLET_CHARS
    max_bullet_chars: int = MAX_BULLET_CHARS
    min_skill_groups: int = 3
    max_skill_groups: int = 5


class ApplicationPlan(BaseModel):
    template: str
    template_reason: str = ""
    experience_ids: list[str] = Field(default_factory=list)
    project_ids: list[str] = Field(default_factory=list)
    skill_groups: list[SkillGroup] = Field(default_factory=list)
    layout: LayoutBudget = Field(default_factory=LayoutBudget)
    cover_letter: bool = True
    cover_letter_source_ids: list[str] = Field(default_factory=list)
    resume_priorities: list[str] = Field(default_factory=list)
    experience_scores: list[RankedSelection] = Field(default_factory=list)
    project_scores: list[RankedSelection] = Field(default_factory=list)


class PipelineMetrics(BaseModel):
    experience_count: int = 0
    project_count: int = 0
    candidate_experiences: int = 0
    selected_experiences: int = 0
    candidate_projects: int = 0
    selected_projects: int = 0
    context_chars: int = 0
    validation_failures: int = 0
    bullet_rewrites: int = 0
    cover_letter_repairs: int = 0
    page_fit_attempts: int = 0
    generation_attempts: int = 0
    semantic_review_failures: int = 0
    initial_resume_pages: int | None = None
    final_resume_pages: int | None = None
    initial_cover_pages: int | None = None
    final_cover_pages: int | None = None
    checker_escalated: bool = False
    semantic_revisions: int = 0


def overlong_answer_issues(result: ApplicationAnswersResult) -> list[str]:
    """Return issues for answers that exceed their per-question max_length."""
    issues: list[str] = []
    for index, item in enumerate(result.answers, start=1):
        if item.max_length is None:
            continue
        length = len(item.answer)
        if length > item.max_length:
            preview = item.answer if length <= 80 else f"{item.answer[:77]}..."
            issues.append(
                f"Answer {index} is {length} characters (limit {item.max_length}): {preview}"
            )
    return issues


def format_answers_markdown(result: ApplicationAnswersResult) -> str:
    """Render application Q&A as a human-readable markdown document."""
    if not result.answers:
        return ""
    parts: list[str] = ["# Application answers", ""]
    for index, item in enumerate(result.answers, start=1):
        limit = "unlimited" if item.max_length is None else f"{item.max_length} characters"
        parts.append(f"## {index}. {item.prompt}")
        parts.append("")
        parts.append(f"*Max length: {limit} · Actual: {len(item.answer)} characters*")
        parts.append("")
        parts.append(item.answer.strip())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def overlong_bullet_issues(resume: TailoredResume, max_chars: int = MAX_BULLET_CHARS) -> list[str]:
    """Return human-readable issues for bullets that exceed one printed line."""
    issues: list[str] = []
    for item in resume.experience:
        for bullet in item.bullets:
            if is_stack_bullet(bullet):
                continue
            length = len(bullet)
            if length > max_chars:
                preview = bullet if length <= 80 else f"{bullet[:77]}..."
                issues.append(
                    f"Experience {item.company} / {item.role}: {length} chars (max {max_chars}): {preview}"
                )
    for item in resume.projects:
        for bullet in item.bullets:
            if is_stack_bullet(bullet):
                continue
            length = len(bullet)
            if length > max_chars:
                preview = bullet if length <= 80 else f"{bullet[:77]}..."
                issues.append(
                    f"Project {item.name}: {length} chars (max {max_chars}): {preview}"
                )
    return issues


def split_skill_items(items: str) -> list[str]:
    return [part.strip() for part in items.split(",") if part.strip()]


def normalize_skill_name(name: str) -> str:
    return " ".join(name.strip().split()).casefold()


def parse_skills_bank(text: str) -> tuple[dict[str, list[str]], list[SkillGroup]]:
    """Parse a markdown skills bank into inventory categories and the default mix.

    The recommended general-purpose section is returned as default groups, not as
    a printed inventory category. Extra technology bullets stay in inventory.
    """
    categories: dict[str, list[str]] = {}
    default_groups: list[SkillGroup] = []
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = _SKILLS_HEADING_RE.match(line)
        if heading:
            current = heading.group(1).strip()
            if current.casefold() != RECOMMENDED_SKILLS_HEADING.casefold():
                categories.setdefault(current, [])
            continue
        if current is None:
            continue
        if current.casefold() == RECOMMENDED_SKILLS_HEADING.casefold():
            bold = _SKILLS_BOLD_LINE_RE.match(line)
            if bold:
                default_groups.append(
                    SkillGroup(category=bold.group(1).strip(), items=bold.group(2).strip())
                )
            continue
        bullet = _SKILLS_BULLET_RE.match(line)
        if bullet:
            categories[current].append(bullet.group(1).strip())
    return categories, default_groups


def allowed_skill_names(text: str) -> set[str]:
    """Case-insensitive allowed skill names from the full bank, including extras."""
    categories, default_groups = parse_skills_bank(text)
    names: set[str] = set()
    for items in categories.values():
        for item in items:
            names.add(normalize_skill_name(item))
    for group in default_groups:
        for item in split_skill_items(group.items):
            names.add(normalize_skill_name(item))
    return names


def skill_is_allowed(item: str, allowed: set[str]) -> bool:
    key = normalize_skill_name(item)
    if key in allowed:
        return True
    if "/" not in item:
        return False
    parts = [part.strip() for part in item.split("/") if part.strip()]
    return len(parts) >= 2 and all(normalize_skill_name(part) in allowed for part in parts)


def unknown_skill_issues(resume: TailoredResume, allowed: set[str]) -> list[str]:
    """Return issues for skill items that are not in the allowed bank."""
    issues: list[str] = []
    for group in resume.skills:
        for item in split_skill_items(group.items):
            if not skill_is_allowed(item, allowed):
                issues.append(
                    f"Skill {group.category}: {item!r} is not in the skills bank"
                )
    return issues


def load_skills_bank(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Skills bank not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Skills bank is empty: {path}")
    return text


def _read_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_job(path: Path) -> Job:
    data = _read_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"Job file must be a YAML mapping: {path}")
    return Job.model_validate(data)


def load_resume(path: Path) -> Resume:
    data = _read_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"Resume file must be a YAML mapping: {path}")
    skills = data.get("skills")
    if isinstance(skills, dict):
        data["skills"] = [{"category": key, "items": value} for key, value in skills.items()]
    return Resume.model_validate(data)


def _load_text_notes(directory: Path, prefix: str = "") -> str:
    if not directory.is_dir():
        return ""
    parts: list[str] = []
    for path in sorted(directory.glob("*.md")) + sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            label = f"{prefix}{path.stem}" if prefix else path.stem
            parts.append(f"### {label}\n{text}")
    return "\n\n".join(parts)


def load_additions(directory: Path) -> str:
    parts: list[str] = []
    for subdir in ("experiences", "projects"):
        notes = _load_text_notes(directory / subdir, prefix=f"{subdir}/")
        if notes:
            parts.append(notes)
    return "\n\n".join(parts)


def load_connections(path: Path) -> list[Connection]:
    if not path.is_file():
        return []
    data = _read_yaml(path)
    if not data:
        return []
    if not isinstance(data, list):
        raise ValueError(f"Connections file must be a YAML list: {path}")
    return [Connection.model_validate(item) for item in data]


def load_cover_letter_examples(directory: Path) -> str:
    return _load_text_notes(directory)


def load_writing_samples(directory: Path) -> str:
    return _load_text_notes(directory)


def dump_yaml(data: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
