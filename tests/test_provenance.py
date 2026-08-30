from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobapps.career import load_career_bank
from jobapps.models import (
    DraftExperience,
    DraftProject,
    DraftResume,
    Job,
    SourcedBullet,
)
from jobapps.plan import build_application_plan, selected_experiences, selected_projects
from jobapps.validate import source_issues, validate_draft_resume, validate_sources

ROOT = Path(__file__).resolve().parents[1]


class ProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = load_career_bank(ROOT / "career")

    def test_canonical_bullets_cite_fact_or_metric_ids(self) -> None:
        for record in [*self.bank.experiences, *self.bank.projects]:
            known = record.fact_metric_ids()
            for bullet in record.bullets:
                if bullet.text.strip().lower().startswith("stack:"):
                    continue
                self.assertTrue(bullet.sources, bullet.id)
                for source_id in bullet.sources:
                    self.assertIn(source_id, known, f"{bullet.id} -> {source_id}")

    def test_draft_rejects_empty_and_unknown_sources(self) -> None:
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
        )
        plan = build_application_plan(job, self.bank)
        experiences = selected_experiences(plan, self.bank)
        projects = selected_projects(plan, self.bank)
        draft = DraftResume(
            experience=[
                DraftExperience(
                    company=item.company,
                    role=item.role,
                    bullets=[
                        SourcedBullet(text="A" * 100, sources=[item.facts[0].id]),
                        SourcedBullet(text="B" * 100, sources=[item.facts[min(1, len(item.facts) - 1)].id]),
                    ],
                )
                for item in experiences
            ],
            projects=[
                DraftProject(
                    name=item.name,
                    bullets=[
                        SourcedBullet(text="P" * 100, sources=[item.facts[0].id]),
                        SourcedBullet(text="Q" * 100, sources=[item.facts[min(1, len(item.facts) - 1)].id]),
                        SourcedBullet(text=f"Stack: {item.stack}", sources=[]),
                    ],
                )
                for item in projects
            ],
            education=list(self.bank.profile.education),
            skills=plan.skill_groups,
        )
        issues = validate_draft_resume(
            draft, plan, self.bank.skills.allowed_names(), self.bank.profile, self.bank
        )
        source_issue_list = [item for item in issues if "source" in item.casefold()]
        self.assertEqual(source_issue_list, [])

        first = experiences[0]
        bad = draft.model_copy(
            update={
                "experience": [
                    DraftExperience(
                        company=first.company,
                        role=first.role,
                        bullets=[
                            SourcedBullet(text="A" * 100, sources=[]),
                            SourcedBullet(text="B" * 100, sources=["not-a-real-id"]),
                        ],
                    ),
                    *draft.experience[1:],
                ]
            }
        )
        issues = validate_draft_resume(
            bad, plan, self.bank.skills.allowed_names(), self.bank.profile, self.bank
        )
        self.assertTrue(any("missing sources" in item for item in issues))
        self.assertTrue(any("unknown sources" in item for item in issues))

    def test_validate_sources_rejects_wrong_record_and_unsupported_metric(self) -> None:
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
        )
        plan = build_application_plan(job, self.bank)
        plan = plan.model_copy(update={"experience_ids": ["mk-lending"]})
        lending = self.bank.experience_by_id()["mk-lending"]
        tena = self.bank.experience_by_id()["tena"]
        wrong_record = DraftResume(
            experience=[
                DraftExperience(
                    company=lending.company,
                    role=lending.role,
                    bullets=[
                        SourcedBullet(
                            text="Improved accuracy by 93% using TypeScript APIs.",
                            sources=[tena.metrics[0].id],
                        ),
                        SourcedBullet(text="B" * 100, sources=[lending.facts[0].id]),
                    ],
                )
            ]
        )
        issues = validate_sources(wrong_record, plan, self.bank)
        self.assertTrue(any("unknown sources" in item for item in issues))

        fact_only = DraftResume(
            experience=[
                DraftExperience(
                    company=lending.company,
                    role=lending.role,
                    bullets=[
                        SourcedBullet(
                            text="Improved accuracy by 93% using TypeScript APIs.",
                            sources=[lending.facts[0].id],
                        ),
                        SourcedBullet(text="B" * 100, sources=[lending.facts[0].id]),
                    ],
                )
            ]
        )
        issues = validate_sources(fact_only, plan, self.bank)
        self.assertTrue(any("numeric claims" in item for item in issues))
        self.assertTrue(any("93" in item or "metric" in item.casefold() for item in issues))

    def test_stack_lines_may_have_empty_sources(self) -> None:
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
        )
        plan = build_application_plan(job, self.bank)
        draft = DraftResume(
            experience=[
                DraftExperience(
                    company=item.company,
                    role=item.role,
                    bullets=[
                        SourcedBullet(text="A" * 100, sources=[item.facts[0].id]),
                        SourcedBullet(text="B" * 100, sources=[item.facts[0].id]),
                    ],
                )
                for item in selected_experiences(plan, self.bank)
            ],
            projects=[
                DraftProject(
                    name=item.name,
                    bullets=[
                        SourcedBullet(text="P" * 100, sources=[item.facts[0].id]),
                        SourcedBullet(text="Q" * 100, sources=[item.facts[0].id]),
                        SourcedBullet(text=f"Stack: {item.stack}", sources=[]),
                    ],
                )
                for item in selected_projects(plan, self.bank)
            ],
            education=list(self.bank.profile.education),
            skills=plan.skill_groups,
        )
        self.assertEqual(source_issues(draft, plan, self.bank), [])


if __name__ == "__main__":
    unittest.main()
