from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobapps.career import load_career_bank
from jobapps.models import Job
from jobapps.plan import build_application_plan

ROOT = Path(__file__).resolve().parents[1]


class PlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = load_career_bank(ROOT / "career")

    def test_plan_respects_layout_budget(self) -> None:
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
        )
        plan = build_application_plan(job, self.bank)
        self.assertLessEqual(len(plan.experience_ids), plan.layout.max_experiences)
        self.assertGreaterEqual(len(plan.experience_ids), plan.layout.min_experiences)
        self.assertLessEqual(len(plan.project_ids), plan.layout.max_projects)
        self.assertGreaterEqual(len(plan.project_ids), plan.layout.min_projects)
        self.assertTrue(plan.skill_groups)
        self.assertLessEqual(len(plan.skill_groups), plan.layout.max_skill_groups)
        self.assertEqual(plan.template, "swe")
        self.assertTrue(plan.cover_letter)

    def test_plan_includes_explanations_and_priorities(self) -> None:
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
        )
        plan = build_application_plan(job, self.bank)
        self.assertEqual(len(plan.experience_scores), len(plan.experience_ids))
        self.assertEqual(plan.experience_scores[0].priority, 1)
        self.assertTrue(plan.experience_scores[0].explanation)
        self.assertEqual(plan.resume_priorities[0], plan.experience_ids[0])

    def test_reused_plan_keeps_ids_and_skills(self) -> None:
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
        )
        original = build_application_plan(job, self.bank)
        similar = job.model_copy(
            update={"description": "Python TypeScript PostgreSQL REST APIs Node.js. Also Kafka."}
        )
        reused = build_application_plan(similar, self.bank, reuse=original)
        self.assertEqual(reused.experience_ids, original.experience_ids)
        self.assertEqual(reused.project_ids, original.project_ids)
        self.assertEqual(reused.skill_groups, original.skill_groups)
        self.assertIn("reused", reused.template_reason.casefold())


if __name__ == "__main__":
    unittest.main()
