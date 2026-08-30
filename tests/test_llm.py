from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobapps.career import load_career_bank
from jobapps.generate import review_materials
from jobapps.llm import anthropic_model_id, estimate_cost_usd, extract_json, resolve_provider
from jobapps.models import (
    Experience,
    Job,
    Project,
    ReviewIssue,
    ReviewResult,
    TailoredResume,
    TemplateChoice,
)
from jobapps.plan import build_application_plan

ROOT = Path(__file__).resolve().parents[1]


class JsonExtractTests(unittest.TestCase):
    def test_extract_json_strips_fences(self) -> None:
        payload = extract_json('```json\n{"resume": {}, "cover_letter": {}}\n```')
        self.assertIn('"resume"', payload)
        self.assertTrue(payload.startswith("{"))

    def test_extract_json_requires_object(self) -> None:
        with self.assertRaises(Exception):
            extract_json("no json here")

    def test_review_result_parses(self) -> None:
        review = ReviewResult.model_validate_json(
            extract_json('{"approved": false, "summary": "Too generic", "issues": ["Cliches"]}')
        )
        self.assertFalse(review.approved)
        self.assertEqual(review.issues[0].message, "Cliches")
        self.assertEqual(review.issues[0].location, "resume")
        self.assertEqual(review.issues[0].code, "other")
        self.assertEqual(review.issues[0].type, "other")
        self.assertEqual(review.issues[0].section, "resume")

    def test_template_choice_parses(self) -> None:
        choice = TemplateChoice.model_validate_json(
            extract_json('{"template": "swe", "reason": "Backend APIs and services."}')
        )
        self.assertEqual(choice.template, "swe")
        self.assertIn("Backend", choice.reason)


class LlmProviderTests(unittest.TestCase):
    def test_resolve_provider_by_model_name(self) -> None:
        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "ak", "OPENAI_API_KEY": "sk", "LLM_PROVIDER": ""},
            clear=False,
        ):
            self.assertEqual(resolve_provider("claude-4.5-sonnet"), "anthropic")
            self.assertEqual(resolve_provider("gpt-4.1"), "openai")
            self.assertEqual(resolve_provider("gpt-5.6-sol"), "openai")

    def test_resolve_provider_falls_back_to_cursor(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": ""}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("LLM_PROVIDER", None)
            self.assertEqual(resolve_provider("claude-4.5-sonnet"), "cursor")
            self.assertEqual(resolve_provider("gpt-4.1"), "cursor")

    def test_anthropic_model_alias(self) -> None:
        self.assertEqual(anthropic_model_id("claude-4.5-sonnet"), "claude-sonnet-4-5")
        self.assertEqual(anthropic_model_id("claude-sonnet-4-5"), "claude-sonnet-4-5")

    def test_estimate_cost_uses_cached_rate(self) -> None:
        uncached = estimate_cost_usd("gpt-4.1", 1_000_000, 0, 0)
        cached = estimate_cost_usd("gpt-4.1", 1_000_000, 1_000_000, 0)
        self.assertAlmostEqual(uncached, 2.00)
        self.assertAlmostEqual(cached, 0.50)
        self.assertLess(cached, uncached)


class CheckerEscalationTests(unittest.TestCase):
    def test_sonnet_approve_skips_opus(self) -> None:
        bank = load_career_bank(ROOT / "career")
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs.",
        )
        plan = build_application_plan(job, bank)
        resume = TailoredResume(
            experience=[
                Experience(
                    company="M.K Lending",
                    role="Software Engineer Intern",
                    bullets=["x" * 100],
                )
            ],
            projects=[Project(name="Canvas-Notion", bullets=["y" * 100, "Stack: Python"])],
            education=bank.profile.education,
            skills=plan.skill_groups,
        )
        with patch("jobapps.generate.checker_model", return_value="claude-4.5-sonnet"):
            with patch("jobapps.generate.llm_review") as mock_run:
                mock_run.return_value = ReviewResult(
                    approved=True, summary="Looks strong.", issues=[]
                )
                review, escalated, model = review_materials(job, plan, bank, resume, None)
        self.assertTrue(review.approved)
        self.assertFalse(escalated)
        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(model, "claude-4.5-sonnet")

    def test_sonnet_reject_calls_opus(self) -> None:
        bank = load_career_bank(ROOT / "career")
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs.",
        )
        plan = build_application_plan(job, bank)
        resume = TailoredResume(
            experience=[
                Experience(
                    company="M.K Lending",
                    role="Software Engineer Intern",
                    bullets=["x" * 100],
                )
            ],
            projects=[Project(name="Canvas-Notion", bullets=["y" * 100, "Stack: Python"])],
            education=bank.profile.education,
            skills=plan.skill_groups,
        )
        with patch("jobapps.generate.checker_model", return_value="claude-4.5-sonnet"):
            with patch("jobapps.generate.escalation_model", return_value="claude-opus-5"):
                with patch("jobapps.generate.llm_review") as mock_run:
                    mock_run.side_effect = [
                        ReviewResult(
                            approved=False,
                            summary="Too generic.",
                            issues=[ReviewIssue(message="Cliches")],
                        ),
                        ReviewResult(
                            approved=True,
                            summary="Acceptable after scrutiny.",
                            issues=[],
                        ),
                    ]
                    review, escalated, model = review_materials(job, plan, bank, resume, None)
        self.assertTrue(review.approved)
        self.assertTrue(escalated)
        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(model, "claude-opus-5")


if __name__ == "__main__":
    unittest.main()
