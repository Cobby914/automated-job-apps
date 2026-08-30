from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobapps.career import load_career_bank
from jobapps.config import openai_reasoning_effort
from jobapps.generate import review_materials
from jobapps.llm import (
    anthropic_model_id,
    begin_usage_collection,
    estimate_cost_usd,
    extract_json,
    generate_structured,
    reset_usage_collection,
    resolve_provider,
    usage_records,
)
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

    def test_reasoning_effort_uses_role_override(self) -> None:
        env = {
            "OPENAI_REASONING_EFFORT": "medium",
            "OPENAI_WRITER_REASONING_EFFORT": "",
            "OPENAI_REVIEWER_REASONING_EFFORT": "low",
            "OPENAI_REPAIR_REASONING_EFFORT": "minimal",
            "OPENAI_ESCALATION_REASONING_EFFORT": "high",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(openai_reasoning_effort("resume_write"), "medium")
            self.assertEqual(openai_reasoning_effort("review"), "low")
            self.assertEqual(openai_reasoning_effort("bullet_shorten"), "minimal")
            self.assertEqual(openai_reasoning_effort("review_escalation"), "high")

    def test_structured_gpt5_uses_responses_reasoning(self) -> None:
        parsed = ReviewResult(approved=True, summary="ok", issues=[])
        usage = MagicMock()
        usage.prompt_tokens = 0
        usage.completion_tokens = 0
        usage.input_tokens = 12
        usage.output_tokens = 8
        usage.prompt_tokens_details = None
        usage.completion_tokens_details = None
        usage.input_tokens_details = MagicMock(cached_tokens=3)
        usage.output_tokens_details = MagicMock(reasoning_tokens=5)
        response = MagicMock(output_parsed=parsed, usage=usage, choices=[])
        client = MagicMock()
        client.responses.parse.return_value = response
        env = {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_REASONING_EFFORT": "low",
            "LLM_MAX_RETRIES": "0",
        }
        begin_usage_collection()
        try:
            with patch.dict(os.environ, env, clear=False):
                with patch("jobapps.llm._openai_client", return_value=client):
                    result = generate_structured(
                        system="sys",
                        user="write it",
                        model="gpt-5.6",
                        schema=ReviewResult,
                        purpose="resume_write",
                    )
            self.assertTrue(result.approved)
            kwargs = client.responses.parse.call_args.kwargs
            self.assertEqual(kwargs["model"], "gpt-5.6")
            self.assertEqual(kwargs["instructions"], "sys")
            self.assertEqual(kwargs["input"], "write it")
            self.assertIs(kwargs["text_format"], ReviewResult)
            self.assertEqual(kwargs["reasoning"], {"effort": "low"})
            records = usage_records()
            self.assertEqual(records[0].input_tokens, 12)
            self.assertEqual(records[0].cached_input_tokens, 3)
            self.assertEqual(records[0].output_tokens, 8)
            self.assertEqual(records[0].reasoning_tokens, 5)
        finally:
            reset_usage_collection()

    def test_structured_gpt41_omits_reasoning(self) -> None:
        parsed = ReviewResult(approved=True, summary="ok", issues=[])
        usage = MagicMock(
            prompt_tokens=4,
            completion_tokens=2,
            input_tokens=0,
            output_tokens=0,
            prompt_tokens_details=None,
            completion_tokens_details=None,
            input_tokens_details=None,
            output_tokens_details=None,
        )
        response = MagicMock(output_parsed=parsed, usage=usage, choices=[])
        client = MagicMock()
        client.responses.parse.return_value = response
        env = {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_REASONING_EFFORT": "high",
            "LLM_MAX_RETRIES": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("jobapps.llm._openai_client", return_value=client):
                generate_structured(
                    system="sys",
                    user="write it",
                    model="gpt-4.1",
                    schema=ReviewResult,
                    purpose="resume_write",
                )
        kwargs = client.responses.parse.call_args.kwargs
        self.assertNotIn("reasoning", kwargs)


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
