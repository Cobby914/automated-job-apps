from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobapps.career import load_career_bank
from jobapps.dedup import find_duplicate, find_reusable_plan, job_fingerprint, strip_tracking_params
from jobapps.errors import INVALID_PROVENANCE, PDF_OVERFLOW, PipelineError
from jobapps.models import (
    ApplicationPlan,
    CoverLetter,
    DraftExperience,
    DraftProject,
    DraftResume,
    Education,
    Experience,
    FitReport,
    Job,
    PipelineMetrics,
    ReviewIssue,
    ReviewResult,
    SourcedBullet,
    TailoredResume,
)
from jobapps.pipeline import _render_and_fit, merge_fitted_into_draft, process_job_file
from jobapps.validate import MAX_COVER_FIT_LLM_REPAIRS, MAX_PDF_FIT_LLM_REPAIRS
from jobapps.plan import build_application_plan
from jobapps.validate import validate_draft_resume

ROOT = Path(__file__).resolve().parents[1]


def _valid_draft(bank, plan) -> DraftResume:
    return DraftResume(
        experience=[
            DraftExperience(
                company=item.company,
                role=item.role,
                bullets=[
                    SourcedBullet(
                        text=f"{item.id}-a".ljust(100, "x"),
                        sources=[item.facts[0].id],
                    ),
                    SourcedBullet(
                        text=f"{item.id}-b".ljust(100, "y"),
                        sources=[item.facts[0].id],
                    ),
                ],
            )
            for item in [bank.experience_by_id()[rid] for rid in plan.experience_ids]
        ],
        research=[
            DraftExperience(
                company=item.company,
                role=item.role,
                bullets=[
                    SourcedBullet(
                        text=f"{item.id}-r".ljust(100, "r"),
                        sources=[item.facts[0].id],
                    ),
                    SourcedBullet(
                        text=f"{item.id}-s".ljust(100, "s"),
                        sources=[item.facts[0].id],
                    ),
                ],
            )
            for item in [bank.experience_by_id()[rid] for rid in plan.research_ids]
        ],
        projects=[
            DraftProject(
                name=item.name,
                bullets=[
                    SourcedBullet(
                        text=f"{item.id}-p".ljust(100, "p"),
                        sources=[item.facts[0].id],
                    ),
                    SourcedBullet(
                        text=f"{item.id}-q".ljust(100, "q"),
                        sources=[item.facts[0].id],
                    ),
                    SourcedBullet(text=f"Stack: {item.stack}", sources=[]),
                ],
            )
            for item in [bank.project_by_id()[rid] for rid in plan.project_ids]
        ],
        education=list(bank.profile.education),
        skills=plan.skill_groups,
    )


def _fake_fit(_job, _contact, resume, cover, _materials_dir, _plan, _bank, metrics):
    metrics.final_resume_pages = 1
    return resume, cover, FitReport()


class DedupTests(unittest.TestCase):
    def test_fingerprint_prefers_portal_url(self) -> None:
        left = Job(
            company="Acme Inc",
            title="SWE",
            portal_url="https://example.com/jobs/1/",
            description="One",
        )
        right = Job(
            company="Other",
            title="Intern",
            portal_url="https://example.com/jobs/1",
            description="Two",
        )
        self.assertEqual(job_fingerprint(left), job_fingerprint(right))

    def test_fingerprint_strips_tracking_params(self) -> None:
        left = Job(
            company="Acme",
            title="SWE",
            portal_url="https://example.com/jobs/1?utm_source=board&fbclid=abc",
            description="One",
        )
        right = Job(
            company="Acme",
            title="SWE",
            portal_url="https://example.com/jobs/1",
            description="Two",
        )
        self.assertEqual(job_fingerprint(left), job_fingerprint(right))
        self.assertNotIn("utm_", strip_tracking_params(left.portal_url))

    def test_fingerprint_normalizes_description_whitespace(self) -> None:
        left = Job(company="Acme", title="SWE", description="Build   APIs.\nPlease apply.")
        right = Job(company="acme", title="swe", description="Build APIs. Please apply.")
        self.assertEqual(job_fingerprint(left), job_fingerprint(right))

    def test_find_duplicate_in_processed(self) -> None:
        job = Job(company="Acme", title="SWE", description="Build APIs.")
        with TemporaryDirectory() as tmp:
            processed = Path(tmp) / "processed"
            processed.mkdir()
            (processed / "acme.yaml").write_text(
                "company: Acme\ntitle: SWE\ndescription: Build APIs.\n",
                encoding="utf-8",
            )
            with patch("jobapps.dedup.PROCESSED_DIR", processed):
                with patch("jobapps.dedup.OUTPUT_DIR", Path(tmp) / "output"):
                    found = find_duplicate(job)
            self.assertIsNotNone(found)
            self.assertEqual(found.name, "acme.yaml")

    def test_reusable_plan_requires_minimum_similarity(self) -> None:
        original = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
        )
        similar = original.model_copy(
            update={
                "description": "Python TypeScript PostgreSQL REST APIs Node.js. Also Kafka."
            }
        )
        unrelated = original.model_copy(
            update={
                "description": "Design visual brand assets in Figma and run marketing campaigns."
            }
        )
        plan = ApplicationPlan(template="swe", experience_ids=["mk-lending"])
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "stripe-backend"
            (folder / "inputs").mkdir(parents=True)
            (folder / "meta").mkdir()
            (folder / "inputs" / "job.yaml").write_text(
                "company: Stripe\ntitle: Backend Engineer\n"
                "description: Python TypeScript PostgreSQL REST APIs Node.js.\n",
                encoding="utf-8",
            )
            (folder / "meta" / "application_plan.json").write_text(
                plan.model_dump_json(),
                encoding="utf-8",
            )
            with patch("jobapps.dedup.OUTPUT_DIR", Path(tmp)):
                reused = find_reusable_plan(similar)
                skipped = find_reusable_plan(unrelated)
                duplicate = find_reusable_plan(original)
        self.assertIsNotNone(reused)
        self.assertEqual(reused.experience_ids, ["mk-lending"])
        self.assertIsNone(skipped)
        self.assertIsNone(duplicate)


class PipelineIntegrationTests(unittest.TestCase):
    def test_merge_fitted_preserves_matching_sources(self) -> None:
        draft = DraftResume(
            experience=[
                DraftExperience(
                    company="Acme",
                    role="Intern",
                    bullets=[
                        SourcedBullet(text="kept", sources=["fact.1"]),
                        SourcedBullet(text="old", sources=["fact.2"]),
                    ],
                )
            ]
        )
        fitted = TailoredResume(
            experience=[
                Experience(company="Acme", role="Intern", bullets=["kept", "shortened"])
            ],
            education=[Education(school="UCI", degree="BS")],
        )
        merged = merge_fitted_into_draft(draft, fitted)
        self.assertEqual(merged.experience[0].bullets[0].sources, ["fact.1"])
        self.assertEqual(merged.experience[0].bullets[1].text, "shortened")
        self.assertEqual(merged.experience[0].bullets[1].sources, ["fact.2"])

    def test_process_resumes_from_reviewed_stage(self) -> None:
        bank = load_career_bank(ROOT / "career")
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
            cover_letter=False,
        )
        plan = build_application_plan(job, bank)
        draft = _valid_draft(bank, plan)
        review = ReviewResult(approved=True, summary="Looks good.", issues=[])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_path = root / "stripe.yaml"
            job_path.write_text(
                "company: Stripe\ntitle: Backend Engineer\n"
                "description: Python TypeScript PostgreSQL REST APIs Node.js.\n"
                "cover_letter: false\n",
                encoding="utf-8",
            )
            destination = root / "output" / "stripe"
            meta = destination / "meta"
            inputs = destination / "inputs"
            materials = destination / "materials"
            for path in (meta, inputs, materials):
                path.mkdir(parents=True)
            (inputs / "job.yaml").write_text(job_path.read_text(encoding="utf-8"), encoding="utf-8")
            (meta / "application_plan.json").write_text(
                plan.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            (meta / "resume_draft.json").write_text(
                draft.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            (meta / "review.json").write_text(
                review.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            (root / "stripe.yaml.progress.json").write_text(
                json.dumps(
                    {
                        "output_dir": str(destination),
                        "stage": "reviewed",
                        "template_request": "auto",
                        "graduation_year": "June 2028",
                        "checker_model": "claude-4.5-sonnet",
                        "escalated": False,
                        "metrics": {},
                    }
                ),
                encoding="utf-8",
            )
            with patch("jobapps.pipeline.find_duplicate", return_value=None):
                with patch("jobapps.pipeline.generate_resume_draft") as mock_draft:
                    with patch("jobapps.pipeline.review_materials") as mock_review:
                        with patch("jobapps.pipeline._render_and_fit", side_effect=_fake_fit):
                            with patch(
                                "jobapps.pipeline.create_application_page",
                                return_value=None,
                            ):
                                result = process_job_file(job_path)
            mock_draft.assert_not_called()
            mock_review.assert_not_called()
            self.assertTrue((destination / "meta" / "resume_final.json").is_file())
            self.assertTrue((destination / "meta" / "meta.yaml").is_file())
            self.assertTrue(result.checker_approved)

    def test_cover_letter_disabled_skips_cover_generation(self) -> None:
        bank = load_career_bank(ROOT / "career")
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
            cover_letter=False,
        )
        plan = build_application_plan(job, bank)
        draft = _valid_draft(bank, plan)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_path = root / "stripe.yaml"
            job_path.write_text(
                "company: Stripe\ntitle: Backend Engineer\n"
                "description: Python TypeScript PostgreSQL REST APIs Node.js.\n"
                "cover_letter: false\n",
                encoding="utf-8",
            )
            with patch("jobapps.pipeline.OUTPUT_DIR", root / "output"):
                with patch("jobapps.pipeline.find_duplicate", return_value=None):
                    with patch(
                        "jobapps.pipeline.generate_resume_draft",
                        return_value=(draft, 120),
                    ):
                        with patch(
                            "jobapps.pipeline.review_materials",
                            return_value=(
                                ReviewResult(approved=True, summary="ok", issues=[]),
                                False,
                                "claude-4.5-sonnet",
                            ),
                        ):
                            with patch(
                                "jobapps.pipeline.generate_cover_letter_draft"
                            ) as mock_cover:
                                with patch(
                                    "jobapps.pipeline._render_and_fit",
                                    side_effect=_fake_fit,
                                ):
                                    with patch(
                                        "jobapps.pipeline.create_application_page",
                                        return_value=None,
                                    ):
                                        result = process_job_file(job_path)
            mock_cover.assert_not_called()
            self.assertFalse(result.job.cover_letter)

    def test_invalid_source_claim_raises_provenance_error(self) -> None:
        bank = load_career_bank(ROOT / "career")
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
            cover_letter=False,
        )
        plan = build_application_plan(job, bank)
        bad = _valid_draft(bank, plan)
        first = bad.experience[0]
        bad = bad.model_copy(
            update={
                "experience": [
                    first.model_copy(
                        update={
                            "bullets": [
                                SourcedBullet(text="A" * 100, sources=["not-a-real-id"]),
                                first.bullets[1],
                            ]
                        }
                    ),
                    *bad.experience[1:],
                ]
            }
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_path = root / "stripe.yaml"
            job_path.write_text(
                "company: Stripe\ntitle: Backend Engineer\n"
                "description: Python TypeScript PostgreSQL REST APIs Node.js.\n"
                "cover_letter: false\n",
                encoding="utf-8",
            )
            with patch("jobapps.pipeline.OUTPUT_DIR", root / "output"):
                with patch("jobapps.pipeline.find_duplicate", return_value=None):
                    with patch(
                        "jobapps.pipeline.generate_resume_draft",
                        return_value=(bad, 120),
                    ):
                        with self.assertRaises(PipelineError) as ctx:
                            process_job_file(job_path)
        self.assertEqual(ctx.exception.category, INVALID_PROVENANCE)

    def test_semantic_repair_then_success(self) -> None:
        bank = load_career_bank(ROOT / "career")
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
            cover_letter=False,
        )
        plan = build_application_plan(job, bank)
        draft = _valid_draft(bank, plan)
        first_id = plan.experience_ids[0]
        failed = ReviewResult(
            approved=False,
            summary="Second bullet is generic.",
            issues=[
                ReviewIssue(
                    type="generic",
                    section="experience",
                    item_id=first_id,
                    bullet_index=1,
                    message="Second bullet is generic.",
                )
            ],
        )
        passed = ReviewResult(approved=True, summary="Looks good.", issues=[])
        rewritten = "Rewritten grounded TypeScript API work with real impact now."
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_path = root / "stripe.yaml"
            job_path.write_text(
                "company: Stripe\ntitle: Backend Engineer\n"
                "description: Python TypeScript PostgreSQL REST APIs Node.js.\n"
                "cover_letter: false\n",
                encoding="utf-8",
            )
            with patch("jobapps.pipeline.OUTPUT_DIR", root / "output"):
                with patch("jobapps.pipeline.find_duplicate", return_value=None):
                    with patch(
                        "jobapps.pipeline.generate_resume_draft",
                        return_value=(draft, 120),
                    ):
                        with patch(
                            "jobapps.pipeline.review_materials",
                            side_effect=[
                                (failed, False, "claude-4.5-sonnet"),
                                (passed, False, "claude-4.5-sonnet"),
                            ],
                        ) as mock_review:
                            with patch(
                                "jobapps.pipeline.rewrite_bullet",
                                return_value=rewritten,
                            ):
                                with patch(
                                    "jobapps.pipeline.validate_draft_resume",
                                    wraps=validate_draft_resume,
                                ):
                                    with patch(
                                        "jobapps.pipeline._render_and_fit",
                                        side_effect=_fake_fit,
                                    ):
                                        with patch(
                                            "jobapps.pipeline.create_application_page",
                                            return_value=None,
                                        ):
                                            result = process_job_file(job_path)
        self.assertEqual(mock_review.call_count, 2)
        self.assertTrue(result.checker_approved)
        self.assertEqual(result.metrics.semantic_revisions, 1)

    def test_duplicate_posting_is_skipped(self) -> None:
        job_yaml = (
            "company: Stripe\ntitle: Backend Engineer\n"
            "description: Python TypeScript PostgreSQL REST APIs Node.js.\n"
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_path = root / "stripe.yaml"
            job_path.write_text(job_yaml, encoding="utf-8")
            with patch("jobapps.pipeline.find_duplicate", return_value=root / "prior"):
                result = process_job_file(job_path)
        self.assertTrue(result.skipped_duplicate)


class CoverFitAttemptTests(unittest.TestCase):
    def _resume(self) -> TailoredResume:
        return TailoredResume(
            experience=[
                Experience(company="Acme", role="Engineer", bullets=["Built a Python service."])
            ],
            education=[Education(school="UCI", degree="B.S. Computer Science")],
        )

    def _cover(self) -> CoverLetter:
        return CoverLetter(
            greeting="Dear Hiring Manager,",
            paragraphs=["One paragraph.", "Two paragraph.", "Three paragraph.", "Four paragraph."],
            closing="Sincerely,",
        )

    def test_cover_only_overflow_allows_four_shorten_attempts(self) -> None:
        self.assertEqual(MAX_COVER_FIT_LLM_REPAIRS, 4)
        self.assertEqual(MAX_PDF_FIT_LLM_REPAIRS, 2)
        calls: list[dict[str, object]] = []

        def shorten(cover: CoverLetter, **kwargs: object) -> CoverLetter:
            calls.append(kwargs)
            return cover

        def pages(path: Path) -> int:
            return 2 if Path(path).name.startswith("cover") else 1

        def compile_tex(path: Path) -> tuple[None, str]:
            if Path(path).name.startswith("cover"):
                return None, r"Overfull \hbox (12.5pt too wide) in paragraph"
            return None, ""

        job = Job(company="Acme", title="Software Engineer Intern", description="Build services.")
        plan = ApplicationPlan(template="swe")
        with TemporaryDirectory() as tmp:
            with patch("jobapps.pipeline.render_documents"):
                with patch("jobapps.pipeline._compile_tex", side_effect=compile_tex):
                    with patch("jobapps.pipeline.pdf_page_count", side_effect=pages):
                        with patch("jobapps.pipeline.shorten_cover_letter", side_effect=shorten):
                            with self.assertRaises(PipelineError) as caught:
                                _render_and_fit(
                                    job,
                                    None,
                                    self._resume(),
                                    self._cover(),
                                    Path(tmp),
                                    plan,
                                    None,  # type: ignore[arg-type]
                                    PipelineMetrics(),
                                )
        self.assertEqual(caught.exception.category, PDF_OVERFLOW)
        self.assertIn("4 LLM repairs", str(caught.exception))
        self.assertEqual([call["attempt"] for call in calls], [1, 2, 3, 4])
        self.assertTrue(all(call["pages"] == 2 for call in calls))
        self.assertTrue(all(call["overfull"] == ["12.5pt"] for call in calls))

    def test_resume_overflow_keeps_the_two_attempt_cap(self) -> None:
        bullet_calls = 0

        def shorten_bullet(current: str, **_kwargs: object) -> str:
            nonlocal bullet_calls
            bullet_calls += 1
            return current

        job = Job(company="Acme", title="Software Engineer", description="Build services.")
        plan = ApplicationPlan(template="swe")
        bank = SimpleNamespace(experiences=[], projects=[])
        with TemporaryDirectory() as tmp:
            with patch("jobapps.pipeline.render_documents"):
                with patch("jobapps.pipeline._compile_tex", return_value=(None, "")):
                    with patch("jobapps.pipeline.pdf_page_count", return_value=2):
                        with patch("jobapps.pipeline.shorten_bullet", side_effect=shorten_bullet):
                            with patch("jobapps.pipeline.shorten_cover_letter") as shorten_cover:
                                with self.assertRaises(PipelineError) as caught:
                                    _render_and_fit(
                                        job,
                                        None,
                                        self._resume(),
                                        self._cover(),
                                        Path(tmp),
                                        plan,
                                        bank,  # type: ignore[arg-type]
                                        PipelineMetrics(),
                                    )
        self.assertEqual(caught.exception.category, PDF_OVERFLOW)
        self.assertEqual(bullet_calls, MAX_PDF_FIT_LLM_REPAIRS)
        shorten_cover.assert_not_called()


if __name__ == "__main__":
    unittest.main()
