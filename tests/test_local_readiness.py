"""Offline checks for local startup and token-saving execution paths."""
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jobapps.career import load_career_bank
from jobapps.generate import generate_resume_draft
from jobapps.models import DraftResume, Job, PipelineMetrics, Project, TailoredResume
from jobapps.pipeline import _render_and_fit
from jobapps.plan import build_application_plan

ROOT = Path(__file__).resolve().parents[1]


class LocalReadinessTests(unittest.TestCase):
    def test_cli_help_starts_without_platform_worker(self):
        result = subprocess.run(
            [sys.executable, "-m", "jobapps", "--help"],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("process", result.stdout)

    def test_resume_prefix_stays_identical_across_jobs(self):
        bank = load_career_bank(ROOT / "career")
        first = Job(company="Acme", title="Backend Engineer", description="Python APIs")
        second = first.model_copy(update={"company": "Beta", "description": "PostgreSQL APIs"})
        plan = build_application_plan(first, bank)
        draft = DraftResume(experience=[], projects=[], education=[], skills=[])
        with patch("jobapps.generate.writer_model", return_value="gpt-4.1"):
            with patch("jobapps.generate.generate_structured", return_value=draft) as call:
                generate_resume_draft(first, plan, bank)
                generate_resume_draft(second, plan, bank)
        left, right = [item.kwargs for item in call.call_args_list]
        self.assertEqual(left["system"], right["system"])
        self.assertNotEqual(left["user"], right["user"])

    def test_vertical_overflow_can_fit_without_model_call(self):
        bank = load_career_bank(ROOT / "career")
        job = Job(company="Acme", title="Backend Engineer", description="Python", cover_letter=False)
        plan = build_application_plan(job, bank)
        resume = TailoredResume(
            experience=[], education=bank.profile.education,
            projects=[Project(name="Demo", bullets=["A" * 100, "B" * 100, "C" * 100, "Stack: Python"])],
        )
        metrics = PipelineMetrics()
        with TemporaryDirectory() as tmp:
            with patch("jobapps.pipeline.render_documents"):
                with patch("jobapps.pipeline._compile_tex", return_value=(None, "")):
                    with patch("jobapps.pipeline.pdf_page_count", side_effect=[2, 1]):
                        with patch("jobapps.pipeline.shorten_bullet") as shorten:
                            _, _, report = _render_and_fit(
                                job, bank.profile.contact, resume, None, Path(tmp), plan, bank, metrics,
                            )
        shorten.assert_not_called()
        self.assertTrue(report.changes)
        self.assertFalse(report.used_llm)
        self.assertEqual(metrics.final_resume_pages, 1)


if __name__ == "__main__":
    unittest.main()
