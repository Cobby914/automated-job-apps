from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobapps.fit import apply_python_trim, apply_python_trim_step, parse_overfull_hbox
from jobapps.latex import render_documents
from jobapps.models import (
    ApplicationPlan,
    Education,
    Experience,
    LayoutBudget,
    Project,
    SkillGroup,
    TailoredResume,
    load_job,
    load_resume,
)

ROOT = Path(__file__).resolve().parents[1]


class FitTests(unittest.TestCase):
    def _resume(self, **kwargs: object) -> TailoredResume:
        base: dict[str, object] = dict(
            experience=[
                Experience(company="Acme", role="Intern", bullets=["A" * 100, "B" * 100]),
                Experience(company="Beta", role="Intern", bullets=["C" * 100, "D" * 100]),
                Experience(company="Gamma", role="Intern", bullets=["E" * 100, "F" * 100]),
            ],
            projects=[
                Project(name="Demo", bullets=["P" * 100, "Q" * 100, "T" * 100, "Stack: Python"]),
                Project(name="Other", bullets=["R" * 100, "S" * 100, "U" * 100, "Stack: C"]),
            ],
            education=[
                Education(
                    school="University of California, Irvine",
                    degree="B.S. Computer Science",
                    year="June 2027",
                )
            ],
            skills=[
                SkillGroup(category="Languages", items="Python, TypeScript"),
                SkillGroup(category="Backend/Data", items="PostgreSQL, REST APIs"),
                SkillGroup(category="AI/ML", items="PyTorch, pandas"),
            ],
        )
        base.update(kwargs)
        return TailoredResume.model_validate(base)

    def test_parse_overfull_hbox(self) -> None:
        log = r"Overfull \hbox (12.3pt too wide) in paragraph at lines 20--22"
        self.assertEqual(parse_overfull_hbox(log), ["12.3pt"])

    def test_python_trim_drops_lowest_priority_project_bullets_first(self) -> None:
        plan = ApplicationPlan(
            template="swe",
            experience_ids=["acme", "beta", "gamma"],
            project_ids=["demo", "other"],
            resume_priorities=["acme", "beta", "gamma", "demo", "other"],
            layout=LayoutBudget(project_bullets=2),
        )
        resume = self._resume()
        trimmed = apply_python_trim(resume, plan)
        self.assertIsNotNone(trimmed)
        assert trimmed is not None
        other = next(item for item in trimmed.projects if item.name == "Other")
        content = [b for b in other.bullets if not b.lower().startswith("stack:")]
        self.assertEqual(len(content), 2)

    def test_trim_step_records_fit_change(self) -> None:
        plan = ApplicationPlan(
            template="swe",
            experience_ids=["acme", "beta", "gamma"],
            project_ids=["demo", "other"],
            resume_priorities=["acme", "beta", "gamma", "demo", "other"],
            layout=LayoutBudget(project_bullets=2),
        )
        result = apply_python_trim_step(self._resume(), plan)
        self.assertIsNotNone(result)
        assert result is not None
        _resume, change = result
        self.assertEqual(change.action, "removed_project_bullet")
        self.assertEqual(change.section, "project")
        self.assertFalse(change.used_llm)
        self.assertIn("other", change.item_id)

    def test_cover_letter_false_skips_cover_render(self) -> None:
        job = load_job(ROOT / "jobs" / "samples" / "stripe-backend.yaml")
        job = job.model_copy(update={"cover_letter": False})
        resume = load_resume(ROOT / "resume_templates" / "swe.yaml")
        tailored = TailoredResume(
            summary=resume.summary,
            experience=resume.experience[:3],
            projects=resume.projects[:2],
            education=resume.education,
            skills=resume.skills,
        )
        with TemporaryDirectory() as tmp:
            resume_tex, cover_tex = render_documents(
                job,
                resume.contact,
                tailored,
                None,
                Path(tmp),
                compile_pdf=False,
            )
            self.assertTrue(resume_tex.is_file())
            self.assertIsNone(cover_tex)
            self.assertFalse((Path(tmp) / "cover_letter.tex").exists())


if __name__ == "__main__":
    unittest.main()
