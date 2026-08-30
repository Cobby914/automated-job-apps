from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobapps.career import load_career_bank
from jobapps.models import (
    MAX_BULLET_CHARS,
    ApplicationPlan,
    CoverLetter,
    Education,
    Experience,
    Job,
    LayoutBudget,
    Project,
    SkillGroup,
    TailoredResume,
    overlong_bullet_issues,
)
from jobapps.validate import validate_cover_letter, validate_resume

ROOT = Path(__file__).resolve().parents[1]


class BulletLengthTests(unittest.TestCase):
    def _resume(self, experience_bullets: list[str], project_bullets: list[str] | None = None) -> TailoredResume:
        return TailoredResume(
            experience=[Experience(company="Acme", role="Intern", bullets=experience_bullets)],
            projects=[Project(name="Demo", bullets=project_bullets or [])],
            education=[Education(school="UCI", degree="BS")],
        )

    def test_accepts_short_bullets(self) -> None:
        resume = self._resume(["Built an API used by three internal teams."])
        self.assertEqual(overlong_bullet_issues(resume), [])

    def test_rejects_long_bullets(self) -> None:
        long = "x" * (MAX_BULLET_CHARS + 1)
        issues = overlong_bullet_issues(self._resume([long]))
        self.assertEqual(len(issues), 1)
        self.assertIn("Experience Acme / Intern", issues[0])

    def test_skips_stack_trailer(self) -> None:
        long_stack = "Stack: " + ("x" * MAX_BULLET_CHARS)
        resume = self._resume(["Short bullet."], [long_stack])
        self.assertEqual(overlong_bullet_issues(resume), [])


class ResumeValidationTests(unittest.TestCase):
    def _resume(self, **kwargs: object) -> TailoredResume:
        base: dict[str, object] = dict(
            experience=[
                Experience(company="Acme", role="Intern", bullets=["A" * 100, "B" * 100]),
                Experience(company="Beta", role="Intern", bullets=["C" * 100, "D" * 100]),
                Experience(company="Gamma", role="Intern", bullets=["E" * 100, "F" * 100]),
            ],
            projects=[
                Project(name="Demo", bullets=["P" * 100, "Q" * 100, "Stack: Python"]),
                Project(name="Other", bullets=["R" * 100, "S" * 100, "Stack: C"]),
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

    def test_rejects_long_bullets_missing_stack_and_invented_skills(self) -> None:
        plan = ApplicationPlan(template="swe", layout=LayoutBudget())
        resume = self._resume(
            experience=[
                Experience(
                    company="Acme",
                    role="Intern",
                    bullets=["x" * (MAX_BULLET_CHARS + 1), "y" * 100],
                ),
                Experience(company="Beta", role="Intern", bullets=["C" * 100, "D" * 100]),
                Experience(company="Gamma", role="Intern", bullets=["E" * 100, "F" * 100]),
            ],
            projects=[
                Project(name="Demo", bullets=["P" * 100, "Q" * 100]),
                Project(name="Other", bullets=["R" * 100, "S" * 100, "Stack: C"]),
            ],
            skills=[
                SkillGroup(category="Cloud", items="AWS, Kubernetes"),
                SkillGroup(category="Languages", items="Python"),
                SkillGroup(category="AI/ML", items="PyTorch"),
            ],
        )
        allowed = load_career_bank(ROOT / "career").skills.allowed_names()
        issues = validate_resume(resume, plan, allowed)
        self.assertTrue(any("chars" in item for item in issues))
        self.assertTrue(any("Stack" in item for item in issues))
        self.assertTrue(any("Kubernetes" in item for item in issues))

    def test_cover_letter_paragraph_count(self) -> None:
        thin = CoverLetter(
            greeting="Dear Hiring Manager,",
            paragraphs=["One", "Two"],
            closing="Sincerely,",
        )
        issues = validate_cover_letter(thin)
        self.assertTrue(any("at least 4" in item for item in issues))
        ok = CoverLetter(
            greeting="Dear Hiring Manager,",
            paragraphs=["One", "Two", "Three", "Four"],
            closing="Sincerely,",
        )
        self.assertEqual(validate_cover_letter(ok), [])

    def test_job_cover_letter_flag_parses(self) -> None:
        from tempfile import TemporaryDirectory
        from jobapps.models import load_job

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.yaml"
            path.write_text(
                "company: Acme\ntitle: Intern\ndescription: Build things.\ncover_letter: false\n",
                encoding="utf-8",
            )
            job = load_job(path)
        self.assertFalse(job.cover_letter)


if __name__ == "__main__":
    unittest.main()
