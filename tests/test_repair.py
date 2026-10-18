from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobapps.career import load_career_bank
from jobapps.generate import COVER_LETTER_WRITER_PROMPT
from jobapps.models import (
    CoverLetter,
    DraftExperience,
    DraftProject,
    DraftResume,
    Job,
    ReviewIssue,
    ReviewResult,
    SourcedBullet,
    parse_issue_location,
)
from jobapps.repair import cover_letter_word_count, shorten_cover_letter
from jobapps.pipeline import _repair_from_review
from jobapps.plan import build_application_plan

ROOT = Path(__file__).resolve().parents[1]


class RepairTests(unittest.TestCase):
    def test_parse_issue_location(self) -> None:
        loc = parse_issue_location("experience[0].bullets[1]")
        self.assertIsNotNone(loc)
        self.assertEqual(loc.kind, "experience")
        self.assertEqual(loc.item_index, 0)
        self.assertEqual(loc.part_index, 1)
        cover = parse_issue_location("cover_letter.paragraphs[2]")
        self.assertIsNotNone(cover)
        self.assertEqual(cover.kind, "cover_letter")
        self.assertEqual(cover.part_index, 2)

    def test_repair_targets_located_bullet(self) -> None:
        bank = load_career_bank(ROOT / "career")
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
        )
        plan = build_application_plan(job, bank)
        original = "B" * 100
        draft = DraftResume(
            experience=[
                DraftExperience(
                    company="M.K Lending",
                    role="Software Engineer Intern",
                    bullets=[
                        SourcedBullet(text="A" * 100, sources=["mk-lending.fact.2"]),
                        SourcedBullet(text=original, sources=["mk-lending.fact.3"]),
                    ],
                )
            ],
            projects=[
                DraftProject(
                    name="Canvas-Notion",
                    bullets=[
                        SourcedBullet(text="P" * 100, sources=["canvas-notion.fact.1"]),
                        SourcedBullet(text="Q" * 100, sources=["canvas-notion.fact.4"]),
                    ],
                )
            ],
            education=list(bank.profile.education),
            skills=plan.skill_groups,
        )
        review = ReviewResult(
            approved=False,
            summary="Second bullet is generic.",
            issues=[
                ReviewIssue(
                    location="experience[0].bullets[1]",
                    code="generic",
                    message="Second bullet is generic.",
                )
            ],
        )
        rewritten = "Rewritten grounded bullet with TypeScript impact."
        with patch("jobapps.pipeline.rewrite_bullet", return_value=rewritten):
            updated, _cover, count, cover_n = _repair_from_review(
                draft, None, review, plan, bank
            )
        self.assertEqual(count, 1)
        self.assertEqual(cover_n, 0)
        self.assertEqual(updated.experience[0].bullets[0].text, "A" * 100)
        self.assertEqual(updated.experience[0].bullets[1].text, rewritten)

    def test_repair_targets_item_id(self) -> None:
        bank = load_career_bank(ROOT / "career")
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
        )
        plan = build_application_plan(job, bank)
        lending = bank.experience_by_id()["mk-lending"]
        other = next(item for item in bank.experiences if item.id != "mk-lending")
        original = "B" * 100
        draft = DraftResume(
            experience=[
                DraftExperience(
                    company=other.company,
                    role=other.role,
                    bullets=[
                        SourcedBullet(text="Z" * 100, sources=[other.facts[0].id]),
                        SourcedBullet(text="Y" * 100, sources=[other.facts[0].id]),
                    ],
                ),
                DraftExperience(
                    company=lending.company,
                    role=lending.role,
                    bullets=[
                        SourcedBullet(text="A" * 100, sources=["mk-lending.fact.2"]),
                        SourcedBullet(text=original, sources=["mk-lending.fact.3"]),
                    ],
                ),
            ],
            projects=[
                DraftProject(
                    name="Canvas-Notion",
                    bullets=[
                        SourcedBullet(text="P" * 100, sources=["canvas-notion.fact.1"]),
                        SourcedBullet(text="Q" * 100, sources=["canvas-notion.fact.4"]),
                    ],
                )
            ],
            education=list(bank.profile.education),
            skills=plan.skill_groups,
        )
        review = ReviewResult(
            approved=False,
            summary="Second lending bullet is generic.",
            issues=[
                ReviewIssue(
                    type="generic",
                    section="experience",
                    item_id="mk-lending",
                    bullet_index=1,
                    message="Second bullet is generic.",
                )
            ],
        )
        rewritten = "Rewritten grounded bullet with TypeScript impact."
        with patch("jobapps.pipeline.rewrite_bullet", return_value=rewritten):
            updated, _cover, count, cover_n = _repair_from_review(
                draft, None, review, plan, bank
            )
        self.assertEqual(count, 1)
        self.assertEqual(cover_n, 0)
        self.assertEqual(updated.experience[0].bullets[1].text, "Y" * 100)
        self.assertEqual(updated.experience[1].bullets[1].text, rewritten)

    def test_structured_issue_fields_round_trip(self) -> None:
        issue = ReviewIssue(
            type="ungrounded",
            section="experience",
            item_id="uci-scalesense",
            bullet_index=2,
            message="Claim is not in the record.",
        )
        self.assertEqual(issue.code, "ungrounded")
        self.assertEqual(issue.section, "experience")
        self.assertEqual(issue.item_id, "uci-scalesense")
        self.assertEqual(issue.bullet_index, 2)


class CoverLetterShortenTests(unittest.TestCase):
    def _cover(self) -> CoverLetter:
        return CoverLetter(
            greeting="Dear Hiring Manager,",
            paragraphs=["alpha " * 90, "beta " * 80, "gamma " * 70, "delta " * 60],
            closing="Sincerely,",
        )

    def _system(self, cover: CoverLetter, **kwargs: object) -> str:
        captured: dict[str, str] = {}

        def fake_generate(**call: object) -> CoverLetter:
            captured["system"] = str(call["system"])
            return cover

        with patch("jobapps.repair.generate_structured", side_effect=fake_generate):
            shorten_cover_letter(cover, **kwargs)
        return captured["system"]

    def test_prompt_reports_length_and_one_page_target(self) -> None:
        cover = self._cover()
        system = self._system(cover, pages=2, overfull=["12.5pt"], attempt=1)
        current = cover_letter_word_count(cover)
        self.assertIn(f"currently {current} words", system)
        self.assertIn("280-300", system)
        self.assertIn("exactly 4 substantive paragraphs", system)
        self.assertIn("The rendered PDF is 2 pages", system)
        self.assertIn("signature must stay on page 1", system)
        self.assertIn("long unbreakable tokens", system)
        self.assertIn("280-300", COVER_LETTER_WRITER_PROMPT)
        self.assertIn("signature", COVER_LETTER_WRITER_PROMPT)

    def test_later_attempts_tighten_the_word_target(self) -> None:
        cover = self._cover()
        self.assertIn("250-270", self._system(cover, pages=2, attempt=2))
        self.assertIn("220-240", self._system(cover, pages=2, attempt=3))
        self.assertIn("200-230", self._system(cover, pages=2, attempt=4))


if __name__ == "__main__":
    unittest.main()
