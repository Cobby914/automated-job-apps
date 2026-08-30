from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobapps.career import ExperienceRecord, load_career_bank
from jobapps.models import Job, resolve_template_request
from jobapps.ranking import RankedItem, rank_experiences, rank_projects, score_record, score_template, select_ranked, select_skills, select_template

ROOT = Path(__file__).resolve().parents[1]


class TemplateSelectionTests(unittest.TestCase):
    def test_resolve_auto_requests(self) -> None:
        self.assertIsNone(resolve_template_request("auto"))
        self.assertIsNone(resolve_template_request(""))

    def test_resolve_explicit_templates(self) -> None:
        self.assertEqual(resolve_template_request("swe"), "swe")
        self.assertEqual(resolve_template_request("AI"), "ai")
        self.assertEqual(resolve_template_request("default"), "default")

    def test_rejects_unknown_template(self) -> None:
        with self.assertRaises(ValueError):
            resolve_template_request("backend")


class RankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = load_career_bank(ROOT / "career")

    def test_explicit_template_override(self) -> None:
        job = Job(company="Acme", title="Backend Engineer", description="Ship APIs.", template="ai")
        template, reason, auto = select_template(job)
        self.assertEqual(template, "ai")
        self.assertFalse(auto)
        self.assertIn("explicitly", reason.lower())

    def test_swe_job_scores_swe(self) -> None:
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Build REST APIs, PostgreSQL services, and TypeScript backends.",
        )
        template, _reason, scores = score_template(job)
        self.assertEqual(template, "swe")
        self.assertGreater(scores["swe"], scores["ai"])

    def test_ai_job_scores_ai(self) -> None:
        job = Job(
            company="Nuro",
            title="Autonomy Perception Intern",
            description="PyTorch computer vision, CARLA simulation, radar-camera fusion research.",
        )
        template, _reason, scores = score_template(job)
        self.assertEqual(template, "ai")
        self.assertGreater(scores["ai"], scores["swe"])

    def test_close_scores_default(self) -> None:
        job = Job(company="Acme", title="Intern", description="Join our team.")
        template, _reason, _scores = score_template(job)
        self.assertEqual(template, "default")

    def test_ranking_prefers_tech_overlap(self) -> None:
        job = Job(
            company="Nuro",
            title="Perception Engineer",
            description="CARLA radar camera PyTorch multimodal perception autonomy.",
            template="ai",
        )
        ranked = rank_experiences(job, self.bank, "ai")
        self.assertEqual(ranked[0].record_id, "uci-scalesense")
        self.assertTrue(ranked[0].explanation)
        self.assertTrue(ranked[0].matched_terms)
        projects = rank_projects(job, self.bank, "ai")
        self.assertIn(
            projects[0].record_id,
            {"tumor-classification", "genome-sequencing", "smart-step", "unity-rl"},
        )

    def test_html_details_do_not_inflate_ai_score(self) -> None:
        job = Job(company="Acme", title="Intern", description="See the details in the email about our html docs.")
        template, _reason, scores = score_template(job)
        self.assertEqual(template, "default")
        self.assertEqual(scores["ai"], 0)

    def test_select_ranked_drops_low_scores_then_pads_to_min(self) -> None:
        ranked = [
            RankedItem("a", 20.0, "experience"),
            RankedItem("b", 18.0, "experience"),
            RankedItem("c", 5.0, "experience"),
            RankedItem("d", 1.0, "experience"),
        ]
        selected = select_ranked(ranked, min_count=3, max_count=4)
        ids = [item.record_id for item in selected]
        self.assertEqual(ids, ["a", "b", "c"])
        self.assertNotIn("d", ids)

    def test_tech_match_does_not_use_substrings(self) -> None:
        record = ExperienceRecord(id="c-only", company="XCorp", role="Builder", technologies=["C"])
        react_job = Job(company="Acme", title="Frontend", description="React frontend work on the web product.")
        self.assertEqual(score_record(react_job, record, "swe"), 0.0)
        c_job = Job(company="Acme", title="Systems Intern", description="C systems programming on Linux.")
        self.assertGreater(score_record(c_job, record, "swe"), 0.0)

    def test_artificial_intelligence_phrase_scores_ai(self) -> None:
        job = Job(
            company="Acme",
            title="Intern",
            description="We apply artificial intelligence to distributed systems.",
        )
        template, _reason, scores = score_template(job)
        self.assertGreaterEqual(scores["ai"], 2)
        self.assertGreaterEqual(scores["swe"], 2)

    def test_skill_selection_never_invents(self) -> None:
        from jobapps.models import Education as Edu
        from jobapps.models import TailoredResume, split_skill_items, unknown_skill_issues

        job = Job(
            company="Acme",
            title="Engineer",
            description="Kubernetes, PyTorch, PostgreSQL, and React experience required.",
        )
        groups = select_skills(job, self.bank.skills, "swe")
        allowed = self.bank.skills.allowed_names()
        resume = TailoredResume(
            experience=[],
            education=[Edu(school="UCI", degree="BS")],
            skills=groups,
        )
        self.assertEqual(unknown_skill_issues(resume, allowed), [])
        names = [item for group in groups for item in split_skill_items(group.items)]
        self.assertNotIn("Kubernetes", names)
        self.assertTrue(any("PyTorch" in group.items for group in groups))
        for group in groups:
            self.assertLessEqual(len(split_skill_items(group.items)), 5)

    def test_skill_selection_puts_jd_matches_first_and_caps_group(self) -> None:
        from jobapps.models import split_skill_items
        from jobapps.ranking import MAX_SKILLS_PER_GROUP

        job = Job(
            company="Acme",
            title="Backend Engineer",
            description="Python PostgreSQL AWS Docker required.",
        )
        groups = select_skills(job, self.bank.skills, "swe")
        by_category = {group.category: split_skill_items(group.items) for group in groups}
        self.assertLessEqual(len(by_category["Languages"]), MAX_SKILLS_PER_GROUP)
        self.assertEqual(by_category["Languages"][0], "Python")
        self.assertEqual(by_category["Backend/Data"][0], "PostgreSQL")
        tools = by_category["Systems/Tools"]
        self.assertIn("AWS", tools[:2])
        self.assertIn("Docker", tools[:2])


if __name__ == "__main__":
    unittest.main()
