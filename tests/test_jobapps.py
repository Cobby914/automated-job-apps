from __future__ import annotations

import os
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobapps.config import COVER_LETTER_EXAMPLES_DIR, WRITING_SAMPLES_DIR
from jobapps.generate import extract_json
from jobapps.graduation import (
    GRADUATION_DEC,
    GRADUATION_JUNE,
    apply_graduation,
    resolve_graduation_date,
    rewrite_graduation_mentions,
)
from jobapps.latex import escape_latex, ensure_url, pdf_page_count
from jobapps.match import match_connection
from jobapps.models import (
    MAX_BULLET_CHARS,
    ApplicationAnswer,
    ApplicationAnswersResult,
    Connection,
    CoverLetter,
    Education,
    Experience,
    GenerationResult,
    Job,
    Project,
    ReviewResult,
    TemplateChoice,
    resolve_template_request,
    SkillGroup,
    TailoredResume,
    allowed_skill_names,
    format_answers_markdown,
    load_additions,
    load_cover_letter_examples,
    load_job,
    load_resume,
    load_skills_bank,
    load_writing_samples,
    overlong_answer_issues,
    overlong_bullet_issues,
    parse_skills_bank,
    unknown_skill_issues,
)
from jobapps.notion import create_application_page, parse_notion_id
from jobapps.pipeline import ensure_output_tree, output_dir_for, slug


ROOT = Path(__file__).resolve().parents[1]


class GenerateTests(unittest.TestCase):
    def test_extract_json_strips_fences(self) -> None:
        payload = extract_json('```json\n{"resume": {}, "cover_letter": {}}\n```')
        self.assertIn('"resume"', payload)
        self.assertTrue(payload.startswith("{"))

    def test_extract_json_requires_object(self) -> None:
        with self.assertRaises(RuntimeError):
            extract_json("no json here")

    def test_review_result_parses(self) -> None:
        review = ReviewResult.model_validate_json(
            extract_json('{"approved": false, "summary": "Too generic", "issues": ["Cliches"]}')
        )
        self.assertFalse(review.approved)
        self.assertEqual(review.issues, ["Cliches"])

    def test_template_choice_parses(self) -> None:
        choice = TemplateChoice.model_validate_json(
            extract_json('{"template": "swe", "reason": "Backend APIs and services."}')
        )
        self.assertEqual(choice.template, "swe")
        self.assertIn("Backend", choice.reason)


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


class GraduationTests(unittest.TestCase):
    def _job(self, **kwargs: object) -> Job:
        base = {
            "company": "Acme",
            "title": "Software Engineer Intern",
            "description": "Build services.",
        }
        base.update(kwargs)
        return Job.model_validate(base)

    def test_defaults_to_june_when_unknown(self) -> None:
        self.assertEqual(resolve_graduation_date(self._job()), GRADUATION_JUNE)

    def test_summer_maps_to_june(self) -> None:
        job = self._job(notes="Summer 2027 internship for graduating seniors.")
        self.assertEqual(resolve_graduation_date(job), GRADUATION_JUNE)

    def test_fall_maps_to_dec(self) -> None:
        job = self._job(starts="Fall 2027")
        self.assertEqual(resolve_graduation_date(job), GRADUATION_DEC)

    def test_september_maps_to_dec(self) -> None:
        job = self._job(starts="September 2027")
        self.assertEqual(resolve_graduation_date(job), GRADUATION_DEC)

    def test_explicit_override(self) -> None:
        job = self._job(starts="Fall 2027", graduation="June 2027")
        self.assertEqual(resolve_graduation_date(job), GRADUATION_JUNE)

    def test_rejects_bad_override(self) -> None:
        with self.assertRaises(ValueError):
            resolve_graduation_date(self._job(graduation="May 2028"))

    def test_rewrites_cover_letter_mentions(self) -> None:
        text = rewrite_graduation_mentions(
            "I am graduating in June 2027 and also mentioned December 2027.",
            GRADUATION_DEC,
        )
        self.assertEqual(text, "I am graduating in Dec. 2027 and also mentioned Dec. 2027.")

    def test_apply_graduation_updates_resume_and_letter(self) -> None:
        materials = GenerationResult(
            resume=TailoredResume(
                experience=[],
                education=[Education(school="UCI", degree="BS", year=GRADUATION_JUNE)],
            ),
            cover_letter=CoverLetter(
                greeting="Dear Hiring Manager,",
                paragraphs=["I graduate in June 2027."],
                closing="Sincerely,",
            ),
        )
        updated = apply_graduation(materials, GRADUATION_DEC)
        self.assertEqual(updated.resume.education[0].year, GRADUATION_DEC)
        self.assertEqual(updated.cover_letter.paragraphs[0], "I graduate in Dec. 2027.")


class BulletLengthTests(unittest.TestCase):
    def _resume(self, experience_bullets: list[str], project_bullets: list[str] | None = None) -> TailoredResume:
        return TailoredResume(
            experience=[
                Experience(
                    company="Acme",
                    role="Intern",
                    bullets=experience_bullets,
                )
            ],
            projects=[
                Project(name="Demo", bullets=project_bullets or []),
            ],
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


class LatexTests(unittest.TestCase):
    def test_escape_specials(self) -> None:
        self.assertIn(r"\&", escape_latex("Acme & Co"))
        self.assertIn(r"\%", escape_latex("100%"))
        self.assertTrue(escape_latex("C++").startswith("C"))

    def test_ensure_url(self) -> None:
        self.assertEqual(ensure_url("linkedin.com/in/a"), "https://linkedin.com/in/a")
        self.assertEqual(ensure_url("https://example.com"), "https://example.com")

    def test_resolve_template_fallback(self) -> None:
        from jobapps.config import COVER_LETTER_TEMPLATES_DIR, RESUME_TEMPLATES_DIR
        from jobapps.latex import _resolve_template

        self.assertEqual(_resolve_template(RESUME_TEMPLATES_DIR, "swe"), "default.tex.j2")
        self.assertEqual(_resolve_template(COVER_LETTER_TEMPLATES_DIR, "ai"), "default.tex.j2")
        self.assertEqual(_resolve_template(RESUME_TEMPLATES_DIR, "default"), "default.tex.j2")

    def test_pdf_page_count(self) -> None:
        # Minimal PDF with a /Pages object declaring /Count 2.
        pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
            b"2 0 obj<< /Type /Pages /Count 2 /Kids [] >>endobj\n"
            b"trailer<< /Root 1 0 R >>\n"
            b"%%EOF\n"
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.pdf"
            path.write_bytes(pdf)
            self.assertEqual(pdf_page_count(path), 2)

    def test_pdf_page_count_flate_object_stream(self) -> None:
        import zlib

        # Mimic PDF 1.5+ object streams: page tree only appears after inflate.
        inner = b"<</Type/Pages/Count 1/Kids[3 0 R]>>"
        compressed = zlib.compress(inner)
        pdf = (
            b"%PDF-1.7\n"
            b"1 0 obj<< /Type /ObjStm /Filter /FlateDecode /Length "
            + str(len(compressed)).encode()
            + b" >>stream\n"
            + compressed
            + b"\nendstream\nendobj\n"
            b"%%EOF\n"
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "compressed.pdf"
            path.write_bytes(pdf)
            self.assertEqual(pdf_page_count(path), 1)


class MatchTests(unittest.TestCase):
    def test_matches_alias(self) -> None:
        connections = [
            Connection(name="Jane", company="Stripe", aliases=["Stripe, Inc."]),
        ]
        hit = match_connection("Stripe Inc", connections)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Jane")

    def test_no_match(self) -> None:
        self.assertIsNone(match_connection("Acme", [Connection(name="Jane", company="Stripe")]))


class OutputLayoutTests(unittest.TestCase):
    def test_slug(self) -> None:
        self.assertEqual(slug("Stripe", "Backend Engineer"), "Stripe-Backend-Engineer")

    def test_output_dir_includes_date_prefix(self) -> None:
        job = Job(company="Stripe", title="Backend Engineer", description="x")
        now = datetime(2026, 8, 25, 12, 0, 0)
        with TemporaryDirectory() as tmp:
            import jobapps.pipeline as pipeline

            original = pipeline.OUTPUT_DIR
            pipeline.OUTPUT_DIR = Path(tmp)
            try:
                path = output_dir_for(job, now=now)
                self.assertEqual(path.name, "2026-08-25_Stripe-Backend-Engineer")
                self.assertEqual(path.parent, Path(tmp))
            finally:
                pipeline.OUTPUT_DIR = original

    def test_ensure_output_tree(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "app"
            inputs, materials, meta = ensure_output_tree(root)
            self.assertTrue(inputs.is_dir())
            self.assertTrue(materials.is_dir())
            self.assertTrue(meta.is_dir())
            self.assertEqual(inputs, root / "inputs")
            self.assertEqual(materials, root / "materials")
            self.assertEqual(meta, root / "meta")


class LoaderTests(unittest.TestCase):
    def test_sample_job_and_resume(self) -> None:
        job = load_job(ROOT / "jobs" / "samples" / "stripe-backend.yaml")
        self.assertIsInstance(job, Job)
        self.assertEqual(job.company, "Stripe")
        self.assertEqual(job.template, "auto")
        resume = load_resume(ROOT / "resume_templates" / "swe.yaml")
        self.assertTrue(resume.experience)
        self.assertTrue(resume.skills)

    def test_ai_and_swe_templates(self) -> None:
        ai = load_resume(ROOT / "resume_templates" / "ai.yaml")
        swe = load_resume(ROOT / "resume_templates" / "swe.yaml")
        ai_companies = {item.company for item in ai.experience}
        swe_companies = {item.company for item in swe.experience}
        self.assertIn("University of California, Irvine", ai_companies)
        self.assertNotIn("University of California, Irvine", swe_companies)
        self.assertIn("FiPET", swe_companies)
        self.assertNotIn("FiPET", ai_companies)
        ai_projects = {item.name for item in ai.projects}
        swe_projects = {item.name for item in swe.projects}
        self.assertIn("Smart Step", ai_projects)
        self.assertNotIn("Smart Step", swe_projects)
        self.assertIn("Reel In", swe_projects)
        self.assertNotIn("Reel In", ai_projects)

    def test_resume_additions_are_text_notes(self) -> None:
        additions = load_additions(ROOT / "resume_additions")
        self.assertIn("### experiences/tena", additions)
        self.assertIn("TENA", additions)
        self.assertIn("### experiences/uci-scalesense", additions)
        self.assertIn("### projects/canvas-notion", additions)
        self.assertIn("Canvas", additions)
        self.assertIn("### projects/tumor-classification", additions)
        self.assertNotIn("Master Experience", additions)
        self.assertNotIn("kind: experience_bullet", additions)
        self.assertNotIn("Master Resume Skills Bank", additions)
        self.assertNotIn("Recommended General-Purpose Skills Section", additions)

    def test_cover_letter_examples_exclude_writing_samples(self) -> None:
        examples = load_cover_letter_examples(COVER_LETTER_EXAMPLES_DIR)
        self.assertIn("Nuro_Cover_Letter", examples)
        self.assertIn("Spacex_Cover_Letter", examples)
        self.assertNotIn("ScaleSense", examples)
        self.assertNotIn("Failure_Analysis", examples)
        self.assertNotIn("Training_a_Sumo", examples)

    def test_cover_letter_writing_samples(self) -> None:
        writing = load_writing_samples(WRITING_SAMPLES_DIR)
        self.assertIn("ScaleSense", writing)
        self.assertIn("Failure_Analysis", writing)
        self.assertIn("Training_a_Sumo", writing)
        self.assertNotIn("Nuro_Cover_Letter", writing)
        self.assertNotIn("Spacex_Cover_Letter", writing)

    def test_job_questions_optional_and_max_length(self) -> None:
        bare = Job(company="Acme", title="Intern", description="Build things.")
        self.assertEqual(bare.questions, [])

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.yaml"
            path.write_text(
                """\
company: Acme
title: Intern
description: Build things.
questions:
  - prompt: Why us?
    max_length: 200
  - prompt: Favorite project?
    max_length: unlimited
  - prompt: Anything else?
""",
                encoding="utf-8",
            )
            job = load_job(path)
        self.assertEqual(len(job.questions), 3)
        self.assertEqual(job.questions[0].max_length, 200)
        self.assertIsNone(job.questions[1].max_length)
        self.assertIsNone(job.questions[2].max_length)

    def test_overlong_answer_issues_and_markdown(self) -> None:
        result = ApplicationAnswersResult(
            answers=[
                ApplicationAnswer(
                    prompt="Why us?",
                    answer="x" * 10,
                    max_length=5,
                ),
                ApplicationAnswer(
                    prompt="Favorite project?",
                    answer="ScaleSense research.",
                    max_length=None,
                ),
            ]
        )
        issues = overlong_answer_issues(result)
        self.assertEqual(len(issues), 1)
        self.assertIn("limit 5", issues[0])

        markdown = format_answers_markdown(result)
        self.assertIn("# Application answers", markdown)
        self.assertIn("Why us?", markdown)
        self.assertIn("Favorite project?", markdown)
        self.assertIn("unlimited", markdown)

    def test_templates_render(self) -> None:
        from jobapps.latex import render_documents
        from jobapps.models import CoverLetter, TailoredResume, load_job, load_resume

        job = load_job(ROOT / "jobs" / "samples" / "stripe-backend.yaml")
        resume = load_resume(ROOT / "resume_templates" / "swe.yaml")
        tailored = TailoredResume(
            summary=resume.summary,
            experience=resume.experience,
            projects=resume.projects,
            education=resume.education,
            skills=resume.skills,
        )
        cover = CoverLetter(
            greeting="Dear Hiring Manager,",
            paragraphs=["I am applying for the role."],
            closing="Sincerely,",
        )
        with TemporaryDirectory() as tmp:
            resume_tex, cover_tex = render_documents(
                job,
                resume.contact,
                tailored,
                cover,
                Path(tmp),
                compile_pdf=False,
            )
            resume_text = resume_tex.read_text(encoding="utf-8")
            self.assertIn("Colin Kwon", resume_text)
            self.assertIn("TENA", resume_text)
            self.assertNotIn("Jake Ryan", resume_text)
            self.assertNotIn("\\section{Summary}", resume_text)
            self.assertIn("Java, Python, C/C++, SQL, JavaScript, HTML/CSS, TypeScript, Assembly", resume_text)
            self.assertNotIn("built-in method", resume_text)
            self.assertIn("Canvas API, Notion API", resume_text)
            self.assertIn("Dear Hiring Manager", cover_tex.read_text(encoding="utf-8"))
            cover_body = cover_tex.read_text(encoding="utf-8")
            self.assertTrue(
                "Times New Roman" in cover_body or "TeX Gyre Termes" in cover_body
            )

    def test_resume_context_exposes_project_stack(self) -> None:
        from jobapps.latex import _resume_context

        tailored = TailoredResume(
            experience=[],
            projects=[
                Project(
                    name="Canvas-Notion",
                    bullets=[
                        "Built a sync engine.",
                        "Stack: Python, Canvas API, Notion API.",
                    ],
                )
            ],
            education=[Education(school="UCI", degree="BS", details="Dean's Honor List")],
            skills=[SkillGroup(category="Languages", items="Python, TypeScript")],
        )
        context = _resume_context(tailored)
        self.assertEqual(context["projects"][0]["stack"], "Python, Canvas API, Notion API.")
        self.assertEqual(context["projects"][0]["bullets"], ["Built a sync engine."])
        self.assertEqual(context["skills"][0]["items"], "Python, TypeScript")
        self.assertIn("Dean's Honor List", context["awards"])

    def test_notion_id_from_url(self) -> None:
        parsed = parse_notion_id("https://www.notion.so/Job-Apps-0123456789abcdef0123456789abcdef")
        self.assertEqual(parsed, "01234567-89ab-cdef-0123-456789abcdef")

    @patch.dict(
        os.environ,
        {
            "NOTION_TOKEN": "test-token",
            "NOTION_DATABASE_ID": "a4577c61-cb41-4314-8a6a-385fc16f2aea",
        },
    )
    @patch("jobapps.notion._client")
    def test_create_application_page_uses_data_source_parent(self, mock_client: MagicMock) -> None:
        notion = MagicMock()
        mock_client.return_value = notion
        notion.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-test-0000-0000-0000-000000000001", "name": "Job Applications"}]
        }
        notion.data_sources.retrieve.return_value = {
            "properties": {
                "Company": {"type": "title"},
                "Role": {"type": "rich_text"},
                "Status": {
                    "type": "select",
                    "select": {"options": [{"name": "Generated"}, {"name": "Applied"}]},
                },
                "Portal URL": {"type": "url"},
                "Referral": {"type": "rich_text"},
                "Output path": {"type": "rich_text"},
                "Created": {"type": "date"},
            }
        }
        notion.pages.create.return_value = {"id": "page-test-0000-0000-0000-000000000002"}

        job = Job(
            company="Prophet Security",
            title="Software Engineer, Backend Intern",
            portal_url="https://jobs.ashbyhq.com/prophet-security/example",
            description="Backend intern role.",
        )
        output_dir = Path("/tmp/output/Prophet-Security")

        url = create_application_page(job, output_dir, None)

        notion.databases.retrieve.assert_called_once_with("a4577c61-cb41-4314-8a6a-385fc16f2aea")
        notion.data_sources.retrieve.assert_called_once_with("ds-test-0000-0000-0000-000000000001")
        notion.pages.create.assert_called_once()
        kwargs = notion.pages.create.call_args.kwargs
        self.assertEqual(
            kwargs["parent"],
            {"type": "data_source_id", "data_source_id": "ds-test-0000-0000-0000-000000000001"},
        )
        props = kwargs["properties"]
        self.assertEqual(props["Company"]["title"][0]["text"]["content"], "Prophet Security")
        self.assertEqual(
            props["Role"]["rich_text"][0]["text"]["content"],
            "Software Engineer, Backend Intern",
        )
        self.assertEqual(props["Portal URL"]["url"], "https://jobs.ashbyhq.com/prophet-security/example")
        self.assertEqual(props["Referral"]["rich_text"][0]["text"]["content"], "No referral match")
        self.assertEqual(props["Output path"]["rich_text"][0]["text"]["content"], str(output_dir))
        self.assertEqual(props["Status"]["select"]["name"], "Generated")
        self.assertEqual(url, "https://www.notion.so/pagetest000000000000000000000002")

    def test_build_application_properties_maps_tracker_schema(self) -> None:
        from jobapps.notion import build_application_properties
        from jobapps.models import Connection

        schema = {
            "Company 1": {"type": "title"},
            "Company Name (text)": {"type": "rich_text"},
            "Role": {"type": "rich_text"},
            "Portal URL": {"type": "url"},
            "Output path": {"type": "rich_text"},
            "Referral": {"type": "checkbox"},
            "Recruiter / Contact": {"type": "rich_text"},
            "Status": {
                "type": "status",
                "status": {
                    "options": [
                        {"name": "Interested"},
                        {"name": "To apply"},
                        {"name": "Applied"},
                    ]
                },
            },
            "Source": {
                "type": "select",
                "select": {
                    "options": [
                        {"name": "LinkedIn"},
                        {"name": "Company site"},
                        {"name": "Referral"},
                        {"name": "Other"},
                    ]
                },
            },
            "Job Type": {
                "type": "select",
                "select": {
                    "options": [
                        {"name": "Full-time"},
                        {"name": "Internship"},
                    ]
                },
            },
            "Created": {"type": "created_time"},
            "Date Applied": {"type": "date"},
        }
        job = Job(
            company="Prophet Security",
            title="Software Engineer, Backend Intern",
            portal_url="https://jobs.ashbyhq.com/prophet-security/example",
            description="Backend intern role.",
        )
        props = build_application_properties(
            schema,
            job,
            Path("/tmp/out"),
            Connection(name="Alex Friend", company="Prophet Security", relationship="alumni"),
        )
        self.assertEqual(props["Company 1"]["title"][0]["text"]["content"], "Prophet Security")
        self.assertEqual(props["Company Name (text)"]["rich_text"][0]["text"]["content"], "Prophet Security")
        self.assertTrue(props["Referral"]["checkbox"])
        self.assertEqual(props["Status"]["status"]["name"], "To apply")
        self.assertEqual(props["Source"]["select"]["name"], "Referral")
        self.assertEqual(props["Job Type"]["select"]["name"], "Internship")
        self.assertEqual(
            props["Recruiter / Contact"]["rich_text"][0]["text"]["content"],
            "Alex Friend — alumni",
        )
        self.assertNotIn("Created", props)
        self.assertNotIn("Date Applied", props)


class SkillsBankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = load_skills_bank(ROOT / "resume_additions" / "skills.md")
        self.categories, self.defaults = parse_skills_bank(self.bank)
        self.allowed = allowed_skill_names(self.bank)

    def _resume(self, skills: list[SkillGroup]) -> TailoredResume:
        return TailoredResume(
            experience=[],
            education=[Education(school="UCI", degree="BS")],
            skills=skills,
        )

    def test_recommended_section_is_default_not_inventory(self) -> None:
        self.assertNotIn("Recommended General-Purpose Skills Section", self.categories)
        self.assertIn("Languages", self.categories)
        self.assertIn("Python", self.categories["Languages"])
        self.assertEqual(
            [group.category for group in self.defaults],
            ["Languages", "Backend/Data", "AI/ML", "Systems/Tools"],
        )
        self.assertIn("C/C++", self.defaults[0].items)

    def test_allowed_set_includes_recommended_and_extra_tech(self) -> None:
        for name in ("python", "c/c++", "linux/unix", "pytorch", "carla", "posix", "canvas api"):
            self.assertIn(name, self.allowed, name)
        self.assertNotIn("kubernetes", self.allowed)

    def test_rejects_invented_skill(self) -> None:
        resume = self._resume(
            [SkillGroup(category="Cloud", items="AWS, Docker, Kubernetes")]
        )
        issues = unknown_skill_issues(resume, self.allowed)
        self.assertEqual(len(issues), 1)
        self.assertIn("Kubernetes", issues[0])

    def test_accepts_combined_slash_names(self) -> None:
        resume = self._resume(
            [
                SkillGroup(
                    category="Languages",
                    items="Python, C/C++, Java, TypeScript",
                ),
                SkillGroup(category="Systems/Tools", items="Linux/Unix, Git, AWS"),
            ]
        )
        self.assertEqual(unknown_skill_issues(resume, self.allowed), [])

    def test_combined_name_allowed_when_both_halves_exist(self) -> None:
        allowed = allowed_skill_names("## Languages\n\n- C\n- C++\n")
        resume = self._resume([SkillGroup(category="Languages", items="C/C++")])
        self.assertEqual(unknown_skill_issues(resume, allowed), [])

    def test_missing_skills_bank_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_skills_bank(Path(tmp) / "skills.md")


class NotifyTests(unittest.TestCase):
    def test_notify_skips_osascript_in_docker(self) -> None:
        from jobapps import notify as notify_mod

        with patch.dict(os.environ, {"JOBAPPS_IN_DOCKER": "1"}):
            with patch.object(notify_mod.subprocess, "run") as run:
                notify_mod.notify("Title", "Body")
                run.assert_not_called()

    def test_reveal_skips_open_in_docker(self) -> None:
        from jobapps import notify as notify_mod

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume.pdf"
            path.write_bytes(b"%PDF")
            with patch.dict(os.environ, {"JOBAPPS_IN_DOCKER": "1"}):
                with patch.object(notify_mod.subprocess, "run") as run:
                    notify_mod.reveal(path)
                    run.assert_not_called()


class WorkerClaimTests(unittest.TestCase):
    def test_second_claim_fails_while_first_holds_lock(self) -> None:
        from jobapps.worker import release_claim, try_claim

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.yaml"
            path.write_text("company: X\ntitle: Y\ndescription: z\n", encoding="utf-8")
            first = try_claim(path)
            self.assertIsNotNone(first)
            second = try_claim(path)
            self.assertIsNone(second)
            release_claim(first)  # type: ignore[arg-type]
            third = try_claim(path)
            self.assertIsNotNone(third)
            release_claim(third)  # type: ignore[arg-type]

    def test_is_job_file_filters(self) -> None:
        from jobapps.config import JOBS_DIR
        from jobapps.jobs_util import is_job_file

        self.assertTrue(is_job_file(JOBS_DIR / "acme.yaml"))
        self.assertFalse(is_job_file(JOBS_DIR / "acme.yaml.error.txt"))
        self.assertFalse(is_job_file(JOBS_DIR / ".hidden.yaml"))
        self.assertFalse(is_job_file(JOBS_DIR / "samples" / "stripe-backend.yaml"))


if __name__ == "__main__":
    unittest.main()
