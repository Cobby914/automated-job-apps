from __future__ import annotations

import os
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
import json
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
    ReviewIssue,
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


class CareerBankTests(unittest.TestCase):
    def setUp(self) -> None:
        from jobapps.career import load_career_bank

        self.bank = load_career_bank(ROOT / "career")

    def test_loads_profile_experiences_projects_skills(self) -> None:
        self.assertEqual(self.bank.profile.contact.name, "Colin Kwon")
        self.assertTrue(self.bank.profile.education)
        ids = {item.id for item in self.bank.experiences}
        self.assertEqual(
            ids,
            {"mk-lending", "tena", "fipet", "uci-scalesense", "commit-the-change"},
        )
        project_ids = {item.id for item in self.bank.projects}
        self.assertIn("genome-sequencing", project_ids)
        self.assertIn("unity-rl", project_ids)
        self.assertIn("unix-shell", project_ids)

    def test_source_ids_are_unique(self) -> None:
        seen: set[str] = set()
        for record in [*self.bank.experiences, *self.bank.projects]:
            for source_id in record.source_ids():
                self.assertNotIn(source_id, seen, source_id)
                seen.add(source_id)
        self.assertGreater(len(seen), 40)

    def test_track_membership(self) -> None:
        exp = self.bank.experience_by_id()
        self.assertIn("ai", exp["uci-scalesense"].tracks)
        self.assertNotIn("swe", exp["uci-scalesense"].tracks)
        self.assertIn("swe", exp["fipet"].tracks)
        self.assertNotIn("ai", exp["fipet"].tracks)
        projects = self.bank.project_by_id()
        self.assertIn("swe", projects["reel-in"].tracks)
        self.assertNotIn("ai", projects["reel-in"].tracks)
        self.assertIn("ai", projects["smart-step"].tracks)

    def test_yaml_skills_whitelist(self) -> None:
        allowed = self.bank.skills.allowed_names()
        for name in ("python", "c/c++", "linux/unix", "pytorch", "carla", "posix", "canvas api"):
            self.assertIn(name, allowed, name)
        self.assertNotIn("kubernetes", allowed)


class RankingAndPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        from jobapps.career import load_career_bank

        self.bank = load_career_bank(ROOT / "career")

    def test_explicit_template_override(self) -> None:
        from jobapps.ranking import select_template

        job = Job(
            company="Acme",
            title="Backend Engineer",
            description="Ship APIs.",
            template="ai",
        )
        template, reason, auto = select_template(job)
        self.assertEqual(template, "ai")
        self.assertFalse(auto)
        self.assertIn("explicitly", reason.lower())

    def test_swe_job_scores_swe(self) -> None:
        from jobapps.ranking import score_template

        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Build REST APIs, PostgreSQL services, and TypeScript backends.",
        )
        template, _reason, scores = score_template(job)
        self.assertEqual(template, "swe")
        self.assertGreater(scores["swe"], scores["ai"])

    def test_ai_job_scores_ai(self) -> None:
        from jobapps.ranking import score_template

        job = Job(
            company="Nuro",
            title="Autonomy Perception Intern",
            description="PyTorch computer vision, CARLA simulation, radar-camera fusion research.",
        )
        template, _reason, scores = score_template(job)
        self.assertEqual(template, "ai")
        self.assertGreater(scores["ai"], scores["swe"])

    def test_close_scores_default(self) -> None:
        from jobapps.ranking import score_template

        job = Job(company="Acme", title="Intern", description="Join our team.")
        template, _reason, _scores = score_template(job)
        self.assertEqual(template, "default")

    def test_ranking_prefers_tech_overlap(self) -> None:
        from jobapps.ranking import rank_experiences, rank_projects

        job = Job(
            company="Nuro",
            title="Perception Engineer",
            description="CARLA radar camera PyTorch multimodal perception autonomy.",
            template="ai",
        )
        ranked = rank_experiences(job, self.bank, "ai")
        self.assertEqual(ranked[0].record_id, "uci-scalesense")
        projects = rank_projects(job, self.bank, "ai")
        self.assertIn(
            projects[0].record_id,
            {"tumor-classification", "genome-sequencing", "smart-step", "unity-rl"},
        )

    def test_plan_respects_layout_budget(self) -> None:
        from jobapps.plan import build_application_plan

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

    def test_html_details_do_not_inflate_ai_score(self) -> None:
        from jobapps.ranking import score_template

        job = Job(
            company="Acme",
            title="Intern",
            description="See the details in the email about our html docs.",
        )
        template, _reason, scores = score_template(job)
        self.assertEqual(template, "default")
        self.assertEqual(scores["ai"], 0)

    def test_select_ranked_drops_low_scores_then_pads_to_min(self) -> None:
        from jobapps.ranking import RankedItem, select_ranked

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
        from jobapps.career import ExperienceRecord
        from jobapps.ranking import score_record

        record = ExperienceRecord(
            id="c-only",
            company="XCorp",
            role="Builder",
            technologies=["C"],
        )
        react_job = Job(
            company="Acme",
            title="Frontend",
            description="React frontend work on the web product.",
        )
        self.assertEqual(score_record(react_job, record, "swe"), 0.0)
        c_job = Job(
            company="Acme",
            title="Systems Intern",
            description="C systems programming on Linux.",
        )
        self.assertGreater(score_record(c_job, record, "swe"), 0.0)

    def test_artificial_intelligence_phrase_scores_ai(self) -> None:
        from jobapps.ranking import score_template

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
        from jobapps.models import split_skill_items, unknown_skill_issues
        from jobapps.ranking import select_skills

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
        from jobapps.ranking import MAX_SKILLS_PER_GROUP, select_skills

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


class ValidateAndFitTests(unittest.TestCase):
    def _resume(self, **kwargs: object) -> TailoredResume:
        base: dict[str, object] = dict(
            experience=[
                Experience(company="Acme", role="Intern", bullets=["A" * 100, "B" * 100]),
                Experience(company="Beta", role="Intern", bullets=["C" * 100, "D" * 100]),
                Experience(company="Gamma", role="Intern", bullets=["E" * 100, "F" * 100]),
            ],
            projects=[
                Project(
                    name="Demo",
                    bullets=["P" * 100, "Q" * 100, "Stack: Python"],
                ),
                Project(
                    name="Other",
                    bullets=["R" * 100, "S" * 100, "Stack: C"],
                ),
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
        from jobapps.career import load_career_bank
        from jobapps.models import ApplicationPlan, LayoutBudget
        from jobapps.validate import validate_resume

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
        from jobapps.validate import validate_cover_letter

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

    def test_parse_overfull_hbox(self) -> None:
        from jobapps.fit import parse_overfull_hbox

        log = r"Overfull \hbox (12.3pt too wide) in paragraph at lines 20--22"
        self.assertEqual(parse_overfull_hbox(log), ["12.3pt"])

    def test_python_trim_drops_lowest_priority_project_bullets_first(self) -> None:
        from jobapps.fit import apply_python_trim
        from jobapps.models import ApplicationPlan, LayoutBudget

        plan = ApplicationPlan(
            template="swe",
            experience_ids=["acme", "beta", "gamma"],
            project_ids=["demo", "other"],
            resume_priorities=["acme", "beta", "gamma", "demo", "other"],
            layout=LayoutBudget(project_bullets=2),
        )
        resume = self._resume(
            projects=[
                Project(
                    name="Demo",
                    bullets=["P" * 100, "Q" * 100, "T" * 100, "Stack: Python"],
                ),
                Project(
                    name="Other",
                    bullets=["R" * 100, "S" * 100, "U" * 100, "Stack: C"],
                ),
            ]
        )
        trimmed = apply_python_trim(resume, plan)
        self.assertIsNotNone(trimmed)
        assert trimmed is not None
        other = next(item for item in trimmed.projects if item.name == "Other")
        content = [b for b in other.bullets if not b.lower().startswith("stack:")]
        self.assertEqual(len(content), 2)

    def test_cover_letter_false_skips_cover_render(self) -> None:
        from jobapps.latex import render_documents
        from jobapps.models import load_resume

        job = load_job(ROOT / "jobs" / "samples" / "stripe-backend.yaml")
        job = job.model_copy(update={"cover_letter": False})
        self.assertFalse(job.cover_letter)
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

    def test_job_cover_letter_flag_parses(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.yaml"
            path.write_text(
                "company: Acme\ntitle: Intern\ndescription: Build things.\ncover_letter: false\n",
                encoding="utf-8",
            )
            job = load_job(path)
        self.assertFalse(job.cover_letter)


class CheckerEscalationTests(unittest.TestCase):
    def test_sonnet_approve_skips_opus(self) -> None:
        from jobapps.career import load_career_bank
        from jobapps.generate import review_materials
        from jobapps.plan import build_application_plan

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
        from jobapps.career import load_career_bank
        from jobapps.generate import review_materials
        from jobapps.plan import build_application_plan

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


class SourceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        from jobapps.career import load_career_bank

        self.bank = load_career_bank(ROOT / "career")

    def test_canonical_bullets_cite_fact_or_metric_ids(self) -> None:
        for record in [*self.bank.experiences, *self.bank.projects]:
            known = record.fact_metric_ids()
            for bullet in record.bullets:
                if bullet.text.strip().lower().startswith("stack:"):
                    continue
                self.assertTrue(bullet.sources, bullet.id)
                for source_id in bullet.sources:
                    self.assertIn(source_id, known, f"{bullet.id} -> {source_id}")

    def test_load_rejects_unknown_canonical_source(self) -> None:
        from jobapps.career import CanonicalBullet, _validate_canonical_sources

        broken = self.bank.model_copy(
            update={
                "experiences": [
                    self.bank.experiences[0].model_copy(
                        update={
                            "bullets": [
                                CanonicalBullet(
                                    id="bad.bullet.1",
                                    text="Invented claim with no grounding.",
                                    sources=["missing.metric.9"],
                                )
                            ]
                        }
                    )
                ]
            }
        )
        with self.assertRaises(ValueError) as ctx:
            _validate_canonical_sources(broken)
        self.assertIn("unknown source", str(ctx.exception))

    def test_draft_rejects_empty_and_unknown_sources(self) -> None:
        from jobapps.models import DraftExperience, SourcedBullet
        from jobapps.plan import build_application_plan, selected_experiences, selected_projects
        from jobapps.validate import validate_draft_resume

        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
        )
        plan = build_application_plan(job, self.bank)
        experiences = selected_experiences(plan, self.bank)
        projects = selected_projects(plan, self.bank)
        from jobapps.models import DraftProject, DraftResume

        draft = DraftResume(
            experience=[
                DraftExperience(
                    company=item.company,
                    role=item.role,
                    bullets=[
                        SourcedBullet(text="A" * 100, sources=[item.facts[0].id]),
                        SourcedBullet(text="B" * 100, sources=[item.facts[min(1, len(item.facts) - 1)].id]),
                    ],
                )
                for item in experiences
            ],
            projects=[
                DraftProject(
                    name=item.name,
                    bullets=[
                        SourcedBullet(text="P" * 100, sources=[item.facts[0].id]),
                        SourcedBullet(text="Q" * 100, sources=[item.facts[min(1, len(item.facts) - 1)].id]),
                        SourcedBullet(text=f"Stack: {item.stack}", sources=[]),
                    ],
                )
                for item in projects
            ],
            education=list(self.bank.profile.education),
            skills=plan.skill_groups,
        )
        issues = validate_draft_resume(
            draft, plan, self.bank.skills.allowed_names(), self.bank.profile, self.bank
        )
        source_issues = [item for item in issues if "source" in item.casefold()]
        self.assertEqual(source_issues, [])

        first = experiences[0]
        bad = draft.model_copy(
            update={
                "experience": [
                    DraftExperience(
                        company=first.company,
                        role=first.role,
                        bullets=[
                            SourcedBullet(text="A" * 100, sources=[]),
                            SourcedBullet(text="B" * 100, sources=["not-a-real-id"]),
                        ],
                    ),
                    *draft.experience[1:],
                ]
            }
        )
        issues = validate_draft_resume(
            bad, plan, self.bank.skills.allowed_names(), self.bank.profile, self.bank
        )
        self.assertTrue(any("missing sources" in item for item in issues))
        self.assertTrue(any("unknown sources" in item for item in issues))

    def test_validate_sources_rejects_unsupported_metric_number(self) -> None:
        from jobapps.models import DraftExperience, DraftResume, SourcedBullet
        from jobapps.plan import build_application_plan
        from jobapps.validate import validate_sources

        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
        )
        plan = build_application_plan(job, self.bank)
        plan = plan.model_copy(update={"experience_ids": ["mk-lending"]})
        lending = self.bank.experience_by_id()["mk-lending"]
        invented = DraftResume(
            experience=[
                DraftExperience(
                    company=lending.company,
                    role=lending.role,
                    bullets=[
                        SourcedBullet(
                            text="Improved accuracy by 93% using TypeScript APIs.",
                            sources=["made-up-source"],
                        ),
                        SourcedBullet(text="B" * 100, sources=[lending.facts[0].id]),
                    ],
                )
            ]
        )
        issues = validate_sources(invented, plan, self.bank)
        self.assertTrue(any("unknown sources" in item for item in issues))

        fact_only = DraftResume(
            experience=[
                DraftExperience(
                    company=lending.company,
                    role=lending.role,
                    bullets=[
                        SourcedBullet(
                            text="Improved accuracy by 93% using TypeScript APIs.",
                            sources=[lending.facts[0].id],
                        ),
                        SourcedBullet(text="B" * 100, sources=[lending.facts[0].id]),
                    ],
                )
            ]
        )
        issues = validate_sources(fact_only, plan, self.bank)
        self.assertTrue(any("numeric claims" in item for item in issues))
        self.assertTrue(any("93" in item or "metric" in item.casefold() for item in issues))

    def test_stack_lines_may_have_empty_sources(self) -> None:
        from jobapps.models import DraftExperience, DraftProject, DraftResume, SourcedBullet
        from jobapps.plan import build_application_plan, selected_experiences, selected_projects
        from jobapps.validate import source_issues

        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
        )
        plan = build_application_plan(job, self.bank)
        draft = DraftResume(
            experience=[
                DraftExperience(
                    company=item.company,
                    role=item.role,
                    bullets=[
                        SourcedBullet(text="A" * 100, sources=[item.facts[0].id]),
                        SourcedBullet(text="B" * 100, sources=[item.facts[0].id]),
                    ],
                )
                for item in selected_experiences(plan, self.bank)
            ],
            projects=[
                DraftProject(
                    name=item.name,
                    bullets=[
                        SourcedBullet(text="P" * 100, sources=[item.facts[0].id]),
                        SourcedBullet(text="Q" * 100, sources=[item.facts[0].id]),
                        SourcedBullet(text=f"Stack: {item.stack}", sources=[]),
                    ],
                )
                for item in selected_projects(plan, self.bank)
            ],
            education=list(self.bank.profile.education),
            skills=plan.skill_groups,
        )
        self.assertEqual(source_issues(draft, plan, self.bank), [])


class ReviewLocationTests(unittest.TestCase):
    def test_parse_issue_location(self) -> None:
        from jobapps.models import parse_issue_location

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
        from jobapps.career import load_career_bank
        from jobapps.models import (
            DraftExperience,
            DraftProject,
            DraftResume,
            ReviewIssue,
            SourcedBullet,
        )
        from jobapps.pipeline import _repair_from_review
        from jobapps.plan import build_application_plan

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
        from jobapps.career import load_career_bank
        from jobapps.models import DraftExperience, DraftProject, DraftResume, SourcedBullet
        from jobapps.pipeline import _repair_from_review
        from jobapps.plan import build_application_plan

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


class DedupAndResumeTests(unittest.TestCase):
    def test_fingerprint_prefers_portal_url(self) -> None:
        from jobapps.dedup import job_fingerprint

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

    def test_fingerprint_uses_description_when_no_url(self) -> None:
        from jobapps.dedup import job_fingerprint

        left = Job(company="Acme", title="SWE", description="Same posting text")
        right = Job(company="Acme", title="SWE", description="Different posting")
        self.assertNotEqual(job_fingerprint(left), job_fingerprint(right))

    def test_find_duplicate_in_processed(self) -> None:
        from jobapps.dedup import find_duplicate

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

    def test_merge_fitted_preserves_matching_sources(self) -> None:
        from jobapps.models import DraftExperience, DraftResume, SourcedBullet
        from jobapps.pipeline import merge_fitted_into_draft

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
        from jobapps.career import load_career_bank
        from jobapps.models import DraftExperience, DraftProject, DraftResume, SourcedBullet
        from jobapps.pipeline import process_job_file
        from jobapps.plan import build_application_plan

        bank = load_career_bank(ROOT / "career")
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
            cover_letter=False,
        )
        plan = build_application_plan(job, bank)
        draft = DraftResume(
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
                        "graduation_year": "June 2027",
                        "checker_model": "claude-4.5-sonnet",
                        "escalated": False,
                        "metrics": {},
                    }
                ),
                encoding="utf-8",
            )

            def fake_fit(_job, _contact, resume, cover, _materials_dir, _plan, _bank, metrics):
                metrics.final_resume_pages = 1
                return resume, cover

            with patch("jobapps.pipeline.find_duplicate", return_value=None):
                with patch("jobapps.pipeline.generate_resume_draft") as mock_draft:
                    with patch("jobapps.pipeline.review_materials") as mock_review:
                        with patch("jobapps.pipeline._render_and_fit", side_effect=fake_fit):
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

    def test_process_validates_after_semantic_repair(self) -> None:
        from jobapps.career import load_career_bank
        from jobapps.models import DraftExperience, DraftProject, DraftResume, SourcedBullet
        from jobapps.pipeline import process_job_file
        from jobapps.plan import build_application_plan
        from jobapps.validate import validate_draft_resume

        bank = load_career_bank(ROOT / "career")
        job = Job(
            company="Stripe",
            title="Backend Engineer",
            description="Python TypeScript PostgreSQL REST APIs Node.js.",
            cover_letter=False,
        )
        plan = build_application_plan(job, bank)
        draft = DraftResume(
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

        def fake_fit(_job, _contact, resume, cover, _materials_dir, _plan, _bank, metrics):
            metrics.final_resume_pages = 1
            return resume, cover

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
                                ) as mock_validate:
                                    with patch(
                                        "jobapps.pipeline._render_and_fit",
                                        side_effect=fake_fit,
                                    ):
                                        with patch(
                                            "jobapps.pipeline.create_application_page",
                                            return_value=None,
                                        ):
                                            result = process_job_file(job_path)
        self.assertEqual(mock_review.call_count, 2)
        self.assertGreaterEqual(mock_validate.call_count, 2)
        self.assertTrue(result.checker_approved)
        self.assertEqual(result.metrics.semantic_revisions, 1)
        self.assertEqual(result.metrics.semantic_review_failures, 1)


class LlmProviderTests(unittest.TestCase):
    def test_resolve_provider_by_model_name(self) -> None:
        from jobapps.llm import resolve_provider

        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "ak", "OPENAI_API_KEY": "sk", "LLM_PROVIDER": ""},
            clear=False,
        ):
            self.assertEqual(resolve_provider("claude-4.5-sonnet"), "anthropic")
            self.assertEqual(resolve_provider("gpt-4.1"), "openai")
            self.assertEqual(resolve_provider("gpt-5.6-sol"), "openai")

    def test_resolve_provider_falls_back_to_cursor(self) -> None:
        from jobapps.llm import resolve_provider

        with patch.dict(os.environ, {"LLM_PROVIDER": ""}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("LLM_PROVIDER", None)
            self.assertEqual(resolve_provider("claude-4.5-sonnet"), "cursor")
            self.assertEqual(resolve_provider("gpt-4.1"), "cursor")

    def test_anthropic_model_alias(self) -> None:
        from jobapps.llm import anthropic_model_id

        self.assertEqual(anthropic_model_id("claude-4.5-sonnet"), "claude-sonnet-4-5")
        self.assertEqual(anthropic_model_id("claude-sonnet-4-5"), "claude-sonnet-4-5")


if __name__ == "__main__":
    unittest.main()

