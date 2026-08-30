# Automated Job Applications

Drop a job YAML into `jobs/`. The pipeline ranks your structured career bank, builds an `ApplicationPlan`, writes a tailored resume (and optional cover letter), validates the result in Python, runs a cheap then expensive semantic review, applies targeted repairs, fits the PDFs to one page, logs a Notion row, and notifies you with the portal URL and any referral match.

Writing and review go through `src/jobapps/llm.py`. Direct OpenAI/Anthropic keys are preferred when set; otherwise the Cursor API is the fallback.

## Architecture

```
Job YAML
  → duplicate fingerprint (skip if already processed)
  → ApplicationPlan (template, ranked experiences/projects, skill whitelist)
  → Resume draft (LLM, selected records only)
  → Python validation (length, skills, source IDs, sections)
  → Semantic review (cheap checker, escalate only on issues)
  → Targeted repair if needed
  → Python validation again
  → One final semantic pass
  → PDF fit (trim / shorten, no full rewrite)
  → Fit-aware Python validation
  → resume_final.json / cover_letter_final.json + PDFs
```

Python owns objective checks (bullet length, skill whitelist, source-ID existence, education, page count). The LLM reviewer owns subjective checks (relevance, voice, misleading claims). Repairs rewrite one located bullet or paragraph, not the whole resume.

Checkpoints live in `{job}.progress.json` (`planned` → `resume_drafted` → `cover_drafted` → `reviewed` → `fitted` → `complete`). A PDF failure restarts from rendering; a cover-letter failure does not regenerate the resume.

## Setup

1. Install Python 3.11 or newer.
2. Install a TeX distribution so `xelatex` and `latexmk` work ([MacTeX](https://www.tug.org/mactex/), BasicTeX, or MiKTeX).
3. Create a virtualenv and install this project:

```bash
cd /path/to/automated-job-apps
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

4. Copy `.env.example` to `.env`. Add a Cursor API key from [Cursor Dashboard → Integrations](https://cursor.com/dashboard/integrations), or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` to call those providers directly. `LLM_PROVIDER` can force `openai`, `anthropic`, or `cursor`.
5. Edit `career/` YAML with your real experiences, projects, skills, and profile.
6. Optional Notion:
   - Create an internal integration at [notion.so/my-integrations](https://www.notion.so/my-integrations).
   - Share **one parent page** with that integration.
   - Put the token and parent page ID (or page URL) in `.env`.
   - Run `python -m jobapps setup-notion` to create the database. The database ID is written back to `.env`.

## Fill in your materials

| Path | What to put there |
| --- | --- |
| `career/profile.yaml` | Contact and education |
| `career/experiences.yaml` | Jobs/internships with facts, metrics, canonical bullets, and source IDs |
| `career/projects.yaml` | Projects with the same sourced structure |
| `career/skills.yaml` | Allowed skill inventory plus recommended groups |
| `resume_templates/*.yaml` | Track layouts (`swe`, `ai`, `default`) |
| `resume_templates/default.tex.j2` | Resume LaTeX layout |
| `cover_letter_templates/default.tex.j2` | Cover letter layout |
| `cover_letter_examples/*.md` | Past cover letters — structure, length, and tone only |
| `connections/connections.yaml` | People who could refer you, keyed by company |

The career YAML is the source of truth. Every generated content bullet must cite fact/metric IDs from the selected records. Canonical bullets should not invent percentages or technologies that are not in those sources.

`resume_additions/` is leftover notes and is not loaded by the generator.

## Job files

Create a YAML file and drop it into `jobs/` (not `jobs/samples/` or `jobs/processed/`):

```yaml
company: Stripe
title: Backend Engineer
portal_url: https://stripe.com/jobs/listing/backend-engineer
description: |
  Paste the full job description here.
notes: ""
```

A copy lives at `jobs/samples/stripe-backend.yaml`.

Optional: `template: auto` (default) scores the job description with token/phrase matching and picks `swe`, `ai`, or `default`. Set `template: swe`, `template: ai`, or `template: default` to force a track. Layout always uses `default.tex.j2` unless a matching `.tex.j2` exists.

Set `cover_letter: false` to skip cover-letter generation, review, and PDF.

Graduation date is either **June 2027** or **Dec. 2027**, chosen from when the role starts (on/before June 2027 → June; after June 2027 → Dec.). Optional fields:

```yaml
starts: Summer 2027          # or Fall 2027, 2027-09, June 2027, …
graduation: Dec. 2027        # force a date; overrides starts / inference
```

If neither is set, the pipeline scans `notes`, `title`, and `description` for season/month hints (e.g. “Summer 2027 internship”). When nothing is found it defaults to June 2027.

Optional screening / portal questions — when present, the writer drafts answers and the checker reviews them after the resume and cover letter. Answers are grounded in selected career records and written to `materials/answers.md`:

```yaml
questions:
  - prompt: Why do you want to work at Stripe?
    max_length: 2000
  - prompt: What is your favorite project and why?
    max_length: 200
  - prompt: Tell us about a time you debugged a hard production issue.
    # omit max_length (or use null / unlimited) for no character cap
```

## Run

Watch the folder (leave this terminal open):

```bash
python -m jobapps watch
```

Or process one file immediately:

```bash
python -m jobapps process jobs/samples/stripe-backend.yaml
```

On success you get:

```
output/2026-08-25_Stripe-Backend-Engineer/
  inputs/job.yaml
  materials/resume.tex
  materials/resume.pdf
  materials/cover_letter.tex
  materials/cover_letter.pdf
  materials/answers.md   # only when job YAML has questions:
  meta/application_plan.json
  meta/resume_draft.json
  meta/resume_final.json
  meta/review.json
  meta/meta.yaml         # includes pipeline metrics
```

The OS file manager opens on the PDF. A notification includes the company, role, portal URL, and referral (or “No referral match”). Duplicate postings (same portal URL, or same company + title + description hash) are skipped. Successful drops in `jobs/` are moved to `jobs/processed/`. Failures leave the YAML in place and write `*.error.txt` next to it; a `.progress.json` sidecar lets the next run resume.

Resumes and cover letters are enforced to one page each. Ranking selects 3–4 experiences and 2–3 projects by relevance (not a fixed top-N of irrelevant items). Resume bullets must fit on a single line (≤113 characters; prefer 90–113 when facts allow). If a PDF still exceeds one page after automatic trim/shorten passes, the run fails.

You can edit the `.tex` files and recompile:

```bash
cd output/2026-08-25_Stripe-Backend-Engineer/materials
latexmk -xelatex resume.tex
```

## Parallel processing with Docker

Run 2–3 workers on one machine so jobs do not wait on each other. Uses Docker Compose (not Kubernetes).

**Do not** run `python -m jobapps watch` on the host at the same time — both would fight over `jobs/`.

```bash
# Build once
docker compose build

# Start three workers
docker compose up --scale worker=3

# In another terminal, drop job YAMLs into jobs/
cp jobs/samples/stripe-backend.yaml jobs/acme-backend.yaml
# edit company/title/description, then save more files as needed

# Logs show which worker claimed each file
docker compose logs -f

# Stop
docker compose down
```

Each worker polls `jobs/`, claims a file with `flock`, runs the same pipeline as `process`, and writes to `output/`. Secrets come from `.env` (`CURSOR_API_KEY`, Notion vars). Cover letters use Times New Roman when available, otherwise TeX Gyre Termes inside the Linux image.

If Cursor rate-limits concurrent runs, start with `--scale worker=2` instead of `3`.

You can also run a single worker without Compose:

```bash
python -m jobapps worker
```
