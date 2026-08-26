# Automated Job Applications

Drop a job YAML into `jobs/`. The watcher uses GPT-5.6 Sol to tailor a resume and cover letter, then Claude Opus 5 reviews the draft. After that it writes LaTeX plus PDFs, logs a row in Notion, and shows a macOS notification with the application portal and any referral match.

## Setup

1. Install Python 3.11 or newer. This Mac’s `/usr/bin/python3` is 3.9; if you have Miniconda, use `/opt/miniconda3/bin/python3.13`.
2. Install a TeX distribution so `xelatex` and `latexmk` work in Terminal ([MacTeX](https://www.tug.org/mactex/) or BasicTeX).
3. Create a virtualenv and install this project:

```bash
cd ~/Desktop/AutomatedJobApps
/opt/miniconda3/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

4. Copy `.env.example` to `.env` and add a Cursor API key from [Cursor Dashboard → Integrations](https://cursor.com/dashboard/integrations). Writing uses `CURSOR_WRITER_MODEL` (default `gpt-5.6-sol`); review uses `CURSOR_CHECKER_MODEL` (default `claude-opus-5`).
5. Replace the sample resume, additions, cover-letter examples, and connections with your real content.
6. Optional Notion:
   - Create an internal integration at [notion.so/my-integrations](https://www.notion.so/my-integrations).
   - Share **one parent page** with that integration.
   - Put the token and parent page ID (or page URL) in `.env`.
   - Run `python -m jobapps setup-notion` to create the database. The database ID is written back to `.env`.

## Fill in your materials

| Path | What to put there |
| --- | --- |
| `resume_templates/swe.yaml` | SWE base resume (default track) |
| `resume_templates/ai.yaml` | AI/ML base resume |
| `resume_templates/default.yaml` | Legacy generic base resume |
| `resume_templates/default.tex.j2` | Resume layout (shared by all tracks) |
| `cover_letter_templates/default.tex.j2` | Cover letter layout |
| `cover_letter_examples/*.md` | Past cover letters — the model matches structure, length, and tone |
| `writing_samples/*.md` | Project/research writeups — cover-letter prose style; also content for screening answers |
| `resume_additions/experiences/*.md` | Notes on jobs and internships the model may turn into bullets |
| `resume_additions/projects/*.md` | Notes on projects the model may include or rewrite |
| `resume_additions/skills.md` | Allowed skills inventory; the model picks a short subset per job |
| `connections/connections.yaml` | People who could refer you, keyed by company |

The sample resume is fictional placeholder data. Do not send it.

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

Optional: `template: auto` (default) lets the model pick `swe`, `ai`, or `default` from the job description. Set `template: swe`, `template: ai`, or `template: default` to force a track. Layout always uses `default.tex.j2` unless a matching `.tex.j2` exists.

Graduation date is either **June 2027** or **Dec. 2027**, chosen from when the role starts (on/before June 2027 → June; after June 2027 → Dec.). Optional fields:

```yaml
starts: Summer 2027          # or Fall 2027, 2027-09, June 2027, …
graduation: Dec. 2027        # force a date; overrides starts / inference
```

If neither is set, the pipeline scans `notes`, `title`, and `description` for season/month hints (e.g. “Summer 2027 internship”). When nothing is found it defaults to June 2027.

Optional screening / portal questions — when present, GPT drafts answers and Claude reviews them after the resume and cover letter. Answers are grounded in the job description, resume, addition summaries, and writing samples, and written to `materials/answers.md`:

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
  meta/meta.yaml
```

Finder opens on the PDF. A notification includes the company, role, portal URL, and referral (or “No referral match”). Successful drops in `jobs/` are moved to `jobs/processed/`. Failures leave the YAML in place and write `*.error.txt` next to it.

Resumes and cover letters are enforced to one page each; resumes should fill most of that page (~4 experiences and 3–4 projects when material allows). Resume bullets must fit on a single line (≤113 characters; prefer 90–113 when facts allow). If a PDF still exceeds one page after two automatic shorten passes, the run fails.

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
