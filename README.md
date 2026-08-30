# Automated Job Applications

Drop a job YAML into `jobs/`. The pipeline treats `career/*.yaml` as the single source of truth, ranks experiences and projects deterministically, builds an `ApplicationPlan`, writes a tailored resume (and optional cover letter), validates provenance in Python, runs a cheap then expensive semantic review, applies targeted repairs, fits the PDFs to one page, logs a Notion row, and notifies you with the portal URL and any referral match.

Writing, review, and repair go through `src/jobapps/llm.py`. Callers never branch on OpenAI, Anthropic, or Cursor. Direct OpenAI/Anthropic keys are preferred when set; otherwise the Cursor API is the fallback.

## Architecture

```
Job YAML
  → duplicate / near-duplicate fingerprint (skip if already processed)
  → optional plan reuse for a similar same-company posting
  → ApplicationPlan (template, ranked ids, priorities, match explanations, skill whitelist)
  → Resume draft (LLM, selected records only, cached career prefix)
  → Python validation (length, skills, source IDs, numeric claims vs metric sources)
  → Semantic review (cheap checker, escalate only on issues)
  → Targeted repair of located bullets/paragraphs (hard retry budget)
  → Python validation again
  → One final semantic pass
  → PDF fit (deterministic trim first; LLM shorten only if needed)
  → Re-review only if fitting used an LLM rewrite
  → resume_final.json / cover_letter_final.json written after layout is done
  → Screening answers checkpointed independently
  → Notion + notification
```

Python owns objective checks: bullet length, skill whitelist, section counts, page counts, duplicate detection, template choice, relevance ranking, trimming, and source-ID / metric validity. The LLM reviewer owns subjective checks (relevance, voice, misleading claims). Repairs rewrite one located bullet or paragraph, not the whole resume.

Every generated experience/project bullet must cite at least one fact or metric ID from that same record. Unknown IDs and numeric claims that are not in the cited metrics are rejected as `invalid_provenance`.

Checkpoints live in `{job}.progress.json`:

`planned` → `resume_drafted` → `cover_drafted` → `reviewed` → `fitted` → `answers_generated` → `answers_validated` → `complete`

A PDF failure restarts from rendering. A Notion failure does not regenerate the resume, cover letter, or answers. Re-running a completed stage overwrites the same artifacts instead of creating duplicates.

## Career knowledge base

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

Metrics must declare `kind: absolute`, `relative`, or `count` so generators cannot turn a 30% relative gain into an absolute score, or F1 0.40→0.70 into “70% improvement.” Canonical bullets may only cite facts/metrics on that same record.

`resume_additions/` is leftover notes and is not loaded by the generator.

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

4. Copy `.env.example` to `.env`. Set provider keys and **explicit model names**. Do not rely on hidden defaults:
   - `OPENAI_WRITER_MODEL` — strong model for the initial resume/cover letter
   - `OPENAI_REVIEWER_MODEL` — cheaper model for the first semantic pass
   - `OPENAI_ESCALATION_MODEL` — strong reviewer only when something is flagged
   - `OPENAI_REPAIR_MODEL` — cheaper model for tiny bullet/paragraph repairs
   - `OPENAI_REASONING_EFFORT` — GPT-5/o-series Responses API effort (`none`/`minimal`/`low`/`medium`/`high`). Role overrides: `OPENAI_WRITER_REASONING_EFFORT`, `OPENAI_REVIEWER_REASONING_EFFORT`, `OPENAI_REPAIR_REASONING_EFFORT`, `OPENAI_ESCALATION_REASONING_EFFORT`
   - `LLM_PROVIDER` can force `openai`, `anthropic`, or `cursor`
   - `LLM_REVIEWER_PROVIDER` can send review to a different provider
   - `LLM_MAX_RETRIES`, `LLM_RETRY_BASE_SECONDS`, `LLM_DAILY_BUDGET_USD`
5. Edit `career/` YAML with your real experiences, projects, skills, and profile.
6. Optional Notion:
   - Create an internal integration at [notion.so/my-integrations](https://www.notion.so/my-integrations).
   - Share **one parent page** with that integration.
   - Put the token and parent page ID (or page URL) in `.env`.
   - Run `python -m jobapps setup-notion` to create the database. The database ID is written back to `.env`.

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

Optional: `template: auto` (default) scores the job description with token/phrase matching and picks `swe`, `ai`, or `default`. Set `template: swe`, `template: ai`, or `template: default` to force a track.

Set `cover_letter: false` to skip cover-letter generation, review, and PDF.

Graduation date is either **June 2027** or **Dec. 2027**, chosen from when the role starts. Optional fields:

```yaml
starts: Summer 2027
graduation: Dec. 2027
```

Optional screening / portal questions — answers are grounded in selected career records, written to `materials/answers.md`, and checkpointed so a later Notion/PDF failure does not regenerate them:

```yaml
questions:
  - prompt: Why do you want to work at Stripe?
    max_length: 2000
  - prompt: What is your favorite project and why?
    max_length: 200
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
  materials/answers.md
  meta/application_plan.json   # ids, scores, priorities, match explanations
  meta/resume_draft.json
  meta/resume_final.json       # content actually used for the PDF
  meta/cover_letter_final.json
  meta/fit_report.json         # what trimming/shortening changed
  meta/review.json
  meta/answers.json
  meta/meta.yaml               # pipeline metrics, cost, manual-review flag
```

Shared telemetry (gitignored under `output/`):

- `output/ranking_log.jsonl` — score of every experience/project, selected or not. After 20–50 applications, inspect whether weak items are being selected and tune thresholds from real results.
- `output/usage.jsonl` — per-call provider, model, input/cached/output/reasoning tokens, latency, estimated cost. Aggregated into cost per application, daily cost, weekly cost, and average cost/application in `meta.yaml`.

Do not add more architecture until you have measured those numbers on a real sample.

Duplicate postings (same portal URL with tracking parameters stripped, or same company + title + whitespace-normalized description) are skipped. A later posting at the same company and title reuses the previous `ApplicationPlan` only when the description is similar enough (default ≥ 0.65) but not a near-duplicate (< 0.92).

Failures leave the YAML in place and write `*.error.txt` with a **failure category**:

`generation_failure` · `invalid_provenance` · `semantic_rejection` · `pdf_overflow` · `latex_compile_failure` · `provider_failure` · `upload_failure`

Retry budgets: semantic rewrite 1, bullet repair 2, PDF-fit LLM repair 1–2, cover-letter repair 1, screening-answer repair 1. After that the application is marked for manual review instead of looping.

Provider 429/5xx errors retry with backoff inside `llm.py`. A provider timeout does not restart earlier pipeline stages.

## Tests

```bash
python -m unittest discover -s tests -q
```

Suites are split by layer: `test_career`, `test_ranking`, `test_plan`, `test_validation`, `test_provenance`, `test_repair`, `test_fit`, `test_llm`, `test_pipeline`. Integration tests mock model responses. An optional live smoke test checks structured outputs, token reporting, and cost calculation:

```bash
JOBAPPS_SMOKE=1 python -m unittest tests.test_smoke
```

## Parallel processing with Docker

Run 2–3 workers on one machine so jobs do not wait on each other. Uses Docker Compose (not Kubernetes).

**Do not** run `python -m jobapps watch` on the host at the same time — both would fight over `jobs/`.

```bash
docker compose build
docker compose up --scale worker=3
```

Each worker polls `jobs/`, claims a file with `flock`, runs the same pipeline as `process`, and writes to `output/`. If the API rate-limits concurrent runs, start with `--scale worker=2`.
