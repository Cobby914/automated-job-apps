# GrokBot ↔ AutomatedJobApps handoff

You find roles and generate application packets by dropping job YAML into this repo. Do **not** shell `python -m jobapps process` per job. Docker workers pick up files from `jobs/`.

**Repo root:** `/Users/colinkwon/Desktop/AutomatedJobApps`

---

## Your job vs this repo

| You (GrokBot) | This repo (workers) |
| --- | --- |
| Find roles | Tailor resume + cover letter |
| Write one YAML per role into `jobs/` | Claude review, LaTeX → PDFs |
| Ensure Compose workers are up | Optional screening answers |
| Wait for success / failure signals | Notion row + move YAML to `processed/` |
| Ping Colin with `output/` paths | |

Portal submit is **out of scope** — Colin does that after reviewing PDFs.

---

## Before dropping files (6am or any batch)

1. `cd /Users/colinkwon/Desktop/AutomatedJobApps`
2. If workers are not already running:

```bash
docker compose up --scale worker=2 -d
```

3. **Never** run `python -m jobapps watch` on the host while Compose workers are up — they fight over `jobs/`.

Workers poll every **2 seconds**, claim with `flock`, run the pipeline, then move or error.

---

## Drop contract

- **Path:** `jobs/` top level only  
  - ✅ `/Users/colinkwon/Desktop/AutomatedJobApps/jobs/acme-backend.yaml`  
  - ❌ `jobs/samples/`  
  - ❌ `jobs/processed/`
- **One file per role.** Extension `.yaml` or `.yml`.
- **Slug filenames:** lowercase kebab-case, unique in the batch (e.g. `stripe-backend-engineer.yaml`).

### Atomic write (required)

Workers ignore names starting with `_` or `.`. To avoid a half-written file being claimed:

1. Write the full YAML to `jobs/_slug.yaml`
2. `mv` it to `jobs/slug.yaml` (same filesystem — atomic rename)

Example:

```bash
# after writing jobs/_acme-swe.yaml completely:
mv jobs/_acme-swe.yaml jobs/acme-swe.yaml
```

---

## Job YAML schema

**Required**

| Field | Notes |
| --- | --- |
| `company` | Company name |
| `title` | Role title |
| `description` | Full job description (paste the whole JD) |

**Optional but recommended**

| Field | Notes |
| --- | --- |
| `portal_url` | Application link |
| `notes` | Anything useful for the writer (referral context, location, etc.) |
| `template` | `auto` (default), `swe`, `ai`, or `default` |
| `starts` | e.g. `Summer 2027`, `Fall 2027`, `2027-09` — used to pick graduation |
| `graduation` | Force `June 2027` or `Dec. 2027` (overrides inference) |
| `questions` | Screening prompts; each needs `prompt`, optional `max_length` |

### Minimal example

```yaml
company: Stripe
title: Backend Engineer
portal_url: https://stripe.com/jobs/listing/backend-engineer
description: |
  Paste the full job description here.
```

### With screening questions

```yaml
company: Acme
title: Software Engineer Intern
portal_url: https://boards.greenhouse.io/acme/jobs/123
description: |
  Full JD text…
starts: Summer 2027
template: swe
questions:
  - prompt: Why do you want to work at Acme?
    max_length: 2000
  - prompt: Describe a project you are proud of.
    max_length: 500
```

Reference copy: `jobs/samples/stripe-backend.yaml` (do not drop from `samples/` — copy fields into a new top-level file).

---

## Wait signals

A file sitting in `jobs/` is **not** failure — it may still be processing.

| Outcome | How you know |
| --- | --- |
| **Success** | YAML leaves `jobs/` and appears under `jobs/processed/` (same basename, or `stem-YYYYMMDD-HHMMSS` if a duplicate name already existed) |
| **Failure** | YAML stays in `jobs/` **and** sidecar `jobs/<filename>.error.txt` appears (e.g. `acme-swe.yaml.error.txt`) |

After success, materials are under:

```
output/YYYY-MM-DD_Company-Title/
  materials/resume.pdf
  materials/cover_letter.pdf
  materials/answers.md    # only if questions: were provided
  meta/meta.yaml
```

If that folder already existed, a time suffix is added: `output/YYYY-MM-DD_Company-Title-HHMMSS/`.

---

## After the batch

1. For each dropped slug, wait until **processed** or **`.error.txt`**.
2. Ping Colin with:
   - Success: list of `output/…` folders (and portal URLs if useful)
   - Failure: filename + first lines of the `.error.txt` sidecar
3. Do not delete error sidecars or re-drop the same broken YAML without fixing the content first.

---

## Hard rules

- Drop YAML only — no per-job `process` CLI.
- Top-level `jobs/` only; atomic `_` → final name rename.
- No host `watch` alongside Compose.
- Do not submit applications or fill portals unless Colin explicitly asks later.
- Prefer full JDs in `description`; thin blurbs produce weak packets.
