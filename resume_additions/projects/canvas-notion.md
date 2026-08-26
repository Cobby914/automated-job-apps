Canvas–Notion Automation
Oct. 2024–May 2025

Built a Python synchronization engine that automatically transfers assignments and task information from Canvas LMS into Notion.

Integrated the Canvas and Notion REST APIs and created a metadata-processing pipeline that translated assignment information into Notion tasks. Added custom heuristics for estimating task priority and effort.

Implemented idempotent synchronization using Canvas Assignment IDs, ensuring repeated executions updated existing tasks rather than producing duplicate database entries.

Used GitHub Actions to automate execution so the synchronization workflow could run without manual intervention.

Best statistic:
- No strong numeric result currently; strongest value is technical depth: two APIs + automation + data transformation + scheduled execution + idempotency.

Resume bullets:
- Made a Python sync engine to automate Canvas-to-Notion task tracking, using REST APIs and GitHub Actions.
- Engineered metadata transformation pipeline to calculate task priority and effort using custom heuristics.
- Implemented idempotent logic with Canvas Assignment IDs to ensure consistency and prevent duplicate entries.
- Stack: Python, Canvas API, Notion API, GitHub Actions.
