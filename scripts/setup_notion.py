#!/usr/bin/env python3
"""Create the Job Applications Notion database. Run from the project root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jobapps.config import load_env
from jobapps.notion import create_database


def main() -> int:
    load_env()
    database_id = create_database()
    print("Created Notion database.")
    print(f"NOTION_DATABASE_ID={database_id}")
    print("This value was written to .env if that file exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
