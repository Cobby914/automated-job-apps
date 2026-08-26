"""Command-line entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jobapps.config import load_env
from jobapps.notion import create_database
from jobapps.pipeline import run_job_file
from jobapps.watch import watch
from jobapps.worker import DEFAULT_POLL_SECONDS, worker_loop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate tailored resumes and cover letters from job files.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("watch", help="Watch jobs/ and generate when a YAML file is dropped")

    process = sub.add_parser("process", help="Process one job YAML file immediately")
    process.add_argument("job_file", type=Path)

    worker = sub.add_parser(
        "worker",
        help="Poll jobs/ with flock claiming (for Docker Compose replicas)",
    )
    worker.add_argument(
        "--poll",
        type=float,
        default=DEFAULT_POLL_SECONDS,
        help=f"Seconds between idle polls (default {DEFAULT_POLL_SECONDS})",
    )
    worker.add_argument(
        "--worker-id",
        default=None,
        help="Label for logs (default: WORKER_ID, HOSTNAME, or pid)",
    )

    sub.add_parser("setup-notion", help="Create the applications database on a shared Notion page")

    args = parser.parse_args(argv)
    load_env()

    if args.command == "watch":
        watch()
        return 0
    if args.command == "process":
        path = args.job_file.expanduser().resolve()
        if not path.is_file():
            print(f"Job file not found: {path}", file=sys.stderr)
            return 1
        try:
            result = run_job_file(path)
        except Exception as error:
            print(error, file=sys.stderr)
            return 1
        print(f"Wrote {result.output_dir}")
        return 0
    if args.command == "worker":
        worker_loop(poll_seconds=args.poll, worker_id=args.worker_id)
        return 0
    if args.command == "setup-notion":
        database_id = create_database()
        print("Created Notion database.")
        print(f"NOTION_DATABASE_ID={database_id}")
        print("This value was written to .env if that file exists.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
