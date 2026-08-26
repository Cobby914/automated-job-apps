"""Poll jobs/ and process files with cross-process flock claiming."""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

from jobapps.config import JOBS_DIR, PROCESSED_DIR, load_env
from jobapps.jobs_util import is_job_file, list_job_files
from jobapps.latex import ensure_latex_tools
from jobapps.pipeline import run_job_file

DEFAULT_POLL_SECONDS = 2.0


def try_claim(path: Path) -> int | None:
    """Open and lock a job file. Returns an open fd, or None if another worker holds it."""
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except OSError:
        os.close(fd)
        return None
    return fd


def release_claim(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def process_claimed(path: Path, fd: int, worker_id: str) -> None:
    print(f"[{worker_id}] claimed {path.name}", flush=True)
    try:
        if not path.is_file() or path.stat().st_size == 0:
            print(f"[{worker_id}] skip empty/missing {path.name}", flush=True)
            return
        result = run_job_file(path)
        print(f"[{worker_id}] wrote {result.output_dir}", flush=True)
    except Exception as error:
        print(f"[{worker_id}] failed {path.name}: {error}", flush=True)
    finally:
        release_claim(fd)


def poll_once(worker_id: str) -> bool:
    """Try to claim and process one job. Returns True if a job was claimed."""
    for path in list_job_files():
        if not is_job_file(path):
            continue
        fd = try_claim(path)
        if fd is None:
            continue
        process_claimed(path, fd, worker_id)
        return True
    return False


def worker_loop(
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    worker_id: str | None = None,
) -> None:
    load_env()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ensure_latex_tools()

    wid = worker_id or os.getenv("WORKER_ID") or os.getenv("HOSTNAME") or str(os.getpid())
    print(f"[{wid}] worker started — polling {JOBS_DIR} every {poll_seconds}s", flush=True)
    try:
        while True:
            claimed = poll_once(wid)
            if not claimed:
                time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print(f"[{wid}] worker stopped", flush=True)


if __name__ == "__main__":
    worker_loop()
