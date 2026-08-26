"""Watch jobs/ and generate materials when a YAML file appears."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from jobapps.config import JOBS_DIR, PROCESSED_DIR, load_env
from jobapps.jobs_util import is_job_file
from jobapps.latex import ensure_latex_tools
from jobapps.notify import notify
from jobapps.pipeline import run_job_file

DEBOUNCE_SECONDS = 1.5


class JobHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        self._timers: dict[str, threading.Timer] = {}
        self._inflight: set[str] = set()
        self._lock = threading.Lock()

    def on_created(self, event: FileSystemEvent) -> None:
        self._schedule(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._schedule(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        dest = getattr(event, "dest_path", None)
        if dest:
            self._schedule_path(Path(str(dest)))

    def _schedule(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._schedule_path(Path(str(event.src_path)))

    def _schedule_path(self, path: Path) -> None:
        if not is_job_file(path):
            return
        key = str(path.resolve())
        with self._lock:
            existing = self._timers.pop(key, None)
            if existing is not None:
                existing.cancel()
            timer = threading.Timer(DEBOUNCE_SECONDS, self._run, args=(path,))
            self._timers[key] = timer
            timer.daemon = True
            timer.start()

    def _run(self, path: Path) -> None:
        key = str(path.resolve())
        with self._lock:
            self._timers.pop(key, None)
            if key in self._inflight:
                return
            self._inflight.add(key)
        try:
            if not path.is_file() or path.stat().st_size == 0:
                return
            run_job_file(path)
        except Exception:
            # Errors are written to a sidecar and notified inside run_job_file.
            pass
        finally:
            with self._lock:
                self._inflight.discard(key)


def watch() -> None:
    load_env()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ensure_latex_tools()

    handler = JobHandler()
    observer = Observer()
    observer.schedule(handler, str(JOBS_DIR), recursive=False)
    observer.start()
    notify("Job apps watcher started", f"Drop a job YAML into {JOBS_DIR}")
    print(f"Watching {JOBS_DIR} — drop a .yaml job file to generate materials.")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    watch()
