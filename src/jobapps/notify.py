"""Desktop notifications and Finder reveal (macOS); no-op elsewhere / in Docker."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _desktop_enabled() -> bool:
    if os.getenv("JOBAPPS_IN_DOCKER", "").strip() in {"1", "true", "yes"}:
        return False
    return sys.platform == "darwin"


def _apple_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def notify(title: str, body: str) -> None:
    print(f"{title}: {body}", flush=True)
    if not _desktop_enabled():
        return
    script = f'display notification "{_apple_string(body)}" with title "{_apple_string(title)}"'
    subprocess.run(["osascript", "-e", script], check=False, capture_output=True)


def reveal(path: Path) -> None:
    if not _desktop_enabled():
        return
    target = path if path.is_file() else path
    if not target.exists():
        return
    subprocess.run(["open", "-R", str(target)], check=False)
