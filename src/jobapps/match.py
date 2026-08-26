"""Company-name matching against the connections list."""

from __future__ import annotations

from jobapps.models import Connection


def _norm(value: str) -> str:
    cleaned = value.lower().strip()
    for suffix in (", inc.", ", inc", " inc.", " inc", ", llc", " llc", ", ltd", " ltd"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return " ".join(cleaned.replace(",", " ").split())


def match_connection(company: str, connections: list[Connection]) -> Connection | None:
    target = _norm(company)
    if not target:
        return None
    for connection in connections:
        names = [connection.company, *connection.aliases]
        if any(_norm(name) == target for name in names if name):
            return connection
    return None
