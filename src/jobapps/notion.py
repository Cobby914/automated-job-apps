"""Notion database setup and application row creation."""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from typing import Any

from notion_client import Client

from jobapps.config import ROOT, require_env
from jobapps.models import Connection, Job


def parse_notion_id(value: str) -> str:
    raw = value.strip().split("?")[0]
    hex_only = re.sub(r"[^0-9a-fA-F]", "", raw)
    if len(hex_only) >= 32:
        hex_only = hex_only[-32:]
        return (
            f"{hex_only[0:8]}-{hex_only[8:12]}-{hex_only[12:16]}-"
            f"{hex_only[16:20]}-{hex_only[20:32]}"
        )
    return value.strip()


STATUS_OPTIONS = [
    {"name": "Generated", "color": "blue"},
    {"name": "Applied", "color": "yellow"},
    {"name": "Interviewing", "color": "purple"},
    {"name": "Offer", "color": "green"},
    {"name": "Rejected", "color": "red"},
    {"name": "Withdrawn", "color": "gray"},
]

# Preferred Status values, in order, depending on property type.
_STATUS_PREFERENCES = (
    "To apply",
    "Interested",
    "Generated",
    "Applied",
)

_READ_ONLY_TYPES = frozenset(
    {
        "created_time",
        "last_edited_time",
        "created_by",
        "last_edited_by",
        "formula",
        "rollup",
        "unique_id",
        "button",
    }
)


def _client() -> Client:
    return Client(auth=require_env("NOTION_TOKEN"))


_DATA_SOURCE_CACHE: dict[str, str] = {}
_SCHEMA_CACHE: dict[str, dict[str, dict[str, Any]]] = {}


def _pick_data_source(sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer a named applications source over empty placeholder data sources."""
    if not sources:
        raise RuntimeError("No data sources on Notion database")
    named = [s for s in sources if (s.get("name") or "").strip()]
    preferred = [
        s
        for s in named
        if "job" in (s.get("name") or "").lower() or "application" in (s.get("name") or "").lower()
    ]
    if preferred:
        return preferred[0]
    if named:
        return named[0]
    return sources[0]


def _data_source_id(notion: Client, database_id: str) -> str:
    normalized = parse_notion_id(database_id)
    cached = _DATA_SOURCE_CACHE.get(normalized)
    if cached is not None:
        return cached
    db = notion.databases.retrieve(normalized)
    sources = db.get("data_sources") or []
    if not sources:
        raise RuntimeError(f"No data sources on Notion database {database_id}")
    source_id = _pick_data_source(sources)["id"]
    _DATA_SOURCE_CACHE[normalized] = source_id
    return source_id


def _data_source_schema(notion: Client, data_source_id: str) -> dict[str, dict[str, Any]]:
    cached = _SCHEMA_CACHE.get(data_source_id)
    if cached is not None:
        return cached
    ds = notion.data_sources.retrieve(data_source_id)
    props = ds.get("properties") or {}
    _SCHEMA_CACHE[data_source_id] = props
    return props


def create_database() -> str:
    parent_page_id = parse_notion_id(require_env("NOTION_PARENT_PAGE_ID"))
    notion = _client()
    database = notion.databases.create(
        parent={"type": "page_id", "page_id": parent_page_id},
        title=[{"type": "text", "text": {"content": "Job Applications"}}],
        initial_data_source={
            "properties": {
                "Company": {"title": {}},
                "Role": {"rich_text": {}},
                "Status": {"select": {"options": STATUS_OPTIONS}},
                "Portal URL": {"url": {}},
                "Referral": {"rich_text": {}},
                "Output path": {"rich_text": {}},
                "Created": {"date": {}},
            }
        },
    )
    database_id = database["id"]
    _write_database_id(database_id)
    return database_id


def _write_database_id(database_id: str) -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    lines = env_path.read_text(encoding="utf-8").splitlines()
    written = False
    updated: list[str] = []
    for line in lines:
        if line.startswith("NOTION_DATABASE_ID="):
            updated.append(f"NOTION_DATABASE_ID={database_id}")
            written = True
        else:
            updated.append(line)
    if not written:
        if updated and updated[-1].strip():
            updated.append("")
        updated.append(f"NOTION_DATABASE_ID={database_id}")
    env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def referral_text(connection: Connection | None) -> str:
    if connection is None:
        return "No referral match"
    extra = f" — {connection.relationship}" if connection.relationship else ""
    return f"{connection.name}{extra}"


def _rich_text(content: str) -> dict[str, Any]:
    return {
        "type": "rich_text",
        "rich_text": [{"type": "text", "text": {"content": content[:2000]}}],
    }


def _title(content: str) -> dict[str, Any]:
    return {
        "type": "title",
        "title": [{"type": "text", "text": {"content": content[:2000]}}],
    }


def _option_names(prop: dict[str, Any], prop_type: str) -> list[str]:
    block = prop.get(prop_type) or {}
    names: list[str] = []
    for option in block.get("options") or []:
        if isinstance(option, dict):
            name = option.get("name")
            if name:
                names.append(str(name))
        elif option:
            names.append(str(option))
    return names


def _status_value(prop: dict[str, Any], prop_type: str) -> str | None:
    names = _option_names(prop, prop_type)
    if not names:
        return None
    lower = {name.lower(): name for name in names}
    for preferred in _STATUS_PREFERENCES:
        if preferred.lower() in lower:
            return lower[preferred.lower()]
    return names[0]


def _find_prop(
    schema: dict[str, dict[str, Any]],
    *,
    names: tuple[str, ...] = (),
    types: tuple[str, ...] = (),
) -> tuple[str, dict[str, Any]] | None:
    """Find a property by exact name (case-insensitive).

    When ``names`` is empty, the first property whose type is in ``types`` wins.
    Named lookups never fall back to an unrelated property of the same type.
    """
    if names:
        lowered = {key.lower(): key for key in schema}
        for name in names:
            key = lowered.get(name.lower())
            if key is None:
                continue
            prop = schema[key]
            if not types or prop.get("type") in types:
                return key, prop
        return None
    if types:
        for key, prop in schema.items():
            if prop.get("type") in types:
                return key, prop
    return None


def _infer_job_type(title: str, options: list[str]) -> str | None:
    if not options:
        return None
    lower = {name.lower(): name for name in options}
    blob = title.lower()
    if any(token in blob for token in ("intern", "internship")) and "internship" in lower:
        return lower["internship"]
    if "contract" in blob and "contract" in lower:
        return lower["contract"]
    if "part-time" in blob or "part time" in blob:
        if "part-time" in lower:
            return lower["part-time"]
    if "full-time" in lower:
        return lower["full-time"]
    return None


def build_application_properties(
    schema: dict[str, dict[str, Any]],
    job: Job,
    output_dir: Path,
    connection: Connection | None,
) -> dict[str, Any]:
    """Map job fields onto whatever property names/types the Notion DB exposes."""
    properties: dict[str, Any] = {}
    has_referral = connection is not None
    referral = referral_text(connection)

    title_match = _find_prop(schema, names=("Company", "Company 1", "Name"), types=("title",))
    if title_match is None:
        title_match = _find_prop(schema, types=("title",))
    if title_match is None:
        raise RuntimeError("Notion data source has no title property")
    title_key, _ = title_match
    properties[title_key] = _title(job.company)

    company_text = _find_prop(
        schema,
        names=("Company Name (text)", "Company Name", "Company"),
        types=("rich_text",),
    )
    if company_text is not None and company_text[0] != title_key:
        properties[company_text[0]] = _rich_text(job.company)

    role = _find_prop(schema, names=("Role", "Title", "Position"), types=("rich_text",))
    if role is not None:
        properties[role[0]] = _rich_text(job.title)

    portal = _find_prop(schema, names=("Portal URL", "URL", "Link", "Application URL"), types=("url",))
    if portal is not None:
        properties[portal[0]] = {
            "type": "url",
            "url": job.portal_url.strip() or None,
        }

    output = _find_prop(schema, names=("Output path", "Output Path", "Output"), types=("rich_text",))
    if output is not None:
        properties[output[0]] = _rich_text(str(output_dir))

    referral_prop = _find_prop(schema, names=("Referral",), types=("checkbox", "rich_text"))
    if referral_prop is not None:
        key, prop = referral_prop
        if prop.get("type") == "checkbox":
            properties[key] = {"type": "checkbox", "checkbox": has_referral}
        else:
            properties[key] = _rich_text(referral)

    contact = _find_prop(
        schema,
        names=("Recruiter / Contact", "Contact", "Referral Contact"),
        types=("rich_text",),
    )
    if contact is not None and has_referral:
        properties[contact[0]] = _rich_text(referral)

    notes = _find_prop(schema, names=("Notes",), types=("rich_text",))
    if notes is not None and has_referral:
        properties[notes[0]] = _rich_text(f"Referral: {referral}")

    status = _find_prop(schema, names=("Status",), types=("status", "select"))
    if status is not None:
        key, prop = status
        prop_type = prop.get("type") or "select"
        value = _status_value(prop, prop_type)
        if value:
            properties[key] = {"type": prop_type, prop_type: {"name": value}}

    source = _find_prop(schema, names=("Source",), types=("select",))
    if source is not None:
        key, prop = source
        options = _option_names(prop, "select")
        lower = {name.lower(): name for name in options}
        if has_referral and "referral" in lower:
            chosen = lower["referral"]
        elif "company site" in lower:
            chosen = lower["company site"]
        elif "other" in lower:
            chosen = lower["other"]
        else:
            chosen = options[0] if options else None
        if chosen:
            properties[key] = {"type": "select", "select": {"name": chosen}}

    job_type = _find_prop(schema, names=("Job Type", "Type"), types=("select",))
    if job_type is not None:
        key, prop = job_type
        chosen = _infer_job_type(job.title, _option_names(prop, "select"))
        if chosen:
            properties[key] = {"type": "select", "select": {"name": chosen}}

    created = _find_prop(schema, names=("Created",), types=("date",))
    if created is not None:
        properties[created[0]] = {
            "type": "date",
            "date": {"start": date.today().isoformat()},
        }

    # Drop anything that somehow targeted a read-only property.
    return {
        key: value
        for key, value in properties.items()
        if schema.get(key, {}).get("type") not in _READ_ONLY_TYPES
    }


def create_application_page(
    job: Job,
    output_dir: Path,
    connection: Connection | None,
) -> str | None:
    if not os.getenv("NOTION_TOKEN", "").strip() or not os.getenv("NOTION_DATABASE_ID", "").strip():
        return None

    notion = _client()
    data_source_id = _data_source_id(notion, require_env("NOTION_DATABASE_ID"))
    schema = _data_source_schema(notion, data_source_id)
    page = notion.pages.create(
        parent={"type": "data_source_id", "data_source_id": data_source_id},
        properties=build_application_properties(schema, job, output_dir, connection),
    )
    page_id = page["id"].replace("-", "")
    return f"https://www.notion.so/{page_id}"
