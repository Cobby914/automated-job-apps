"""Render Jinja2 LaTeX templates and compile them to PDF."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import zlib
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from jobapps.config import COVER_LETTER_TEMPLATES_DIR, RESUME_TEMPLATES_DIR
from jobapps.models import Contact, CoverLetter, Job, TailoredResume

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(value: str) -> str:
    return "".join(_LATEX_SPECIALS.get(char, char) for char in value)


def escape_url(value: str) -> str:
    return value.replace("\\", "/").replace("%", r"\%").replace("#", r"\#").replace("&", r"\&")


def ensure_url(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if text.startswith(("http://", "https://", "mailto:")):
        return text
    return f"https://{text}"


def _jinja(directory: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(directory)),
        block_start_string=r"\BLOCK{",
        block_end_string="}",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
        comment_start_string=r"\#{",
        comment_end_string="}",
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _contact_context(contact: Contact) -> dict[str, str]:
    return {
        "name": escape_latex(contact.name),
        "email": escape_latex(contact.email),
        "email_href": escape_url(contact.email),
        "phone": escape_latex(contact.phone),
        "location": escape_latex(contact.location),
        "linkedin": escape_latex(contact.linkedin),
        "github": escape_latex(contact.github),
        "website": escape_latex(contact.website),
        "linkedin_url": escape_url(ensure_url(contact.linkedin)),
        "github_url": escape_url(ensure_url(contact.github)),
        "website_url": escape_url(ensure_url(contact.website)),
    }


def _split_stack(bullets: list[str]) -> tuple[list[str], str]:
    if not bullets:
        return [], ""
    last = bullets[-1].strip()
    if last.lower().startswith("stack:"):
        return bullets[:-1], last.split(":", 1)[1].strip()
    return bullets, ""


def _resume_context(resume: TailoredResume) -> dict[str, object]:
    awards: list[str] = []
    for item in resume.education:
        for part in item.details.split(";"):
            text = part.strip()
            if text:
                awards.append(escape_latex(text))

    projects = []
    for item in resume.projects:
        bullets, stack = _split_stack(item.bullets)
        projects.append(
            {
                "name": escape_latex(item.name),
                "url": escape_latex(item.url),
                "href": escape_url(ensure_url(item.url)),
                "stack": escape_latex(stack),
                "bullets": [escape_latex(bullet) for bullet in bullets],
            }
        )

    return {
        "summary": "",  # Summary section removed from the template
        "experience": [
            {
                "company": escape_latex(item.company),
                "role": escape_latex(item.role),
                "location": escape_latex(item.location),
                "start": escape_latex(item.start),
                "end": escape_latex(item.end),
                "bullets": [escape_latex(bullet) for bullet in item.bullets],
            }
            for item in resume.experience
        ],
        "projects": projects,
        "education": [
            {
                "school": escape_latex(item.school),
                "degree": escape_latex(item.degree),
                "location": escape_latex(item.location),
                "year": escape_latex(item.year),
                "details": escape_latex(item.details),
            }
            for item in resume.education
        ],
        "awards": awards,
        "skills": [
            {"category": escape_latex(group.category), "items": escape_latex(group.items)}
            for group in resume.skills
        ],
    }


def _tex_search_path() -> str:
    extras = [
        "/Library/TeX/texbin",
        "/usr/local/texlive/2026/bin/universal-darwin",
        "/usr/local/texlive/2025/bin/universal-darwin",
        "/usr/local/texlive/2024/bin/universal-darwin",
        "/usr/local/texlive/2023/bin/universal-darwin",
    ]
    parts = [os.environ.get("PATH", "")]
    for extra in extras:
        if Path(extra).is_dir():
            parts.append(extra)
    return os.pathsep.join(parts)


def _which_tex(name: str) -> str | None:
    return shutil.which(name, path=_tex_search_path())


def ensure_latex_tools() -> None:
    missing = [name for name in ("latexmk", "xelatex") if _which_tex(name) is None]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"{joined} not found on PATH. Install MacTeX or BasicTeX and make sure "
            "latexmk and xelatex are available in the terminal."
        )


def compile_tex_with_log(tex_path: Path) -> tuple[Path, str]:
    """Compile a .tex file and return (pdf_path, log_text) before aux cleanup."""
    ensure_latex_tools()
    env = os.environ.copy()
    env["PATH"] = _tex_search_path()
    result = subprocess.run(
        [
            "latexmk",
            "-xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            tex_path.name,
        ],
        cwd=tex_path.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    pdf_path = tex_path.with_suffix(".pdf")
    log_path = tex_path.with_suffix(".log")
    log_text = ""
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if result.returncode != 0 or not pdf_path.is_file():
        details = result.stderr.strip() or result.stdout.strip()
        if log_text:
            details = log_text[-4000:]
        raise RuntimeError(f"LaTeX compile failed for {tex_path.name}:\n{details}")
    subprocess.run(
        ["latexmk", "-c", tex_path.name],
        cwd=tex_path.parent,
        env=env,
        capture_output=True,
        check=False,
    )
    return pdf_path, log_text


def compile_tex(tex_path: Path) -> Path:
    pdf_path, _log = compile_tex_with_log(tex_path)
    return pdf_path


def _pdf_page_count_from_bytes(data: bytes) -> int | None:
    """Parse page count from uncompressed PDF bytes; None if not found."""
    # Prefer /Count on a /Type /Pages node (order of keys varies).
    for pattern in (
        rb"/Type\s*/Pages\b[^>]*?/Count\s+(\d+)",
        rb"/Count\s+(\d+)[^>]*?/Type\s*/Pages\b",
    ):
        match = re.search(pattern, data, flags=re.DOTALL)
        if match:
            return int(match.group(1))
    # Fallback: count page objects (exclude /Type /Pages).
    pages = re.findall(rb"/Type\s*/Page(?!\s*s)\b", data)
    if pages:
        return len(pages)
    return None


def _pdf_inflated_streams(data: bytes) -> list[bytes]:
    """Inflate FlateDecode stream bodies (common in PDF 1.5+ object streams)."""
    inflated: list[bytes] = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, flags=re.DOTALL):
        raw = match.group(1)
        for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
            try:
                inflated.append(zlib.decompress(raw, wbits))
                break
            except zlib.error:
                continue
    return inflated


def pdf_page_count(path: Path) -> int:
    """Return the number of pages in a PDF using a stdlib parse (no extra deps)."""
    data = path.read_bytes()
    count = _pdf_page_count_from_bytes(data)
    if count is not None:
        return count
    # latexmk/pdfTeX often compresses the page tree into object streams.
    for stream in _pdf_inflated_streams(data):
        count = _pdf_page_count_from_bytes(stream)
        if count is not None:
            return count
    raise RuntimeError(f"Could not determine page count for {path}")


def _resolve_template(directory: Path, template_name: str) -> str:
    """Return template filename, falling back to default.tex.j2 when missing."""
    named = f"{template_name}.tex.j2"
    if (directory / named).is_file():
        return named
    fallback = "default.tex.j2"
    if template_name != "default" and (directory / fallback).is_file():
        return fallback
    return named


def render_documents(
    job: Job,
    contact: Contact,
    resume: TailoredResume,
    cover: CoverLetter | None,
    output_dir: Path,
    template_name: str = "default",
    compile_pdf: bool = True,
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_template = _resolve_template(RESUME_TEMPLATES_DIR, template_name)
    if not (RESUME_TEMPLATES_DIR / resume_template).is_file():
        raise FileNotFoundError(f"Missing resume template: {RESUME_TEMPLATES_DIR / resume_template}")

    context = {
        "contact": _contact_context(contact),
        "resume": _resume_context(resume),
        "cover": {
            "greeting": escape_latex(cover.greeting) if cover else "",
            "paragraphs": [escape_latex(paragraph) for paragraph in cover.paragraphs] if cover else [],
            "closing": escape_latex(cover.closing) if cover else "",
        },
        "job": {
            "company": escape_latex(job.company),
            "title": escape_latex(job.title),
        },
    }

    resume_tex = output_dir / "resume.tex"
    resume_tex.write_text(_jinja(RESUME_TEMPLATES_DIR).get_template(resume_template).render(**context), encoding="utf-8")
    cover_tex: Path | None = None
    if cover is not None:
        cover_template = _resolve_template(COVER_LETTER_TEMPLATES_DIR, template_name)
        if not (COVER_LETTER_TEMPLATES_DIR / cover_template).is_file():
            raise FileNotFoundError(
                f"Missing cover letter template: {COVER_LETTER_TEMPLATES_DIR / cover_template}"
            )
        cover_tex = output_dir / "cover_letter.tex"
        cover_tex.write_text(
            _jinja(COVER_LETTER_TEMPLATES_DIR).get_template(cover_template).render(**context),
            encoding="utf-8",
        )
    if compile_pdf:
        compile_tex(resume_tex)
        if cover_tex is not None:
            compile_tex(cover_tex)
    return resume_tex, cover_tex
