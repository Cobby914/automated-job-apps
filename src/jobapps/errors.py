"""Pipeline failure categories. Recorded on errors so ~30 apps/day stay diagnosable."""

from __future__ import annotations

GENERATION_FAILURE = "generation_failure"
INVALID_PROVENANCE = "invalid_provenance"
SEMANTIC_REJECTION = "semantic_rejection"
PDF_OVERFLOW = "pdf_overflow"
LATEX_COMPILE_FAILURE = "latex_compile_failure"
PROVIDER_FAILURE = "provider_failure"
UPLOAD_FAILURE = "upload_failure"

FAILURE_CATEGORIES = frozenset(
    {
        GENERATION_FAILURE,
        INVALID_PROVENANCE,
        SEMANTIC_REJECTION,
        PDF_OVERFLOW,
        LATEX_COMPILE_FAILURE,
        PROVIDER_FAILURE,
        UPLOAD_FAILURE,
    }
)


class PipelineError(RuntimeError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category if category in FAILURE_CATEGORIES else GENERATION_FAILURE


def infer_failure_category(error: Exception) -> str:
    category = getattr(error, "category", None)
    if isinstance(category, str) and category in FAILURE_CATEGORIES:
        return category
    text = str(error).casefold()
    if "latex" in text or "xelatex" in text or "latexmk" in text:
        return LATEX_COMPILE_FAILURE
    if "source" in text or "provenance" in text or "numeric claim" in text:
        return INVALID_PROVENANCE
    if "overflow" in text or "1 page" in text or "overfull" in text:
        return PDF_OVERFLOW
    if "openai" in text or "anthropic" in text or "cursor" in text or "rate limit" in text:
        return PROVIDER_FAILURE
    if "notion" in text:
        return UPLOAD_FAILURE
    return GENERATION_FAILURE
