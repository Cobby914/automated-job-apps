FROM python:3.13-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JOBAPPS_ROOT=/app \
    JOBAPPS_IN_DOCKER=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-xetex \
    texlive-latex-recommended \
    texlive-fonts-recommended \
    latexmk \
    fontconfig \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY resume_templates ./resume_templates
COPY cover_letter_templates ./cover_letter_templates
COPY cover_letter_examples ./cover_letter_examples
COPY writing_samples ./writing_samples
COPY resume_additions ./resume_additions
COPY connections ./connections
COPY jobs/samples ./jobs/samples
COPY jobs/processed/.gitkeep ./jobs/processed/.gitkeep
COPY output/.gitkeep ./output/.gitkeep

RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "jobapps", "worker"]
