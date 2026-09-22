FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a source edit does not re-resolve the whole RDKit
# wheel on every rebuild.
COPY pyproject.toml README.md ./
COPY src/sorbent/__init__.py src/sorbent/__init__.py
RUN pip install --no-cache-dir -e .

COPY src/ src/

# The pool forks children; running as root is unnecessary.
RUN useradd --create-home --uid 10001 sorbent && chown -R sorbent /app
USER sorbent

EXPOSE 8000

# One uvicorn worker on purpose. Parallelism here comes from the process pool,
# not from --workers, and the in-memory job store is not shared between
# uvicorn workers: with --workers 4 a status poll would hit a process that has
# never heard of the job. Swap in a shared JobStore before scaling this out.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "sorbent.main:app", "--host", "0.0.0.0", "--port", "8000"]
