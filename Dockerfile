# Build context is the workspace root (ai-studio/), because docqa needs the sibling package llm-client.
# docker-compose.yml sets this up:  docker compose --profile app up -d --build
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY packages/llm-client /app/packages/llm-client
COPY projects/a1-rag-docqa /app/projects/a1-rag-docqa

# Editable install keeps ROOT (prices.json, data/) pointing at /app/projects/a1-rag-docqa.
RUN pip install /app/packages/llm-client \
 && pip install -e "/app/projects/a1-rag-docqa[api]"

WORKDIR /app/projects/a1-rag-docqa
RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "docqa.main:app", "--host", "0.0.0.0", "--port", "8000"]
