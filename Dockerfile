FROM node:24-bookworm-slim AS web
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --ignore-scripts
COPY web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /usr/local/bin/uv
RUN useradd -m -u 1000 user
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/home/user/.cache/huggingface \
    SCENETRACE_DATA=/home/user/data \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PATH=/opt/venv/bin:$PATH
COPY pyproject.toml uv.lock README.md LICENSE NOTICE.md ./
COPY backend/ backend/
RUN uv sync --frozen --no-dev
COPY scripts/download_models.py scripts/download_models.py
RUN python scripts/download_models.py && chown -R user:user /home/user
COPY --from=web /build/web/dist/ web/dist/
COPY assets/ assets/
COPY artifacts/released/ artifacts/released/
COPY artifacts/evaluation.json artifacts/training-history.json artifacts/
COPY docs/ docs/
COPY licenses/ licenses/
RUN mkdir -p /home/user/data && chown -R user:user /home/user/data
USER user
EXPOSE 7860
CMD ["uvicorn", "scenetrace.app:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log", "--proxy-headers", "--forwarded-allow-ips", "*"]
