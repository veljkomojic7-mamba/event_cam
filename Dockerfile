# Runtime image for the traffic analysis CLI. Torch's CUDA wheels bundle their own
# CUDA runtime, so the host needs only an NVIDIA driver plus the container GPU runtime.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    PATH=/app/.venv/bin:$PATH

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

ENTRYPOINT ["traffic-analysis"]
CMD ["--help"]
