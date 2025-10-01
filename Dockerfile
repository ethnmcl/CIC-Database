FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /app/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=America/New_York

# Pre-download the model to reduce cold starts (optional)
ARG MODEL_NAME=BAAI/bge-small-en-v1.5
RUN python - <<'PY'\nfrom sentence_transformers import SentenceTransformer\nSentenceTransformer("${MODEL_NAME}")\nPY

EXPOSE 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
