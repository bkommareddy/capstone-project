FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the ChromaDB index at container build time so the image serves
# requests immediately (MOCK_LLM defaults to "1" -> fully offline mock mode,
# no LLM API key or network call needed at request time).
RUN python ingest.py



CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
