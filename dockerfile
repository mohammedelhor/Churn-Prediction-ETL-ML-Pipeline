# syntax=docker/dockerfile:1
FROM python:3.11-slim

# pandas/pyarrow/scikit-learn sometimes need a compiler for wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (separate layer) so Docker caches pip install
# and only re-runs it when requirements.txt actually changes, not on every code edit
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project
COPY . .

EXPOSE 8000

# Default command: serve the FastAPI app.
# Overridden by docker-compose for the ETL service (see docker-compose.yaml)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
