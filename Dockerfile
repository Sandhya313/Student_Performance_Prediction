# Use an official lightweight Python image compatible with scikit-learn 1.6.1
FROM python:3.11-slim

# Set environment variables to optimize Python runtime in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory inside the container
WORKDIR /app

# Install curl for healthcheck support
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files and model into the container
COPY main.py model.joblib ./

# Create a non-privileged user and adjust permissions for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose FastAPI port
EXPOSE 8000

# Healthcheck to monitor API liveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server binding to 0.0.0.0:8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
