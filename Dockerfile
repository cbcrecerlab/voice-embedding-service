FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY embedding_service.py .
EXPOSE 8000
CMD ["uvicorn", "embedding_service:app", "--host", "0.0.0.0", "--port", "8000"]
