FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn torchaudio speechbrain torch
COPY embedding_service.py .
EXPOSE 8000
CMD ["uvicorn", "embedding_service:app", "--host", "0.0.0.0", "--port", "8000"]
