FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir speechbrain==0.5.16 fastapi uvicorn python-multipart soundfile
COPY embedding_service.py .
CMD ["uvicorn", "embedding_service:app", "--host", "0.0.0.0", "--port", "8000"]
