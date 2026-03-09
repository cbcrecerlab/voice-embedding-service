from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import soundfile as sf
import io
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    from speechbrain.pretrained import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/app/model_cache"
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    data, samplerate = sf.read(io.BytesIO(audio_bytes))
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    # Resample to 16kHz if needed
    if samplerate != 16000:
        from scipy.signal import resample
        num_samples = int(len(data) * 16000 / samplerate)
        data = resample(data, num_samples)
    
    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    embedding = model.encode_batch(waveform)
    
    return {"embedding": embedding.squeeze().tolist()}
