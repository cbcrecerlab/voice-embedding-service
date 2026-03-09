import io
import numpy as np
import soundfile as sf
from scipy.signal import resample
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None

@app.on_event("startup")
def load_model():
    global model
    from speechbrain.pretrained import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/ecapa_model"
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/embed")
async def embed(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    data, sr = sf.read(io.BytesIO(audio_bytes))
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        num_samples = int(len(data) * 16000 / sr)
        data = resample(data, num_samples)
    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_batch(waveform)
    vec = embedding.squeeze().cpu().numpy().tolist()
    return {"embedding": vec, "dim": len(vec)}
