import io, logging
import numpy as np
import soundfile as sf
from scipy.signal import resample
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
startup_error = None

@app.on_event("startup")
def load_model():
    global model, startup_error
    try:
        from speechbrain.pretrained import EncoderClassifier
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/ecapa_model"
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        startup_error = str(e)
        logger.error(f"Failed to load model: {e}")

@app.get("/health")
def health():
    if model is not None:
        return {"status": "ok", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False, "error": startup_error}

@app.post("/embed")
async def embed(audio: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded", "detail": startup_error}, 503
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
