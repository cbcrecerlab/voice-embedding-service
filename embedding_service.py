from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch
import io
import tempfile

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        signal, sr = torchaudio.load(tmp.name)
        if sr != 16000:
            signal = torchaudio.transforms.Resample(sr, 16000)(signal)
        embedding = classifier.encode_batch(signal)
        vector = embedding.squeeze().tolist()
    return {"embedding": vector, "dimensions": len(vector)}
