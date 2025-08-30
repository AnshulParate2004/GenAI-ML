import os
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import tempfile
import traceback

app = FastAPI(title="Voice Emotion API")

# --- Model Setup ---
MODEL_DIR = os.path.abspath(r"D:\GenAI-ML\Backend\Voice\Model\voiceemotionmodel")

processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# 🔥 Define your real emotion labels (order must match training dataset)
CUSTOM_LABELS = ["angry: 0.08", "calm: 0.72", "disgust: 0.05", "fearful: 0.18", "happy: 0.64", "neutral: 0.81", "sad: 0.10", "surprised: 0.22"]

# Overwrite the model config with correct labels
model.config.id2label = {i: lab for i, lab in enumerate(CUSTOM_LABELS)}
model.config.label2id = {lab: i for i, lab in enumerate(CUSTOM_LABELS)}

# Use labels from the updated model config
id2label = model.config.id2label
label2id = model.config.label2id

# --- Prediction ---
def predict_emotion(audio_tensor, sampling_rate):
    inputs = processor(
        audio_tensor,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    return id2label[predicted_id]

# --- API Endpoint ---
@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file to temp path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load audio
        audio = AudioSegment.from_file(tmp_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Ensure mono, 16kHz
        duration_ms = len(audio)

        results = []

        for start_ms in range(0, duration_ms, 4000):  # split into 4s chunks
            end_ms = min(start_ms + 4000, duration_ms)
            chunk = audio[start_ms:end_ms]

            # Convert to numpy
            samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0

            # Skip chunks shorter than 0.5 sec (8000 samples)
            if len(samples) < 8000:
                continue

            waveform = torch.tensor(samples)

            # Emotion prediction
            emotion = predict_emotion(waveform, 16000)

            results.append({
                "start": round(start_ms / 1000, 2),
                "end": round(end_ms / 1000, 2),
                "emotion": emotion
            })

        return JSONResponse(content={"chunks": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "trace": traceback.format_exc()
        })