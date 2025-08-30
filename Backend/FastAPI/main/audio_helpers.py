# audio_helpers.py

import subprocess
import os
import numpy as np
from pydub import AudioSegment
import torch
from config import EMOTION_PATH
import logging

logger = logging.getLogger(__name__)

# Convert MP4 to WAV using ffmpeg
def convert_to_wav(input_path, output_path):
    logger.info("Converting to WAV")
    subprocess.run([
        "ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", output_path, "-y"
    ], check=True, capture_output=True)

# Process chunks for emotion detection
def process_chunk(samples, start_ms, end_ms, emotion_model, emotion_processor, device):
    try:
        waveform = torch.tensor(samples)
        emotion = predict_emotion(waveform, 16000, emotion_model, emotion_processor, device)
        return {
            "start": round(start_ms / 1000, 2),
            "end": round(end_ms / 1000, 2),
            "emotion": emotion
        }
    except Exception as e:
        return {"start": start_ms, "end": end_ms, "error": str(e)}

# Emotion prediction logic (import from emotion_model.py)
def predict_emotion(audio_tensor, sampling_rate, emotion_model, emotion_processor, device):
    inputs = emotion_processor(audio_tensor, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    return emotion_model.config.id2label[predicted_id]
