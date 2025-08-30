from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import numpy as np
import tempfile
import os
from pydub import AudioSegment
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------
# Init FastAPI app
# ------------------------------
app = FastAPI(title="Unified Voice Pipeline (Threaded)")

# ------------------------------
# Load Whisper (translation model)
# ------------------------------
WHISPER_PATH = r"D:\GenAI-ML\Backend\Voice\Model\WhisperModels"
whisper_processor = WhisperProcessor.from_pretrained(WHISPER_PATH)
whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper_model.to(device)

# ------------------------------
# Load Emotion Model
# ------------------------------
EMOTION_PATH = r"D:\GenAI-ML\Backend\Voice\Model\voiceemotionmodel"
emotion_processor = Wav2Vec2Processor.from_pretrained(EMOTION_PATH)
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_PATH)
emotion_model.eval()

CUSTOM_LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
emotion_model.config.id2label = {i: lab for i, lab in enumerate(CUSTOM_LABELS)}

# ------------------------------
# Helper: Emotion prediction
# ------------------------------
def predict_emotion(audio_tensor, sampling_rate):
    inputs = emotion_processor(audio_tensor, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    return emotion_model.config.id2label[predicted_id]

# ------------------------------
# Worker for chunk
# ------------------------------
def process_chunk(samples, start_ms, end_ms):
    try:
        waveform = torch.tensor(samples)

        # Emotion
        emotion = predict_emotion(waveform, 16000)

        # Translation
        input_features = whisper_processor(samples, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        predicted_ids = whisper_model.generate(
            input_features,
            forced_decoder_ids=whisper_processor.get_decoder_prompt_ids(language="en", task="translate")
        )
        translation = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return {
            "start": round(start_ms / 1000, 2),
            "end": round(end_ms / 1000, 2),
            "emotion": emotion,
            "translation": translation
        }
    except Exception as e:
        return {"start": start_ms, "end": end_ms, "error": str(e)}

# ------------------------------
# Root
# ------------------------------
@app.get("/")
def root():
    return {"status": "Unified voice pipeline running ✅ (threaded)"}

# ------------------------------
# Unified Endpoint: Translation + Emotion (Parallel per chunk)
# ------------------------------
@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Convert to standard format
        audio = AudioSegment.from_file(tmp_path).set_frame_rate(16000).set_channels(1)
        duration_ms = len(audio)

        futures = []
        results = []

        with ThreadPoolExecutor(max_workers=4) as executor:  # adjust workers based on CPU/GPU
            for start_ms in range(0, duration_ms, 4000):
                end_ms = min(start_ms + 4000, duration_ms)
                chunk = audio[start_ms:end_ms]
                samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0

                if len(samples) < 8000:  # skip too short
                    continue

                futures.append(executor.submit(process_chunk, samples, start_ms, end_ms))

            for f in as_completed(futures):
                results.append(f.result())

        # sort by start time (since threads may complete out of order)
        results = sorted(results, key=lambda x: x["start"])

        os.remove(tmp_path)
        return JSONResponse(content={"chunks": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})
