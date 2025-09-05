from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
import time
import numpy as np
from pydub import AudioSegment   # 👈 ADD THIS
import logging
import asyncio
import aiohttp

from audio_helpers import convert_to_wav, process_chunk
from transcription_helpers import (
    upload_audio_async,
    request_transcription_async,
    get_transcription_result_async,
)
from emotion_model import emotion_model, emotion_processor, device

app = FastAPI(title="Sequential MP4/WAV Voice Pipeline")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
def root():
    return {"status": "Sequential MP4/WAV voice pipeline running ✅"}

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    start_total = time.time()
    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in [".mp4", ".wav"]:
            return JSONResponse(status_code=400, content={"error": "File must be MP4 or WAV"})

        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Convert to WAV
        wav_path = tmp_path if file_extension == ".wav" else tmp_path.replace(".mp4", ".wav")
        if file_extension == ".mp4":
            convert_to_wav(tmp_path, wav_path)
        else:
            audio = AudioSegment.from_file(tmp_path, format="wav")
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_path, format="wav", codec="pcm_s16le")

        audio = AudioSegment.from_file(wav_path)
        duration_ms = len(audio)

        # Async AssemblyAI transcription
        async with aiohttp.ClientSession() as session:
            upload_url = await upload_audio_async(wav_path, session)
            transcript_id = await request_transcription_async(upload_url, session)
            transcription = await get_transcription_result_async(transcript_id, session)

        # Process chunks for emotion detection sequentially
        results = []
        chunk_size_ms = 4000
        for start_ms in range(0, duration_ms, chunk_size_ms):
            end_ms = min(start_ms + chunk_size_ms, duration_ms)
            voice_model_chunk = audio[start_ms:end_ms]
            samples = np.array(voice_model_chunk.get_array_of_samples()).astype(np.float32) / 32768.0
            if len(samples) < 4000:  # Skip short chunks
                continue
            result = process_chunk(samples, start_ms, end_ms, emotion_model, emotion_processor, device)
            results.append(result)

        # Cleanup
        os.remove(tmp_path)
        if file_extension == ".mp4":
            os.remove(wav_path)

        return JSONResponse(content={"transcription": transcription, "voice_model_chunk": results})

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
