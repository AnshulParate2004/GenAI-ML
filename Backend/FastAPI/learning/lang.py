# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# import torch
# import numpy as np
# import tempfile
# import os
# from pydub import AudioSegment
# import requests
# import traceback
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification  # <-- Added import here

# # ------------------------------
# # Init FastAPI app
# # ------------------------------
# app = FastAPI(title="Unified Voice Pipeline (Threaded)")

# # ------------------------------
# # Set up AssemblyAI API
# # ------------------------------
# ASSEMBLYAI_API_KEY = "e889fade24c44b288322796a23305744"
# ASSEMBLYAI_URL = "https://api.assemblyai.com/v2"

# headers = {
#     'authorization': ASSEMBLYAI_API_KEY
# }

# # ------------------------------
# # Load Emotion Model
# # ------------------------------
# EMOTION_PATH = r"D:/GenAI-ML/Backend/Voice/Model/voiceemotionmodel"
# emotion_processor = Wav2Vec2Processor.from_pretrained(EMOTION_PATH)
# emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_PATH)
# emotion_model.eval()

# CUSTOM_LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
# emotion_model.config.id2label = {i: lab for i, lab in enumerate(CUSTOM_LABELS)}

# # ------------------------------
# # Helper: Emotion prediction
# # ------------------------------
# def predict_emotion(audio_tensor, sampling_rate):
#     inputs = emotion_processor(audio_tensor, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         logits = emotion_model(**inputs).logits
#     predicted_id = torch.argmax(logits, dim=-1).item()
#     return emotion_model.config.id2label[predicted_id]

# # ------------------------------
# # Helper: AssemblyAI Transcription (Upload and Get Transcript)
# # ------------------------------
# def upload_audio(file_path):
#     """Upload audio file to AssemblyAI"""
#     upload_url = f"{ASSEMBLYAI_URL}/upload"
#     with open(file_path, 'rb') as f:
#         response = requests.post(upload_url, headers=headers, files={"file": f})
#         response_data = response.json()
#     if response.status_code == 200:
#         return response_data['upload_url']
#     else:
#         raise Exception(f"Error uploading audio: {response_data}")

# def request_transcription(upload_url):
#     """Request transcription from AssemblyAI"""
#     transcript_request = {
#         "audio_url": upload_url
#     }
#     response = requests.post(f"{ASSEMBLYAI_URL}/transcript", headers=headers, json=transcript_request)
#     response_data = response.json()
#     if response.status_code == 200:
#         return response_data['id']
#     else:
#         raise Exception(f"Error requesting transcription: {response_data}")

# def get_transcription_result(transcript_id):
#     """Poll AssemblyAI for the transcription result"""
#     polling_url = f"{ASSEMBLYAI_URL}/transcript/{transcript_id}"
#     while True:
#         response = requests.get(polling_url, headers=headers)
#         response_data = response.json()
#         if response.status_code == 200:
#             if response_data['status'] == 'completed':
#                 return response_data['text']
#             elif response_data['status'] == 'failed':
#                 raise Exception(f"Transcription failed: {response_data}")
#         time.sleep(5)  # Wait 5 seconds before polling again

# # ------------------------------
# # Worker for chunk
# # ------------------------------
# def process_chunk(samples, start_ms, end_ms, file_path):
#     try:
#         waveform = torch.tensor(samples)

#         # Emotion
#         emotion = predict_emotion(waveform, 16000)

#         return {
#             "start": round(start_ms / 1000, 2),
#             "end": round(end_ms / 1000, 2),
#             "emotion": emotion
#         }
#     except Exception as e:
#         return {"start": start_ms, "end": end_ms, "error": str(e)}

# # ------------------------------
# # Root
# # ------------------------------
# @app.get("/")
# def root():
#     return {"status": "Unified voice pipeline running ✅ (threaded)"}

# # ------------------------------
# # Unified Endpoint: Translation + Emotion (Parallel per chunk)
# # ------------------------------
# @app.post("/process-audio/")
# async def process_audio(file: UploadFile = File(...)):
#     try:
#         # Save uploaded file
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#             tmp.write(await file.read())
#             tmp_path = tmp.name

#         # Convert to standard format
#         audio = AudioSegment.from_file(tmp_path).set_frame_rate(16000).set_channels(1)
#         duration_ms = len(audio)

#         # Upload to AssemblyAI and get transcription result
#         upload_url = upload_audio(tmp_path)
#         transcript_id = request_transcription(upload_url)
#         transcription = get_transcription_result(transcript_id)

#         # Prepare to process chunks
#         futures = []
#         results = []

#         with ThreadPoolExecutor(max_workers=4) as executor:  # adjust workers based on CPU/GPU
#             for start_ms in range(0, duration_ms, 4000):  # Processing 4-second chunks
#                 end_ms = min(start_ms + 4000, duration_ms)
#                 chunk = audio[start_ms:end_ms]
#                 samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0

#                 if len(samples) < 8000:  # skip too short chunks
#                     continue

#                 # Process each chunk for emotion detection only
#                 futures.append(executor.submit(process_chunk, samples, start_ms, end_ms, tmp_path))

#             # Gather results as they complete
#             for f in as_completed(futures):
#                 results.append(f.result())

#         # Sort results by start time (threads might finish out of order)
#         results = sorted(results, key=lambda x: x["start"])

#         os.remove(tmp_path)

#         # Return combined results with transcription at the top and emotion per chunk
#         return JSONResponse(content={"transcription": transcription, "chunks": results})

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})











# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# import tempfile
# import os
# from pydub import AudioSegment
# import requests
# import traceback
# import time

# # ------------------------------
# # Init FastAPI app
# # ------------------------------
# app = FastAPI(title="MP4 → WAV → AssemblyAI")

# # ------------------------------
# # AssemblyAI Config
# # ------------------------------
# ASSEMBLYAI_API_KEY = "e889fade24c44b288322796a23305744"  # put your real key
# ASSEMBLYAI_URL = "https://api.assemblyai.com/v2"

# headers = {
#     "authorization": ASSEMBLYAI_API_KEY
# }

# # ------------------------------
# # Helper: Upload to AssemblyAI
# # ------------------------------
# def upload_audio(file_path):
#     """Upload audio file to AssemblyAI"""
#     upload_url = f"{ASSEMBLYAI_URL}/upload"
#     with open(file_path, "rb") as f:
#         response = requests.post(upload_url, headers=headers, data=f)
#         response_data = response.json()
#     if response.status_code == 200:
#         return response_data["upload_url"]
#     else:
#         raise Exception(f"Upload error: {response_data}")

# def request_transcription(upload_url):
#     """Ask AssemblyAI to transcribe"""
#     payload = {"audio_url": upload_url}
#     response = requests.post(f"{ASSEMBLYAI_URL}/transcript", headers=headers, json=payload)
#     response_data = response.json()
#     if response.status_code == 200:
#         return response_data["id"]
#     else:
#         raise Exception(f"Transcription request error: {response_data}")

# def get_transcription_result(transcript_id):
#     """Poll until transcription is ready"""
#     polling_url = f"{ASSEMBLYAI_URL}/transcript/{transcript_id}"
#     while True:
#         response = requests.get(polling_url, headers=headers)
#         data = response.json()
#         if response.status_code == 200:
#             if data["status"] == "completed":
#                 return data["text"]
#             elif data["status"] == "failed":
#                 raise Exception(f"Transcription failed: {data}")
#         time.sleep(5)

# # ------------------------------
# # Endpoint: Upload MP4 → WAV → AssemblyAI
# # ------------------------------
# @app.post("/transcribe/")
# async def transcribe(file: UploadFile = File(...)):
#     try:
#         # Save MP4 temporarily
#         with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
#             tmp.write(await file.read())
#             mp4_path = tmp.name

#         # Convert MP4 → WAV using pydub (ffmpeg required on system)
#         wav_path = mp4_path.replace(".mp4", ".wav")
#         audio = AudioSegment.from_file(mp4_path, format="mp4")
#         audio = audio.set_frame_rate(16000).set_channels(1)  # standardize
#         audio.export(wav_path, format="wav")

#         # Send to AssemblyAI
#         upload_url = upload_audio(wav_path)
#         transcript_id = request_transcription(upload_url)
#         transcription = get_transcription_result(transcript_id)

#         # Cleanup
#         os.remove(mp4_path)
#         os.remove(wav_path)

#         return JSONResponse(content={"transcription": transcription})

#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": str(e), "trace": traceback.format_exc()}
#         )













# ------------------------------
# Voice MOdel + translation (AssemblyAI)
# ------------------------------

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import numpy as np
import tempfile
import os
import subprocess
from pydub import AudioSegment
import aiohttp
import asyncio
import traceback
import time
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import logging

# ------------------------------
# Init FastAPI app and logging
# ------------------------------
app = FastAPI(title="Sequential MP4/WAV Voice Pipeline")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# AssemblyAI Config
# ------------------------------
ASSEMBLYAI_API_KEY = "e889fade24c44b288322796a23305744"
ASSEMBLYAI_URL = "https://api.assemblyai.com/v2"

headers = {
    "authorization": ASSEMBLYAI_API_KEY
}

# ------------------------------
# Load Emotion Model
# ------------------------------
EMOTION_PATH = r"D:/GenAI-ML/Backend/Voice/Model/voiceemotionmodel"
emotion_processor = Wav2Vec2Processor.from_pretrained(EMOTION_PATH)
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model.to(device)
emotion_model.eval()

CUSTOM_LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
emotion_model.config.id2label = {i: lab for i, lab in enumerate(CUSTOM_LABELS)}

# ------------------------------
# Helper: Emotion prediction
# ------------------------------
def predict_emotion(audio_tensor, sampling_rate=16000):
    logger.info("Predicting emotion for chunk")
    start_time = time.time()
    inputs = emotion_processor(audio_tensor, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    logger.info(f"Emotion prediction completed in {time.time() - start_time} seconds")
    return emotion_model.config.id2label[predicted_id]

# ------------------------------
# Helper: AssemblyAI Transcription (async)
# ------------------------------
async def upload_audio_async(file_path, session):
    logger.info("Uploading audio to AssemblyAI")
    start_time = time.time()
    upload_url = f"{ASSEMBLYAI_URL}/upload"
    with open(file_path, 'rb') as f:
        async with session.post(upload_url, headers=headers, data=f) as response:
            response_data = await response.json()
    logger.info(f"Upload completed in {time.time() - start_time} seconds")
    if response.status == 200:
        return response_data['upload_url']
    raise Exception(f"Error uploading audio: {response_data}")

async def request_transcription_async(upload_url, session):
    logger.info("Requesting transcription")
    start_time = time.time()
    transcript_request = {"audio_url": upload_url}
    async with session.post(f"{ASSEMBLYAI_URL}/transcript", headers=headers, json=transcript_request) as response:
        response_data = await response.json()
    logger.info(f"Transcription request completed in {time.time() - start_time} seconds")
    if response.status == 200:
        return response_data['id']
    raise Exception(f"Error requesting transcription: {response_data}")

async def get_transcription_result_async(transcript_id, session):
    logger.info("Polling for transcription result")
    start_time = time.time()
    polling_url = f"{ASSEMBLYAI_URL}/transcript/{transcript_id}"
    wait_time = 2
    max_wait = 30
    while True:
        async with session.get(polling_url, headers=headers) as response:
            response_data = await response.json()
        if response.status == 200:
            if response_data['status'] == 'completed':
                logger.info(f"Transcription completed in {time.time() - start_time} seconds")
                return response_data['text']
            elif response_data['status'] == 'failed':
                raise Exception(f"Transcription failed: {response_data}")
        await asyncio.sleep(min(wait_time, max_wait))
        wait_time = min(wait_time * 1.5, max_wait)

# ------------------------------
# Helper: Convert MP4 to WAV using ffmpeg
# ------------------------------
def convert_to_wav(input_path, output_path):
    logger.info("Converting to WAV")
    start_time = time.time()
    subprocess.run([
        "ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", output_path, "-y"
    ], check=True, capture_output=True)
    logger.info(f"Conversion completed in {time.time() - start_time} seconds")

# ------------------------------
# Worker for chunk processing (sequential)
# ------------------------------
def process_chunk(samples, start_ms, end_ms):
    try:
        waveform = torch.tensor(samples)
        emotion = predict_emotion(waveform, 16000)
        return {
            "start": round(start_ms / 1000, 2),
            "end": round(end_ms / 1000, 2),
            "emotion": emotion
        }
    except Exception as e:
        return {"start": start_ms, "end": end_ms, "error": str(e)}

# ------------------------------
# Root
# ------------------------------
@app.get("/")
def root():
    return {"status": "Sequential MP4/WAV voice pipeline running ✅"}

# ------------------------------
# Unified Endpoint: MP4/WAV → Transcription + Emotion
# ------------------------------
@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    start_total = time.time()
    try:
        # Validate file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in [".mp4", ".wav"]:
            return JSONResponse(status_code=400, content={"error": "File must be MP4 or WAV"})

        # Save uploaded file
        logger.info(f"Processing file: {file.filename}")
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Convert to WAV (16kHz, mono)
        wav_path = tmp_path if file_extension == ".wav" else tmp_path.replace(".mp4", ".wav")
        if file_extension == ".mp4":
            convert_to_wav(tmp_path, wav_path)
        else:
            audio = AudioSegment.from_file(tmp_path, format="wav")
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_path, format="wav", codec="pcm_s16le")

        # Load audio for chunking
        audio = AudioSegment.from_file(wav_path)
        duration_ms = len(audio)

        # Async AssemblyAI transcription
        async with aiohttp.ClientSession() as session:
            upload_url = await upload_audio_async(wav_path, session)
            transcript_id = await request_transcription_async(upload_url, session)
            transcription = await get_transcription_result_async(transcript_id, session)

        # Process chunks for emotion detection sequentially
        results = []
        chunk_size_ms = 4000  # 4-second chunks
        logger.info("Starting sequential emotion detection")
        start_emotion = time.time()
        for start_ms in range(0, duration_ms, chunk_size_ms):
            end_ms = min(start_ms + chunk_size_ms, duration_ms)
            voice_model_chunk = audio[start_ms:end_ms]
            samples = np.array(voice_model_chunk.get_array_of_samples()).astype(np.float32) / 32768.0
            if len(samples) < 4000:  # Skip short chunks
                continue
            result = process_chunk(samples, start_ms, end_ms)
            results.append(result)
        logger.info(f"Emotion detection completed in {time.time() - start_emotion} seconds")

        # Sort results by start time (optional, since sequential processing preserves order)
        results = sorted(results, key=lambda x: x["start"])

        # Cleanup
        os.remove(tmp_path)
        if file_extension == ".mp4":
            os.remove(wav_path)

        logger.info(f"Total processing time: {time.time() - start_total} seconds")
        return JSONResponse(content={"transcription": transcription, "voice_model_chunk": results})

    except Exception as e:
        logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


