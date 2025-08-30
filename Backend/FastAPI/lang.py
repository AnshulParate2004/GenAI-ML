# from fastapi import FastAPI, File, UploadFile
# import os
# import requests
# import time

# # FastAPI app instance
# app = FastAPI()

# # Your AssemblyAI API Key
# API_KEY = "e889fade24c44b288322796a23305744"

# # API URL for AssemblyAI
# ASSEMBLYAI_API_URL = "https://api.assemblyai.com/v2"

# headers = {
#     'authorization': API_KEY
# }

# # Upload audio file to AssemblyAI and get the transcript
# @app.post("/transcribe/")
# async def transcribe_audio(file: UploadFile = File(...)):
#     # Save uploaded audio file
#     file_path = f"temp_{file.filename}"
#     with open(file_path, "wb") as buffer:
#         buffer.write(await file.read())

#     # Upload audio file to AssemblyAI
#     upload_url = upload_audio(file_path)
    
#     # Request transcription
#     transcript_id = request_transcription(upload_url)
    
#     # Poll for the transcription result
#     result = get_transcription_result(transcript_id)
    
#     # Delete the temporary file
#     os.remove(file_path)
    
#     # Return the transcription result
#     return result


# def upload_audio(file_path: str):
#     """Upload the audio file to AssemblyAI"""
#     upload_url = f"{ASSEMBLYAI_API_URL}/upload"
    
#     with open(file_path, 'rb') as f:
#         response = requests.post(upload_url, headers=headers, files={"file": f})
#         response_data = response.json()

#     if response.status_code == 200:
#         return response_data['upload_url']
#     else:
#         return {"error": "Error uploading audio file", "details": response_data}

# def request_transcription(upload_url: str):
#     """Request transcription from AssemblyAI"""
#     transcript_request = {
#         "audio_url": upload_url
#     }

#     response = requests.post(f"{ASSEMBLYAI_API_URL}/transcript", headers=headers, json=transcript_request)
#     response_data = response.json()

#     if response.status_code == 200:
#         return response_data['id']
#     else:
#         return {"error": "Error requesting transcription", "details": response_data}

# def get_transcription_result(transcript_id: str):
#     """Poll AssemblyAI for the transcription result"""
#     polling_url = f"{ASSEMBLYAI_API_URL}/transcript/{transcript_id}"
    
#     while True:
#         response = requests.get(polling_url, headers=headers)
#         response_data = response.json()
        
#         if response.status_code == 200:
#             if response_data['status'] == 'completed':
#                 return {"transcription": response_data['text']}
#             elif response_data['status'] == 'failed':
#                 return {"error": "Transcription failed", "details": response_data}
        
#         time.sleep(5)  # Wait 5 seconds before polling again






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

#         # Transcription using AssemblyAI
#         upload_url = upload_audio(file_path)
#         transcript_id = request_transcription(upload_url)
#         transcription = get_transcription_result(transcript_id)

#         return {
#             "start": round(start_ms / 1000, 2),
#             "end": round(end_ms / 1000, 2),
#             "emotion": emotion,
#             "transcription": transcription
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

#         futures = []
#         results = []

#         with ThreadPoolExecutor(max_workers=4) as executor:  # adjust workers based on CPU/GPU
#             for start_ms in range(0, duration_ms, 4000):
#                 end_ms = min(start_ms + 4000, duration_ms)
#                 chunk = audio[start_ms:end_ms]
#                 samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0

#                 if len(samples) < 8000:  # skip too short
#                     continue

#                 futures.append(executor.submit(process_chunk, samples, start_ms, end_ms, tmp_path))

#             for f in as_completed(futures):
#                 results.append(f.result())

#         # sort by start time (since threads may complete out of order)
#         results = sorted(results, key=lambda x: x["start"])

#         os.remove(tmp_path)
#         return JSONResponse(content={"chunks": results})

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})




from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import numpy as np
import tempfile
import os
from pydub import AudioSegment
import requests
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# ------------------------------
# Init FastAPI app
# ------------------------------
app = FastAPI(title="Unified Voice Pipeline (Threaded)")

# ------------------------------
# Set up AssemblyAI API
# ------------------------------
ASSEMBLYAI_API_KEY = "e889fade24c44b288322796a23305744"
ASSEMBLYAI_URL = "https://api.assemblyai.com/v2"

headers = {
    'authorization': ASSEMBLYAI_API_KEY
}

# ------------------------------
# Load Emotion Model
# ------------------------------
EMOTION_PATH = r"D:/GenAI-ML/Backend/Voice/Model/voiceemotionmodel"
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
# Helper: AssemblyAI Transcription (Upload and Get Transcript)
# ------------------------------
def upload_audio(file_path):
    """Upload audio file to AssemblyAI"""
    upload_url = f"{ASSEMBLYAI_URL}/upload"
    with open(file_path, 'rb') as f:
        response = requests.post(upload_url, headers=headers, files={"file": f})
        response_data = response.json()
    if response.status_code == 200:
        return response_data['upload_url']
    else:
        raise Exception(f"Error uploading audio: {response_data}")

def request_transcription(upload_url):
    """Request transcription from AssemblyAI"""
    transcript_request = {
        "audio_url": upload_url
    }
    response = requests.post(f"{ASSEMBLYAI_URL}/transcript", headers=headers, json=transcript_request)
    response_data = response.json()
    if response.status_code == 200:
        return response_data['id']
    else:
        raise Exception(f"Error requesting transcription: {response_data}")

def get_transcription_result(transcript_id):
    """Poll AssemblyAI for the transcription result"""
    polling_url = f"{ASSEMBLYAI_URL}/transcript/{transcript_id}"
    while True:
        response = requests.get(polling_url, headers=headers)
        response_data = response.json()
        if response.status_code == 200:
            if response_data['status'] == 'completed':
                return response_data['text']
            elif response_data['status'] == 'failed':
                raise Exception(f"Transcription failed: {response_data}")
        time.sleep(5)  # Wait 5 seconds before polling again

# ------------------------------
# Worker for chunk
# ------------------------------
def process_chunk(samples, start_ms, end_ms, file_path):
    try:
        waveform = torch.tensor(samples)

        # Emotion detection
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
    return {"status": "Unified voice pipeline running ✅ (threaded)"}

# ------------------------------
# Unified Endpoint: Emotion + Full Audio Transcription (Parallel per chunk)
# ------------------------------
@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Convert to standard format (16kHz, mono)
        audio = AudioSegment.from_file(tmp_path).set_frame_rate(16000).set_channels(1)
        duration_ms = len(audio)

        # Upload to AssemblyAI and get transcription result
        upload_url = upload_audio(tmp_path)
        transcript_id = request_transcription(upload_url)
        transcription = get_transcription_result(transcript_id)

        futures = []
        results = []

        with ThreadPoolExecutor(max_workers=4) as executor:  # adjust workers based on CPU/GPU
            for start_ms in range(0, duration_ms, 4000):  # Process chunks in 4-second intervals
                end_ms = min(start_ms + 4000, duration_ms)
                chunk = audio[start_ms:end_ms]
                samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0

                if len(samples) < 8000:  # skip too short
                    continue

                futures.append(executor.submit(process_chunk, samples, start_ms, end_ms, tmp_path))

            for f in as_completed(futures):
                results.append(f.result())

        # sort by start time (since threads may complete out of order)
        results = sorted(results, key=lambda x: x["start"])

        os.remove(tmp_path)
        
        # Return combined results with transcription and emotion
        return JSONResponse(content={"transcription": transcription, "chunks": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})
