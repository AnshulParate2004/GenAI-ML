# import os
# import time
# import tempfile
# import logging
# import asyncio
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import numpy as np
# import cv2
# from PIL import Image
# from pydub import AudioSegment
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import aiohttp
# from tensorflow.keras.models import load_model

# # ------------------ IMPORT HELPERS ------------------
# from audio_helpers import convert_to_wav, process_chunk
# from transcription_helpers import (
#     upload_audio_async,
#     request_transcription_async,
#     get_transcription_result_async,
# )
# from emotion_model import emotion_model, emotion_processor, device

# # ------------------ LOGGING ------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ------------------ FASTAPI APP ------------------
# app = FastAPI(title="Unified Voice + Face Emotion Pipeline")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ------------------ LOAD FACE MODEL ------------------
# face_model = load_model(r"D:/GenAI-ML/Backend/Face/Model/Face_Emotion_Recognition.keras")
# face_model.load_weights(r"D:/GenAI-ML/Backend/Face/Model/Face_Emotion_Recognition_Weights.weights.h5")
# emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# # ------------------ HELPERS ------------------
# def predict_face_frame(image):
#     try:
#         image = image.convert("L")  # grayscale
#         image = image.resize((48, 48))
#         img_array = np.array(image) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = np.expand_dims(img_array, axis=-1)
#         prediction = face_model.predict(img_array)
#         idx = np.argmax(prediction)
#         emotion = emotion_labels[idx]
#         confidence = float(np.max(prediction)) * 100
#         return emotion, confidence
#     except Exception as e:
#         logger.error(f"Face processing error: {e}")
#         return "error", 0

# # ------------------ MAIN PIPELINE ------------------
# @app.post("/analyze/")
# async def analyze(file: UploadFile = File(...)):
#     try:
#         file_extension = os.path.splitext(file.filename)[1].lower()
#         if file_extension not in [".mp4", ".wav"]:
#             return JSONResponse(status_code=400, content={"error": "File must be MP4 or WAV"})

#         # Save file
#         with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
#             tmp.write(await file.read())
#             tmp_path = tmp.name

#         # ---------- VOICE PIPELINE ----------
#         wav_path = tmp_path if file_extension == ".wav" else tmp_path.replace(".mp4", ".wav")
#         if file_extension == ".mp4":
#             convert_to_wav(tmp_path, wav_path)
#         else:
#             audio = AudioSegment.from_file(tmp_path, format="wav")
#             audio = audio.set_frame_rate(16000).set_channels(1)
#             audio.export(wav_path, format="wav", codec="pcm_s16le")

#         audio = AudioSegment.from_file(wav_path)
#         duration_ms = len(audio)

#         async with aiohttp.ClientSession() as session:
#             upload_url = await upload_audio_async(wav_path, session)
#             transcript_id = await request_transcription_async(upload_url, session)
#             transcription = await get_transcription_result_async(transcript_id, session)

#         voice_results = []
#         chunk_size_ms = 4000
#         for start_ms in range(0, duration_ms, chunk_size_ms):
#             end_ms = min(start_ms + chunk_size_ms, duration_ms)
#             chunk = audio[start_ms:end_ms]
#             samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0
#             if len(samples) < 4000:
#                 continue
#             result = process_chunk(samples, start_ms, end_ms, emotion_model, emotion_processor, device)
#             voice_results.append(result)

#         # ---------- FACE PIPELINE ----------
#         face_results = []
#         if file_extension == ".mp4":
#             cap = cv2.VideoCapture(tmp_path)
#             if not cap.isOpened():
#                 raise Exception("Error opening video file")
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             frame_count = 0
#             frame_interval = int(fps)  # 1 fps sampling
#             futures = []
#             with ThreadPoolExecutor(max_workers=10) as executor:
#                 while True:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                     if frame_count % frame_interval == 0:
#                         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                         futures.append(executor.submit(predict_face_frame, img))
#                         start_time = frame_count / fps
#                         end_time = (frame_count + frame_interval) / fps
#                         face_results.append({"start": round(start_time, 2), "end": round(end_time, 2)})
#                     frame_count += 1
#                 for i, future in enumerate(as_completed(futures)):
#                     emotion, confidence = future.result()
#                     face_results[i]["emotion"] = emotion
#                     face_results[i]["confidence_of_model"] = confidence
#             cap.release()

#         # ---------- CLEANUP ----------
#         os.remove(tmp_path)
#         if os.path.exists(wav_path) and file_extension == ".mp4":
#             os.remove(wav_path)

#         # ---------- RESPONSE ----------
#         return JSONResponse(content={
#             "transcription": transcription,
#             "voice_model_chunk": voice_results,
#             "face_model_prediction": face_results
#         })

#     except Exception as e:
#         logger.error(f"Pipeline error: {e}")
#         return JSONResponse(status_code=500, content={"error": str(e)})

#     except asyncio.CancelledError as ce:
#         logger.error(f"Task cancelled: {ce}")
#         return {"error": "Request was canceled."}







import os
import tempfile
import logging
import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiohttp

from voice_pipeline import process_voice
from face_pipeline import process_faces

# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ FASTAPI APP ------------------
app = FastAPI(title="Unified Voice + Face Emotion Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in [".mp4", ".wav"]:
            return JSONResponse(
                status_code=400,
                content={"error": "File must be MP4 or WAV"}
            )

        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Run voice pipeline
        async with aiohttp.ClientSession() as session:
            transcription, voice_results = await process_voice(
                tmp_path, file_extension, session
            )

        # Run face pipeline (only for mp4)
        face_results = []
        if file_extension == ".mp4":
            face_results = process_faces(tmp_path)

        # ---------- MERGE RESULTS ----------
        merged_results = []
        for v_chunk in voice_results:
            start_v, end_v = v_chunk["start"], v_chunk["end"]

            # find face preds inside this voice chunk window
            face_emotions, face_confidences = [], []
            for f_chunk in face_results:
                if f_chunk["start"] >= start_v and f_chunk["end"] <= end_v:
                    face_emotions.append(f_chunk["emotion"])
                    face_confidences.append(str(f_chunk["confidence_of_model"]))

            merged_results.append({
                "start": start_v,
                "end": end_v,
                "emotion_by_voice_model": v_chunk["emotion"],
                "emotion_by_face_model": ",".join(face_emotions) if face_emotions else "",
                "confidence_of_face_model": ",".join(face_confidences) if face_confidences else ""
            })

        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return JSONResponse(content={
            "transcription": transcription,
            "model_chunk": merged_results
        })

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    except asyncio.CancelledError:
        return {"error": "Request was canceled."}
