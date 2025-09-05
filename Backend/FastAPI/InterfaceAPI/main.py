from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from GenAI.Intelligent_bot import IntelligentBot

# ---------------------------
# FastAPI app initialization
# ---------------------------
app = FastAPI()

# ✅ Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠ Replace "" with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

# ---------------------------
# Initialize bot + directories
# ---------------------------
bot = IntelligentBot()

UPLOAD_DIR = r"D:\GenAI-ML\Backend\VedioRecording"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------
# API Endpoints
# ---------------------------

@app.post("/start_interview")
async def start_interview():
    try:
        print(">>> /start_interview called")
        wav_path = bot.get_first_question()
        print(f">>> Got wav path: {wav_path}")
        return FileResponse(
            wav_path,
            media_type="audio/wav",
            filename=os.path.basename(wav_path)
        )
    except Exception as e:
        import traceback
        print(">>> ERROR in /start_interview:", e)
        traceback.print_exc()
        return {"error": str(e)}



@app.post("/submit_answer")
async def submit_answer(audio: UploadFile = File(...), video: UploadFile = File(...)):
    """Receive candidate's answer (audio+video), process, return next bot question"""

    # Save audio file

    audio_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    # Save video file
    video_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.mp4")
    with open(video_path, "wb") as f:
        f.write(await video.read())

    # Process the answer through IntelligentBot
    wav_path = bot.process_answer(audio_path)

    # Return the next bot question as WAV file
    return FileResponse(
        wav_path,
        media_type="audio/wav",
        filename=os.path.basename(wav_path)
    )