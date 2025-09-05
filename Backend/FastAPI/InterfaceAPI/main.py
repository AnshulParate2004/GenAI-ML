from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import uuid

# Import bot from GenAI
from GenAI.InteligentBot import IntelligentBot

app = FastAPI()
bot = IntelligentBot()

# Directory to save uploaded files
UPLOAD_DIR = r"D:\GenAI-ML\Backend\VedioRecording"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/start_interview")
async def start_interview():
    """Start interview and return first question as wav"""
    wav_file = bot.get_first_question()
    return FileResponse(
        path=wav_file,
        media_type="audio/wav",
        filename=os.path.basename(wav_file)
    )


@app.post("/submit_answer")
async def submit_answer(
    audio: UploadFile = File(...),
    video: UploadFile = File(...)
):
    """Submit answer (audio + video), return next bot question wav"""

    # Save audio file
    audio_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    # Save video file
    video_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.mp4")
    with open(video_path, "wb") as f:
        f.write(await video.read())

    # Process answer with IntelligentBot
    bot_question_wav = bot.process_answer(audio_path)

    # Return next bot question as WAV
    return FileResponse(
        path=bot_question_wav,
        media_type="audio/wav",
        filename=os.path.basename(bot_question_wav)
    )
