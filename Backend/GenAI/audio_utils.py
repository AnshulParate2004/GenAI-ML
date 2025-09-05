import sounddevice as sd
import wave
from GenAI.logger import log
import os
import assemblyai as aai
from dotenv import load_dotenv

# Load AssemblyAI API key
load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

def record_audio(audio_path, duration=15, samplerate=16000):
    """Record microphone audio and save as WAV"""
    log("info", "🎙️ Recording audio...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()

    with wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    log("info", f"✅ Audio saved: {audio_path}")
    return audio_path

def listen(audio_file: str) -> str:
    """Transcribe a given WAV file using AssemblyAI"""
    log("info", f"🎧 Transcribing audio file: {audio_file}")
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        log("error", f"❌ Transcription failed: {transcript.error}")
        return ""
    log("info", f"📝 Transcript: {transcript.text}")
    return transcript.text
