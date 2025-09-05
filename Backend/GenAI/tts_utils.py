# D:\GenAI-ML\Backend\GenAI\tts_utils.py
import pyttsx3
from GenAI.logger import log

def speak(text: str, output_path: str):
    """
    Convert given text to speech and save as a .wav file.
    Uses pyttsx3 (offline TTS engine).
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)   # speed
        engine.setProperty("volume", 0.9) # volume
        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)  # use first available voice
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        log("info", f"✅ TTS saved: {output_path}")
        return output_path
    except Exception as e:
        log("error", f"TTS generation failed: {e}")
        raise
