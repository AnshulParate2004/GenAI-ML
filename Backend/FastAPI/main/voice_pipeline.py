import os
import numpy as np
from pydub import AudioSegment
from audio_helpers import convert_to_wav, process_chunk
from transcription_helpers import (
    upload_audio_async,
    request_transcription_async,
    get_transcription_result_async,
)
from emotion_model import emotion_model, emotion_processor, device

async def process_voice(file_path: str, file_extension: str, session):
    # Convert file to wav
    wav_path = file_path if file_extension == ".wav" else file_path.replace(".mp4", ".wav")
    if file_extension == ".mp4":
        convert_to_wav(file_path, wav_path)
    else:
        audio = AudioSegment.from_file(file_path, format="wav")
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav", codec="pcm_s16le")

    audio = AudioSegment.from_file(wav_path)
    duration_ms = len(audio)

    # Transcription
    upload_url = await upload_audio_async(wav_path, session)
    transcript_id = await request_transcription_async(upload_url, session)
    transcription = await get_transcription_result_async(transcript_id, session)

    # Process chunks
    voice_results = []
    chunk_size_ms = 4000
    for start_ms in range(0, duration_ms, chunk_size_ms):
        end_ms = min(start_ms + chunk_size_ms, duration_ms)
        chunk = audio[start_ms:end_ms]
        samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0
        if len(samples) < 4000:
            continue
        result = process_chunk(samples, start_ms, end_ms, emotion_model, emotion_processor, device)
        voice_results.append(result)

    # cleanup intermediate wav
    if file_extension == ".mp4" and os.path.exists(wav_path):
        os.remove(wav_path)

    return transcription, voice_results
