# transcription_helpers.py

import aiohttp
import asyncio
import time
from config import ASSEMBLYAI_API_KEY, ASSEMBLYAI_URL
import logging

logger = logging.getLogger(__name__)

headers = {
    "authorization": ASSEMBLYAI_API_KEY
}

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
