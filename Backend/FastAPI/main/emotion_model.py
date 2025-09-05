# emotion_model.py

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from config import EMOTION_PATH, CUSTOM_LABELS
import logging
import time

logger = logging.getLogger(__name__)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_processor = Wav2Vec2Processor.from_pretrained(EMOTION_PATH)
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_PATH)
emotion_model.to(device)
emotion_model.eval()
emotion_model.config.id2label = {i: lab for i, lab in enumerate(CUSTOM_LABELS)}

def predict_emotion(audio_tensor, sampling_rate=16000):
    try:
        logger.info("Predicting emotion for chunk")
        start_time = time.time()
        inputs = emotion_processor(audio_tensor, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = emotion_model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        logger.info(f"Emotion prediction completed in {time.time() - start_time} seconds")
        return emotion_model.config.id2label[predicted_id]
    except Exception as e:
        logger.error(f"Error predicting emotion: {e}")
        return None
