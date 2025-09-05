import os
import uuid
from GenAI.logger import log
from GenAI.tts_utils import speak   # converts text -> wav
from GenAI.audio_utils import listen   # converts wav -> text
from GenAI.ai_helper import analyze_answer
from GenAI.storage import load_history, save_history, INTERVIEW_FILE

BOT_Q_DIR = r"D:\GenAI-ML\Backend\BotQuestions"
os.makedirs(BOT_Q_DIR, exist_ok=True)

MAIN_QUESTIONS = [
    "Tell me about yourself.",
    "Why do you want to work at our company?",
    "What are your biggest strengths?",
    "What are your biggest weaknesses?",
    "Where do you see yourself in 5 years?"
]

class IntelligentBot:
    def __init__(self):
        self.history = load_history()
        self.q_index = 0

    def get_first_question(self):
        """Start interview with the first question"""
        if os.path.exists(INTERVIEW_FILE):
            os.remove(INTERVIEW_FILE)
        self.q_index = 0
        return self._make_question(MAIN_QUESTIONS[self.q_index])

    def process_answer(self, audio_path: str) -> str:
        """Handle user answer, analyze it, and generate next question or finish"""
        # ✅ Transcribe only once
        answer_text = listen(audio_path)
        current_question = MAIN_QUESTIONS[self.q_index]

        log("info", f"Transcript for Q{self.q_index+1}: {answer_text}")

        # ✅ Pass transcript text instead of audio file
        result = analyze_answer(
            history=self.history,
            question=current_question,
            answer=answer_text,
            clarifications=[]
        )

        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        elif not isinstance(result, dict):
            result = {}

        score = result.get("score", 0)
        next_question = result.get("sub_question")

        self.history.append({
            "question": current_question,
            "answer": answer_text,
            "score": score,
            "audio_file": audio_path
        })
        save_history(self.history)

        # If follow-up not provided, move to next main question
        if not next_question:
            self.q_index += 1
            if self.q_index < len(MAIN_QUESTIONS):
                next_question = MAIN_QUESTIONS[self.q_index]
            else:
                next_question = "Interview finished! Thank you."

        return self._make_question(next_question)

    def _make_question(self, text: str) -> str:
        """Convert text question into wav and return path"""
        log("info", f"🎤 Generating TTS for: {text}")
        bot_wav = os.path.join(BOT_Q_DIR, f"bot_{uuid.uuid4()}.wav")
        speak(text, bot_wav)
        log("info", f"✅ Bot question saved as WAV: {bot_wav}")
        return bot_wav
