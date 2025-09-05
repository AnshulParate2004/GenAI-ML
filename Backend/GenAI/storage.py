# D:\GenAI-ML\Backend\GenAI\storage.py
import json
import os

# File where interview history is stored
INTERVIEW_FILE = r"D:\GenAI-ML\Backend\interview.json"


def load_history():
    """Load interview history from JSON file"""
    if os.path.exists(INTERVIEW_FILE):
        with open(INTERVIEW_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_history(history):
    """Save interview history to JSON file"""
    with open(INTERVIEW_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)


def add_main_question(history, question):
    """Start a new main question block"""
    block = {
        "question": question,
        "answer": None,
        "score": None,
        "audio_file": None,
        "clarifications": []
    }
    history.append(block)
    return history


def update_main_answer(history, answer, score, audio_file=None):
    """Update answer for the last main question"""
    if not history:
        return history
    history[-1]["answer"] = answer
    history[-1]["score"] = score
    if audio_file:
        history[-1]["audio_file"] = audio_file
    return history


def add_clarification(history, sub_q, answer=None, score=None, audio_file=None):
    """Add a clarification Q&A to the last main question"""
    if not history:
        return history
    clarification = {
        "sub_question": sub_q,
        "answer": answer,
        "score": score,
        "audio_file": audio_file
    }
    history[-1]["clarifications"].append(clarification)
    return history


def update_clarification_answer(history, answer, score, audio_file=None):
    """Update the last clarification answer"""
    if not history or not history[-1]["clarifications"]:
        return history
    history[-1]["clarifications"][-1]["answer"] = answer
    history[-1]["clarifications"][-1]["score"] = score
    if audio_file:
        history[-1]["clarifications"][-1]["audio_file"] = audio_file
    return history
