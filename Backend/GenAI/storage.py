# D:\GenAI-ML\Backend\GenAI\storage.py
import json
import os

# File where interview history is stored
INTERVIEW_FILE = r"D:\GenAI-ML\Backend\interview_history.json"


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
