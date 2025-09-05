# D:\GenAI-ML\Backend\GenAI\client_config.py
import os
from dotenv import load_dotenv
from groq import Groq

# Load .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

# Initialize Groq client
client = Groq(api_key=api_key)
