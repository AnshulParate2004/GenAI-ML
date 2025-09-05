
def log(level: str, message: str):
    logos = {
        "info": "🟢 [GROQ INFO]",
        "warn": "🟡 [GROQ WARN]",
        "error": "🔴 [GROQ ERROR]",
        "ask": "❓ [GROQ ASK]",
        "brain": "🧠 [GROQ PLAN]",
        "bot": "🤖 [GROQ BOT]",
        "mem": "📚 [GROQ MEMORY]"
    }
    prefix = logos.get(level.lower(), "ℹ️ [GROQ]")
    print(f"{prefix} {message}")