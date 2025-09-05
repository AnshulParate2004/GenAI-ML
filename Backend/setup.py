from setuptools import setup, find_packages

setup(
    name="GenAI",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv",
        "neo4j",
        "groq",
        "mem0",
        "qdrant-client",
        "watchdog",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "genai=GenAI.InteligentBot:main",  # assumes you define a main() in InteligentBot.py
        ]
    },
    python_requires=">=3.8",
)
