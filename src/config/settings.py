import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

# LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 허깅페이스
HF_TOKEN = os.getenv("HF_TOKEN")

# 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# GCP
IMAGE_BUCKET_NAME = os.getenv("IMAGE_BUCKET_NAME")
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
GCP_SERVICE_ACCOUNT_JSON = os.getenv("GCP_SERVICE_ACCOUNT_JSON")