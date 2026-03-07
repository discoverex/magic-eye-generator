import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

# LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# 허깅페이스
HF_TOKEN = os.getenv("HF_TOKEN")

# 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent

# GCP
BUCKET_NAME = os.getenv("BUCKET_NAME")
GCP_SERVICE_ACCOUNT_JSON = os.getenv("GCP_SERVICE_ACCOUNT_JSON")