import os
from dotenv import load_dotenv

# Путь до .env от текущего файла
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")

USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
PROXY_URL = os.getenv("PROXY_URL", "")

def debug_settings():
    print(f"✅ Loaded settings from .env:")
    print(f"OPENAI_MODEL = {OPENAI_MODEL}")
    print(f"USE_PROXY = {USE_PROXY}")
    print(f"PROXY_URL = {PROXY_URL}")
