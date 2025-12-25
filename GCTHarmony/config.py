import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda *args, **kwargs: None

def load_openai_key(api_key: str = None, env_file: str = ".env") -> str:
    if api_key:
        return api_key.strip()
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key.strip()
    p = Path(env_file)
    if p.exists():
        load_dotenv(dotenv_path=p)
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return key.strip()
    raise ValueError(
        "OpenAI API key not found. "
        "Set OPENAI_API_KEY, add a .env file, or pass it via `api_key`."
    )
