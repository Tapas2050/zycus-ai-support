from dataclasses import dataclass
from datetime import datetime, timezone
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: str = os.getenv("DATA_DIR", "data")
    knowledge_base_dir: str = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge-base")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")
    llm_model: str | None = os.getenv("LLM_MODEL")
    llm_base_url: str | None = os.getenv("LLM_BASE_URL")
    llm_cache_dir: str = os.getenv("LLM_CACHE_DIR", ".cache/llm")
    data_as_of: str | None = os.getenv("DATA_AS_OF")


settings = Settings()


def parse_as_of(value: str) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
