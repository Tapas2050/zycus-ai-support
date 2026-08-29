import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app.llm_client import LLMClient
from app.triage_agent import TriageResult


llm = LLMClient()

result = llm.generate_json(
    system_prompt=(
        "You are a support ticket classifier. "
        "Return only the requested structured data."
    ),
    user_prompt=(
        "Classify this ticket:\n\n"
        "Subject: Dashboard is extremely slow\n\n"
        "Body: Our AnalyticsHub dashboard takes more than "
        "100 seconds to load."
    ),
    response_model=TriageResult,
    temperature=0,
)

print("\nRESULT TYPE:")
print(type(result))

print("\nRESULT:")
print(result)

print("\nAS JSON:")
print(result.model_dump_json(indent=2))